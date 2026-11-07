"""Run one prepared experiment on a GPU that vast-rental rents, checks and returns.

prepare freezes the current sources with the prepared runtime as a job in queue/. start checks that the job may run
today and runs it in the background as a systemd user service, with its output in logs/; run does the same in the
foreground. A run rents a checked machine, sends the runtime and the sources, trains and evaluates once, collects the
results, notes how the machine did, returns it and records the result. finish returns the machines of a job whose run
ended early and records it as failed. vast-rental returns every machine at the end of its rental length if nothing
else has.
"""

import argparse
import ast
import csv
import datetime as dt
from decimal import Decimal
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gpu_config import active_gpu, get_profile, profile_for_job, read_config, source_files
from tools.runtime_archives import archive_specs, deployment_names, encoding_cache_asset_path, UNPACK_XZ
from tools.parallel_ssh import stream_archive
from tools import r2, vast_backend

OLMES_SOURCES = tuple('evaluation/' + name for name in (
    'arc_macro.py', 'checkpoint.py', 'count_tokens.py', 'count_gold_tokens.py', 'gold_bpb.py', 'model.py', 'prepare_requests.py',
    'protocol.py', 'run_prepared.py', 'tokenizer_adapter.py', 'verify_requests.py',
    'encoding_cache.py', 'prepare_encodings.py',
    'pyproject.toml', 'uv.lock', 'patches/olmes-arc-bpb-macro.patch',
))
LEASE_SECONDS = 3600  # The rental length of jobs prepared before GPU tracks.
CLEANUP_SECONDS = 60  # A run ends this long before vast-rental's own deadline for the machine.
EXECUTION_SETUP_SECONDS = 180  # Post-transfer data setup and compilation estimate.
COLLECTION_SECONDS = 120  # Copying every result file back, the checkpoint included.
LABEL_PREFIX = 'autoresearch-'  # Rentals whose label starts with this belong to this loop.
BOOTSTRAP = 'gpu-bootstrap.tar.gz'
FAILED_COST_LIMIT = Decimal('0.50')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def job_path(identifier):
    if not re.fullmatch(r'[a-z0-9-]+', identifier):
        raise ValueError('Invalid experiment identifier.')
    return ROOT / 'queue' / identifier / 'job.json'


def save(job):
    path = job_path(job['id'])
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = json.loads(path.read_text()) if path.exists() else {}
        current.update(job)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(current, indent=2) + '\n')
        temporary.replace(path)


def load(identifier):
    return json.loads(job_path(identifier).read_text())


def training_seconds(job):
    if 'gpu_settings' in job:
        return job['gpu_settings']['training_seconds']
    if job.get('id'):
        archive_path = job_path(job['id']).parent / 'source.tar.gz'
        if archive_path.exists():
            with tarfile.open(archive_path, 'r:gz') as archive:
                tree = ast.parse(archive.extractfile('prepare.py').read())
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == 'TIME_BUDGET' for target in node.targets):
                    return ast.literal_eval(node.value)
    return profile_for_job(job).settings()['training_seconds']


def lease_seconds(job):
    return job.get('gpu_settings', {}).get('rental_seconds', LEASE_SECONDS)


def artifact_directory(job):
    if job.get('gpu_id'):
        return profile_for_job(job).artifacts_directory(ROOT)
    return ROOT / 'artifacts'


def r2_store(manifest):
    """Select the prepared GPU's archive delivery without changing another GPU's policy."""
    if not manifest.get('transport'):
        return None
    if profile_for_job(manifest).values.get('archive_delivery', 'auto') == 'ssh':
        return None
    return r2.settings()


def verify_archives(manifest, artifacts, failure):
    """Check every runtime archive against its frozen digest before a rental is requested.

    With R2, everything except the small bootstrap lives there; a local copy is needed only
    to upload an archive that R2 does not hold yet.
    """
    store = r2_store(manifest)
    sizes = {}
    for name, sha256 in archive_specs(manifest):
        path = artifacts / name
        if store is None or name == BOOTSTRAP:
            if not path.is_file() or digest(path) != sha256:
                raise RuntimeError(failure)
            sizes[name] = path.stat().st_size
        else:
            size = r2.stored(store, name, sha256)
            if size is None:
                if not path.is_file():
                    raise RuntimeError(failure)
                r2.upload(store, path, sha256, name)
                size = path.stat().st_size
            sizes[name] = size
    return sizes


def results_directory(job):
    directory = profile_for_job(job).results_directory(ROOT)
    if job.get('gpu_id') or directory.exists():
        return directory
    return ROOT / 'results'


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def decisions():
    found = {}
    for path in [ROOT / 'results/decisions.json', *ROOT.glob('gpus/*/results/decisions.json')]:
        if path.is_file():
            found.update({row['id']: row['status'] for row in json.loads(path.read_text())})
    return found


def measured(job, statuses):
    status = statuses.get(job['id'])
    return status in ('keep', 'discard') or (job.get('phase') == 'evaluated' and status != 'crash')


def failed_cost_today(now):
    """What this loop's failed attempts have cost so far on the UTC day of now, when the loop may rent again that day;
    otherwise raise.

    The loop rents one GPU at a time and measures one run a day. After failures it may try again while they have cost
    no more than FAILED_COST_LIMIT in all, as the vast-rental ledger records. Other projects' rentals do not count.
    """
    day = dt.datetime.fromtimestamp(now, dt.timezone.utc).date()
    jobs = {}
    for path in (ROOT / 'queue').glob('*/job.json'):
        job = json.loads(path.read_text())
        jobs[job.get('label')] = job
    statuses = decisions()
    entries = vast_backend.entries()
    returned = {row['instance']: row for row in entries if row.get('event') == 'returned'}
    total = Decimal('0')
    for row in entries:
        if row.get('event') != 'rented' or not str(row.get('label', '')).startswith(LABEL_PREFIX):
            continue
        back = returned.get(row['instance'])
        if back is None:
            raise RuntimeError(f"Instance {row['instance']} of this loop has not been returned.")
        if dt.datetime.fromisoformat(row['at']).astimezone(dt.timezone.utc).date() != day:
            continue
        job = jobs.get(row['label'])
        if job is not None and measured(job, statuses):
            raise RuntimeError('This loop already measured a run today.')
        dollars = back.get('dollars')
        if (isinstance(dollars, bool) or not isinstance(dollars, (int, float))
                or not math.isfinite(dollars) or dollars < 0):
            raise RuntimeError(f"The ledger holds no valid cost for instance {row['instance']}.")
        total += Decimal(str(dollars))
    if total > FAILED_COST_LIMIT:
        raise RuntimeError(f"Today's failed attempts have cost ${total}, more than ${FAILED_COST_LIMIT}.")
    return total


def check_ready(job):
    """Raise unless the job may start now; return what today's failed attempts have cost."""
    profile = profile_for_job(job)
    if os.environ.get('AUTORESEARCH_RENTAL_HOLD_' + profile.identifier.upper()) == '1':
        raise RuntimeError(f'{profile.identifier} rentals are on hold by user request.')
    if profile.identifier != get_profile().identifier:
        raise RuntimeError('Only the GPU selected in gpus.json may be rented.')
    if job['phase'] != 'prepared':
        raise RuntimeError('This job has already been started; finish it instead.')
    directory = job_path(job['id']).parent
    if digest(directory / 'runtime.json') != job['runtime_manifest_sha256']:
        raise RuntimeError('The prepared runtime manifest changed.')
    if digest(directory / 'source.tar.gz') != job['source_archive_sha256']:
        raise RuntimeError('The prepared experiment sources changed.')
    execution_budget(job)
    validate_environment()
    return failed_cost_today(time.time())


def validate_environment():
    if not os.environ.get('VAST_API_KEY'):
        raise RuntimeError('VAST_API_KEY is required in the environment.')
    with tempfile.TemporaryDirectory(prefix='autoresearch-key-check-') as temporary:
        key = Path(temporary) / 'key'
        key.write_text(os.environ['VAST_SSH_KEY'].rstrip() + '\n')
        key.chmod(0o600)
        return subprocess.check_output(['ssh-keygen', '-y', '-f', str(key)], text=True,
                                       stderr=subprocess.DEVNULL).strip()


def execution_budget(job):
    expected_eval = float(job['expected_eval_seconds'])
    if not math.isfinite(expected_eval) or expected_eval <= 0:
        raise ValueError('Reserve a positive finite duration for BPB evaluation.')
    required = training_seconds(job) + expected_eval + EXECUTION_SETUP_SECONDS + COLLECTION_SECONDS
    if required >= lease_seconds(job) - CLEANUP_SECONDS:
        raise ValueError('The prepared runtime estimate does not fit inside the rental.')
    return required


def require_execution_time(job):
    if job['expires_at'] - time.time() < execution_budget(job):
        raise TimeoutError('Insufficient time remains for training and the complete fixed evaluation.')


def prepare_job(commit, baseline, description, expected_eval_seconds=180):
    import prepare

    profile = active_gpu()
    settings = profile.settings()
    artifacts = profile.artifacts_directory(ROOT)

    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Pass the full commit from the local Git checkout.')
    from tools.package_runtime import verify_tokenizer
    tokenizer_hashes = verify_tokenizer()
    execution_budget(dict(expected_eval_seconds=expected_eval_seconds, gpu_settings=settings))
    manifest = json.loads((artifacts / 'runtime.json').read_text())
    if manifest.get('gpu_id') != profile.identifier:
        raise RuntimeError('Prepare the selected GPU track runtime before creating its job.')
    if manifest['tokenizer'] != tokenizer_hashes:
        raise RuntimeError('The tokenizer differs from the packaged BPB assets.')
    metric = manifest.get('evaluation_metric', 'val_bpb')
    sources = source_files(profile)
    sources += OLMES_SOURCES if metric == 'olmes_bpb' else ()
    hashes = {name: digest(ROOT / name) for name in sources}
    verify_archives(manifest, artifacts, 'A runtime archive is incomplete or changed.')
    identifier = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
    directory = job_path(identifier).parent
    directory.mkdir(parents=True)
    manifest['sources'] = hashes
    runtime_path = directory / 'runtime.json'
    runtime_path.write_text(json.dumps(manifest, indent=2) + '\n')
    source_dir = directory / 'source'
    source_dir.mkdir()
    for name in sources:
        (source_dir / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, source_dir / name)
    shutil.copy2(ROOT / 'tools/gpu_run.sh', source_dir / 'gpu_run.sh')
    with tarfile.open(directory / 'source.tar.gz', 'w:gz') as archive:
        for path in sorted(source_dir.iterdir()):
            archive.add(path, arcname=path.name)
    ledger = profile.results_directory(ROOT) / 'decisions.json'
    if not ledger.exists() and profile.identifier == 'h100':
        ledger = ROOT / 'results/decisions.json'
    records = json.loads(ledger.read_text()) if ledger.exists() else []
    baseline = baseline or not any(row['status'] == 'keep'
                                   and profile_for_job(load(row['id'])).identifier == profile.identifier
                                   for row in records)
    job = dict(id=identifier, label='autoresearch-' + profile.identifier + '-' + identifier,
               gpu_id=profile.identifier, gpu_name=profile.gpu_name, gpu_settings=settings,
               model_source=f'gpus/{profile.identifier}/train.py',
               commit=commit, baseline=baseline,
               description=description, phase='prepared', source_sha256=hashes,
               source_archive_sha256=digest(directory / 'source.tar.gz'),
               runtime_manifest_sha256=digest(runtime_path), max_hourly_cost=settings['max_hourly_cost'],
               expected_eval_seconds=expected_eval_seconds, metric=metric)
    save(job)
    print(json.dumps({'id': identifier, 'commit': commit, 'phase': job['phase']}, indent=2))


def requirements(profile):
    """The machine the job needs, as a vast-rental offer query."""
    values = profile.values
    return {name: {'gte': values[field]} for name, field in (
        ('gpu_ram', 'minimum_gpu_ram_mb'), ('cpu_ram', 'minimum_cpu_ram_mb'),
        ('cpu_cores_effective', 'minimum_cpu_cores'), ('inet_down', 'minimum_download_mbps'),
        ('reliability2', 'minimum_reliability'), ('cuda_max_good', 'cuda_minimum'))}


def ssh_command(job, key):
    """ssh to the job's machine through the configuration that vast-rental wrote, with the loop's key."""
    return ['ssh', '-F', str(job['ssh_config']), '-i', str(key), '-o', 'IdentitiesOnly=yes',
            '-o', 'IdentityAgent=none', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', job['ssh_host']]


def run_lock():
    """The lock that one run of this loop holds from its checks until its machines are back."""
    path = ROOT / 'results/operations/rental.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open('a')


def hold(lock):
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another run of this loop is in progress.') from None


def record_gpu_observation(remote, destination):
    observation = dict(observed_at=time.time())
    sample_deadline = time.monotonic() + 5
    try:
        try:
            result = remote(
                'nvidia-smi --query-gpu=name,memory.used,utilization.gpu,clocks.sm,clocks.max.sm,'
                'clocks.mem,power.draw,power.limit,temperature.gpu --format=csv,noheader,nounits',
                timeout=5, capture_output=True, text=True,
            )
        except (subprocess.SubprocessError, OSError) as error:
            remaining = sample_deadline - time.monotonic()
            if remaining <= 0:
                raise
            observation['extended_error_type'] = type(error).__name__
            result = remote(
                'nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader,nounits',
                timeout=remaining, capture_output=True, text=True,
            )
        rows = list(csv.reader(result.stdout.splitlines()))
        if len(rows) != 1 or len(rows[0]) not in (3, 9):
            raise ValueError('Expected one GPU observation.')
        name, memory, utilization = rows[0][:3]
        memory, utilization = float(memory), float(utilization)
        if not (math.isfinite(memory) and memory >= 0 and 0 <= utilization <= 100):
            raise ValueError('Invalid GPU observation.')
        observation.update(gpu_name=name.strip(), memory_mib=memory, utilization_percent=utilization)
        optional_fields = ('sm_clock_mhz', 'max_sm_clock_mhz', 'memory_clock_mhz',
                           'power_draw_w', 'power_limit_w', 'temperature_c')
        for index, field in enumerate(optional_fields, start=3):
            try:
                value = float(rows[0][index])
                value = value if math.isfinite(value) and value >= 0 else None
            except (IndexError, ValueError):
                value = None
            observation[field] = value
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        # Optional telemetry never changes the training lifecycle or decision.
        observation['error_type'] = type(error).__name__
    try:
        with (destination / 'gpu-observations.jsonl').open('a') as output:
            output.write(json.dumps(observation, allow_nan=False) + '\n')
    except OSError:
        print('The GPU observation could not be saved; retaining the original run.', flush=True)


def record_preparation_timing(remote, destination):
    observation = dict(observed_at=time.time(), files={},
                       time_resolution_seconds=1,
                       limitation='GPU file modification times approximate wrapper and preparation time, '
                       'including Python/uv startup, data checks/downloads and tokenizer preparation. '
                       'This is not a measurement of network transfer alone. The training log mtime '
                       'does not identify the start of training.')
    try:
        result = remote(
            'cd autoresearch && for timing_file in run.pid prepare.log run.log run.exit; do '
            'if test -f "$timing_file"; then stat --printf=\'%n\\t%Y\\t%s\\n\' "$timing_file"; fi; done',
            timeout=5, capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            name, modified, size = line.split('\t')
            if name not in ('run.pid', 'prepare.log', 'run.log', 'run.exit'):
                raise ValueError('Unexpected preparation timing file.')
            modified, size = int(modified), int(size)
            if modified < 0 or size < 0 or name in observation['files']:
                raise ValueError('Invalid preparation file metadata.')
            observation['files'][name] = dict(mtime_seconds=modified, bytes=size)
        try:
            with (destination / 'prepare.log').open('rb') as source:
                source.seek(0, 2)
                source.seek(max(0, source.tell() - 4096))
                tail = source.read()
        except OSError:
            tail = b''
        observation['prepare_log_completed_marker'] = b'Done! Ready to train.' in tail
        ready = re.search(rb'Data: (\d+)/(\d+) shards ready', tail)
        cached = re.search(rb'Data: all (\d+) shards already downloaded', tail)
        if ready:
            observation['ready_shards'], observation['requested_shards'] = map(int, ready.groups())
        elif cached:
            observation['ready_shards'] = observation['requested_shards'] = int(cached.group(1))
        files = observation['files']
        if observation['prepare_log_completed_marker'] and all(name in files for name in ('run.pid', 'prepare.log')):
            elapsed = files['prepare.log']['mtime_seconds'] - files['run.pid']['mtime_seconds']
            if elapsed < 0:
                raise ValueError('The preparation file times are out of order.')
            observation['approximate_preparation_seconds'] = elapsed
    except (subprocess.SubprocessError, OSError, TimeoutError, ValueError, TypeError) as error:
        # Optional timing never changes the training lifecycle or decision.
        observation['error_type'] = type(error).__name__
    try:
        (destination / 'preparation-timing.json').write_text(json.dumps(observation, indent=2, allow_nan=False) + '\n')
    except OSError:
        print('The preparation timing could not be saved; retaining the original run.', flush=True)


def execute(job, connection):
    require_execution_time(job)
    deadline = job['expires_at'] - 5
    observe_until = job['expires_at'] - COLLECTION_SECONDS
    destination = results_directory(job) / 'runs' / job['id']
    destination.mkdir(parents=True, exist_ok=True)

    def remote(command, timeout=30, **kwargs):
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError('The rental budget is exhausted.')
        return run(connection + [command], timeout=min(timeout, remaining), **kwargs)

    try:
        gpu = remote('nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits',
                     capture_output=True, text=True).stdout.strip().splitlines()
    except subprocess.CalledProcessError:
        job['failure_kind'] = 'machine'
        save(job)
        raise
    job['initial_gpu_observation'] = gpu
    save(job)
    profile = profile_for_job(job)
    pattern = profile.values.get('device_name_pattern', re.escape(profile.gpu_name))
    if (len(gpu) != 1 or not re.search(pattern, gpu[0])
            or float(gpu[0].rsplit(',', 1)[1]) > 512):
        job['failure_kind'] = 'machine'
        save(job)
        raise RuntimeError(f'The rental does not provide one idle {profile.gpu_name}.')
    remote('mkdir -p autoresearch')
    job['transfer_started_at'] = time.time()
    save(job)
    artifacts = artifact_directory(job)
    runtime_path = job_path(job['id']).parent / 'runtime.json'
    manifest = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    names = deployment_names(manifest)
    hashes = dict(archive_specs(manifest)) if manifest else {}
    hashes['source.tar.gz'] = job.get('source_archive_sha256')
    store = r2_store(manifest)
    interpreter = None
    if manifest.get('transport'):
        version = manifest['python_version']
        if not re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', version):
            raise ValueError('Invalid packaged Python version.')
        interpreter = f'.bootstrap/.python/cpython-{version}-linux-x86_64-gnu/bin/python3.12'
    transfers = []
    for path in [*(artifacts / name for name in names), job_path(job['id']).parent / 'source.tar.gz']:
        if path.name == BOOTSTRAP:
            remote('mkdir -p autoresearch/.bootstrap')
            unpack = 'tar -xzf - -C autoresearch/.bootstrap'
        elif path.suffix == '.xz':
            command = [interpreter]
            if manifest.get('transport', {}).get('format') == 'python-xz-mt':
                command += ['.bootstrap/runtime_unpack.py', 'extract']
            else:
                command += ['-c', UNPACK_XZ]
            unpack = 'cd autoresearch && ' + shlex.join(command)
        else:
            unpack = 'tar -xzf - -C autoresearch'
        if store is not None and path.parent == artifacts and path.name != BOOTSTRAP:
            # The rental downloads directly from R2; the processing server sends only small files.
            size = r2.stored(store, path.name, hashes[path.name])
            if size is None:
                raise RuntimeError('A runtime archive is missing from R2.')
            receipt = r2.fetch(connection, store, path.name, hashes[path.name], size, unpack, interpreter,
                               timeout=min(900, deadline - time.time()),
                               expires=max(60, int(job['expires_at'] - time.time())))
        else:
            receipt = stream_archive(connection, path, unpack, sha256=hashes.get(path.name) or digest(path),
                                     timeout=min(900, deadline - time.time()))
            if path.name == BOOTSTRAP and store is not None:
                remote('cat > autoresearch/.bootstrap/r2_fetch.py', input=r2.FETCHER.read_bytes())
        transfers.append(receipt)
        job['archive_transfers'] = transfers
        save(job)
        if receipt.get('cleanup_error'):
            remote('true', capture_output=True, text=True)
    job['transfer_completed_at'] = time.time()
    save(job)
    require_execution_time(job)
    job['launch_requested_at'] = time.time()
    save(job)
    try:
        environment = {'AUTORESEARCH_GPU': profile.identifier}
        if manifest.get('encoding_cache'):
            path = manifest['encoding_cache']['asset_path']
            if path != encoding_cache_asset_path(profile.identifier):
                raise ValueError('The encoding cache must belong to the prepared GPU track.')
            environment['OLMES_ENCODING_CACHE'] = path
        remote('cd autoresearch && export ' + shlex.join(f'{name}={value}' for name, value in environment.items())
               + ' && if mkdir .started 2>/dev/null; then '
               'nohup bash gpu_run.sh > bootstrap.log 2>&1 < /dev/null & echo $! > run.pid; fi')
    except (subprocess.SubprocessError, OSError):
        print('The launch acknowledgement was lost; observing the original job handle.', flush=True)
    job['phase'] = 'running'
    save(job)
    status = None
    while time.time() < observe_until:
        try:
            probe = remote('cd autoresearch && if test -f run.exit; then printf "exit "; cat run.exit; '
                           'elif test -f run.pid && kill -0 "$(cat run.pid)" 2>/dev/null; then echo running; '
                           'else echo missing; fi', capture_output=True, text=True).stdout.strip()
            if probe.startswith('exit '):
                status = int(probe.split()[1])
                job['completed_observed_at'] = time.time()
                break
            if probe == 'missing':
                raise RuntimeError('The remote training process handle is missing.')
            if probe == 'running':
                record_gpu_observation(remote, destination)
                print('Observed the same live training process.', flush=True)
            else:
                print('The observation was incomplete; rechecking the same process.', flush=True)
        except (subprocess.SubprocessError, OSError) as error:
            # A failed observation never starts another training process or rental.
            print('Observation failed; rechecking the same rental:', type(error).__name__, flush=True)
        time.sleep(min(30, max(0, observe_until - time.time())))
    artifacts = ('run.log', 'prepare.log', 'bootstrap.log')
    if job.get('metric') == 'olmes_bpb':
        artifacts += ('olmes.log', 'olmes-summary.json', 'olmes-metrics.jsonl',
                      'olmes-details.tar.gz', 'final-model.pt')

    failures = {}

    def copy(name):
        """Copy one file of the rental; a failed copy keeps the previous one."""
        path = destination / Path(name).name
        temporary = path.with_name(path.name + '.tmp')
        try:
            with temporary.open('wb') as output:
                # Only the rental's deadline bounds a copy, so a large checkpoint arrives whole.
                remote('cd autoresearch && cat ' + shlex.quote(name), timeout=math.inf,
                       stdout=output, stderr=subprocess.DEVNULL)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def collect(names):
        """Copy the files back until each equals the rental's own, three tries at most, and record the rest."""
        pending = list(names)
        for _ in range(3):
            for name in pending:
                try:
                    copy(name)
                    failures.pop(name, None)
                except (subprocess.SubprocessError, OSError, TimeoutError) as error:
                    failures[name] = type(error).__name__
            try:
                listing = remote('cd autoresearch && sha256sum -- ' + shlex.join(pending) + ' 2>/dev/null; true',
                                 capture_output=True, text=True).stdout
            except (subprocess.SubprocessError, OSError, TimeoutError):
                listing = None
            held = dict(line.split('  ', 1)[::-1] for line in (listing or '').splitlines() if '  ' in line)
            retry = []
            for name in pending:
                path = destination / Path(name).name
                if name in held:
                    if path.exists() and digest(path) == held[name]:
                        failures.pop(name, None)
                    else:
                        failures.setdefault(name, 'different')
                        retry.append(name)
                elif name in failures:
                    # Without a listing a failed copy is tried again; a listing without it means the rental lacks it.
                    if listing is None:
                        retry.append(name)
                    else:
                        failures[name] = 'absent'
            pending = retry
            if not pending:
                break
        for name in names:
            if name in failures:
                print(f'{name} was not collected: {failures[name]}.', flush=True)
        job['collection_errors'] = dict(failures)
        save(job)

    collect((*artifacts, 'run.exit'))
    try:
        text = (destination / 'run.exit').read_text().strip()
        collected_status = int(text) if re.fullmatch(r'[0-9]{1,3}', text) and int(text) <= 255 else None
    except (OSError, UnicodeError):
        collected_status = None
    if status is not None and collected_status is not None and status != collected_status:
        raise RuntimeError('Exit status changed between observation and collection.')
    if status is None and collected_status is not None:
        status = collected_status
        job['exit_code_observed_in_collection'] = True
        job['completed_observed_at'] = time.time()
        # The process is now terminal; refresh its final files without restarting it.
        collect(artifacts)
    record_preparation_timing(remote, destination)
    job['exit_code'] = status
    job['phase'] = 'evaluated' if status == 0 else 'failed'
    if status != 0:
        job.setdefault('failure_kind', 'unknown')
    job['collected_at'] = time.time()
    save(job)


def return_machines(job):
    """Return every machine rented under the job's label that the ledger has not seen returned; give their ids."""
    entries = vast_backend.entries()
    instances = sorted({row['instance'] for row in entries
                        if row.get('event') == 'rented' and row.get('label') == job['label']})
    returned = {row['instance'] for row in entries if row.get('event') == 'returned'}
    for instance in instances:
        if instance not in returned:
            vast_backend.command(['return', str(instance)], timeout=120)
    returned = {row['instance'] for row in vast_backend.entries() if row.get('event') == 'returned'}
    if not set(instances) <= returned:
        raise RuntimeError('vast-rental has not recorded every machine of this job as returned.')
    if instances:
        update = dict(id=job['id'], deleted_at=time.time())
        save(update)
        job.update(update)
    return instances


def finalize(job):
    """Note how the job's machine did, return it, then record the result of a run that had a machine."""
    if job.get('instance_id'):
        try:
            values = {name: job.get(name) for name in ('id', 'phase', 'exit_code', 'failure_kind')}
            vast_backend.command(['note', str(job['instance_id']),
                                  'autoresearch ' + job['id'], '--values', json.dumps(values)], timeout=30)
        except (OSError, subprocess.SubprocessError):
            # A note must never delay returning a paid machine.
            pass
    return_machines(job)
    current = load(job['id'])
    if current.get('instance_id') and current['phase'] in ('evaluated', 'failed'):
        # Recorded only after the machines are back; the recorder is idempotent.
        run([sys.executable, str(ROOT / 'tools/record_run.py'), current['id']])


def run_job(job):
    """Rent a checked machine through vast-rental and run the job on it once, returning the machine at the end."""
    with run_lock() as lock:
        hold(lock)
        check_ready(job)
        directory = job_path(job['id']).parent
        profile = profile_for_job(job)
        manifest = json.loads((directory / 'runtime.json').read_text())
        sizes = verify_archives(manifest, artifact_directory(job),
                                'A prepared runtime archive changed; no rental was requested.')
        required = execution_budget(job)
        now = time.time()
        job.update(phase='renting', rental_backend='vast-rental', requested_at=now, creation_requested_at=now,
                   expires_at=now + lease_seconds(job) - CLEANUP_SECONDS,
                   deployment_bytes=sum(sizes[name] for name in deployment_names(manifest))
                   + (directory / 'source.tar.gz').stat().st_size)
        save(job)
        try:
            with tempfile.TemporaryDirectory(prefix='autoresearch-ssh-') as temporary:
                key = Path(temporary) / 'key'
                key.write_text(os.environ['VAST_SSH_KEY'].rstrip() + '\n')
                key.chmod(0o600)
                while True:
                    failed_cost_today(time.time())
                    remaining = job['expires_at'] - time.time() - required
                    if remaining <= 0:
                        raise TimeoutError('No time remains for another checked machine and the complete run.')
                    response, attempts = vast_backend.rent(job, manifest, profile, key, requirements(profile),
                                                           remaining, directory)
                    job.setdefault('selection_attempts', []).extend(attempts)
                    if response.get('rented') is not None:
                        job.update(instance_id=response['rented'], provider_machine_id=response['machine'],
                                   hourly_cost=response['price'], phase='starting', ready_at=time.time(),
                                   ssh_config=response['ssh_config'], ssh_host=response['ssh_host'])
                        save(job)
                        execute(job, ssh_command(job, key))
                        break
                    save(job)
                    if response.get('failed', 0) == 0:
                        raise RuntimeError('vast-rental found no available eligible machine.')
        except BaseException as error:
            if job.get('instance_id') is None:
                # No machine ever ran the job, so it may start again later, such as when cheaper machines are free.
                job.update(phase='prepared', error_type=type(error).__name__)
            else:
                job.update(phase='failed', error_type=type(error).__name__)
                job.setdefault('failure_kind', 'unknown')
            try:
                with (directory / 'failure.log').open('a') as log:
                    log.write(type(error).__name__ + ': ' + str(error) + '\n')
                    for name in ('stdout', 'stderr'):
                        value = getattr(error, name, None)
                        if value:
                            if isinstance(value, bytes):
                                value = value.decode(errors='replace')
                            log.write(name + ':\n' + value + '\n')
            except OSError:
                print('The failure details could not be saved; returning the same machine.', flush=True)
            save(job)
            raise
        finally:
            finalize(job)


def start(job):
    """Check that the job may run now, then run it in the background as a systemd user service, which outlives the
    session that starts it and writes to logs/<id>.log."""
    cost = check_ready(job)
    unit = 'autoresearch-run-' + job['id']
    log = ROOT / 'logs' / (job['id'] + '.log')
    log.parent.mkdir(exist_ok=True)
    run(['systemd-run', '--user', '--unit=' + unit, '--collect',
         '--property=WorkingDirectory=' + str(ROOT), '--property=EnvironmentFile=' + str(ROOT / '.env'),
         '--property=StandardOutput=append:' + str(log), '--property=StandardError=append:' + str(log),
         sys.executable, str(ROOT / 'tools/rental.py'), 'run', job['id']])
    print(json.dumps({'id': job['id'], 'service': unit + '.service', 'log': str(log.relative_to(ROOT)),
                      'failed_cost_today': str(cost)}), flush=True)


def finish(job):
    """Return the machines of a job whose run ended early, and record it as failed."""
    with run_lock() as lock:
        hold(lock)
        job = load(job['id'])
        if job['phase'] not in ('prepared', 'evaluated', 'failed'):
            job.update(phase='failed', error_type='RunEnded')
            job.setdefault('failure_kind', 'unknown')
            save(job)
        finalize(job)


def stop(signum, frame):
    # Stopping the service ends the run through finalize, which returns the machine; a second signal must not cut
    # that short.
    signal.signal(signum, signal.SIG_IGN)
    sys.exit(128 + signum)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--gpu', choices=tuple(read_config()['gpus']),
                        help='Select a GPU for preparation, or verify the GPU of an existing job.')
    sub = parser.add_subparsers(dest='action', required=True)
    preparation = sub.add_parser('prepare', help='freeze the current sources with the prepared runtime as a job')
    preparation.add_argument('--commit', required=True)
    preparation.add_argument('--baseline', action='store_true')
    preparation.add_argument('--description', required=True)
    preparation.add_argument('--expected-eval-seconds', type=float, default=180)
    for action, text in (('start', 'check the job, then run it in the background'),
                         ('run', 'run the job in the foreground'),
                         ('finish', "return the machines of a job whose run ended early, and record it")):
        sub.add_parser(action, help=text).add_argument('identifier')
    args = parser.parse_args()
    if args.gpu:
        os.environ['AUTORESEARCH_GPU'] = args.gpu
    if args.action == 'prepare':
        prepare_job(args.commit, args.baseline, args.description, args.expected_eval_seconds)
        return
    job = load(args.identifier)
    profile = profile_for_job(job)
    if args.gpu and args.gpu != profile.identifier:
        parser.error('The specified GPU differs from the prepared job.')
    os.environ['AUTORESEARCH_GPU'] = profile.identifier
    if args.action == 'start':
        start(job)
    elif args.action == 'run':
        signal.signal(signal.SIGTERM, stop)
        run_job(job)
    else:
        finish(job)


if __name__ == '__main__':
    main()
