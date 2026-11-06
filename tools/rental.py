"""Run one prepared experiment, with a separate deletion timer and no training retries."""

import argparse
import ast
import csv
import datetime as dt
from decimal import Decimal, ROUND_CEILING
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
import urllib.error

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.vast_control import api, list_instances
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
LEASE_SECONDS = 3600  # Immutable lease limit for pre-track historical jobs only.
CLEANUP_SECONDS = 60
EXECUTION_SETUP_SECONDS = 180  # Post-transfer data setup and compilation estimate.
COLLECTION_SECONDS = 45
PROJECT_LABELS = ('autoresearch-',)
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


def label_gpu(label):
    if not str(label).startswith(PROJECT_LABELS):
        return None
    for identifier in read_config()['gpus']:
        if label.startswith('autoresearch-' + identifier + '-'):
            return identifier
    # All rental labels before GPU tracks were introduced belonged to the H100.
    return 'h100'


def recent_rentals(now, profile=None):
    """Check live instances and actual contract start times in recent billing history."""
    profile = active_gpu() if profile is None else profile
    whole_loop = read_config().get('rental_scope', 'loop') == 'loop'

    def belongs(row):
        identifier = label_gpu(row.get('label', ''))
        return identifier is not None and (whole_loop or identifier == profile.identifier)

    rows = list_instances(timeout=10)
    owned = [row for row in rows if belongs(row)]
    if owned:
        return owned
    invoices = api('GET', 'users/me/invoices', params={
        'owner': 'me', 'sdate': now - 86400, 'edate': now, 'inc_charges': 'true',
    })
    identifiers = sorted({row['instance_id'] for row in invoices['invoices'] if row.get('instance_id')})
    if identifiers:
        contracts = api('POST', 'contracts/fetch/', {'contract_ids': identifiers})
        rows.extend(contracts['contracts'])
    return [row for row in rows if belongs(row)
            and now - float(row['start_date']) < 86400]


def check_daily_budget(now, profile=None, ignore_job=None):
    """Permit another attempt after failures within $0.50, but never a second valid daily run."""
    if any(label_gpu(row.get('label', '')) is not None for row in list_instances(timeout=10)):
        raise RuntimeError('A GPU belonging to this loop is still active.')
    day = dt.datetime.fromtimestamp(now, dt.timezone.utc).date()
    jobs = [json.loads(path.read_text()) for path in (ROOT / 'queue').glob('*/job.json')]
    labels = {job.get('label') for job in jobs}
    decisions = {}
    for path in [ROOT / 'results/decisions.json', *ROOT.glob('gpus/*/results/decisions.json')]:
        if path.is_file():
            decisions.update({row['id']: row['status'] for row in json.loads(path.read_text())})
    entries = vast_backend.entries()
    rentals = {row['instance']: row for row in entries
               if row.get('event') == 'rented' and row.get('label') in labels}
    returned = {row['instance']: row for row in entries if row.get('event') == 'returned'}
    unusable = {row['instance'] for row in entries if row.get('event') == 'unusable'}
    failed_ids, total = set(), Decimal('0')
    for previous in jobs:
        requested = previous.get('creation_requested_at')
        if requested is None or previous.get('creation_rejected') or previous['id'] == ignore_job:
            continue
        if dt.datetime.fromtimestamp(requested, dt.timezone.utc).date() != day:
            continue
        status = decisions.get(previous['id'])
        if status in ('keep', 'discard') or (previous.get('phase') == 'evaluated' and status != 'crash'):
            raise RuntimeError('This loop already completed a measured run today.')
        if (previous.get('phase') != 'failed' and status != 'crash') or not previous.get('deleted_at'):
            raise RuntimeError('A previous attempt has not finished and been returned.')
        ids = {identifier for identifier, row in rentals.items() if row['label'] == previous.get('label')}
        failed_ids.update(ids)
        if not ids:
            # Before the shared tool was adopted, cost was recorded by duration and hourly price.
            if previous.get('rental_backend') == 'vast-rental':
                raise RuntimeError('The shared ledger has not recorded the previous attempt cost.')
            duration = max(0, previous['deleted_at'] - requested)
            charge = Decimal(str(duration)) * Decimal(str(previous['hourly_cost'])) / 3600
            if not charge.is_finite() or charge < 0:
                raise RuntimeError('The historical rental cost is invalid.')
            total += charge.quantize(Decimal('0.01'), rounding=ROUND_CEILING)
    for identifier, row in rentals.items():
        if (identifier in unusable and
                dt.datetime.fromisoformat(row['at']).astimezone(dt.timezone.utc).date() == day):
            failed_ids.add(identifier)
    for identifier in failed_ids:
        if identifier not in returned:
            raise RuntimeError('A failed rental has not been returned in the shared ledger.')
        charge = Decimal(str(returned[identifier]['dollars']))
        if not charge.is_finite() or charge < 0:
            raise RuntimeError('The shared ledger returned an invalid rental cost.')
        total += charge
    if total > FAILED_COST_LIMIT:
        raise RuntimeError("Today's failed attempts have cost more than $0.50.")
    return total


def timer_unit(job, action):
    return 'autoresearch-' + action + '-' + profile_for_job(job).identifier + '-' + job['id']


def rental_lock(profile):
    scope = 'loop' if read_config().get('rental_scope', 'loop') == 'loop' else profile.identifier
    return ROOT / 'results/operations' / ('rental-' + scope + '.lock')


def check_gpu_selection(profile):
    if (read_config().get('rental_scope', 'loop') == 'loop'
            and profile.identifier != get_profile().identifier):
        raise RuntimeError('Only the GPU selected in gpus.json may be rented in loop scope.')


def arm_expiry(job):
    unit = timer_unit(job, 'expire')
    when = dt.datetime.fromtimestamp(job['expires_at'], dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    run(['systemd-run', '--user', '--unit=' + unit, '--on-calendar=' + when,
         '--timer-property=AccuracySec=1s', '--timer-property=Persistent=yes',
         '--property=WorkingDirectory=' + str(ROOT), '--property=EnvironmentFile=' + str(ROOT / '.env'),
         '--property=Restart=on-failure', '--property=RestartSec=10s',
         sys.executable, str(ROOT / 'tools/rental.py'), '--gpu', profile_for_job(job).identifier,
         'expire', job['id']])
    state = subprocess.check_output(
        ['systemctl', '--user', 'show', unit + '.timer', '--property=ActiveState', '--value'], text=True,
    ).strip()
    if state != 'active':
        raise RuntimeError('The independent deletion timer did not become active.')
    job['expiry_unit'] = unit
    save(job)


def owned_instances(job, rows):
    matches = [row for row in rows if row.get('label') == job['label']]
    identifier = job.get('instance_id')
    if identifier is not None:
        for row in rows:
            if row['id'] == identifier and row.get('label') != job['label']:
                raise RuntimeError('The recorded instance does not belong to this experiment label.')
        if identifier not in {row['id'] for row in matches}:
            # The create response is an ownership record even before list propagation.
            matches.append({'id': identifier, 'label': job['label']})
    return matches


def delete_instances(job):
    rows = list_instances(timeout=10)
    matches = owned_instances(job, rows)
    if not matches and job.get('creation_uncertain') and time.time() < job['expires_at']:
        raise RuntimeError('Creation is still unconfirmed; retain the expiry job and observe the same label again.')
    acknowledged = load(job['id']).get('delete_acknowledged_at')
    visible_ids = {row['id'] for row in rows}
    for row in matches:
        if acknowledged and row['id'] not in visible_ids:
            continue
        try:
            if job.get('rental_backend') == 'vast-rental':
                vast_backend.command(['return', str(row['id'])], timeout=30)
                response = {'success': True}
            else:
                response = api('DELETE', f"instances/{row['id']}/", {}, timeout=10)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            if any(item['id'] == row['id'] for item in list_instances(timeout=10)):
                raise RuntimeError('The provider still lists the instance after reporting it missing.') from error
            response = {'success': True}
        if response.get('success') is not True:
            raise RuntimeError('The provider did not confirm the deletion request.')
        save(dict(id=job['id'], delete_acknowledged_at=time.time()))
    remaining = list_instances(timeout=10) if matches and not acknowledged else rows
    identifiers = {row['id'] for row in matches}
    if any(row['id'] in identifiers or row.get('label') == job['label'] for row in remaining):
        raise RuntimeError('The instance is still listed; deletion remains pending.')
    update = dict(id=job['id'], deleted_at=time.time(), creation_uncertain=False)
    save(update)
    job.update(update)
    if job.get('expiry_unit'):
        subprocess.run(['systemctl', '--user', 'stop', job['expiry_unit'] + '.timer'], check=False)


def finalize(job):
    if job.get('rental_backend') == 'vast-rental' and job.get('instance_id'):
        try:
            values = {name: job.get(name) for name in ('id', 'phase', 'exit_code', 'failure_kind')}
            vast_backend.command(['note', str(job['instance_id']),
                                  'autoresearch ' + job['id'], '--values', json.dumps(values)], timeout=30)
        except (OSError, subprocess.SubprocessError):
            # A note must never delay returning a paid machine.
            pass
    delete_instances(job)
    current = load(job['id'])
    if (current.get('creation_requested_at') is not None
            and not current.get('creation_rejected')
            and current['phase'] in ('evaluated', 'failed')):
        # Record only after deletion; the existing ledger validates the log and is idempotent.
        run([sys.executable, str(ROOT / 'tools/record_run.py'), current['id']])


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


def lowest_cost_offer(offers, download_bytes, rental_seconds, max_hourly_cost):
    """Prefer known transfer savings over the cheapest hourly offer, within its budget."""
    cheapest = min(offers, key=lambda row: row['dph_total'])

    def rates(row):
        try:
            values = (row['inet_down_cost'], row['inet_up_cost'])
            if all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   and math.isfinite(value) and value >= 0 for value in values):
                return values
        except KeyError:
            pass
        return None

    base_rates = rates(cheapest)
    if download_bytes <= 0 or base_rates is None:
        return cheapest
    # GiB understates savings if the provider's advertised $/GB uses decimal bytes.
    download_gib = download_bytes / 1024 ** 3
    # Savings at the maximum lease also hold at shorter leases after deployment.
    # Unknown output traffic may not become more expensive than the hourly choice.
    hours = rental_seconds / 3600
    chosen = cheapest
    cost = hours * chosen['dph_total'] + download_gib * base_rates[0]
    for row in offers:
        prices = rates(row)
        if (row['dph_total'] > max_hourly_cost or prices is None
                or prices[1] > base_rates[1]):
            continue
        estimate = hours * row['dph_total'] + download_gib * prices[0]
        if estimate < cost:
            chosen, cost = row, estimate
    return chosen


def excluded_machines():
    unavailable_machines, unavailable_offers = set(), set()
    failures_path = ROOT / 'results/operations/machine-failures.json'
    if failures_path.is_file():
        for failure in json.loads(failures_path.read_text()):
            if failure.get('machine_id') is not None:
                unavailable_machines.add(failure['machine_id'])
    for path in (ROOT / 'queue').glob('*/job.json'):
        previous = json.loads(path.read_text())
        requested = previous.get('creation_requested_at')
        if (requested is not None and not previous.get('creation_rejected')
                and (previous.get('failure_kind') == 'machine'
                     or previous.get('error_type') == 'SSHUnavailable')):
            if previous.get('provider_machine_id') is not None:
                unavailable_machines.add(previous['provider_machine_id'])
            if previous.get('offer_id') is not None:
                unavailable_offers.add(previous['offer_id'])
    return unavailable_machines, unavailable_offers


def choose_offer(profile=None, *, download_bytes=0, settings=None):
    profile = active_gpu() if profile is None else profile
    values = profile.values
    unavailable_machines, unavailable_offers = excluded_machines()
    query = {'external': {'eq': False}, 'rentable': {'eq': True}, 'verified': {'eq': True},
             'rented': {'eq': False}, 'num_gpus': {'eq': 1},
             'gpu_name': {'eq': profile.gpu_name}, 'gpu_ram': {'gte': values['minimum_gpu_ram_mb']},
             'cpu_ram': {'gte': values['minimum_cpu_ram_mb']},
             'cpu_cores_effective': {'gte': values['minimum_cpu_cores']},
             'disk_space': {'gte': values['disk_gb']}, 'inet_down': {'gte': values['minimum_download_mbps']},
             'reliability2': {'gte': values['minimum_reliability']},
             'cuda_max_good': {'gte': values['cuda_minimum']},
             'order': [['dph_total', 'asc']], 'type': 'ondemand',
             'allocated_storage': values['disk_gb'], 'limit': 20}
    if unavailable_machines:
        query['machine_id'] = {'notin': sorted(unavailable_machines)}
    offers = [row for row in api('POST', 'bundles/', query)['offers']
              if row.get('num_gpus') == 1 and row.get('gpu_name') == profile.gpu_name
              and row.get('verification') == 'verified'
              and float(row.get('gpu_ram', 0)) >= values['minimum_gpu_ram_mb']
              and float(row.get('cuda_max_good', 0)) >= values['cuda_minimum']
              and row.get('machine_id') not in unavailable_machines
              and row.get('id') not in unavailable_offers
              and math.isfinite(row['dph_total']) and row['dph_total'] >= 0]
    if not offers:
        raise RuntimeError(f'No suitable single {profile.gpu_name} is available; no rental was requested.')
    settings = profile.settings() if settings is None else settings
    return lowest_cost_offer(offers, download_bytes, settings['rental_seconds'], settings['max_hourly_cost'])


def ssh_connection(instance, key, prefer_direct=True):
    if instance.get('ssh_config'):
        return ['ssh', '-F', str(instance['ssh_config']), '-i', str(key),
                '-o', 'IdentitiesOnly=yes', '-o', 'IdentityAgent=none',
                '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', instance['ssh_host']]
    ports = instance.get('ports') or {}
    direct = ports.get('22/tcp')
    if prefer_direct and direct and instance.get('public_ipaddr'):
        host, port = instance['public_ipaddr'], direct[0]['HostPort']
    else:
        host, port = instance['ssh_host'], instance['ssh_port']
    port = int(port)
    if not host or not 0 < port < 65536:
        raise RuntimeError('SSH is not ready yet.')
    return ['ssh', '-F', '/dev/null', '-i', str(key), '-p', str(port), '-o', 'BatchMode=yes',
            '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'UserKnownHostsFile=' + str(key.parent / 'known_hosts'), 'root@' + host]


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


def execute(job, instance, key):
    require_execution_time(job)
    connection = ssh_connection(instance, key)
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
            connection = ssh_connection(instance, key, prefer_direct=False)
        time.sleep(min(30, max(0, observe_until - time.time())))
    artifacts = ('run.log', 'prepare.log', 'bootstrap.log')
    if job.get('metric') == 'olmes_bpb':
        artifacts += ('olmes.log', 'olmes-summary.json', 'olmes-metrics.jsonl',
                      'olmes-details.tar.gz', 'final-model.pt')

    def collect(names):
        for name in names:
            path = destination / Path(name).name
            temporary = path.with_name(path.name + '.tmp')
            try:
                with temporary.open('wb') as output:
                    remote('cd autoresearch && cat ' + shlex.quote(name), stdout=output, stderr=subprocess.DEVNULL)
                temporary.replace(path)
            except (subprocess.SubprocessError, OSError, TimeoutError):
                temporary.unlink(missing_ok=True)

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


def start(job):
    profile = profile_for_job(job)
    if os.environ.get('AUTORESEARCH_RENTAL_HOLD_' + profile.identifier.upper()) == '1':
        raise RuntimeError(f'{profile.identifier} rentals are on hold by user request.')
    if job['phase'] != 'prepared':
        raise RuntimeError('This job already has a creation attempt; observe or expire it instead.')
    check_gpu_selection(profile)
    runtime_path = job_path(job['id']).parent / 'runtime.json'
    if digest(runtime_path) != job['runtime_manifest_sha256']:
        raise RuntimeError('The prepared runtime manifest changed.')
    if digest(job_path(job['id']).parent / 'source.tar.gz') != job['source_archive_sha256']:
        raise RuntimeError('The prepared experiment sources changed.')
    manifest = json.loads(runtime_path.read_text())
    sizes = verify_archives(manifest, artifact_directory(job),
                            'A prepared runtime archive changed; no rental was requested.')
    download_bytes = sum(sizes[name] for name in deployment_names(manifest))
    download_bytes += (job_path(job['id']).parent / 'source.tar.gz').stat().st_size
    required_seconds = execution_budget(job)
    check_daily_budget(time.time(), profile)
    validate_environment()
    values = profile.values
    query = {name: {'gte': values[field]} for name, field in (
        ('gpu_ram', 'minimum_gpu_ram_mb'), ('cpu_ram', 'minimum_cpu_ram_mb'),
        ('cpu_cores_effective', 'minimum_cpu_cores'), ('inet_down', 'minimum_download_mbps'),
        ('reliability2', 'minimum_reliability'), ('cuda_max_good', 'cuda_minimum'))}
    excluded, _ = excluded_machines()
    if excluded:
        query['machine_id'] = {'notin': sorted(excluded)}
    reservation = ROOT / 'results/operations/rental-reservation.lock'
    reservation.parent.mkdir(parents=True, exist_ok=True)
    with reservation.open('a') as lock:
        # Serialize only quota reservation, so changing scope cannot race a second creation.
        fcntl.flock(lock, fcntl.LOCK_EX)
        check_gpu_selection(profile)
        check_daily_budget(time.time(), profile)
        job.update(requested_at=time.time(), expires_at=time.time() + lease_seconds(job) - CLEANUP_SECONDS,
                   phase='arming', gpu_name=profile.gpu_name, rental_backend='vast-rental',
                   deployment_bytes=download_bytes)
        save(job)
        arm_expiry(job)
        job.update(phase='creation_requested', creation_uncertain=True, creation_requested_at=time.time())
        save(job)
    try:
        with tempfile.TemporaryDirectory(prefix='autoresearch-ssh-') as temporary:
            key = Path(temporary) / 'key'
            key.write_text(os.environ['VAST_SSH_KEY'].rstrip() + '\n')
            key.chmod(0o600)
            while True:
                check_daily_budget(time.time(), profile, ignore_job=job['id'])
                remaining = job['expires_at'] - time.time() - required_seconds
                if remaining <= 0:
                    raise TimeoutError('No time remains for another checked machine and the complete run.')
                job['creation_uncertain'] = True
                save(job)
                response, attempts = vast_backend.rent(job, manifest, profile, key, query, remaining,
                                                       job_path(job['id']).parent)
                job.setdefault('selection_attempts', []).extend(attempts)
                if response.get('rented') is not None:
                    job.update(instance_id=response['rented'], provider_machine_id=response['machine'],
                               hourly_cost=response['price'], host_verification='verified',
                               creation_uncertain=False, phase='starting', ready_at=time.time(),
                               ssh_config=response['ssh_config'], ssh_host=response['ssh_host'])
                    save(job)
                    execute(job, dict(id=job['instance_id'], ssh_config=job['ssh_config'],
                                      ssh_host=job['ssh_host']), key)
                    break
                job['creation_uncertain'] = False
                save(job)
                if response.get('failed', 0) == 0:
                    created = any(row.get('event') == 'rented' and row.get('label') == job['label']
                                  for row in vast_backend.entries())
                    job['creation_rejected'] = not created
                    raise RuntimeError('vast-rental found no available eligible machine.')
    except BaseException as error:
        job.update(phase='failed', error_type=type(error).__name__)
        job.setdefault('failure_kind', 'unknown')
        try:
            with (job_path(job['id']).parent / 'failure.log').open('a') as log:
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
        # Expiry has its own service, credentials and timer if this controller disappears.
        finalize(job)


def schedule(job, when):
    if job['phase'] != 'prepared':
        raise RuntimeError('Only a prepared, unattempted job can be scheduled.')
    parsed = dt.datetime.fromisoformat(when.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.timestamp() < time.time() + 60:
        raise ValueError('Specify a future time with an explicit timezone.')
    unit = timer_unit(job, 'start')
    calendar = parsed.astimezone(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    run(['systemd-run', '--user', '--unit=' + unit, '--on-calendar=' + calendar,
         '--timer-property=AccuracySec=1s', '--timer-property=Persistent=yes',
         '--property=WorkingDirectory=' + str(ROOT), '--property=EnvironmentFile=' + str(ROOT / '.env'),
         '--property=StandardOutput=append:' + str(ROOT / 'logs' / (job['id'] + '.log')),
         '--property=StandardError=append:' + str(ROOT / 'logs' / (job['id'] + '.log')),
         sys.executable, str(ROOT / 'tools/rental.py'), '--gpu', profile_for_job(job).identifier,
         'start', job['id']])
    state = subprocess.check_output(
        ['systemctl', '--user', 'show', unit + '.timer', '--property=ActiveState', '--value'], text=True,
    ).strip()
    if state != 'active':
        raise RuntimeError('The launch timer did not become active.')
    save(dict(id=job['id'], launch_unit=unit, scheduled_for=parsed.timestamp()))


def check_expiry():
    identifier = 'expiry-check-' + uuid.uuid4().hex[:8]
    job_path(identifier).parent.mkdir(parents=True)
    job = dict(id=identifier, label='autoresearch-' + identifier, phase='diagnostic',
               gpu_id=active_gpu().identifier, diagnostic=True,
               expires_at=time.time() + 20, creation_uncertain=False)
    save(job)
    arm_expiry(job)
    print('Scheduled an expiry check with no rental or destroy request.', flush=True)
    print(identifier, flush=True)


def cancel(job):
    if job.get('launch_unit'):
        for suffix in ('.timer', '.service'):
            unit = job['launch_unit'] + suffix
            state = subprocess.run(['systemctl', '--user', 'show', unit, '--property=LoadState', '--value'],
                                   capture_output=True, text=True)
            if state.stdout.strip() == 'not-found':
                continue
            state.check_returncode()
            run(['systemctl', '--user', 'stop', unit])
    job = load(job['id'])
    if job.get('creation_requested_at') and not job.get('deleted_at'):
        delete_instances(job)
    save(dict(id=job['id'], phase='cancelled'))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=tuple(read_config()['gpus']),
                        help='Select a GPU for preparation, or verify the GPU of an existing job.')
    sub = parser.add_subparsers(dest='action', required=True)
    preparation = sub.add_parser('prepare')
    preparation.add_argument('--commit', required=True)
    preparation.add_argument('--baseline', action='store_true')
    preparation.add_argument('--description', required=True)
    preparation.add_argument('--expected-eval-seconds', type=float, default=180)
    sub.add_parser('check-expiry')
    for action in ('start', 'expire', 'status', 'cancel', 'upload'):
        command = sub.add_parser(action)
        command.add_argument('identifier')
    scheduling = sub.add_parser('schedule')
    scheduling.add_argument('identifier')
    scheduling.add_argument('--at', required=True)
    args = parser.parse_args()
    if args.gpu:
        os.environ['AUTORESEARCH_GPU'] = args.gpu
    if args.action == 'prepare':
        prepare_job(args.commit, args.baseline, args.description, args.expected_eval_seconds)
        return
    if args.action == 'check-expiry':
        check_expiry()
        return
    job = load(args.identifier)
    profile = profile_for_job(job)
    if args.gpu and args.gpu != profile.identifier:
        parser.error('The specified GPU differs from the prepared job.')
    os.environ['AUTORESEARCH_GPU'] = profile.identifier
    if args.action == 'expire':
        if job.get('diagnostic'):
            validate_environment()
        finalize(job)
    elif args.action == 'schedule':
        schedule(job, args.at)
    elif args.action == 'cancel':
        cancel(job)
    elif args.action == 'status':
        print(json.dumps(dict(job=job, instances=[row for row in list_instances() if row.get('label') == job['label']]), indent=2))
    elif args.action == 'upload':
        runtime_path = job_path(job['id']).parent / 'runtime.json'
        if digest(runtime_path) != job['runtime_manifest_sha256']:
            raise RuntimeError('The prepared runtime manifest changed.')
        manifest = json.loads(runtime_path.read_text())
        if r2_store(manifest) is None:
            raise RuntimeError('R2 holds only transport runtimes, and every R2 variable must be set.')
        verify_archives(manifest, artifact_directory(job), 'A prepared runtime archive is missing or changed.')
        print('R2 holds every runtime archive of this job except its bootstrap.', flush=True)
    else:
        lock_path = rental_lock(profile)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            start(job)


if __name__ == '__main__':
    main()
