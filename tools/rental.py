"""Run one prepared experiment, with a separate deletion timer and no training retries."""

import argparse
import datetime as dt
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

SOURCES = ('train.py', 'prepare.py', 'program.md', 'pyproject.toml', 'uv.lock')
FRAME_SOURCES = SOURCES[1:]
LEASE_SECONDS = 3600
CLEANUP_SECONDS = 60
PROJECT_LABELS = ('autoresearch-',)


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


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def recent_rentals(now):
    """Check live instances and actual contract start times in recent billing history."""
    rows = list_instances(timeout=10)
    invoices = api('GET', 'users/me/invoices', params={
        'owner': 'me', 'sdate': now - 86400, 'edate': now, 'inc_charges': 'true',
    })
    identifiers = sorted({row['instance_id'] for row in invoices['invoices'] if row.get('instance_id')})
    if identifiers:
        contracts = api('POST', 'contracts/fetch/', {'contract_ids': identifiers})
        rows.extend(contracts['contracts'])
    return [row for row in rows if str(row.get('label', '')).startswith(PROJECT_LABELS)
            and now - float(row['start_date']) < 86400]


def check_daily_budget(now):
    for path in (ROOT / 'queue').glob('*/job.json'):
        previous = json.loads(path.read_text())
        requested = previous.get('creation_requested_at')
        if requested is not None and not previous.get('creation_rejected') and now - requested < 86400:
            raise RuntimeError('A daily research rental was already requested within the last 24 hours.')
    if recent_rentals(now):
        raise RuntimeError('The previous daily research rental is less than 24 hours old.')


def arm_expiry(job):
    unit = 'autoresearch-expire-' + job['id']
    when = dt.datetime.fromtimestamp(job['expires_at'], dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    run(['systemd-run', '--user', '--unit=' + unit, '--on-calendar=' + when,
         '--timer-property=AccuracySec=1s', '--timer-property=Persistent=yes',
         '--property=WorkingDirectory=' + str(ROOT), '--property=EnvironmentFile=' + str(ROOT / '.env'),
         '--property=Restart=on-failure', '--property=RestartSec=10s',
         sys.executable, str(ROOT / 'tools/rental.py'), 'expire', job['id']])
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


def validate_environment():
    if not os.environ.get('VAST_API_KEY'):
        raise RuntimeError('VAST_API_KEY is required in the environment.')
    with tempfile.TemporaryDirectory(prefix='autoresearch-key-check-') as temporary:
        key = Path(temporary) / 'key'
        key.write_text(os.environ['VAST_SSH_KEY'].rstrip() + '\n')
        key.chmod(0o600)
        return subprocess.check_output(['ssh-keygen', '-y', '-f', str(key)], text=True,
                                       stderr=subprocess.DEVNULL).strip()


def prepare_job(commit, baseline, description):
    import prepare

    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Pass the full commit from the local Git checkout.')
    prepare.verify_core_assets()
    preflight = json.loads((ROOT / 'results/core-preflight.json').read_text())
    if (len(preflight['tasks']) != 22 or preflight['bundle_sha256'] != prepare.CORE_BUNDLE_SHA256
            or preflight['nanochat_revision'] != prepare.CORE_NANOCHAT_REVISION):
        raise RuntimeError('The complete prompt preflight is not ready for this frame.')
    if preflight['tokenizer_sha256'] != digest(Path(prepare.TOKENIZER_DIR) / 'tokenizer.pkl'):
        raise RuntimeError('The tokenizer differs from the completed prompt preflight.')
    manifest = json.loads((ROOT / 'artifacts/runtime.json').read_text())
    hashes = {name: digest(ROOT / name) for name in SOURCES}
    for name in FRAME_SOURCES:
        if hashes[name] != manifest['sources'][name]:
            raise RuntimeError('The prepared runtime belongs to a different experiment frame.')
    for name, field in (('gpu-env.tar.gz', 'gpu_env_sha256'), ('gpu-assets.tar.gz', 'gpu_assets_sha256')):
        if digest(ROOT / 'artifacts' / name) != manifest[field]:
            raise RuntimeError('A runtime archive is incomplete or changed.')
    identifier = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
    directory = job_path(identifier).parent
    directory.mkdir(parents=True)
    source_dir = directory / 'source'
    source_dir.mkdir()
    for name in SOURCES:
        shutil.copy2(ROOT / name, source_dir / name)
    shutil.copy2(ROOT / 'tools/gpu_run.sh', source_dir / 'gpu_run.sh')
    with tarfile.open(directory / 'source.tar.gz', 'w:gz') as archive:
        for path in sorted(source_dir.iterdir()):
            archive.add(path, arcname=path.name)
    job = dict(id=identifier, label='autoresearch-' + identifier, commit=commit, baseline=baseline,
               description=description, phase='prepared', source_sha256=hashes,
               source_archive_sha256=digest(directory / 'source.tar.gz'),
               runtime_manifest_sha256=digest(ROOT / 'artifacts/runtime.json'), max_hourly_cost=5.0)
    save(job)
    print(json.dumps({'id': identifier, 'commit': commit, 'phase': job['phase']}, indent=2))


def choose_offer():
    query = {'external': {'eq': False}, 'rentable': {'eq': True},
             'rented': {'eq': False}, 'num_gpus': {'eq': 1},
             'gpu_name': {'eq': 'H100 SXM'}, 'gpu_ram': {'gte': 79000},
             'cpu_ram': {'gte': 32000}, 'cpu_cores_effective': {'gte': 12}, 'disk_space': {'gte': 90},
             'inet_down': {'gte': 800}, 'reliability2': {'gte': 0.98}, 'cuda_max_good': {'gte': 12.8},
             'order': [['dph_total', 'asc']], 'type': 'ondemand', 'allocated_storage': 90, 'limit': 20}
    offers = [row for row in api('POST', 'bundles/', query)['offers']
              if row.get('num_gpus') == 1 and row.get('gpu_name') == 'H100 SXM'
              and float(row.get('gpu_ram', 0)) >= 79000 and float(row.get('cuda_max_good', 0)) >= 12.8
              and math.isfinite(row['dph_total']) and row['dph_total'] >= 0]
    if not offers:
        raise RuntimeError('No suitable single H100 SXM is available; no rental was requested.')
    return min(offers, key=lambda row: row['dph_total'])


def ssh_connection(instance, key, prefer_direct=True):
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


def execute(job, instance, key):
    connection = ssh_connection(instance, key)
    deadline = job['expires_at'] - 5
    observe_until = job['expires_at'] - 45
    destination = ROOT / 'results/runs' / job['id']
    destination.mkdir(parents=True, exist_ok=True)

    def remote(command, timeout=30, **kwargs):
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError('The rental budget is exhausted.')
        return run(connection + [command], timeout=min(timeout, remaining), **kwargs)

    gpu = remote('nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits',
                 capture_output=True, text=True).stdout.strip().splitlines()
    job['initial_gpu_observation'] = gpu
    save(job)
    if (len(gpu) != 1 or not re.search(r'H100 (?:SXM|80GB HBM3)', gpu[0])
            or float(gpu[0].rsplit(',', 1)[1]) > 512):
        raise RuntimeError('The rental does not provide one idle H100 SXM.')
    remote('mkdir -p autoresearch')
    for path in [ROOT / 'artifacts/gpu-env.tar.gz', ROOT / 'artifacts/gpu-assets.tar.gz',
                 job_path(job['id']).parent / 'source.tar.gz']:
        with path.open('rb') as source:
            remote('tar -xzf - -C autoresearch', timeout=900, stdin=source)
    try:
        remote('cd autoresearch && if mkdir .started 2>/dev/null; then '
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
                break
            if probe == 'missing':
                raise RuntimeError('The remote training process handle is missing.')
            if probe == 'running':
                print('Observed the same live training process.', flush=True)
            else:
                print('The observation was incomplete; rechecking the same process.', flush=True)
        except (subprocess.SubprocessError, OSError) as error:
            # A failed observation never starts another training process or rental.
            print('Observation failed; rechecking the same rental:', type(error).__name__, flush=True)
            connection = ssh_connection(instance, key, prefer_direct=False)
        time.sleep(min(30, max(0, observe_until - time.time())))
    for name in ('run.log', 'prepare.log', 'bootstrap.log', 'results/core.json', 'results/core.csv'):
        try:
            with (destination / Path(name).name).open('wb') as output:
                remote('cd autoresearch && cat ' + shlex.quote(name), stdout=output, stderr=subprocess.DEVNULL)
        except (subprocess.SubprocessError, OSError, TimeoutError):
            pass
    job['exit_code'] = status
    job['phase'] = 'evaluated' if status == 0 else 'failed'
    save(job)


def start(job):
    if job['phase'] != 'prepared':
        raise RuntimeError('This job already has a creation attempt; observe or expire it instead.')
    if digest(ROOT / 'artifacts/runtime.json') != job['runtime_manifest_sha256']:
        raise RuntimeError('The prepared runtime manifest changed.')
    if digest(job_path(job['id']).parent / 'source.tar.gz') != job['source_archive_sha256']:
        raise RuntimeError('The prepared experiment sources changed.')
    manifest = json.loads((ROOT / 'artifacts/runtime.json').read_text())
    for name, field in (('gpu-env.tar.gz', 'gpu_env_sha256'), ('gpu-assets.tar.gz', 'gpu_assets_sha256')):
        if digest(ROOT / 'artifacts' / name) != manifest[field]:
            raise RuntimeError('A prepared runtime archive changed; no rental was requested.')
    for name in FRAME_SOURCES:
        if digest(ROOT / name) != job['source_sha256'][name]:
            raise RuntimeError('The experiment frame changed after this job was prepared.')
    check_daily_budget(time.time())
    public = validate_environment()
    offer = choose_offer()
    if offer['dph_total'] > job['max_hourly_cost']:
        raise RuntimeError('The current H100 offer exceeds the prepared cost limit.')
    job.update(requested_at=time.time(), expires_at=time.time() + LEASE_SECONDS - CLEANUP_SECONDS,
               phase='arming', gpu_name=offer['gpu_name'])
    save(job)
    arm_expiry(job)
    job.update(phase='creation_requested', creation_uncertain=True, creation_requested_at=time.time())
    save(job)
    try:
        response = api('PUT', f"asks/{offer['id']}/", {
            'client_id': 'me', 'image': manifest['container_image'], 'disk': 90,
            'label': job['label'], 'runtype': 'ssh_direc ssh_proxy', 'env': {}, 'cancel_unavail': True,
        })
        if response.get('success') is not True or not response.get('new_contract'):
            job.update(creation_uncertain=False, creation_rejected=True)
            raise RuntimeError('The provider declined the rental request.')
        job.update(instance_id=int(response['new_contract']), creation_uncertain=False, phase='starting')
        save(job)
        with tempfile.TemporaryDirectory(prefix='autoresearch-ssh-') as temporary:
            key = Path(temporary) / 'key'
            key.write_text(os.environ['VAST_SSH_KEY'].rstrip() + '\n')
            key.chmod(0o600)
            try:
                api('POST', f"instances/{job['instance_id']}/ssh/", {'ssh_key': public})
            except (urllib.error.URLError, TimeoutError):
                print('SSH-key acknowledgement unavailable; observing access to the same instance.', flush=True)
            while time.time() < job['expires_at'] - 1900:
                try:
                    instance = next((row for row in list_instances() if row['id'] == job['instance_id']), None)
                except (urllib.error.URLError, TimeoutError, RuntimeError) as error:
                    print('Readiness observation failed; retaining the same instance:', type(error).__name__, flush=True)
                    time.sleep(15)
                    continue
                if instance and instance.get('actual_status') == 'running' and instance.get('ssh_host'):
                    try:
                        run(ssh_connection(instance, key) + ['true'], timeout=15,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except (subprocess.SubprocessError, OSError):
                        time.sleep(15)
                        continue
                    execute(job, instance, key)
                    break
                time.sleep(15)
            else:
                raise TimeoutError('The rental did not become ready in time for a full training run.')
    except BaseException as error:
        if isinstance(error, urllib.error.HTTPError) and error.code in (400, 401, 402, 403, 404, 409, 410, 422, 429):
            if not job.get('instance_id'):
                job.update(creation_uncertain=False, creation_rejected=True)
        job.update(phase='failed', error_type=type(error).__name__)
        save(job)
        raise
    finally:
        # Expiry has its own service, credentials and timer if this controller disappears.
        delete_instances(job)


def schedule(job, when):
    if job['phase'] != 'prepared':
        raise RuntimeError('Only a prepared, unattempted job can be scheduled.')
    parsed = dt.datetime.fromisoformat(when.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.timestamp() < time.time() + 60:
        raise ValueError('Specify a future time with an explicit timezone.')
    unit = 'autoresearch-start-' + job['id']
    calendar = parsed.astimezone(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    run(['systemd-run', '--user', '--unit=' + unit, '--on-calendar=' + calendar,
         '--timer-property=AccuracySec=1s', '--timer-property=Persistent=yes',
         '--property=WorkingDirectory=' + str(ROOT), '--property=EnvironmentFile=' + str(ROOT / '.env'),
         '--property=StandardOutput=append:' + str(ROOT / 'logs' / (job['id'] + '.log')),
         '--property=StandardError=append:' + str(ROOT / 'logs' / (job['id'] + '.log')),
         sys.executable, str(ROOT / 'tools/rental.py'), 'start', job['id']])
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
               diagnostic=True, expires_at=time.time() + 20, creation_uncertain=False)
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
    sub = parser.add_subparsers(dest='action', required=True)
    preparation = sub.add_parser('prepare')
    preparation.add_argument('--commit', required=True)
    preparation.add_argument('--baseline', action='store_true')
    preparation.add_argument('--description', required=True)
    sub.add_parser('check-expiry')
    for action in ('start', 'expire', 'status', 'cancel'):
        command = sub.add_parser(action)
        command.add_argument('identifier')
    scheduling = sub.add_parser('schedule')
    scheduling.add_argument('identifier')
    scheduling.add_argument('--at', required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare_job(args.commit, args.baseline, args.description)
        return
    if args.action == 'check-expiry':
        check_expiry()
        return
    job = load(args.identifier)
    if args.action == 'expire':
        if job.get('diagnostic'):
            validate_environment()
        delete_instances(job)
    elif args.action == 'schedule':
        schedule(job, args.at)
    elif args.action == 'cancel':
        cancel(job)
    elif args.action == 'status':
        print(json.dumps(dict(job=job, instances=[row for row in list_instances() if row.get('label') == job['label']]), indent=2))
    else:
        lock_path = ROOT / 'results/operations/rental.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            start(job)


if __name__ == '__main__':
    main()
