"""Run one prepared job on a GPU that vast-rental rents, checks, watches and returns.

start checks that the job may run today and runs it in the background as a systemd user service that writes to
logs/<id>.log; run does the same in the foreground; finish gives back the machines of a run that ended early and
records it as failed. A run asks vast-rental for a checked machine within the day's failure budget, sends the runtime
and the sources, checks that the GPU is idle, trains and evaluates once, has the machine put its results in R2 and
brings them back checked against the machine's own digests, gives the machine back and records the result. Every step
ends by the machine's deadline, at which vast-rental's watcher sends the machine back whatever becomes of the run.
"""

import argparse
import contextlib
import datetime as dt
from decimal import Decimal
import fcntl
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
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu_config import get_profile, profile_for_job
from tools import delivery, jobs, r2, telemetry, vast_backend
from tools.jobs import load, save
from tools.runtime_archives import encoding_cache_asset_path

LABEL_PREFIX = 'autoresearch-'  # Rentals whose label starts with this belong to this loop.
FAILED_COST_LIMIT = Decimal('0.50')  # What a day's failed attempts may cost before the loop rents no more that day.
KEEP_FREE_BYTES = 5 * 10 ** 9  # program.md never leaves less than this free on the processing server.
RESULT_BYTES = 2 * 10 ** 9  # Room for one run's results, its checkpoint included.
# run.exit goes first: once it exists, every other file is final.
RESULTS = ('run.exit', 'run.log', 'prepare.log', 'bootstrap.log', 'olmes.log', 'olmes-summary.json',
           'olmes-metrics.jsonl', 'olmes-details.tar.gz', 'final-model.pt')
PROBE = ('cd autoresearch && if test -f run.exit; then printf "exit "; cat run.exit; '
         'elif test -f run.pid && kill -0 "$(cat run.pid)" 2>/dev/null; then echo running; else echo missing; fi')
LAUNCH = ('if mkdir .started 2>/dev/null; then '
          'nohup bash gpu_run.sh > bootstrap.log 2>&1 < /dev/null & echo $! > run.pid; fi')


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def decisions():
    """Each recorded job's status, from every GPU track's ledger."""
    found = {}
    for path in jobs.ROOT.glob('gpus/*/results/decisions.json'):
        found.update({row['id']: row['status'] for row in json.loads(path.read_text())})
    return found


def measured(job, statuses):
    """Whether the job's run counts as the day's measurement: recorded with a result, or evaluated and not yet
    recorded as a crash."""
    status = statuses.get(job['id'])
    return status in ('keep', 'discard') or (job.get('phase') == 'evaluated' and status != 'crash')


def failed_cost_today(now):
    """What this loop's failed attempts have cost so far on the UTC day of now, when the loop may rent again that day;
    otherwise raise.

    The loop rents one GPU at a time and measures one run a day. After failures it may try again while they have cost
    no more than FAILED_COST_LIMIT in all, as the vast-rental ledger records. Other projects' rentals do not count.
    """
    day = dt.datetime.fromtimestamp(now, dt.timezone.utc).date()
    labelled = {}
    for path in (jobs.ROOT / 'queue').glob('*/job.json'):
        job = json.loads(path.read_text())
        labelled[job['label']] = job
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
        job = labelled.get(row['label'])
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


@contextlib.contextmanager
def key_file():
    """The loop's SSH key as a private file, for as long as the work needs it."""
    if not os.environ.get('VAST_SSH_KEY'):
        raise RuntimeError('VAST_SSH_KEY is required in the environment.')
    with tempfile.TemporaryDirectory(prefix='autoresearch-key-') as temporary:
        key = Path(temporary) / 'key'
        key.write_text(os.environ['VAST_SSH_KEY'].rstrip() + '\n')
        key.chmod(0o600)
        yield key


def check_ready(job):
    """Raise unless the job may start now; give what today's failed attempts have cost."""
    profile = profile_for_job(job)
    if os.environ.get('AUTORESEARCH_RENTAL_HOLD_' + profile.identifier.upper()) == '1':
        raise RuntimeError(f'{profile.identifier} rentals are on hold by user request.')
    if profile.identifier != get_profile().identifier:
        raise RuntimeError('Only the GPU selected in gpus.json may be rented.')
    if job['phase'] != 'prepared':
        raise RuntimeError('This job has already been started; finish it instead.')
    directory = jobs.job_path(job['id']).parent
    if jobs.digest(directory / 'runtime.json') != job['runtime_manifest_sha256']:
        raise RuntimeError('The prepared runtime manifest changed.')
    if jobs.digest(directory / 'source.tar.gz') != job['source_archive_sha256']:
        raise RuntimeError('The prepared experiment sources changed.')
    jobs.check_fit(job)
    delivery.store()
    with key_file() as key:
        subprocess.check_output(['ssh-keygen', '-y', '-f', str(key)], stderr=subprocess.DEVNULL)
    free = shutil.disk_usage(jobs.ROOT).free
    if free < KEEP_FREE_BYTES + RESULT_BYTES:
        raise RuntimeError(f'The processing server has {free / 10 ** 9:.1f} GB free, and a run needs '
                           f'{(KEEP_FREE_BYTES + RESULT_BYTES) / 10 ** 9:.0f} GB.')
    return failed_cost_today(time.time())


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


def unreachable(error):
    """Whether the error says that the machine could not be reached, or did not answer in time."""
    return (isinstance(error, subprocess.TimeoutExpired)
            or (isinstance(error, subprocess.CalledProcessError) and error.returncode == 255))


def execute(job, connection):
    """Run the job once on its machine and bring its results back, each step ending by the machine's deadline."""
    stop_at = job['deadline'] - jobs.RETURN_SECONDS
    destination = jobs.results_directory(job) / 'runs' / job['id']
    destination.mkdir(parents=True, exist_ok=True)

    def remote(command, timeout=30, **kwargs):
        remaining = stop_at - time.time()
        if remaining <= 0:
            raise TimeoutError('The rental has no time left.')
        return run(connection + [command], timeout=min(timeout, remaining), **kwargs)

    job['transfer_started_at'] = time.time()
    save(job)
    try:
        delivery.send(job, connection, remote, until=stop_at - jobs.machine_seconds(job))
        job['transfer_completed_at'] = time.time()
        save(job)
        if stop_at - time.time() < jobs.machine_seconds(job):
            raise TimeoutError('Too little of the rental remains for the training and the full evaluation.')
        check_gpu(job, remote)
    except BaseException as error:
        if unreachable(error):
            job['failure_kind'] = 'machine'
        raise
    launch(job, remote)
    status = wait(remote, destination, until=stop_at - jobs.COLLECTION_SECONDS)
    if status is not None:
        job['completed_observed_at'] = time.time()
    problems = collect(job, connection, destination, until=stop_at)
    collected = None if 'run.exit' in problems else exit_status(destination / 'run.exit')
    if status is not None and collected is not None and status != collected:
        raise RuntimeError('The exit status changed between observation and collection.')
    if status is None and collected is not None:
        status = collected
        job.update(exit_code_observed_in_collection=True, completed_observed_at=time.time())
    telemetry.record_preparation_time(remote, destination)
    job.update(exit_code=status, phase='evaluated' if status == 0 else 'failed', collected_at=time.time(),
               collection_errors=problems)
    if status != 0:
        job.setdefault('failure_kind', 'unknown')
    save(job)


def check_gpu(job, remote):
    """Raise unless the machine shows one idle GPU of the job's kind right before the work; a failure here is the
    machine's."""
    try:
        rows = remote('nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits',
                      capture_output=True, text=True).stdout.strip().splitlines()
    except subprocess.CalledProcessError:
        job['failure_kind'] = 'machine'
        raise
    job['initial_gpu_observation'] = rows
    save(job)
    profile = profile_for_job(job)
    try:
        name, memory = rows[0].rsplit(',', 1)
        idle = len(rows) == 1 and re.search(profile.values['device_name_pattern'], name) and float(memory) <= 512
    except (IndexError, ValueError):
        idle = False
    if not idle:
        job['failure_kind'] = 'machine'
        raise RuntimeError(f'The machine does not show one idle {profile.gpu_name}.')


def launch(job, remote):
    """Start the training and its evaluation once; after a lost acknowledgement, launching again starts nothing."""
    manifest = json.loads((jobs.job_path(job['id']).parent / 'runtime.json').read_text())
    environment = {'AUTORESEARCH_GPU': job['gpu_id']}
    if 'encoding_cache' in manifest:
        path = manifest['encoding_cache']['asset_path']
        if path != encoding_cache_asset_path(job['gpu_id']):
            raise ValueError('The encoding cache must belong to the prepared GPU track.')
        environment['OLMES_ENCODING_CACHE'] = path
    job['launch_requested_at'] = time.time()
    save(job)
    try:
        remote('cd autoresearch && export ' + shlex.join(f'{name}={value}' for name, value in environment.items())
               + ' && ' + LAUNCH)
    except (subprocess.SubprocessError, OSError):
        print('The launch acknowledgement was lost; watching the one process it may have started.', flush=True)
    job['phase'] = 'running'
    save(job)


def wait(remote, destination, until):
    """Watch the training process until it exits or until `until`: its exit status, or None if it had not exited."""
    while time.time() < until:
        try:
            probe = remote(PROBE, capture_output=True, text=True).stdout.strip()
        except (subprocess.SubprocessError, OSError) as error:
            # A failed look never starts another training process or rental.
            print('Observation failed; looking at the same machine again:', type(error).__name__, flush=True)
        else:
            found = re.fullmatch(r'exit ([0-9]{1,3})', probe)
            if found and int(found.group(1)) <= 255:
                return int(found.group(1))
            if probe == 'missing':
                raise RuntimeError('The training process is gone without an exit status.')
            if probe == 'running':
                telemetry.record_gpu_sample(remote, destination)
            else:
                print('The observation was incomplete; looking again.', flush=True)
        time.sleep(min(30, max(0, until - time.time())))
    return None


def collect(job, connection, destination, until):
    """Have the machine put its result files in R2 and bring each one here, checked against the machine's own size and
    digest, trying again while time remains; give each file that could not be brought back, with why. A file that R2
    holds but that did not arrive here stays in R2."""
    config = delivery.store()
    manifest = json.loads((jobs.job_path(job['id']).parent / 'runtime.json').read_text())
    python = delivery.interpreter(manifest)
    keys = {name: f"results/{job['id']}/{name}" for name in RESULTS}
    problems, pending, arrived = {}, list(RESULTS), []
    for _ in range(3):
        try:
            remaining = until - time.time()
            if remaining <= 0:
                raise TimeoutError('The rental has no time left to collect the results.')
            sent, absent = r2.send(connection, config, {name: keys[name] for name in pending}, python,
                                   timeout=remaining, expires=max(60, int(job['deadline'] - time.time())))
        except (subprocess.SubprocessError, OSError, ValueError, RuntimeError, TimeoutError) as error:
            problems.update({name: type(error).__name__ for name in pending})
            continue
        problems.update({name: 'absent' for name in absent})
        retry = []
        for name, receipt in sent.items():
            try:
                r2.download(config, keys[name], destination / name, receipt['sha256'], receipt['bytes'], until)
            except (OSError, ValueError, TimeoutError) as error:
                problems[name] = type(error).__name__
                retry.append(name)
            else:
                problems.pop(name, None)
                arrived.append(name)
        pending = retry
        if not pending:
            break
    for name in arrived:
        try:
            r2.delete(config, keys[name])
        except OSError:
            print(f'{name} arrived, but its copy in R2 could not be deleted.', flush=True)
    for name, why in problems.items():
        print(f'{name} was not collected: {why}.' + (f' R2 keeps it as {keys[name]}.' if why != 'absent' else ''),
              flush=True)
    return problems


def exit_status(path):
    try:
        text = path.read_text().strip()
    except (OSError, UnicodeError):
        return None
    return int(text) if re.fullmatch(r'[0-9]{1,3}', text) and int(text) <= 255 else None


def finalize(job):
    """Note how the machine did, rejecting it when it failed the work, give back every machine of the job, then
    record a run that had a machine."""
    if job.get('instance_id'):
        values = {name: job.get(name) for name in ('id', 'phase', 'exit_code', 'failure_kind')}
        try:
            if job.get('failure_kind') == 'machine':
                vast_backend.reject(job['instance_id'], f"autoresearch {job['id']}: {job.get('error_detail')}")
            vast_backend.note(job['instance_id'], 'autoresearch ' + job['id'], values)
        except (OSError, subprocess.SubprocessError):
            # Neither may delay giving a paid machine back.
            pass
    vast_backend.give_back(job['label'])
    save(dict(id=job['id'], returned_at=time.time()))
    current = load(job['id'])
    if current.get('instance_id') and current['phase'] in ('evaluated', 'failed'):
        # Recorded only once the machines are back; the recorder is idempotent.
        run([sys.executable, str(jobs.ROOT / 'tools/record_run.py'), current['id']])


def run_lock():
    """The lock that one run of this loop holds from its checks until its machines are back."""
    path = jobs.ROOT / 'results/operations/rental.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open('a')


def hold(lock):
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another run of this loop is in progress.') from None


def run_job(job):
    """Rent a checked machine through vast-rental and run the job on it once, giving the machine back at the end."""
    with run_lock() as lock:
        hold(lock)
        spent = check_ready(job)
        directory = jobs.job_path(job['id']).parent
        manifest = json.loads((directory / 'runtime.json').read_text())
        delivery.verify_archives(manifest, jobs.artifact_directory(job),
                                 'A prepared runtime archive changed; no machine was rented.')
        profile = profile_for_job(job)
        job.update(phase='renting', requested_at=time.time())
        save(job)
        try:
            with key_file() as key:
                machine = vast_backend.rent(job, manifest['container_image'], profile, key, requirements(profile),
                                            FAILED_COST_LIMIT - spent, directory)
                if machine is None:
                    raise RuntimeError('vast-rental kept no machine within its tries and the failure budget.')
                job.update(instance_id=machine['rented'], provider_machine_id=machine['machine'],
                           hourly_cost=machine['price'], ssh_config=machine['ssh_config'],
                           ssh_host=machine['ssh_host'], phase='starting', ready_at=time.time(),
                           deadline=dt.datetime.fromisoformat(machine['deadline']).timestamp())
                save(job)
                execute(job, ssh_command(job, key))
        except BaseException as error:
            job.update(error_type=type(error).__name__, error_detail=str(error)[:300])
            if job.get('instance_id') is None:
                # No machine ran the job, so it may start again later, such as when cheaper machines are free.
                job['phase'] = 'prepared'
            else:
                job['phase'] = 'failed'
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
                print('The failure details could not be saved; giving the same machine back.', flush=True)
            save(job)
            raise
        finally:
            finalize(job)


def start(job):
    """Check that the job may run now, then run it in the background as a systemd user service, which outlives the
    session that starts it and writes to logs/<id>.log."""
    cost = check_ready(job)
    unit = 'autoresearch-run-' + job['id']
    log = jobs.ROOT / 'logs' / (job['id'] + '.log')
    log.parent.mkdir(exist_ok=True)
    run(['systemd-run', '--user', '--unit=' + unit, '--collect',
         '--property=WorkingDirectory=' + str(jobs.ROOT), '--property=EnvironmentFile=' + str(jobs.ROOT / '.env'),
         '--property=StandardOutput=append:' + str(log), '--property=StandardError=append:' + str(log),
         sys.executable, str(jobs.ROOT / 'tools/rental.py'), 'run', job['id']])
    print(json.dumps({'id': job['id'], 'service': unit + '.service', 'log': str(log.relative_to(jobs.ROOT)),
                      'failed_cost_today': str(cost)}), flush=True)


def finish(job):
    """Give back the machines of a job whose run ended early, and record it as failed."""
    with run_lock() as lock:
        hold(lock)
        job = load(job['id'])
        if job['phase'] not in ('prepared', 'evaluated', 'failed'):
            job.update(phase='failed', error_type='RunEnded')
            job.setdefault('failure_kind', 'unknown')
            save(job)
        finalize(job)


def stop(signum, frame):
    # Stopping the service ends the run through finalize, which gives the machine back; a second signal must not cut
    # that short.
    signal.signal(signum, signal.SIG_IGN)
    sys.exit(128 + signum)


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='action', required=True)
    for action, text in (('start', 'check the job, then run it in the background'),
                         ('run', 'run the job in the foreground'),
                         ('finish', 'give back the machines of a job whose run ended early, and record it')):
        commands.add_parser(action, help=text).add_argument('identifier')
    args = parser.parse_args()
    job = load(args.identifier)
    os.environ['AUTORESEARCH_GPU'] = job['gpu_id']
    if args.action == 'start':
        start(job)
    elif args.action == 'run':
        signal.signal(signal.SIGTERM, stop)
        run_job(job)
    else:
        finish(job)


if __name__ == '__main__':
    main()
