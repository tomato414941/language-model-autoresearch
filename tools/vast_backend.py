"""Call vast-rental, which rents, checks, watches and returns this loop's machines and keeps their ledger."""

import json
import os
from pathlib import Path
import shutil
import subprocess

# Machines that may fail vast-rental's check in one rent, within the day's failure budget.
TRIES = 10
# vast-rental gives each machine at most about half an hour to start, take the key and pass its check.
RENT_SECONDS = TRIES * 35 * 60
# Each of vast-rental's calls to Vast.ai is tried four times over about five minutes; a machine that is still not
# gone after this is the watcher's to send back.
RETURN_SECONDS = 15 * 60


def executable():
    return Path(shutil.which('vast-rental') or Path.home() / '.local/bin/vast-rental').resolve()


def command(arguments, *, key=None, timeout=60, check=True):
    environment = os.environ.copy()
    if key is not None:
        environment['VAST_SSH_KEY_FILE'] = str(key)
    return subprocess.run([str(executable()), *arguments], env=environment,
                          capture_output=True, text=True, timeout=timeout, check=check)


def entries():
    """The ledger's events, each once, read without changing it."""
    path = executable().parent / 'data/ledger.jsonl'
    if not path.exists():
        return []
    found = {}
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except ValueError:  # a line that a writer left unfinished
            continue
        found[entry.get('id', line)] = entry
    return list(found.values())


def rent(job, image, profile, key, query, failure_budget, directory):
    """The machine that vast-rental kept for the job, as it printed it, or None when none passed within the tries and
    the failure budget. Its whole output goes to vast-rental.log in the job's folder."""
    arguments = ['rent', profile.gpu_name, '--image', image, '--label', job['label'],
                 '--hours', str(job['gpu_settings']['rental_seconds'] / 3600),
                 '--max-price', str(job['max_hourly_cost']), '--disk', str(profile.values['disk_gb']),
                 '--query', json.dumps(query), '--tries', str(TRIES), '--failure-budget', str(failure_budget)]
    try:
        result = command(arguments, key=key, timeout=RENT_SECONDS, check=False)
    except subprocess.TimeoutExpired as error:
        with (directory / 'vast-rental.log').open('a') as log:
            for value in (error.stdout, error.stderr):
                if value:
                    log.write(value.decode(errors='replace') if isinstance(value, bytes) else value)
        raise
    with (directory / 'vast-rental.log').open('a') as log:
        log.write(result.stdout + result.stderr)
    rows = []
    for line in result.stdout.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    last = rows[-1] if rows else {}
    if result.returncode == 0 and type(last.get('rented')) is int and last['rented'] > 0:
        if (not last.get('ssh_config') or last.get('ssh_host') != f"vast-{last['rented']}"
                or not last.get('deadline')):
            raise RuntimeError('vast-rental kept a machine without saying how to reach it or when it goes back.')
        return last
    if result.returncode == 1 and 'rented' in last and last['rented'] is None:
        return None
    raise RuntimeError('vast-rental failed; its full output is in vast-rental.log.')


def note(instance, text, values):
    command(['note', str(instance), text, '--values', json.dumps(values)])


def reject(instance, detail):
    command(['reject', str(instance), detail])


def give_back(label):
    """Send back every machine with the label; a machine that Vast.ai does not destroy now is the watcher's to send
    back, and makes this raise."""
    command(['return', '--label', label], timeout=RETURN_SECONDS)
