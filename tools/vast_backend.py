"""Use the shared Vast rental tool without taking over its machine checks or ledger."""

import json
import os
from pathlib import Path
import shutil
import subprocess


def executable():
    return Path(shutil.which('vast-rental') or Path.home() / '.local/bin/vast-rental').resolve()


def command(arguments, *, key=None, timeout=30, check=True):
    environment = os.environ.copy()
    if key is not None:
        environment['VAST_SSH_KEY_FILE'] = str(key)
    return subprocess.run([str(executable()), *arguments], env=environment,
                          capture_output=True, text=True, timeout=timeout, check=check)


def entries():
    path = executable().parent / 'data/ledger.jsonl'
    if not path.exists():
        return []
    found = {}
    for line in path.read_text().splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        found[entry.get('id', line)] = entry
    return list(found.values())


def rent(job, manifest, profile, key, query, timeout, directory):
    arguments = ['rent', profile.gpu_name, '--image', manifest['container_image'],
                 '--label', job['label'], '--hours', str(job['gpu_settings']['rental_seconds'] / 3600),
                 '--max-price', str(job['max_hourly_cost']), '--disk', str(profile.values['disk_gb']),
                 '--query', json.dumps(query), '--tries', '1']
    try:
        result = command(arguments, key=key, timeout=timeout, check=False)
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
    if not rows:
        raise RuntimeError('vast-rental returned no result; its full log was preserved.')
    last = rows[-1]
    if result.returncode == 0 and type(last.get('rented')) is int and last['rented'] > 0:
        if not last.get('ssh_config') or last.get('ssh_host') != f"vast-{last['rented']}":
            raise RuntimeError('vast-rental did not return the connection configuration.')
    elif result.returncode != 1 or last.get('rented') is not None or 'failed' not in last:
        raise RuntimeError('vast-rental failed; its full log was preserved.')
    return last, rows
