"""Record completed attempts and apply the fixed CORE adoption rule to native artifacts."""

import argparse
import csv
import fcntl
import io
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import prepare
from tools import rental

HEADER = ['commit', 'core_metric', 'val_bpb', 'memory_gb', 'status', 'description']


def read_metrics(job, directory):
    if job.get('exit_code') != 0 or job['phase'] != 'evaluated':
        raise ValueError('The experiment did not exit successfully.')
    result = json.loads((directory / 'core.json').read_text())
    labels = [task['label'] for task in prepare.verify_core_assets()]
    for name in ('results', 'centered_results'):
        if list(result[name]) != labels or not all(math.isfinite(value) for value in result[name].values()):
            raise ValueError('The native CORE result must contain all 22 finite task scores in order.')
    core = result['core_metric']
    if not math.isfinite(core):
        raise ValueError('The aggregate CORE score must be finite.')
    log = (directory / 'run.log').read_text()
    fields = dict(re.findall(r'^([a-z_]+):\s+(\S+)', log, flags=re.MULTILINE))
    values = {name: float(fields[name]) for name in
              ('core_metric', 'peak_vram_mb', 'training_seconds', 'total_seconds')}
    if not all(math.isfinite(value) for value in values.values()) or values['core_metric'] != core:
        raise ValueError('The summary must contain finite metrics and the full-precision native CORE score.')
    if not prepare.TIME_BUDGET <= values['training_seconds'] <= values['total_seconds'] <= rental.LEASE_SECONDS:
        raise ValueError('The summary does not satisfy the fixed training and rental time budgets.')
    with (directory / 'core.csv').open(newline='') as source:
        rows = [[value.strip() for value in row] for row in csv.reader(source)]
    expected = [['Task', 'Accuracy', 'Centered']]
    expected.extend([[label, f"{result['results'][label]:.6f}",
                      f"{result['centered_results'][label]:.6f}"] for label in labels])
    expected.append(['CORE', '', f'{core:.6f}'])
    if rows != expected:
        raise ValueError('The native CSV is incomplete or differs from the native JSON.')
    try:
        reference = float(fields.get('val_bpb', 'nan'))
        reference = reference if math.isfinite(reference) else None
    except ValueError:
        reference = None
    return dict(core_metric=core, val_bpb=reference, memory_gb=values['peak_vram_mb'] / 1024)


def decide(job, directory, incumbent):
    frame = {name: job['source_sha256'][name] for name in rental.FRAME_SOURCES}
    frame['gpu_name'] = job['gpu_name']
    if incumbent and frame != incumbent['frame'] and not job['baseline']:
        raise ValueError('A candidate requires a baseline measured under the same frame.')
    if job['baseline'] and incumbent and frame == incumbent['frame']:
        raise ValueError('This frame already has a baseline; do not replace it with another run.')
    if not job['baseline'] and incumbent is None:
        raise ValueError('Record a successful baseline before deciding on candidates.')
    record = dict(id=job['id'], commit=job['commit'], frame=frame, recorded_at=time.time(),
                  description=' '.join(job['description'].split()),
                  incumbent=None if job['baseline'] else incumbent['id'])
    try:
        metrics = read_metrics(job, directory)
    except (OSError, KeyError, ValueError, TypeError) as error:
        record.update(status='crash', core_metric=None, val_bpb=None, memory_gb=0.0,
                      validation_error=str(error))
        log_path = directory / 'run.log'
        if log_path.exists():
            reference = re.findall(r'^val_bpb:\s+(\S+)', log_path.read_text(), flags=re.MULTILINE)
            if reference:
                try:
                    value = float(reference[-1])
                    if math.isfinite(value):
                        record['val_bpb'] = value
                except ValueError:
                    pass
    else:
        accepted = job['baseline'] or prepare.core_improves(metrics['core_metric'], incumbent['core_metric'])
        record.update(metrics, status='keep' if accepted else 'discard')
    record['artifacts'] = {name: rental.digest(directory / name)
                           for name in ('run.log', 'core.json', 'core.csv') if (directory / name).exists()}
    return record


def write_outputs(records):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
    writer.writerow(HEADER)
    for row in records:
        writer.writerow([row['commit'][:7],
                         'NA' if row['core_metric'] is None else format(row['core_metric'], '.17g'),
                         'NA' if row['val_bpb'] is None else format(row['val_bpb'], '.6f'),
                         format(row['memory_gb'], '.1f'), row['status'], row['description']])
    temporary = ROOT / 'results.tsv.tmp'
    temporary.write_text(stream.getvalue())
    temporary.replace(ROOT / 'results.tsv')


def record_run(identifier):
    job = rental.load(identifier)
    if job['phase'] not in ('evaluated', 'failed'):
        raise ValueError('The original job is still pending; observe it without recording a failed attempt.')
    directory = ROOT / 'results/runs' / identifier
    directory.mkdir(parents=True, exist_ok=True)
    ledger = ROOT / 'results/decisions.json'
    with ledger.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = json.loads(ledger.read_text()) if ledger.exists() else []
        record = next((row for row in records if row['id'] == identifier), None)
        if record is not None:
            for name, sha in record['artifacts'].items():
                if rental.digest(directory / name) != sha:
                    raise ValueError('A previously recorded native artifact changed.')
        else:
            incumbent = next((row for row in reversed(records) if row['status'] == 'keep'), None)
            record = decide(job, directory, incumbent)
            records.append(record)
            temporary = ledger.with_suffix('.tmp')
            temporary.write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
            temporary.replace(ledger)
        # The ledger is authoritative; rebuilding the TSV repairs an interrupted export.
        write_outputs(records)
    print(json.dumps(record, indent=2, allow_nan=False))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('identifier')
    args = parser.parse_args()
    record_run(args.identifier)


if __name__ == '__main__':
    main()
