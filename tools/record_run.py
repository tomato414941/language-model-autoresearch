"""Record completed attempts and apply the user's fixed BPB adoption rule."""

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
from tools import measurement, rental

HEADER = ['commit', 'val_bpb', 'memory_gb', 'status', 'description']


def read_metrics(job, directory):
    if job.get('exit_code') != 0 or job['phase'] != 'evaluated':
        raise ValueError('The experiment did not exit successfully.')
    log = (directory / 'run.log').read_text()
    values = {}
    for name in ('val_bpb', 'peak_vram_mb', 'training_seconds', 'total_seconds'):
        matches = re.findall(r'^' + name + r':\s+(\S+)', log, flags=re.MULTILINE)
        if len(matches) != 1:
            raise ValueError('The summary must contain one value for each required metric.')
        values[name] = float(matches[0])
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError('The summary metrics must be finite.')
    if values['peak_vram_mb'] < 0:
        raise ValueError('Peak memory must be nonnegative.')
    if not prepare.TIME_BUDGET <= values['training_seconds'] <= values['total_seconds'] <= rental.LEASE_SECONDS:
        raise ValueError('The summary does not satisfy the fixed training and rental time budgets.')
    return dict(val_bpb=values['val_bpb'], memory_gb=values['peak_vram_mb'] / 1024)


def verify_artifacts(record, directory):
    for name, sha in record['artifacts'].items():
        if rental.digest(directory / name) != sha:
            raise ValueError('A previously recorded artifact changed.')


def decide(job, directory, incumbent):
    frame = {name: job['source_sha256'][name] for name in rental.FRAME_SOURCES}
    frame['gpu_name'] = job['gpu_name']
    conditions = measurement.from_snapshot(job, rental.job_path(job['id']).parent)
    previous = None
    if incumbent:
        previous_job = rental.load(incumbent['id'])
        previous_frame = {name: previous_job['source_sha256'][name] for name in rental.FRAME_SOURCES}
        previous_frame['gpu_name'] = previous_job['gpu_name']
        if previous_frame != incumbent['frame']:
            raise ValueError('The incumbent source provenance differs from its recorded frame.')
        previous = measurement.from_snapshot(previous_job, rental.job_path(previous_job['id']).parent)
        recorded = incumbent.get('measurement')
        if recorded and recorded['definition'].get('metric') == 'val_bpb' and recorded != previous:
            raise ValueError('The incumbent measurement differs from its recorded identity.')
        previous_directory = ROOT / 'results/runs' / incumbent['id']
        verify_artifacts(incumbent, previous_directory)
        logged = re.findall(r'^val_bpb:\s+(\S+)', (previous_directory / 'run.log').read_text(),
                            flags=re.MULTILINE)
        if len(logged) != 1 or float(logged[0]) != incumbent['val_bpb']:
            raise ValueError('The incumbent BPB differs from its original log.')
    if previous and conditions != previous and not job['baseline']:
        raise ValueError('A candidate requires a baseline under the same measurement conditions.')
    if job['baseline'] and previous and conditions == previous:
        raise ValueError('This measurement already has a baseline; do not repeat it for a rule change.')
    if not job['baseline'] and incumbent is None:
        raise ValueError('Record a successful baseline before deciding on candidates.')
    record = dict(id=job['id'], commit=job['commit'], frame=frame, measurement=conditions,
                  metric='val_bpb', recorded_at=time.time(),
                  description=' '.join(job['description'].split()),
                  incumbent=None if job['baseline'] else incumbent['id'])
    try:
        metrics = read_metrics(job, directory)
    except (OSError, KeyError, ValueError, TypeError) as error:
        record.update(status='crash', val_bpb=None, memory_gb=0.0, validation_error=str(error))
        log_path = directory / 'run.log'
        if log_path.exists():
            observed = re.findall(r'^val_bpb:\s+(\S+)', log_path.read_text(), flags=re.MULTILINE)
            if observed:
                try:
                    value = float(observed[-1])
                    if math.isfinite(value):
                        record['val_bpb'] = value
                except ValueError:
                    pass
    else:
        accepted = job['baseline'] or prepare.bpb_improves(metrics['val_bpb'], incumbent['val_bpb'])
        record.update(metrics, status='keep' if accepted else 'discard')
    record['artifacts'] = {name: rental.digest(directory / name) for name in
                           ('run.log', 'prepare.log', 'bootstrap.log', 'run.exit', 'gpu-observations.jsonl')
                           if (directory / name).exists()}
    return record


def write_outputs(records):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
    writer.writerow(HEADER)
    for row in records:
        writer.writerow([row['commit'][:7],
                         '0.000000' if row['status'] == 'crash' else format(row['val_bpb'], '.17g'),
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
            verify_artifacts(record, directory)
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
