"""Record completed attempts and apply their fixed suite or temporary BPB adoption rule."""

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

SUITE_METRICS = ('qa_bpb', 'math_bpb', 'code_bpb')
FRAME_SOURCES = ('prepare.py', 'program.md', 'pyproject.toml', 'uv.lock')
HEADER = ['commit', *SUITE_METRICS, 'val_bpb', 'memory_gb', 'status', 'description']


def read_metrics(job, directory):
    if job.get('exit_code') != 0 or job['phase'] != 'evaluated':
        raise ValueError('The experiment did not exit successfully.')
    log = (directory / 'run.log').read_text()
    values = {}
    native = job.get('metric') == measurement.OLMES_METRIC
    required = SUITE_METRICS if native else ('val_bpb',)
    for name in (*required, 'peak_vram_mb', 'training_seconds', 'total_seconds'):
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
    result = dict(val_bpb=values.get('val_bpb'), memory_gb=values['peak_vram_mb'] / 1024)
    if native:
        if not all(values[name] > 0 for name in SUITE_METRICS):
            raise ValueError('Every native suite must have finite positive BPB.')
        observed = re.findall(r'^val_bpb:\s+(\S+)', log, flags=re.MULTILINE)
        if len(observed) == 1:
            try:
                value = float(observed[0])
                if math.isfinite(value):
                    result['val_bpb'] = value
            except ValueError:
                pass
        summary = json.loads((directory / 'olmes-summary.json').read_text())
        runtime = json.loads((rental.job_path(job['id']).parent / 'runtime.json').read_text())
        expected = runtime['olmes']
        if (not summary['complete'] or summary['olmes_revision'] != expected['revision']
                or summary['approved_config_fix'] != expected['approved_config_fix']
                or summary['model_source_sha256'] != job['source_sha256']['train.py']
                or summary['native_request_manifest_sha256'] != expected['request_manifest_sha256']
                or summary['tokenizer_sha256'] != runtime['tokenizer']['tokenizer.pkl']
                or any(summary[name] != expected[name] for name in ('leaf_tasks', 'examples', 'requests'))
                or set(summary['suites']) != set(SUITE_METRICS)
                or any(summary['suites'][name] != values[name] for name in SUITE_METRICS)):
            raise ValueError('The complete native artifacts must match the prepared suites and summary.')
        if (rental.digest(directory / 'olmes-metrics.jsonl') != summary['metrics_all_sha256']
                or rental.digest(directory / 'final-model.pt') != summary['checkpoint_sha256']):
            raise ValueError('The native metrics or model checkpoint are incomplete.')
        for name in ('computation', 'computed_requests'):
            if name in expected and summary.get(name) != expected[name]:
                raise ValueError('The computed correct-answer measurements differ from the prepared plan.')
        result.update({name: values[name] for name in SUITE_METRICS})
    return result


def verify_artifacts(record, directory):
    for name, sha in record['artifacts'].items():
        if rental.digest(directory / name) != sha:
            raise ValueError('A previously recorded artifact changed.')


def decide(job, directory, incumbent):
    frame = {name: job['source_sha256'][name] for name in FRAME_SOURCES}
    frame['gpu_name'] = job['gpu_name']
    conditions = measurement.from_snapshot(job, rental.job_path(job['id']).parent)
    previous = None
    shared_metric = job.get('metric', 'val_bpb')
    comparison = conditions
    if incumbent:
        previous_job = rental.load(incumbent['id'])
        previous_frame = {name: previous_job['source_sha256'][name] for name in FRAME_SOURCES}
        previous_frame['gpu_name'] = previous_job['gpu_name']
        if previous_frame != incumbent['frame']:
            raise ValueError('The incumbent source provenance differs from its recorded frame.')
        previous = measurement.from_snapshot(previous_job, rental.job_path(previous_job['id']).parent)
        recorded = incumbent.get('measurement')
        if recorded and recorded['definition'].get('metric') in ('val_bpb', measurement.OLMES_METRIC) and recorded != previous:
            raise ValueError('The incumbent measurement differs from its recorded identity.')
        previous_directory = ROOT / 'results/runs' / incumbent['id']
        verify_artifacts(incumbent, previous_directory)
        previous_log = (previous_directory / 'run.log').read_text()
        names = SUITE_METRICS if previous['definition']['metric'] == measurement.OLMES_METRIC else ('val_bpb',)
        for name in names:
            logged = re.findall(r'^' + name + r':\s+(\S+)', previous_log, flags=re.MULTILINE)
            if len(logged) != 1 or float(logged[0]) != incumbent[name]:
                raise ValueError('The incumbent metric differs from its original log.')
        shared_metric = (measurement.OLMES_METRIC if
                         job.get('metric') == measurement.OLMES_METRIC
                         and previous_job.get('metric') == measurement.OLMES_METRIC else 'val_bpb')
        comparison = measurement.for_metric(job, rental.job_path(job['id']).parent, shared_metric)
        previous = measurement.for_metric(previous_job, rental.job_path(previous_job['id']).parent,
                                          shared_metric)
    if previous and comparison != previous and not job['baseline']:
        raise ValueError('A candidate requires a baseline under the same measurement conditions.')
    if job['baseline'] and previous and comparison == previous:
        raise ValueError('This measurement already has a baseline; do not repeat it for a rule change.')
    if not job['baseline'] and incumbent is None:
        raise ValueError('Record a successful baseline before deciding on candidates.')
    record = dict(id=job['id'], commit=job['commit'], frame=frame, measurement=conditions,
                  metric=job.get('metric', 'val_bpb'), recorded_at=time.time(),
                  description=' '.join(job['description'].split()),
                  decision_metric=shared_metric, comparison_measurement=comparison,
                  incumbent=None if job['baseline'] else incumbent['id'])
    try:
        metrics = read_metrics(job, directory)
        if shared_metric == 'val_bpb' and metrics['val_bpb'] is None:
            raise ValueError('A comparison on shared val_bpb requires its finite result.')
    except (OSError, KeyError, ValueError, TypeError) as error:
        record.update(status='crash', val_bpb=None, memory_gb=0.0, validation_error=str(error))
        log_path = directory / 'run.log'
        if log_path.exists():
            observed = re.findall(r'^(?:reference_)?val_bpb:\s+(\S+)', log_path.read_text(), flags=re.MULTILINE)
            if observed:
                try:
                    value = float(observed[-1])
                    if math.isfinite(value):
                        record['val_bpb'] = value
                except ValueError:
                    pass
    else:
        if job['baseline']:
            accepted = True
        elif shared_metric == measurement.OLMES_METRIC:
            helper = measurement.decision_helper(job, rental.job_path(job['id']).parent, 'suites_improve')
            accepted = helper({name: metrics[name] for name in SUITE_METRICS},
                              {name: incumbent[name] for name in SUITE_METRICS})
        else:
            accepted = prepare.bpb_improves(metrics['val_bpb'], incumbent['val_bpb'])
        record.update(metrics, status='keep' if accepted else 'discard')
    record['artifacts'] = {name: rental.digest(directory / name) for name in
                           ('run.log', 'prepare.log', 'bootstrap.log', 'run.exit', 'gpu-observations.jsonl',
                            'olmes.log', 'olmes-summary.json', 'olmes-metrics.jsonl',
                            'olmes-details.tar.gz', 'final-model.pt')
                           if (directory / name).exists()}
    return record


def write_outputs(records):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
    writer.writerow(HEADER)
    for row in records:
        suites = [('0.000000' if row['status'] == 'crash' else
                   'NA' if row.get(name) is None else format(row[name], '.17g'))
                  for name in SUITE_METRICS]
        writer.writerow([row['commit'][:7],
                         *suites,
                         '0.000000' if row['val_bpb'] is None else format(row['val_bpb'], '.17g'),
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
