"""Record a finished attempt in its GPU track and decide it by the fixed rule on the three OLMES suites."""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import jobs, measurement
from gpu_config import profile_for_job

SUITE_METRICS = ('qa_bpb', 'math_bpb', 'code_bpb')
FRAME_SOURCES = ('prepare.py', 'program.md', 'pyproject.toml', 'uv.lock')
HEADER = ['commit', *SUITE_METRICS, 'val_bpb', 'memory_gb', 'status', 'description']
ARTIFACTS = ('run.log', 'prepare.log', 'bootstrap.log', 'run.exit', 'gpu-observations.jsonl', 'olmes.log',
             'olmes-summary.json', 'olmes-metrics.jsonl', 'olmes-details.tar.gz', 'final-model.pt')


def read_metrics(job, directory):
    """The suites, the reference val_bpb and the peak memory of a complete run, checked against its summary, its
    prepared evaluation and its fixed time budgets."""
    if job.get('exit_code') != 0 or job['phase'] != 'evaluated':
        raise ValueError('The experiment did not exit successfully.')
    log = (directory / 'run.log').read_text()
    values = {}
    for name in (*SUITE_METRICS, 'peak_vram_mb', 'training_seconds', 'total_seconds'):
        matches = re.findall(r'^' + name + r':\s+(\S+)', log, flags=re.MULTILINE)
        if len(matches) != 1:
            raise ValueError('The summary must contain one value for each required metric.')
        values[name] = float(matches[0])
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError('The summary metrics must be finite.')
    if values['peak_vram_mb'] < 0:
        raise ValueError('Peak memory must be nonnegative.')
    if not jobs.training_seconds(job) <= values['training_seconds'] <= values['total_seconds'] <= jobs.rental_seconds(job):
        raise ValueError('The summary does not satisfy the fixed training and rental time budgets.')
    if not all(values[name] > 0 for name in SUITE_METRICS):
        raise ValueError('Every native suite must have finite positive BPB.')
    result = dict(val_bpb=reference_val_bpb(log, r'^val_bpb:\s+(\S+)', single=True),
                  memory_gb=values['peak_vram_mb'] / 1024)
    summary = json.loads((directory / 'olmes-summary.json').read_text())
    runtime = json.loads((jobs.job_path(job['id']).parent / 'runtime.json').read_text())
    expected = runtime['olmes']
    if (not summary['complete'] or summary['olmes_revision'] != expected['revision']
            or summary['approved_config_fix'] != expected['approved_config_fix']
            or summary['model_source_sha256'] != job['source_sha256'][job['model_source']]
            or summary['native_request_manifest_sha256'] != expected['request_manifest_sha256']
            or summary['tokenizer_sha256'] != runtime['tokenizer']['tokenizer.pkl']
            or any(summary[name] != expected[name] for name in ('leaf_tasks', 'examples', 'requests'))
            or set(summary['suites']) != set(SUITE_METRICS)
            or any(summary['suites'][name] != values[name] for name in SUITE_METRICS)):
        raise ValueError('The complete native artifacts must match the prepared suites and summary.')
    if (jobs.digest(directory / 'olmes-metrics.jsonl') != summary['metrics_all_sha256']
            or jobs.digest(directory / 'final-model.pt') != summary['checkpoint_sha256']):
        raise ValueError('The native metrics or model checkpoint are incomplete.')
    # A runtime fixes only the parts of the computation that it names.
    for name in ('computation', 'computed_requests', 'inference_batch_size'):
        if name in expected and summary.get(name) != expected[name]:
            raise ValueError('The evaluation computation differs from the prepared plan.')
    result.update({name: values[name] for name in SUITE_METRICS})
    return result


def reference_val_bpb(log, pattern, single=False):
    """The finite reference val_bpb that the log printed, or None."""
    observed = re.findall(pattern, log, flags=re.MULTILINE)
    if not observed or (single and len(observed) != 1):
        return None
    try:
        value = float(observed[-1])
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def verify_artifacts(record, directory):
    for name, sha in record['artifacts'].items():
        if jobs.digest(directory / name) != sha:
            raise ValueError('A previously recorded artifact changed.')


def decide(job, directory, incumbent):
    """The record of the attempt: its provenance and measurement, and its metrics and decision against the incumbent,
    or a crash with what went wrong."""
    frame = {name: job['source_sha256'][name] for name in FRAME_SOURCES}
    frame['gpu_name'] = job['gpu_name']
    conditions = measurement.from_snapshot(job, jobs.job_path(job['id']).parent)
    if incumbent is not None:
        previous_job = jobs.load(incumbent['id'])
        if previous_job['gpu_id'] != job['gpu_id']:
            raise ValueError('Compare a candidate only within its own GPU track.')
        previous_frame = {name: previous_job['source_sha256'][name] for name in FRAME_SOURCES}
        previous_frame['gpu_name'] = previous_job['gpu_name']
        if previous_frame != incumbent['frame']:
            raise ValueError('The incumbent source provenance differs from its recorded frame.')
        previous = measurement.from_snapshot(previous_job, jobs.job_path(previous_job['id']).parent)
        if previous != incumbent['measurement']:
            raise ValueError('The incumbent measurement differs from its recorded identity.')
        previous_directory = jobs.results_directory(previous_job) / 'runs' / incumbent['id']
        verify_artifacts(incumbent, previous_directory)
        previous_log = (previous_directory / 'run.log').read_text()
        for name in SUITE_METRICS:
            logged = re.findall(r'^' + name + r':\s+(\S+)', previous_log, flags=re.MULTILINE)
            if len(logged) != 1 or float(logged[0]) != incumbent[name]:
                raise ValueError('The incumbent metric differs from its original log.')
        if conditions != previous and not job['baseline']:
            raise ValueError('A candidate requires a baseline under the same measurement conditions.')
        if job['baseline'] and conditions == previous:
            raise ValueError('This measurement already has a baseline; do not repeat it for a rule change.')
    elif not job['baseline']:
        raise ValueError('Record a successful baseline before deciding on candidates.')
    record = dict(id=job['id'], commit=job['commit'], frame=frame, measurement=conditions, gpu_id=job['gpu_id'],
                  metric=job['metric'], recorded_at=time.time(), description=' '.join(job['description'].split()),
                  incumbent=None if job['baseline'] else incumbent['id'])
    try:
        metrics = read_metrics(job, directory)
    except (OSError, KeyError, ValueError, TypeError) as error:
        log_path = directory / 'run.log'
        reference = (reference_val_bpb(log_path.read_text(), r'^(?:reference_)?val_bpb:\s+(\S+)')
                     if log_path.exists() else None)
        record.update(status='crash', val_bpb=reference, memory_gb=0.0, validation_error=str(error))
    else:
        accepted = job['baseline'] or measurement.decision_helper(
            job, jobs.job_path(job['id']).parent, 'suites_improve')(
            {name: metrics[name] for name in SUITE_METRICS}, {name: incumbent[name] for name in SUITE_METRICS})
        record.update(metrics, status='keep' if accepted else 'discard')
    record['artifacts'] = {name: jobs.digest(directory / name) for name in ARTIFACTS if (directory / name).exists()}
    return record


def write_outputs(records, destination):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
    writer.writerow(HEADER)
    for row in records:
        suites = ['0.000000' if row['status'] == 'crash' else format(row[name], '.17g') for name in SUITE_METRICS]
        writer.writerow([row['commit'][:7],
                         *suites,
                         '0.000000' if row['val_bpb'] is None else format(row['val_bpb'], '.17g'),
                         format(row['memory_gb'], '.1f'), row['status'], row['description']])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tsv.tmp')
    temporary.write_text(stream.getvalue())
    temporary.replace(destination)


def write_best(records, profile):
    directory = profile.directory(jobs.ROOT)
    directory.mkdir(parents=True, exist_ok=True)
    best = next((row for row in reversed(records) if row['status'] == 'keep'), None)
    if best is not None:
        job = jobs.load(best['id'])
        if job['gpu_id'] != profile.identifier:
            raise ValueError('The current best must belong to its GPU track.')
        archive = jobs.job_path(job['id']).parent / 'source.tar.gz'
        if jobs.digest(archive) != job['source_archive_sha256']:
            raise ValueError('The current best source archive differs from its recorded hash.')
        own = directory / 'best'
        own.mkdir(exist_ok=True)
        temporary_archive = own / 'source.tar.gz.tmp'
        shutil.copy2(archive, temporary_archive)
        temporary_archive.replace(own / 'source.tar.gz')
        model_source = job['model_source']
        with tarfile.open(archive) as saved:
            source = saved.extractfile(model_source).read()
        if hashlib.sha256(source).hexdigest() != job['source_sha256'][model_source]:
            raise ValueError('The current best training source differs from its recorded hash.')
        temporary_source = own / 'train.py.tmp'
        temporary_source.write_bytes(source)
        temporary_source.replace(own / 'train.py')
        best = dict(best, model_source=model_source, train_source=str((own / 'train.py').relative_to(jobs.ROOT)),
                    source_archive=str((own / 'source.tar.gz').relative_to(jobs.ROOT)))
    path = directory / 'best.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(best, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def record_run(identifier):
    """Record the finished job once in its track's ledger, then rebuild the track's TSV and current best from the
    ledger."""
    job = jobs.load(identifier)
    if job['phase'] not in ('evaluated', 'failed'):
        raise ValueError('The original job is still pending; observe it without recording a failed attempt.')
    profile = profile_for_job(job)
    results = profile.results_directory(jobs.ROOT)
    directory = results / 'runs' / identifier
    directory.mkdir(parents=True, exist_ok=True)
    ledger = results / 'decisions.json'
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
        write_outputs(records, profile.directory(jobs.ROOT) / 'results.tsv')
        write_best(records, profile)
    print(json.dumps(record, indent=2, allow_nan=False))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('identifier')
    args = parser.parse_args()
    record_run(args.identifier)


if __name__ == '__main__':
    main()
