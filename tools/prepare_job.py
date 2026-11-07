"""Freeze the current sources with the active GPU's prepared runtime as a job in queue/, for tools/rental.py to run."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpu_config import active_gpu, read_config, source_files
from tools import delivery, jobs, measurement

OLMES_SOURCES = tuple('evaluation/' + name for name in (
    'arc_macro.py', 'checkpoint.py', 'count_tokens.py', 'count_gold_tokens.py', 'gold_bpb.py', 'model.py',
    'prepare_requests.py', 'protocol.py', 'run_prepared.py', 'tokenizer_adapter.py', 'verify_requests.py',
    'encoding_cache.py', 'prepare_encodings.py', 'pyproject.toml', 'uv.lock', 'patches/olmes-arc-bpb-macro.patch',
))


def prepare_job(commit, baseline, description, expected_eval_seconds):
    """Freeze the sources, the runtime manifest and the GPU's settings as a prepared job; a GPU without a kept
    result measures its baseline first."""
    from tools.package_runtime import verify_tokenizer

    profile = active_gpu()
    settings = profile.settings()
    artifacts = profile.artifacts_directory(jobs.ROOT)
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Pass the full commit from the local Git checkout.')
    tokenizer_hashes = verify_tokenizer()
    manifest = json.loads((artifacts / 'runtime.json').read_text())
    if manifest['gpu_id'] != profile.identifier:
        raise RuntimeError('Prepare the selected GPU track runtime before creating its job.')
    if manifest['evaluation_metric'] != measurement.OLMES_METRIC:
        raise RuntimeError('Package the OLMES evaluation into the runtime before creating a job.')
    if manifest['tokenizer'] != tokenizer_hashes:
        raise RuntimeError('The tokenizer differs from the packaged BPB assets.')
    sources = source_files(profile) + OLMES_SOURCES
    hashes = {name: jobs.digest(jobs.ROOT / name) for name in sources}
    delivery.verify_archives(manifest, artifacts, 'A runtime archive is incomplete or changed.')
    identifier = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
    job = dict(id=identifier, label='autoresearch-' + profile.identifier + '-' + identifier,
               gpu_id=profile.identifier, gpu_name=profile.gpu_name, gpu_settings=settings,
               model_source=f'gpus/{profile.identifier}/train.py', commit=commit, description=description,
               phase='prepared', source_sha256=hashes, max_hourly_cost=settings['max_hourly_cost'],
               expected_eval_seconds=expected_eval_seconds, metric=measurement.OLMES_METRIC)
    jobs.check_fit(job)
    ledger = profile.results_directory(jobs.ROOT) / 'decisions.json'
    records = json.loads(ledger.read_text()) if ledger.exists() else []
    job['baseline'] = baseline or not any(row['status'] == 'keep' for row in records)
    directory = jobs.job_path(identifier).parent
    directory.mkdir(parents=True)
    manifest['sources'] = hashes
    runtime_path = directory / 'runtime.json'
    runtime_path.write_text(json.dumps(manifest, indent=2) + '\n')
    source_dir = directory / 'source'
    source_dir.mkdir()
    for name in sources:
        (source_dir / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(jobs.ROOT / name, source_dir / name)
    shutil.copy2(jobs.ROOT / 'tools/gpu_run.sh', source_dir / 'gpu_run.sh')
    with tarfile.open(directory / 'source.tar.gz', 'w:gz') as archive:
        for path in sorted(source_dir.iterdir()):
            archive.add(path, arcname=path.name)
    job.update(source_archive_sha256=jobs.digest(directory / 'source.tar.gz'),
               runtime_manifest_sha256=jobs.digest(runtime_path))
    jobs.save(job)
    print(json.dumps({'id': identifier, 'commit': commit, 'phase': job['phase'], 'baseline': job['baseline']},
                     indent=2))
    return job


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', choices=tuple(read_config()['gpus']), help='prepare for this GPU instead of the '
                                                                           'active one')
    parser.add_argument('--commit', required=True)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--description', required=True)
    parser.add_argument('--expected-eval-seconds', type=float, required=True)
    args = parser.parse_args()
    if args.gpu:
        os.environ['AUTORESEARCH_GPU'] = args.gpu
    prepare_job(args.commit, args.baseline, args.description, args.expected_eval_seconds)


if __name__ == '__main__':
    main()
