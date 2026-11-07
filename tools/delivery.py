"""Check a job's runtime archives in R2 before its machine is rented, and deliver them with its sources to it.

The machine fetches the large archives straight from R2. Only the small bootstrap, whose Python does that fetching,
and the job's sources go over SSH.
"""

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import jobs, r2
from tools.parallel_ssh import stream_archive
from tools.runtime_archives import archive_specs, deployment_names

BOOTSTRAP = 'gpu-bootstrap.tar.gz'
SOURCES = 'source.tar.gz'


def store():
    """The R2 settings, which a run cannot do without."""
    config = r2.settings()
    if config is None:
        raise RuntimeError('Set the R2 variables: large files travel between a rented machine and R2 only.')
    return config


def verify_archives(manifest, artifacts, failure):
    """Check that R2 holds each runtime archive under its frozen digest, uploading one that it lacks from here, and
    that the bootstrap here matches its digest; raise `failure` when one is wrong or missing."""
    deployment_names(manifest)
    config = store()
    for name, sha256 in archive_specs(manifest):
        path = artifacts / name
        if name == BOOTSTRAP:
            if not path.is_file() or jobs.digest(path) != sha256:
                raise RuntimeError(failure)
        elif r2.stored(config, name, sha256) is None:
            if not path.is_file():
                raise RuntimeError(failure)
            r2.upload(config, path, sha256, name)


def interpreter(manifest):
    """The bootstrap's own Python on the machine, relative to its work folder."""
    version = manifest['python_version']
    if not all(part.isdigit() for part in version.split('.')) or version.count('.') != 2:
        raise ValueError('Invalid packaged Python version.')
    return f'.bootstrap/.python/cpython-{version}-linux-x86_64-gnu/bin/python3.12'


def send(job, connection, remote, until):
    """Put the job's runtime and sources on its machine by `until`, each archive checked against its frozen digest as
    it unpacks, and keep each receipt in the job."""
    directory = jobs.job_path(job['id']).parent
    manifest = json.loads((directory / 'runtime.json').read_text())
    hashes = dict(archive_specs(manifest))
    config = store()
    python = interpreter(manifest)
    remote('mkdir -p autoresearch/.bootstrap')
    job['archive_transfers'] = []

    def keep(receipt):
        job['archive_transfers'].append(receipt)
        jobs.save(job)
        if receipt.get('cleanup_error'):
            # The archive arrived whole but its cleanup lost the connection: go on only while the machine answers.
            remote('true', capture_output=True, text=True)

    def remaining():
        left = until - time.time()
        if left <= 0:
            raise TimeoutError('The transfers took the time that the run needs.')
        return left

    keep(stream_archive(connection, jobs.artifact_directory(job) / BOOTSTRAP, 'tar -xzf - -C autoresearch/.bootstrap',
                        sha256=hashes[BOOTSTRAP], timeout=remaining()))
    remote('cat > autoresearch/.bootstrap/r2_transfer.py', input=r2.TRANSFER.read_bytes())
    for name, unpack in (('gpu-env.tar.xz', f'cd autoresearch && {python} .bootstrap/runtime_unpack.py extract'),
                         ('gpu-assets.tar.gz', 'tar -xzf - -C autoresearch')):
        size = r2.stored(config, name, hashes[name])
        if size is None:
            raise RuntimeError('A runtime archive is missing from R2.')
        keep(r2.fetch(connection, config, name, hashes[name], size, unpack, python, timeout=remaining(),
                      expires=max(60, int(job['deadline'] - time.time()))))
    keep(stream_archive(connection, directory / SOURCES, 'tar -xzf - -C autoresearch',
                        sha256=job['source_archive_sha256'], timeout=remaining()))
