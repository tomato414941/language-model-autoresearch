"""Compress the immutable runtime without changing its installed files or dependencies."""

import argparse
import copy
import gzip
import hashlib
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gpu_config import active_gpu
from tools.package_runtime import PYTHON_VERSION, digest

INTERPRETER = f'.python/cpython-{PYTHON_VERSION}-linux-x86_64-gnu/bin/python3.12'


def identity(rows):
    encoded = json.dumps(sorted(rows, key=lambda row: Path(row['path'])),
                         sort_keys=True, separators=(',', ':')).encode()
    return dict(sha256=hashlib.sha256(encoded).hexdigest(), entries=len(rows))


def tree_identity(root):
    """Identify file bytes, modes and symlink targets independently of compression."""
    rows = []
    for path in root.rglob('*'):
        info = path.lstat()
        row = dict(path=str(path.relative_to(root)), mode=stat.S_IMODE(info.st_mode))
        if stat.S_ISLNK(info.st_mode):
            row.update(type='symlink', target=os.readlink(path))
        elif stat.S_ISREG(info.st_mode):
            row.update(type='file', bytes=info.st_size, sha256=digest(path))
        elif stat.S_ISDIR(info.st_mode):
            row.update(type='directory')
        else:
            raise ValueError('The runtime contains an unsupported special file.')
        rows.append(row)
    return identity(rows)


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.sha256 = hashlib.sha256()

    def read(self, size=-1):
        value = self.stream.read(size)
        self.sha256.update(value)
        return value


def archive_identity(reference, bootstrap=None):
    """Hash canonical files and optionally copy its standalone Python into a bootstrap."""
    rows = {}
    with tarfile.open(reference, 'r|gz') as archive:
        for member in archive:
            name = str(PurePosixPath(member.name))
            if name == '.':
                continue
            row = dict(path=name, mode=member.mode)
            selected = bootstrap is not None and (name == 'uv' or name == '.python' or name.startswith('.python/'))
            info = copy.copy(member)
            info.name = name
            if member.isdir():
                row.update(type='directory')
            elif member.issym():
                row.update(type='symlink', target=member.linkname)
            elif member.islnk():
                target = rows[str(PurePosixPath(member.linkname))]
                row.update(type='file', bytes=target['bytes'], sha256=target['sha256'])
                target_name = str(PurePosixPath(member.linkname))
                if selected and not (target_name == 'uv' or target_name.startswith('.python/')):
                    # A bootstrap cannot refer to a file omitted from that archive.
                    info.type, info.linkname, info.size = tarfile.REGTYPE, '', target['bytes']
                    info.pax_headers = {key: value for key, value in info.pax_headers.items() if key != 'linkpath'}
                    with tarfile.open(reference, 'r:gz') as original:
                        with original.extractfile(member.linkname) as source:
                            bootstrap.addfile(info, source)
                    selected = False
            elif member.isfile():
                with archive.extractfile(member) as source:
                    stream = HashingReader(source)
                    if selected:
                        bootstrap.addfile(info, stream)
                    else:
                        while stream.read(1024*1024):
                            pass
                    row.update(type='file', bytes=member.size, sha256=stream.sha256.hexdigest())
            else:
                raise ValueError('The canonical runtime contains an unsupported special file.')
            if selected and not member.isfile():
                bootstrap.addfile(info)
            rows[name] = row
    return identity(list(rows.values()))


def attach(path, report):
    original = path.read_bytes()
    manifest = json.loads(original)
    if not report['unchanged_runtime_verified'] or manifest['gpu_env_sha256'] != report['canonical_gpu_env_sha256']:
        raise ValueError('The deployment must be verified against this manifest\'s immutable runtime.')
    for name, field in (('gpu-bootstrap.tar.gz', 'bootstrap_sha256'), ('gpu-env.tar.xz', 'gpu_env_xz_sha256')):
        if digest(path.parent / name) != report[field]:
            raise ValueError('The verified compressed deployment changed.')
    history = path.parent / 'history'
    history.mkdir(exist_ok=True)
    (history / ('runtime-' + hashlib.sha256(original).hexdigest() + '.json')).write_bytes(original)
    manifest['transport'] = {name: report[name] for name in
                             ('format', 'canonical_gpu_env_sha256', 'bootstrap_sha256',
                              'gpu_env_xz_sha256', 'runtime_tree', 'decoder_source_sha256')}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(path)


def add_parallel_decoder(bootstrap):
    """Add only the stdlib helper; the existing OS supplies liblzma through dpkg."""
    helper = ROOT / 'tools/unpack_runtime.py'
    payload = helper.read_bytes()
    temporary = bootstrap.with_suffix('.parallel.tmp')
    with tarfile.open(bootstrap, 'r|gz') as source, tarfile.open(temporary, 'w:gz', compresslevel=9) as target:
        for member in source:
            if str(PurePosixPath(member.name)) == 'runtime_unpack.py':
                continue
            if member.isfile():
                with source.extractfile(member) as contents:
                    target.addfile(member, contents)
            else:
                target.addfile(member)
        entry = tarfile.TarInfo('runtime_unpack.py')
        entry.size, entry.mode = len(payload), 0o644
        target.addfile(entry, io.BytesIO(payload))
    temporary.replace(bootstrap)
    return digest(helper)


def package(reference, artifacts, verify_only=False, manifest=None, verification_directory=None,
            relocation_check=False):
    reference, artifacts = Path(reference).resolve(), Path(artifacts).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    verification_directory = Path(verification_directory).resolve() if verification_directory else artifacts
    verification_directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    canonical_sha256 = digest(reference)
    bootstrap = artifacts / 'gpu-bootstrap.tar.gz'
    payload = artifacts / 'gpu-env.tar.xz'
    if verify_only:
        original = archive_identity(reference)
    else:
        with tarfile.open(bootstrap.with_suffix('.tmp'), 'w:gz', compresslevel=9) as archive:
            original = archive_identity(reference, archive)
        bootstrap.with_suffix('.tmp').replace(bootstrap)
        print('Canonical bootstrap packaged; transcoding the immutable archive.', flush=True)
        with gzip.open(reference, 'rb') as source, payload.with_suffix('.tmp').open('wb') as output:
            compressor = shutil.which('xz')
            if compressor:
                with subprocess.Popen([compressor, '--threads=' + str(min(4, os.cpu_count() or 1)),
                                       '-6', '-c'], stdin=subprocess.PIPE, stdout=output) as process:
                    try:
                        shutil.copyfileobj(source, process.stdin, 1024*1024)
                    finally:
                        process.stdin.close()
                    if process.wait() != 0:
                        raise RuntimeError('Runtime compression failed.')
            else:
                with lzma.LZMAFile(output, 'wb', preset=6) as compressed:
                    shutil.copyfileobj(source, compressed, 1024*1024)
        payload.with_suffix('.tmp').replace(payload)
    decoder_sha256 = add_parallel_decoder(bootstrap)
    print('Runtime compressed; verifying every decoded file with the standalone Python.', flush=True)
    imports = {}
    with tempfile.TemporaryDirectory(dir=artifacts, prefix='.bootstrap-check-') as boot_dir:
        boot = Path(boot_dir)
        with tarfile.open(bootstrap) as archive:
            archive.extractall(boot, filter='fully_trusted')
        if digest(boot / 'runtime_unpack.py') != decoder_sha256:
            raise ValueError('The parallel decoder differs from its packaged source.')
        verify_command = [str(boot / INTERPRETER), str(boot / 'runtime_unpack.py'), 'identity']
        verification_started = time.monotonic()
        with payload.open('rb') as source:
            restored = json.loads(subprocess.check_output(
                verify_command, stdin=source, text=True))
        decoder_verification_seconds = time.monotonic() - verification_started
        if restored != original:
            raise ValueError('The compressed runtime must preserve every canonical file byte, mode and link.')
        if relocation_check:
            with tempfile.TemporaryDirectory(dir=verification_directory, prefix='.transport-check-') as target_dir:
                target = Path(target_dir)
                with payload.open('rb') as source:
                    unpack_command = [str(boot / INTERPRETER), str(boot / 'runtime_unpack.py'), 'extract']
                    subprocess.run(unpack_command, cwd=target, stdin=source, check=True)
                if tree_identity(target) != original:
                    raise ValueError('The relocated runtime differs from the immutable archive.')
                environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
                for name in ('.venv', 'evaluation/.venv'):
                    code = ('import sys,torch,numpy,pyarrow,kernels,lzma; '
                            'print({"python":sys.version.split()[0],"torch":torch.__version__, '
                            '"numpy":numpy.__version__,"pyarrow":pyarrow.__version__})')
                    if name.startswith('evaluation/'):
                        code += '; from oe_eval import run_eval'
                    imports[name] = subprocess.check_output(
                        [str(target / name / 'bin/python'), '-c', code], cwd=target,
                        text=True, env=environment).strip()
    if digest(reference) != canonical_sha256:
        raise ValueError('The immutable runtime archive changed during compression.')
    report = dict(format='python-xz-mt', decoder_source_sha256=decoder_sha256,
                  runtime_tree=original, canonical_gpu_env_sha256=canonical_sha256,
                  bootstrap_sha256=digest(bootstrap), gpu_env_xz_sha256=digest(payload),
                  bootstrap_bytes=bootstrap.stat().st_size, gpu_env_xz_bytes=payload.stat().st_size,
                  imports=imports, relocation_imports_checked=relocation_check,
                  unchanged_runtime_verified=True, elapsed_seconds=time.time() - started,
                  decoder_verification_seconds=decoder_verification_seconds)
    path = artifacts / 'transport.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    if manifest is not None:
        attach(Path(manifest).resolve(), report)
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--verification-directory', type=Path)
    parser.add_argument('--check-relocation', action='store_true')
    args = parser.parse_args()
    artifacts = args.output or active_gpu().artifacts_directory(ROOT)
    package(args.archive or artifacts / 'gpu-env.tar.gz', artifacts, args.verify_only, args.manifest,
            args.verification_directory, args.check_relocation)


if __name__ == '__main__':
    main()
