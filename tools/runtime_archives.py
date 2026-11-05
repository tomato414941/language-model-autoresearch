"""Describe verified deployment files without importing the model or Torch."""

import hashlib
import json
from pathlib import PurePosixPath
import sys
import tarfile

UNPACK_XZ = ('import sys,tarfile; '
             'tarfile.open(fileobj=sys.stdin.buffer,mode="r|xz",bufsize=1024*1024)'
             '.extractall(".",filter="fully_trusted")')


def deployment_names(manifest):
    transport = manifest.get('transport')
    if transport is None:
        return ('gpu-env.tar.gz', 'gpu-assets.tar.gz')
    if (transport.get('format') != 'python-xz'
            or transport.get('canonical_gpu_env_sha256') != manifest['gpu_env_sha256']):
        raise ValueError('The compressed deployment must preserve the canonical runtime.')
    return ('gpu-bootstrap.tar.gz', 'gpu-env.tar.xz', 'gpu-assets.tar.gz')


def archive_specs(manifest):
    hashes = {'gpu-env.tar.gz': manifest['gpu_env_sha256'],
              'gpu-assets.tar.gz': manifest['gpu_assets_sha256']}
    if manifest.get('transport'):
        hashes.update({'gpu-bootstrap.tar.gz': manifest['transport']['bootstrap_sha256'],
                       'gpu-env.tar.xz': manifest['transport']['gpu_env_xz_sha256']})
    names = deployment_names(manifest)
    if manifest.get('transport'):
        names = ('gpu-env.tar.gz', *names)
    return [(name, hashes[name]) for name in names]


def stream_identity(stream):
    """Verify all decoded files with the standalone deployment interpreter."""
    rows = {}
    with tarfile.open(fileobj=stream, mode='r|xz', bufsize=1024*1024) as archive:
        for member in archive:
            name = str(PurePosixPath(member.name))
            if name == '.':
                continue
            row = dict(path=name, mode=member.mode)
            if member.isdir():
                row.update(type='directory')
            elif member.issym():
                row.update(type='symlink', target=member.linkname)
            elif member.islnk():
                target = rows[str(PurePosixPath(member.linkname))]
                row.update(type='file', bytes=target['bytes'], sha256=target['sha256'])
            elif member.isfile():
                value = hashlib.sha256()
                with archive.extractfile(member) as source:
                    for block in iter(lambda: source.read(1024*1024), b''):
                        value.update(block)
                row.update(type='file', bytes=member.size, sha256=value.hexdigest())
            else:
                raise ValueError('The runtime contains an unsupported special file.')
            rows[name] = row
    ordered = sorted(rows.values(), key=lambda row: PurePosixPath(row['path']))
    encoded = json.dumps(ordered, sort_keys=True, separators=(',', ':')).encode()
    return dict(sha256=hashlib.sha256(encoded).hexdigest(), entries=len(rows))


if __name__ == '__main__':
    if sys.argv[1:] != ['verify-stream']:
        raise SystemExit('Usage: runtime_archives.py verify-stream < gpu-env.tar.xz')
    print(json.dumps(stream_identity(sys.stdin.buffer)))
