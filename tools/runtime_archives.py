"""Name the runtime archives that a manifest freezes and a rented machine receives."""

from pathlib import PurePosixPath


def encoding_cache_asset_path(gpu_id):
    return str(PurePosixPath('autoresearch', 'gpus', gpu_id, 'olmes', 'native-encodings.sqlite'))


def deployment_names(manifest):
    """The archives a rented machine receives: the bootstrap, the runtime compressed for its parallel decoder, and the
    assets."""
    transport = manifest['transport']
    if (transport['format'] != 'python-xz-mt'
            or transport['canonical_gpu_env_sha256'] != manifest['gpu_env_sha256']):
        raise ValueError('A deployed runtime must be compressed for the parallel decoder from its canonical archive.')
    return ('gpu-bootstrap.tar.gz', 'gpu-env.tar.xz', 'gpu-assets.tar.gz')


def archive_specs(manifest):
    """Each archive that the manifest names, with its frozen digest: the canonical runtime and the assets, and once the
    runtime is compressed for transport, the bootstrap and the compressed runtime too."""
    specs = [('gpu-env.tar.gz', manifest['gpu_env_sha256'])]
    if 'transport' in manifest:
        specs += [('gpu-bootstrap.tar.gz', manifest['transport']['bootstrap_sha256']),
                  ('gpu-env.tar.xz', manifest['transport']['gpu_env_xz_sha256'])]
    return specs + [('gpu-assets.tar.gz', manifest['gpu_assets_sha256'])]
