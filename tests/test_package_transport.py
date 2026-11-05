import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools import package_transport
from tools.runtime_archives import archive_specs, deployment_names


class RuntimeTransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / 'runtime'
        interpreter = self.source / package_transport.INTERPRETER
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        (self.source / 'uv').write_bytes(b'fixed launcher fixture')
        (self.source / 'uv').chmod(0o700)
        (self.source / 'relative-link').symlink_to('uv')
        os.link(self.source / 'uv', self.source / 'hard-link')
        self.reference = self.root / 'canonical.tar.gz'
        with tarfile.open(self.reference, 'w:gz') as archive:
            archive.add(self.source, arcname='.')
        self.output = self.root / 'deployment'
        self.output.mkdir()
        self.manifest = self.output / 'runtime.json'
        self.original = dict(gpu_env_sha256=package_transport.digest(self.reference),
                             gpu_assets_sha256='fixed assets', gpu_id='h100')
        self.manifest.write_text(json.dumps(self.original))

    def tearDown(self):
        self.temporary.cleanup()

    def package(self):
        actual = package_transport.subprocess.check_output

        def execute(command, **kwargs):
            if command[1] == '-c':
                return 'locked versions\n'
            return actual(command, **kwargs)

        with patch.object(package_transport.subprocess, 'check_output', side_effect=execute):
            return package_transport.package(self.reference, self.output, manifest=self.manifest,
                                             verification_directory=self.root / 'checks', relocation_check=True)

    def test_preserves_all_canonical_bytes_modes_links_and_manifest_identity_after_transcoding(self):
        original_tree = package_transport.tree_identity(self.source)
        self.assertEqual(package_transport.archive_identity(self.reference), original_tree)
        report = self.package()
        self.assertEqual(report['runtime_tree'], original_tree)
        self.assertTrue(report['unchanged_runtime_verified'])
        manifest = json.loads(self.manifest.read_text())
        self.assertEqual(manifest['gpu_env_sha256'], self.original['gpu_env_sha256'])
        self.assertEqual(deployment_names(manifest), ('gpu-bootstrap.tar.gz', 'gpu-env.tar.xz', 'gpu-assets.tar.gz'))
        self.assertEqual(dict(archive_specs(manifest))['gpu-env.tar.xz'], report['gpu_env_xz_sha256'])

    def test_requires_the_verified_compressed_bytes_when_attaching_a_deployment(self):
        report = self.package()
        (self.output / 'gpu-env.tar.xz').write_bytes(b'incomplete transport')
        with self.assertRaisesRegex(ValueError, 'changed'):
            package_transport.attach(self.manifest, report)


if __name__ == '__main__':
    unittest.main()
