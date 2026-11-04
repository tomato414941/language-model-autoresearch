import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import package_runtime


class RuntimeReuseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.root_patch = patch.object(package_runtime, 'ROOT', self.root)
        self.root_patch.start()
        self.assets_patch = patch.object(package_runtime.prepare, 'verify_core_assets')
        self.assets_patch.start()
        for name in package_runtime.SOURCES:
            (self.root / name).write_text(name)
        artifacts = self.root / 'artifacts'
        artifacts.mkdir()
        manifest = {'sources': {name: self.sha(self.root / name) for name in package_runtime.SOURCES}}
        for name, field in [('gpu-env.tar.gz', 'gpu_env_sha256'), ('gpu-assets.tar.gz', 'gpu_assets_sha256')]:
            (artifacts / name).write_bytes(b'fixture')
            manifest[field] = self.sha(artifacts / name)
        self.manifest = artifacts / 'runtime.json'
        self.manifest.write_text(json.dumps(manifest))

    def tearDown(self):
        self.assets_patch.stop()
        self.root_patch.stop()
        self.directory.cleanup()

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_reuses_verified_archives_for_updated_training_and_repository_rules(self):
        previous = json.loads(self.manifest.read_text())
        (self.root / 'train.py').write_text('candidate')
        (self.root / 'program.md').write_text('updated repository rules')
        package_runtime.refresh_source_hashes()
        manifest = json.loads(self.manifest.read_text())
        self.assertEqual(manifest['sources']['program.md'], self.sha(self.root / 'program.md'))
        self.assertEqual(manifest['sources']['train.py'], self.sha(self.root / 'train.py'))
        self.assertEqual(manifest['gpu_env_sha256'], previous['gpu_env_sha256'])

    def test_requires_rebuilding_when_the_runtime_or_evaluation_changes(self):
        for name in ('prepare.py', 'pyproject.toml', 'uv.lock'):
            with self.subTest(name=name):
                original = (self.root / name).read_text()
                (self.root / name).write_text('changed')
                with self.assertRaises(RuntimeError):
                    package_runtime.refresh_source_hashes()
                (self.root / name).write_text(original)

    def test_requires_intact_archives_before_refreshing_source_hashes(self):
        previous = self.manifest.read_text()
        (self.root / 'artifacts/gpu-assets.tar.gz').write_bytes(b'incomplete')
        with self.assertRaises(RuntimeError):
            package_runtime.refresh_source_hashes()
        self.assertEqual(self.manifest.read_text(), previous)


if __name__ == '__main__':
    unittest.main()
