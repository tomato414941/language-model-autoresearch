import hashlib
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch

from tools import package_runtime


class RuntimeReuseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.root_patch = patch.object(package_runtime, 'ROOT', self.root)
        self.root_patch.start()
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

    def test_repackages_tokenizer_and_offline_kernel_while_preserving_previous_payloads(self):
        original = json.loads(self.manifest.read_text())
        tokenizer = self.root / 'tokenizer'
        tokenizer.mkdir()
        for name in ('tokenizer.pkl', 'token_bytes.pt'):
            (tokenizer / name).write_bytes(name.encode())
        cache = self.root / 'artifacts/gpu-assets/hf-cache'
        cache.mkdir(parents=True)
        (cache / 'kernel.bin').write_bytes(b'locked kernel fixture')
        hashes = {name: self.sha(tokenizer / name) for name in ('tokenizer.pkl', 'token_bytes.pt')}
        with patch.object(package_runtime.prepare, 'TOKENIZER_DIR', str(tokenizer)):
            with patch.object(package_runtime, 'verify_tokenizer', return_value=hashes):
                package_runtime.refresh_assets()
        manifest = json.loads(self.manifest.read_text())
        self.assertEqual(manifest['gpu_env_sha256'], original['gpu_env_sha256'])
        self.assertEqual(manifest['tokenizer'], hashes)
        archive = self.root / 'artifacts/gpu-assets.tar.gz'
        self.assertEqual(self.sha(archive), manifest['gpu_assets_sha256'])
        with tarfile.open(archive, 'r:gz') as packed:
            self.assertEqual(packed.extractfile('./autoresearch/tokenizer/tokenizer.pkl').read(), b'tokenizer.pkl')
            self.assertEqual(packed.extractfile('./hf-cache/kernel.bin').read(), b'locked kernel fixture')
        preserved = self.root / ('artifacts/history/gpu-assets-' + original['gpu_assets_sha256'] + '.tar.gz')
        self.assertEqual(self.sha(preserved), original['gpu_assets_sha256'])


if __name__ == '__main__':
    unittest.main()
