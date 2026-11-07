import hashlib
import json
import os
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch

from gpu_config import source_files
from tools import package_runtime


class RuntimeReuseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.root_patch = patch.object(package_runtime, 'ROOT', self.root)
        self.root_patch.start()
        self.environment = patch.dict(os.environ, {'AUTORESEARCH_GPU': 'h100'})
        self.environment.start()
        self.sources = source_files()
        for name in self.sources:
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text(name)
        artifacts = self.root / 'gpus/h100/artifacts'
        artifacts.mkdir(parents=True)
        manifest = {'sources': {name: self.sha(self.root / name) for name in self.sources}}
        for name, field in [('gpu-env.tar.gz', 'gpu_env_sha256'), ('gpu-assets.tar.gz', 'gpu_assets_sha256')]:
            (artifacts / name).write_bytes(b'fixture')
            manifest[field] = self.sha(artifacts / name)
        self.manifest = artifacts / 'runtime.json'
        self.manifest.write_text(json.dumps(manifest))

    def tearDown(self):
        self.environment.stop()
        self.root_patch.stop()
        self.directory.cleanup()

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_reuses_verified_archives_for_updated_training_rules_and_evaluation_sources(self):
        previous = json.loads(self.manifest.read_text())
        (self.root / 'train.py').write_text('candidate')
        (self.root / 'program.md').write_text('updated repository rules')
        (self.root / 'prepare.py').write_text('updated evaluation connection')
        package_runtime.refresh_source_hashes()
        manifest = json.loads(self.manifest.read_text())
        self.assertEqual(manifest['sources']['program.md'], self.sha(self.root / 'program.md'))
        self.assertEqual(manifest['sources']['train.py'], self.sha(self.root / 'train.py'))
        self.assertEqual(manifest['sources']['prepare.py'], self.sha(self.root / 'prepare.py'))
        self.assertEqual(manifest['gpu_env_sha256'], previous['gpu_env_sha256'])

    def test_requires_the_locked_dependencies_when_reusing_the_runtime(self):
        for name in ('pyproject.toml', 'uv.lock'):
            with self.subTest(name=name):
                original = (self.root / name).read_text()
                (self.root / name).write_text('changed')
                with self.assertRaises(RuntimeError):
                    package_runtime.refresh_source_hashes()
                (self.root / name).write_text(original)

    def test_requires_intact_archives_before_refreshing_source_hashes(self):
        previous = self.manifest.read_text()
        (self.root / 'gpus/h100/artifacts/gpu-assets.tar.gz').write_bytes(b'incomplete')
        with self.assertRaises(RuntimeError):
            package_runtime.refresh_source_hashes()
        self.assertEqual(self.manifest.read_text(), previous)

    def test_packs_the_tokenizer_and_offline_kernel_while_preserving_the_previous_assets(self):
        artifacts = self.root / 'gpus/h100/artifacts'
        previous = self.sha(artifacts / 'gpu-assets.tar.gz')
        tokenizer = self.root / 'tokenizer'
        tokenizer.mkdir()
        for name in ('tokenizer.pkl', 'token_bytes.pt'):
            (tokenizer / name).write_bytes(name.encode())
        cache = artifacts / 'gpu-assets/hf-cache'
        cache.mkdir(parents=True)
        (cache / 'kernel.bin').write_bytes(b'locked kernel fixture')
        with patch.object(package_runtime.prepare, 'TOKENIZER_DIR', str(tokenizer)):
            packed_sha256 = package_runtime.pack_runtime_assets(artifacts)
        archive = artifacts / 'gpu-assets.tar.gz'
        self.assertEqual(self.sha(archive), packed_sha256)
        with tarfile.open(archive, 'r:gz') as packed:
            self.assertEqual(packed.extractfile('./autoresearch/gpus/h100/tokenizer/tokenizer.pkl').read(), b'tokenizer.pkl')
            self.assertEqual(packed.extractfile('./hf-cache/kernel.bin').read(), b'locked kernel fixture')
        self.assertEqual(self.sha(artifacts / ('history/gpu-assets-' + previous + '.tar.gz')), previous)

if __name__ == '__main__':
    unittest.main()
