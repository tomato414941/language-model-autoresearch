import ast
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import gpu_config
from tools import jobs, prepare_job

R2 = dict(R2_ENDPOINT='https://store.example', R2_BUCKET='runtime', R2_ACCESS_KEY_ID='key',
          R2_SECRET_ACCESS_KEY='secret')
REPOSITORY = Path(__file__).resolve().parents[1]


class PrepareJobTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for patcher in (patch.object(jobs, 'ROOT', self.root), patch.dict(os.environ, {'AUTORESEARCH_GPU': 'h100', **R2}),
                        patch.object(prepare_job.delivery.r2, 'stored', return_value=15),
                        patch('builtins.print')):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.profile = gpu_config.get_profile('h100')
        artifacts = self.profile.artifacts_directory(self.root)
        artifacts.mkdir(parents=True)
        (artifacts / 'gpu-bootstrap.tar.gz').write_bytes(b'bootstrap fixture')
        self.manifest = dict(gpu_id='h100', evaluation_metric='olmes_bpb', tokenizer={'fixture': 'tokenizer'},
                             sources={}, python_version='3.12.16', gpu_env_sha256='a' * 64, gpu_assets_sha256='b' * 64,
                             transport=dict(format='python-xz-mt', canonical_gpu_env_sha256='a' * 64,
                                            gpu_env_xz_sha256='c' * 64,
                                            bootstrap_sha256=jobs.digest(artifacts / 'gpu-bootstrap.tar.gz')))
        (artifacts / 'runtime.json').write_text(json.dumps(self.manifest))
        for name in gpu_config.source_files(self.profile) + prepare_job.OLMES_SOURCES:
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if name.startswith('evaluation/'):
                destination.write_bytes((REPOSITORY / name).read_bytes())
            else:
                destination.write_text('current ' + name)
        (self.root / 'tools').mkdir(exist_ok=True)
        (self.root / 'tools/gpu_run.sh').write_text('fixture')

    def prepare(self, baseline=False, expected_eval_seconds=900):
        with patch('tools.package_runtime.verify_tokenizer', return_value=self.manifest['tokenizer']):
            return prepare_job.prepare_job('a' * 40, baseline, 'fixture', expected_eval_seconds)

    def test_freezes_the_current_sources_with_the_prepared_runtime_and_the_gpus_settings(self):
        job = self.prepare()
        directory = jobs.job_path(job['id']).parent
        runtime = json.loads((directory / 'runtime.json').read_text())
        self.assertEqual(runtime['sources'], job['source_sha256'])
        self.assertEqual((job['gpu_id'], job['gpu_settings'], job['metric']),
                         ('h100', self.profile.settings(), 'olmes_bpb'))
        self.assertEqual(job['label'], 'autoresearch-h100-' + job['id'])
        with tarfile.open(directory / 'source.tar.gz') as archive:
            self.assertEqual(archive.extractfile('prepare.py').read(), b'current prepare.py')
            self.assertEqual(archive.extractfile('gpu_run.sh').read(), b'fixture')
        self.assertEqual(jobs.load(job['id'])['source_archive_sha256'], jobs.digest(directory / 'source.tar.gz'))

    def test_snapshots_the_native_evaluator_with_every_imported_local_module(self):
        job = self.prepare()
        with tarfile.open(jobs.job_path(job['id']).parent / 'source.tar.gz') as archive:
            names = set(archive.getnames())
            for name in names:
                if name.startswith('evaluation/') and name.endswith('.py'):
                    for node in ast.walk(ast.parse(archive.extractfile(name).read())):
                        if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('evaluation.'):
                            module = node.module.replace('.', '/') + '.py'
                            self.assertIn(module, names, f'{name} imports {module}')

    def test_measures_a_baseline_first_on_a_gpu_without_a_kept_result(self):
        self.assertTrue(self.prepare()['baseline'])
        ledger = self.profile.results_directory(self.root) / 'decisions.json'
        ledger.parent.mkdir(parents=True)
        ledger.write_text(json.dumps([dict(id='earlier', status='keep')]))
        self.assertFalse(self.prepare()['baseline'])
        self.assertTrue(self.prepare(baseline=True)['baseline'])

    def test_requires_a_runtime_with_the_olmes_evaluation(self):
        path = self.profile.artifacts_directory(self.root) / 'runtime.json'
        path.write_text(json.dumps(dict(self.manifest, evaluation_metric='something else')))
        with self.assertRaisesRegex(RuntimeError, 'OLMES'):
            self.prepare()

    def test_prepares_nothing_that_would_not_fit_its_rental(self):
        with self.assertRaisesRegex(ValueError, 'does not fit'):
            self.prepare(expected_eval_seconds=3600)
        self.assertFalse((self.root / 'queue').exists())


if __name__ == '__main__':
    unittest.main()
