import contextlib
import csv
import inspect
import io
import json
import math
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch

from gpu_config import read_config
import prepare
from tools import jobs, measurement, record_run, rental

SOURCE = ("BASE_URL = 'https://data.example/fixed'\nDATA_DIR = 'data'\nTIME_BUDGET = 1800\n"
          "def download_data():\n    return BASE_URL, DATA_DIR\n"
          "def evaluate_bpb():\n    return 'reference'\n")


class RecordedDecisionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.root_patch = patch.object(jobs, 'ROOT', self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.directory.cleanup()

    def make_run(self, identifier, val_bpb=1.0, baseline=False, *, gpu_id='h100', default_gpu=None,
                 rules='fixture rules', measurement_source=None, gpu_name=None, runtime_changes=None, suites=None):
        """A finished OLMES job of the GPU track with its snapshot and collected results."""
        config = read_config()
        config['active_gpu'] = default_gpu or gpu_id
        if gpu_id == 'rtx5090':
            config['gpus'][gpu_id].update(training_seconds=900, rental_seconds=2700, max_hourly_cost=0.8)
        values = config['gpus'][gpu_id]
        settings = {name: values[name] for name in ('training_seconds', 'rental_seconds', 'max_hourly_cost')}
        job = dict(id=identifier, commit='a' * 40, baseline=baseline, description='fixture source URL',
                   gpu_id=gpu_id, gpu_name=gpu_name or values['gpu_name'], gpu_settings=settings,
                   model_source=f'gpus/{gpu_id}/train.py', phase='evaluated', exit_code=0,
                   metric=measurement.OLMES_METRIC)
        snapshot = jobs.job_path(identifier).parent
        source = snapshot / 'source'
        source.mkdir(parents=True)
        files = {job['model_source']: 'MODEL_WIDTH = 64\n', 'program.md': rules,
                 'prepare.py': (measurement_source or SOURCE).replace('1800', str(settings['training_seconds']))
                 + '\n' + inspect.getsource(prepare.suites_improve),
                 'pyproject.toml': '[project]\nname = "fixture"\n', 'uv.lock': 'fixed dependencies\n',
                 'evaluation/pyproject.toml': 'native dependency fixture\n',
                 'evaluation/uv.lock': 'fixed native dependencies\n', 'gpus.json': json.dumps(config)}
        for name, content in files.items():
            (source / name).parent.mkdir(parents=True, exist_ok=True)
            (source / name).write_text(content)
        job['source_sha256'] = {name: jobs.digest(source / name) for name in files}
        with tarfile.open(snapshot / 'source.tar.gz', 'w:gz') as archive:
            for name in files:
                archive.add(source / name, arcname=name)
        job['source_archive_sha256'] = jobs.digest(snapshot / 'source.tar.gz')
        runtime = dict.fromkeys(measurement.RUNTIME_FIELDS, 'fixed runtime')
        runtime.update(gpu_assets_sha256='fixed assets', tokenizer={'tokenizer.pkl': 'fixture tokenizer'}, olmes=dict(
            revision=prepare.OLMES_REVISION, suites=['native qa', 'native math', 'native code'],
            approved_config_fix={'arc:rc:bpb::olmes:full': {'primary_metric': 'macro'}},
            native_datasets_sha256='fixed complete native datasets', request_manifest_sha256='fixed full requests',
            leaf_tasks=109, computation='correct_answer_only', computed_requests=99886, examples=99886,
            requests=654536))
        runtime.update(runtime_changes or {})
        (snapshot / 'runtime.json').write_text(json.dumps(runtime))
        job['runtime_manifest_sha256'] = jobs.digest(snapshot / 'runtime.json')
        jobs.save(job)
        directory = self.root / 'gpus' / gpu_id / 'results/runs' / identifier
        directory.mkdir(parents=True)
        suites = suites or dict(qa_bpb=1.0, math_bpb=1.0, code_bpb=100.0)
        log = (f'val_bpb: {val_bpb:.17g}\npeak_vram_mb: 45056.0\n'
               f'training_seconds: {settings["training_seconds"] + 0.2}\n'
               f'total_seconds: {settings["training_seconds"] + 700:.1f}\n')
        log += ''.join(f'{name}: {value:.17g}\n' for name, value in suites.items())
        (directory / 'run.log').write_text(log)
        (directory / 'final-model.pt').write_bytes(b'own model fixture')
        (directory / 'olmes-metrics.jsonl').write_text('{"synthetic_native_metrics": true}\n')
        summary = dict(complete=True, olmes_revision=runtime['olmes']['revision'],
                       approved_config_fix=runtime['olmes']['approved_config_fix'],
                       model_source_sha256=job['source_sha256'][job['model_source']],
                       native_request_manifest_sha256=runtime['olmes']['request_manifest_sha256'],
                       tokenizer_sha256=runtime['tokenizer']['tokenizer.pkl'],
                       suites=suites, leaf_tasks=109, examples=99886, requests=654536,
                       computation='correct_answer_only', computed_requests=99886,
                       metrics_all_sha256=jobs.digest(directory / 'olmes-metrics.jsonl'),
                       checkpoint_sha256=jobs.digest(directory / 'final-model.pt'))
        (directory / 'olmes-summary.json').write_text(json.dumps(summary))
        return job, directory

    def record(self, identifier):
        with contextlib.redirect_stdout(io.StringIO()):
            return record_run.record_run(identifier)

    def ledger(self, gpu='h100'):
        return self.root / 'gpus' / gpu / 'results/decisions.json'

    def rows(self, gpu='h100'):
        with (self.root / 'gpus' / gpu / 'results.tsv').open(newline='') as source:
            return list(csv.reader(source, delimiter='\t'))

    def test_compares_and_exports_each_gpu_against_only_its_own_incumbent(self):
        self.make_run('h100-baseline', baseline=True, suites=dict(qa_bpb=0.9, math_bpb=0.9, code_bpb=90))
        self.make_run('rtx-baseline', baseline=True, gpu_id='rtx5090', suites=dict(qa_bpb=2, math_bpb=2, code_bpb=200))
        self.record('h100-baseline')
        self.record('rtx-baseline')
        self.make_run('rtx-gain', gpu_id='rtx5090', suites=dict(qa_bpb=1.8, math_bpb=1.8, code_bpb=180))
        gained = self.record('rtx-gain')
        self.assertEqual((gained['status'], gained['incumbent']), ('keep', 'rtx-baseline'))
        self.make_run('h100-loss', suites=dict(qa_bpb=1.0, math_bpb=1.0, code_bpb=100))
        discarded = self.record('h100-loss')
        self.assertEqual((discarded['status'], discarded['incumbent']), ('discard', 'h100-baseline'))
        for gpu, best in (('h100', 'h100-baseline'), ('rtx5090', 'rtx-gain')):
            folder = self.root / 'gpus' / gpu
            self.assertEqual(json.loads((folder / 'best.json').read_text())['id'], best)
            records = json.loads(self.ledger(gpu).read_text())
            self.assertEqual(len(records), 2)
            self.assertTrue(all(row['gpu_id'] == gpu for row in records))
            self.assertEqual(len(self.rows(gpu)), 3)

    def test_records_a_jobs_own_gpu_settings_when_the_shared_default_selects_the_other_gpu(self):
        for gpu, default in (('h100', 'rtx5090'), ('rtx5090', 'h100')):
            with self.subTest(gpu=gpu):
                self.make_run(gpu + '-base', baseline=True, gpu_id=gpu, default_gpu=default)
                self.assertEqual(self.record(gpu + '-base')['status'], 'keep')
                self.make_run(gpu + '-gain', gpu_id=gpu, default_gpu=default,
                              suites=dict(qa_bpb=0.9, math_bpb=0.9, code_bpb=90))
                record = self.record(gpu + '-gain')
                self.assertEqual((record['status'], record['incumbent']), ('keep', gpu + '-base'))

    def test_resolves_the_frozen_training_budget_in_the_measurement(self):
        source = ('from gpu_config import active_gpu\nTIME_BUDGET = active_gpu().settings()["training_seconds"]\n'
                  'DATA_DIR = "data"\ndef evaluate_bpb():\n    return 1\n')
        reference = measurement.prepare_semantics(source.replace('active_gpu().settings()["training_seconds"]',
                                                                 '1800'), dict(training_seconds=3600))
        self.assertEqual(measurement.prepare_semantics(source, dict(training_seconds=1800)), reference)
        self.assertNotEqual(measurement.prepare_semantics(source, dict(training_seconds=900)), reference)

    def test_records_one_complete_baseline_and_exports_its_suites_and_reference_bpb(self):
        self.make_run('baseline', 0.974083, baseline=True)
        first = self.record('baseline')
        self.assertEqual(first, self.record('baseline'))
        self.assertEqual(first['status'], 'keep')
        rows = self.rows()
        self.assertEqual(rows[0], ['commit', 'qa_bpb', 'math_bpb', 'code_bpb', 'val_bpb', 'memory_gb', 'status',
                                   'description'])
        self.assertEqual([float(value) for value in rows[1][1:5]], [1.0, 1.0, 100.0, 0.974083])
        self.assertEqual(len(rows), 2)

    def test_decides_by_the_mean_suite_improvement_rate_and_records_val_only_as_reference(self):
        self.make_run('baseline', baseline=True)
        self.record('baseline')
        self.make_run('gain', 2.0, suites=dict(qa_bpb=0.9, math_bpb=0.9, code_bpb=105.0))
        gained = self.record('gain')
        self.assertEqual((gained['status'], gained['val_bpb']), ('keep', 2.0))
        self.make_run('loss', 0.1, suites=dict(qa_bpb=1.0, math_bpb=1.0, code_bpb=100.0))
        self.assertEqual(self.record('loss')['status'], 'discard')
        self.assertEqual([float(value) for value in self.rows()[-2][1:4]], [0.9, 0.9, 105])
        self.make_run('later', suites=dict(qa_bpb=0.8, math_bpb=0.9, code_bpb=105.0))
        later = self.record('later')
        self.assertEqual((later['status'], later['incumbent']), ('keep', 'gain'))

    def test_records_complete_results_with_an_unavailable_reference_val(self):
        self.make_run('baseline', math.nan, baseline=True)
        record = self.record('baseline')
        self.assertEqual((record['status'], record['val_bpb']), ('keep', None))
        self.assertEqual(self.rows()[-1][4], '0.000000')

    def test_requires_complete_artifacts_and_finite_positive_suite_values(self):
        self.make_run('baseline', baseline=True)
        self.record('baseline')
        for identifier, values in (
            ('missing', dict(qa_bpb=0.9, math_bpb=0.9)),
            ('nonfinite', dict(qa_bpb=0.9, math_bpb=math.nan, code_bpb=90)),
            ('zero', dict(qa_bpb=0.9, math_bpb=0, code_bpb=90)),
        ):
            self.make_run(identifier, suites=values)
            self.assertEqual(self.record(identifier)['status'], 'crash')
            self.assertEqual(self.rows()[-1][1], '0.000000')
        _, directory = self.make_run('partial', suites=dict(qa_bpb=0.9, math_bpb=0.9, code_bpb=90))
        path = directory / 'olmes-summary.json'
        summary = json.loads(path.read_text())
        summary['complete'] = False
        path.write_text(json.dumps(summary))
        self.assertEqual(self.record('partial')['status'], 'crash')
        _, directory = self.make_run('lost-checkpoint', suites=dict(qa_bpb=0.9, math_bpb=0.9, code_bpb=90))
        (directory / 'final-model.pt').unlink()
        self.assertEqual(self.record('lost-checkpoint')['status'], 'crash')

    def test_requires_complete_finite_summary_metrics_and_fixed_time_budgets(self):
        cases = [
            ('short', 'training_seconds: 1800.2', 'training_seconds: 1799.9'),
            ('late', 'total_seconds: 2500.0', 'total_seconds: 3600.1'),
            ('missing-memory', 'peak_vram_mb: 45056.0', 'unavailable: 45056.0'),
            ('bad-memory', 'peak_vram_mb: 45056.0', 'peak_vram_mb: nan'),
            ('duplicate', 'qa_bpb: 1', 'qa_bpb: 1\nqa_bpb: 0.5'),
        ]
        for identifier, old, new in cases:
            with self.subTest(identifier=identifier):
                _, directory = self.make_run(identifier, baseline=True)
                path = directory / 'run.log'
                path.write_text(path.read_text().replace(old, new))
                self.assertEqual(self.record(identifier)['status'], 'crash')

    def test_preserves_the_reference_val_of_a_failed_run(self):
        self.make_run('baseline', baseline=True)
        self.record('baseline')
        _, directory = self.make_run('failure', 0.95)
        (directory / 'run.log').write_text('reference_val_bpb: 0.95\nNative evaluation failed.\n')
        jobs.save(dict(id='failure', phase='failed', exit_code=1))
        failed = self.record('failure')
        self.assertEqual((failed['status'], failed['val_bpb']), ('crash', 0.95))
        self.assertEqual(self.rows()[-1][1:4], ['0.000000'] * 3)

    def test_requires_a_new_baseline_when_data_changes_and_then_compares_to_it_once(self):
        self.make_run('baseline', baseline=True)
        self.record('baseline')
        changed = SOURCE.replace('fixed', 'changed')
        gain = dict(qa_bpb=0.9, math_bpb=0.9, code_bpb=90)
        self.make_run('candidate', measurement_source=changed, suites=gain)
        with self.assertRaises(ValueError):
            self.record('candidate')
        self.make_run('new-baseline', baseline=True, measurement_source=changed)
        self.assertEqual(self.record('new-baseline')['status'], 'keep')
        self.make_run('new-candidate', measurement_source=changed, suites=gain)
        record = self.record('new-candidate')
        self.assertEqual((record['status'], record['incumbent']), ('keep', 'new-baseline'))
        self.make_run('duplicate-baseline', baseline=True, measurement_source=changed)
        with self.assertRaises(ValueError):
            self.record('duplicate-baseline')

    def test_compares_across_a_rule_change_and_refuses_a_second_baseline_for_it(self):
        self.make_run('baseline', baseline=True)
        self.record('baseline')
        self.make_run('candidate', rules='measure once', suites=dict(qa_bpb=0.9, math_bpb=0.9, code_bpb=90))
        self.assertEqual(self.record('candidate')['status'], 'keep')
        self.make_run('repeat-baseline', baseline=True, rules='another rule change')
        with self.assertRaises(ValueError):
            self.record('repeat-baseline')

    def test_requires_matching_data_time_gpu_runtime_and_evaluation(self):
        self.make_run('baseline', baseline=True)
        self.record('baseline')
        variants = [
            ('data', {'measurement_source': SOURCE.replace("'data'", "'other-data'")}),
            ('time', {'measurement_source': SOURCE.replace('TIME_BUDGET = 1800', 'TIME_BUDGET = 1800 + 1')}),
            ('gpu', {'gpu_name': 'different GPU'}),
            ('runtime', {'runtime_changes': {'kernel_revision': 'different kernel'}}),
            ('evaluation', {'runtime_changes': {'olmes': dict(
                revision='another revision', suites=['native qa', 'native math', 'native code'],
                approved_config_fix={}, native_datasets_sha256='other datasets', request_manifest_sha256='other',
                leaf_tasks=109, examples=99886, requests=654536)}}),
        ]
        for name, changes in variants:
            with self.subTest(name=name):
                self.make_run(name, suites=dict(qa_bpb=0.9, math_bpb=0.9, code_bpb=90), **changes)
                with self.assertRaises(ValueError):
                    self.record(name)

    def test_tracks_transitive_data_source_changes(self):
        original = SOURCE.replace('return BASE_URL, DATA_DIR', 'return BASE_URL, DATA_DIR, shard_name()')
        original += 'def shard_name():\n    return "shard-1"\n'
        self.make_run('baseline', baseline=True, measurement_source=original)
        self.record('baseline')
        self.make_run('candidate', measurement_source=original.replace('shard-1', 'shard-2'),
                      suites=dict(qa_bpb=0.9, math_bpb=0.9, code_bpb=90))
        with self.assertRaises(ValueError):
            self.record('candidate')

    def test_accepts_a_replacement_tokenizer_under_the_same_measurement(self):
        self.make_run('baseline', baseline=True, runtime_changes={'gpu_assets_sha256': 'first tokenizer'})
        baseline = self.record('baseline')
        self.make_run('candidate', runtime_changes={'gpu_assets_sha256': 'different tokenizer'},
                      suites=dict(qa_bpb=0.9, math_bpb=0.9, code_bpb=90))
        candidate = self.record('candidate')
        self.assertEqual(candidate['status'], 'keep')
        self.assertEqual(candidate['measurement'], baseline['measurement'])

    def test_checks_archived_source_and_runtime_against_their_recorded_hashes(self):
        self.make_run('baseline', baseline=True)
        self.record('baseline')
        for name, filename in (('source', 'source.tar.gz'), ('runtime', 'runtime.json')):
            with self.subTest(name=name):
                self.make_run(name)
                path = jobs.job_path(name).parent / filename
                path.write_bytes(path.read_bytes() + b'changed')
                with self.assertRaises(ValueError):
                    self.record(name)

    def test_checks_a_previously_recorded_log_before_exporting_it_again(self):
        _, directory = self.make_run('baseline', baseline=True)
        self.record('baseline')
        (directory / 'run.log').write_text('altered log')
        with self.assertRaises(ValueError):
            self.record('baseline')

    def test_observes_an_unfinished_job_without_recording_a_failure(self):
        self.make_run('baseline', baseline=True)
        jobs.save(dict(id='baseline', phase='running', exit_code=None))
        with self.assertRaises(ValueError):
            self.record('baseline')

    def test_records_once_after_the_machine_is_returned_and_exports_the_decision(self):
        self.make_run('baseline', baseline=True)
        self.record('baseline')
        self.make_run('candidate', suites=dict(qa_bpb=0.9, math_bpb=0.9, code_bpb=90))
        jobs.save(dict(id='candidate', instance_id=42, label='autoresearch-h100-candidate'))
        returned = []

        def record(command):
            self.assertEqual(returned[-1], 'autoresearch-h100-candidate')
            self.assertIn('returned_at', jobs.load(command[-1]))
            return self.record(command[-1])

        with patch.object(rental.vast_backend, 'note'), \
                patch.object(rental.vast_backend, 'give_back', side_effect=returned.append), \
                patch.object(rental, 'run', side_effect=record):
            rental.finalize(jobs.load('candidate'))
            rental.finalize(jobs.load('candidate'))
        records = json.loads(self.ledger().read_text())
        self.assertEqual(len(records), 2)
        self.assertEqual(records[-1]['status'], 'keep')
        self.assertEqual(self.rows()[-1][6], 'keep')

    def test_uses_the_suite_decision_saved_before_the_measurement(self):
        self.make_run('baseline', baseline=True)
        self.record('baseline')
        self.make_run('gain', suites=dict(qa_bpb=0.99, math_bpb=0.99, code_bpb=99))
        with patch.object(prepare, 'suites_improve', return_value=False):
            self.assertEqual(self.record('gain')['status'], 'keep')

    def test_recognizes_the_same_native_task_data_in_a_different_execution_order(self):
        tasks = [dict(alias=name, task_hash={'hash': name}, dataset_fingerprints={'test': 'fixed'},
                      examples=100, requests_sha256='first tokenizer payload') for name in ('qa', 'math')]
        first = measurement.native_datasets_identity(tasks)
        changed_order = [dict(entry, requests_sha256='another tokenizer payload') for entry in reversed(tasks)]
        self.assertEqual(measurement.native_datasets_identity(changed_order), first)
        changed_data = [dict(entry, dataset_fingerprints={'test': 'different'}) for entry in tasks]
        self.assertNotEqual(measurement.native_datasets_identity(changed_data), first)

    def test_requires_one_correct_answer_measurement_for_every_original_example(self):
        self.make_run('baseline', baseline=True)
        self.record('baseline')
        _, directory = self.make_run('missing-example')
        path = directory / 'olmes-summary.json'
        summary = json.loads(path.read_text())
        summary['computed_requests'] -= 1
        path.write_text(json.dumps(summary))
        self.assertEqual(self.record('missing-example')['status'], 'crash')

    def test_validates_the_inference_batch_that_the_runtime_fixes(self):
        job, directory = self.make_run('batch-plan', baseline=True)
        runtime_path = jobs.job_path(job['id']).parent / 'runtime.json'
        runtime = json.loads(runtime_path.read_text())
        runtime['olmes']['inference_batch_size'] = 32
        runtime_path.write_text(json.dumps(runtime))
        summary_path = directory / 'olmes-summary.json'
        summary = json.loads(summary_path.read_text())
        summary['inference_batch_size'] = 32
        summary_path.write_text(json.dumps(summary))
        self.assertEqual(record_run.read_metrics(job, directory)['qa_bpb'], 1.0)
        summary['inference_batch_size'] = 8
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'computation'):
            record_run.read_metrics(job, directory)


if __name__ == '__main__':
    unittest.main()
