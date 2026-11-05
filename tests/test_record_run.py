import contextlib
import csv
import io
import inspect
import json
import math
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch

from tools import measurement, record_run, rental

SOURCE = ("DATA_REVISION = 'fixed'\nTIME_BUDGET = 1800\nEVAL_TOKENS = 20971520\n"
          "def evaluate_bpb():\n    return DATA_REVISION, EVAL_TOKENS\n")


class RecordedDecisionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.patches = [patch.object(record_run, 'ROOT', self.root), patch.object(rental, 'ROOT', self.root)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def make_run(self, identifier, bpb, baseline=False, *, rules='fixture rules',
                 measurement_source=None, gpu_name='H100 SXM', runtime_changes=None,
                 suites=None, metric='val_bpb'):
        job = dict(id=identifier, commit='a' * 40, baseline=baseline, description='fixture source URL',
                   gpu_name=gpu_name, phase='evaluated', exit_code=0, metric=metric)
        snapshot = rental.job_path(identifier).parent
        source = snapshot / 'source'
        source.mkdir(parents=True)
        files = {'train.py': 'MODEL_WIDTH = 64\n', 'program.md': rules,
                 'prepare.py': measurement_source or SOURCE,
                 'pyproject.toml': '[project]\nname = "fixture"\n', 'uv.lock': 'fixed dependencies\n'}
        if metric == measurement.OLMES_METRIC:
            files['prepare.py'] += '\n' + inspect.getsource(record_run.prepare.suites_improve)
            files.update({'evaluation/pyproject.toml': 'native dependency fixture\n',
                          'evaluation/uv.lock': 'fixed native dependencies\n'})
        for name, content in files.items():
            (source / name).parent.mkdir(parents=True, exist_ok=True)
            (source / name).write_text(content)
        job['source_sha256'] = {name: rental.digest(source / name) for name in files}
        with tarfile.open(snapshot / 'source.tar.gz', 'w:gz') as archive:
            for name in files:
                archive.add(source / name, arcname=name)
        job['source_archive_sha256'] = rental.digest(snapshot / 'source.tar.gz')
        runtime = dict.fromkeys(measurement.RUNTIME_FIELDS, 'fixed runtime')
        if metric == measurement.OLMES_METRIC:
            runtime.update(tokenizer={'tokenizer.pkl': 'fixture tokenizer'}, olmes=dict(
                revision=record_run.prepare.OLMES_REVISION,
                suites=['native qa', 'native math', 'native code'],
                approved_config_fix={'arc:rc:bpb::olmes:full': {'primary_metric': 'macro'}},
                native_datasets_sha256='fixed complete native datasets',
                request_manifest_sha256='fixed full requests', leaf_tasks=109,
                computation='correct_answer_only', computed_requests=99886,
                examples=99886, requests=654536))
        runtime.update(runtime_changes or {})
        (snapshot / 'runtime.json').write_text(json.dumps(runtime))
        job['runtime_manifest_sha256'] = rental.digest(snapshot / 'runtime.json')
        rental.save(job)
        directory = self.root / 'results/runs' / identifier
        directory.mkdir(parents=True)
        (directory / 'run.log').write_text(
            f'val_bpb: {bpb:.17g}\npeak_vram_mb: 45056.0\n'
            'training_seconds: 1800.2\ntotal_seconds: 2500.0\n')
        if metric == measurement.OLMES_METRIC:
            values = suites or dict(qa_bpb=1.0, math_bpb=1.0, code_bpb=100.0)
            with (directory / 'run.log').open('a') as log:
                for name, value in values.items():
                    log.write(f'{name}: {value:.17g}\n')
            (directory / 'final-model.pt').write_bytes(b'own model fixture')
            (directory / 'olmes-metrics.jsonl').write_text('{"synthetic_native_metrics": true}\n')
            summary = dict(complete=True, olmes_revision=runtime['olmes']['revision'],
                           approved_config_fix=runtime['olmes']['approved_config_fix'],
                           model_source_sha256=job['source_sha256']['train.py'],
                           native_request_manifest_sha256=runtime['olmes']['request_manifest_sha256'],
                           tokenizer_sha256=runtime['tokenizer']['tokenizer.pkl'],
                           suites=values, leaf_tasks=109, examples=99886, requests=654536,
                           computation='correct_answer_only', computed_requests=99886,
                           metrics_all_sha256=rental.digest(directory / 'olmes-metrics.jsonl'),
                           checkpoint_sha256=rental.digest(directory / 'final-model.pt'))
            (directory / 'olmes-summary.json').write_text(json.dumps(summary))
        return job, directory

    def record(self, identifier):
        with contextlib.redirect_stdout(io.StringIO()):
            return record_run.record_run(identifier)

    def rows(self):
        with (self.root / 'results.tsv').open(newline='') as source:
            return list(csv.reader(source, delimiter='\t'))

    def test_records_one_complete_baseline_and_exports_reference_bpb_with_unmeasured_suites(self):
        self.make_run('baseline', 0.974083, baseline=True)
        first = self.record('baseline')
        self.assertEqual(first, self.record('baseline'))
        self.assertEqual(first['status'], 'keep')
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], ['commit', 'qa_bpb', 'math_bpb', 'code_bpb', 'val_bpb', 'memory_gb', 'status', 'description'])
        self.assertEqual(rows[1][1:4], ['NA', 'NA', 'NA'])
        self.assertEqual(float(rows[1][4]), first['val_bpb'])
        self.assertEqual(len(rows[1]), 8)

    def test_adopts_any_lower_bpb_and_discards_equal_or_higher_bpb(self):
        self.make_run('baseline', 0.974083, baseline=True)
        self.record('baseline')
        lower = math.nextafter(0.974083, -math.inf)
        self.make_run('small-gain', lower)
        self.assertEqual(self.record('small-gain')['status'], 'keep')
        for identifier, bpb in (('tie', lower), ('higher', 0.98)):
            self.make_run(identifier, bpb)
            self.assertEqual(self.record(identifier)['status'], 'discard')
        self.make_run('later', 0.97)
        later = self.record('later')
        self.assertEqual((later['status'], later['incumbent']), ('keep', 'small-gain'))

    def test_records_missing_and_nonfinite_bpb_as_failed_attempts(self):
        self.make_run('baseline', 1.0, baseline=True)
        self.record('baseline')
        for identifier, bpb in (('nan', math.nan), ('inf', math.inf), ('missing', 0.5)):
            _, directory = self.make_run(identifier, bpb)
            if identifier == 'missing':
                path = directory / 'run.log'
                path.write_text(path.read_text().replace('val_bpb:', 'unavailable:'))
            record = self.record(identifier)
            self.assertEqual(record['status'], 'crash')
            self.assertIsNone(record['val_bpb'])
            self.assertEqual(self.rows()[-1][1], '0.000000')

    def test_records_a_failed_exit_even_when_it_printed_a_lower_bpb(self):
        self.make_run('baseline', 1.0, baseline=True)
        self.record('baseline')
        self.make_run('failed', 0.5)
        rental.save(dict(id='failed', phase='failed', exit_code=1))
        failed = self.record('failed')
        self.assertEqual((failed['status'], failed['val_bpb']), ('crash', 0.5))
        self.assertEqual(self.rows()[-1][1], '0.000000')

    def test_requires_complete_finite_summary_metrics_and_fixed_time_budgets(self):
        cases = [
            ('short', 'training_seconds: 1800.2', 'training_seconds: 1799.9'),
            ('late', 'total_seconds: 2500.0', 'total_seconds: 3600.1'),
            ('missing-memory', 'peak_vram_mb: 45056.0', 'unavailable: 45056.0'),
            ('bad-memory', 'peak_vram_mb: 45056.0', 'peak_vram_mb: nan'),
            ('duplicate', 'val_bpb: 1', 'val_bpb: 1\nval_bpb: 0.5'),
        ]
        for identifier, old, new in cases:
            with self.subTest(identifier=identifier):
                _, directory = self.make_run(identifier, 1.0, baseline=True)
                path = directory / 'run.log'
                path.write_text(path.read_text().replace(old, new))
                self.assertEqual(self.record(identifier)['status'], 'crash')

    def test_requires_a_new_baseline_when_data_changes_and_then_compares_to_it_once(self):
        self.make_run('baseline', 1.0, baseline=True)
        self.record('baseline')
        changed = SOURCE.replace("'fixed'", "'changed'")
        self.make_run('candidate', 0.9, measurement_source=changed)
        with self.assertRaises(ValueError):
            self.record('candidate')
        self.make_run('new-baseline', 1.1, baseline=True, measurement_source=changed)
        self.assertEqual(self.record('new-baseline')['status'], 'keep')
        self.make_run('new-candidate', 1.0, measurement_source=changed)
        record = self.record('new-candidate')
        self.assertEqual((record['status'], record['incumbent']), ('keep', 'new-baseline'))
        self.make_run('duplicate-baseline', 0.5, baseline=True, measurement_source=changed)
        with self.assertRaises(ValueError):
            self.record('duplicate-baseline')

    def test_reuses_historical_bpb_after_a_rule_change_while_preserving_the_original_record(self):
        original = SOURCE + "CORE_MIN_DELTA = .01\ndef core_improves(a, b):\n    return a - b >= CORE_MIN_DELTA\ndef evaluate_core():\n    return 22\n"
        self.make_run('baseline', 0.974083, baseline=True, measurement_source=original)
        self.record('baseline')
        ledger = self.root / 'results/decisions.json'
        legacy = json.loads(ledger.read_text())
        legacy[0].update(core_metric=0.10065166465157924, measurement={'id': 'legacy', 'definition': {'version': 1}})
        historical = self.root / 'results/runs/baseline/core.json'
        historical.write_text('{"historical": true}\n')
        legacy[0]['artifacts']['core.json'] = rental.digest(historical)
        ledger.write_text(json.dumps(legacy))
        preserved = ledger.read_bytes()
        self.record('baseline')
        self.assertEqual(ledger.read_bytes(), preserved)
        self.make_run('candidate', 0.974082, rules='BPB only; measure once')
        candidate = self.record('candidate')
        self.assertEqual((candidate['status'], candidate['incumbent']), ('keep', 'baseline'))
        self.assertEqual(json.loads(ledger.read_text())[0], legacy[0])
        self.assertEqual(rental.digest(historical), legacy[0]['artifacts']['core.json'])
        self.make_run('repeat-baseline', 0.5, baseline=True, rules='another rule change')
        with self.assertRaises(ValueError):
            self.record('repeat-baseline')

    def test_requires_matching_evaluation_time_gpu_and_runtime(self):
        self.make_run('baseline', 1.0, baseline=True)
        self.record('baseline')
        variants = [
            ('evaluation', {'measurement_source': SOURCE.replace('20971520', '10485760')}),
            ('time', {'measurement_source': SOURCE.replace('1800', '900')}),
            ('gpu', {'gpu_name': 'different GPU'}),
            ('runtime', {'runtime_changes': {'kernel_revision': 'different kernel'}}),
        ]
        for name, changes in variants:
            with self.subTest(name=name):
                self.make_run(name, 0.9, **changes)
                with self.assertRaises(ValueError):
                    self.record(name)

    def test_tracks_transitive_byte_counting_and_data_source_changes(self):
        original = SOURCE.replace('return DATA_REVISION, EVAL_TOKENS',
                                  'return DATA_REVISION, EVAL_TOKENS, byte_count()')
        original += 'def byte_count():\n    return 1\n'
        self.make_run('baseline', 1.0, baseline=True, measurement_source=original)
        self.record('baseline')
        self.make_run('candidate', 0.9, measurement_source=original.replace('return 1\n', 'return 2\n'))
        with self.assertRaises(ValueError):
            self.record('candidate')

    def test_accepts_a_replacement_tokenizer_under_the_same_fixed_bpb_evaluation(self):
        self.make_run('baseline', 1.0, baseline=True, runtime_changes={'gpu_assets_sha256': 'first tokenizer'})
        baseline = self.record('baseline')
        self.make_run('candidate', 0.9, runtime_changes={'gpu_assets_sha256': 'different tokenizer'})
        candidate = self.record('candidate')
        self.assertEqual(candidate['status'], 'keep')
        self.assertEqual(candidate['measurement'], baseline['measurement'])

    def test_checks_archived_source_and_runtime_against_their_recorded_hashes(self):
        self.make_run('baseline', 1.0, baseline=True)
        self.record('baseline')
        for name, filename in (('source', 'source.tar.gz'), ('runtime', 'runtime.json')):
            with self.subTest(name=name):
                self.make_run(name, 0.9)
                path = rental.job_path(name).parent / filename
                path.write_bytes(path.read_bytes() + b'changed')
                with self.assertRaises(ValueError):
                    self.record(name)

    def test_checks_a_previously_recorded_log_before_exporting_it_again(self):
        self.make_run('baseline', 1.0, baseline=True)
        self.record('baseline')
        (self.root / 'results/runs/baseline/run.log').write_text('altered log')
        with self.assertRaises(ValueError):
            self.record('baseline')

    def test_observes_an_unfinished_job_without_recording_a_failure(self):
        self.make_run('baseline', 1.0, baseline=True)
        rental.save(dict(id='baseline', phase='running', exit_code=None))
        with self.assertRaises(ValueError):
            self.record('baseline')

    def test_finalizes_after_deletion_once_and_exports_the_bpb_decision(self):
        self.make_run('baseline', 1.0, baseline=True)
        self.record('baseline')
        self.make_run('candidate', 0.9)
        rental.save(dict(id='candidate', creation_requested_at=1000))
        def delete(job):
            rental.save(dict(id=job['id'], deleted_at=1100))
        def record(command):
            self.assertEqual(rental.load(command[-1])['deleted_at'], 1100)
            return self.record(command[-1])
        with patch.object(rental, 'delete_instances', side_effect=delete):
            with patch.object(rental, 'run', side_effect=record):
                for _ in range(2):
                    rental.finalize(rental.load('candidate'))
        records = json.loads((self.root / 'results/decisions.json').read_text())
        self.assertEqual(len(records), 2)
        self.assertEqual((records[-1]['status'], records[-1]['val_bpb']), ('keep', 0.9))
        self.assertEqual(self.rows()[-1][6], 'keep')

    def test_decides_by_the_mean_suite_improvement_rate_and_records_val_only_as_reference(self):
        native = measurement.OLMES_METRIC
        self.make_run('native-baseline', 1.0, baseline=True, metric=native)
        self.record('native-baseline')
        self.make_run('native-gain', 2.0, metric=native,
                      suites=dict(qa_bpb=0.9, math_bpb=0.9, code_bpb=105.0))
        gained = self.record('native-gain')
        self.assertEqual((gained['status'], gained['val_bpb']), ('keep', 2.0))
        self.make_run('native-loss', 0.1, metric=native,
                      suites=dict(qa_bpb=1.0, math_bpb=1.0, code_bpb=100.0))
        self.assertEqual(self.record('native-loss')['status'], 'discard')
        self.assertEqual([float(value) for value in self.rows()[-2][1:4]], [0.9, 0.9, 105])

    def test_records_complete_native_results_with_an_unavailable_reference_val(self):
        self.make_run('native-baseline', math.nan, baseline=True, metric=measurement.OLMES_METRIC)
        record = self.record('native-baseline')
        self.assertEqual((record['status'], record['val_bpb']), ('keep', None))
        self.assertEqual(self.rows()[-1][4], '0.000000')

    def test_requires_complete_native_artifacts_and_finite_positive_suite_values(self):
        native = measurement.OLMES_METRIC
        self.make_run('native-baseline', 1.0, baseline=True, metric=native)
        self.record('native-baseline')
        for identifier, values in (
            ('missing', dict(qa_bpb=0.9, math_bpb=0.9)),
            ('nonfinite', dict(qa_bpb=0.9, math_bpb=math.nan, code_bpb=90)),
            ('zero', dict(qa_bpb=0.9, math_bpb=0, code_bpb=90)),
        ):
            self.make_run(identifier, 0.5, metric=native, suites=values)
            self.assertEqual(self.record(identifier)['status'], 'crash')
        _, directory = self.make_run('partial', 0.5, metric=native,
                                     suites=dict(qa_bpb=0.9, math_bpb=0.9, code_bpb=90))
        path = directory / 'olmes-summary.json'
        summary = json.loads(path.read_text())
        summary['complete'] = False
        path.write_text(json.dumps(summary))
        self.assertEqual(self.record('partial')['status'], 'crash')

    def test_uses_the_suite_decision_saved_before_the_measurement(self):
        native = measurement.OLMES_METRIC
        self.make_run('native-baseline', 1.0, baseline=True, metric=native)
        self.record('native-baseline')
        self.make_run('native-gain', 1.0, metric=native,
                      suites=dict(qa_bpb=0.99, math_bpb=0.99, code_bpb=99))
        with patch.object(record_run.prepare, 'suites_improve', return_value=False):
            self.assertEqual(self.record('native-gain')['status'], 'keep')

    def test_compares_the_first_native_candidates_on_their_shared_val_then_uses_all_suites(self):
        self.make_run('old-baseline', 0.974083, baseline=True)
        self.record('old-baseline')
        native = measurement.OLMES_METRIC
        runtime = {'gpu_env_sha256': 'combined training and isolated evaluation',
                   'training_gpu_env_sha256': 'fixed runtime'}
        self.make_run('first-native-loss', 0.98, metric=native, runtime_changes=runtime)
        lost = self.record('first-native-loss')
        self.assertEqual((lost['status'], lost['decision_metric']), ('discard', 'val_bpb'))
        self.make_run('first-native-gain', 0.97, metric=native, runtime_changes=runtime)
        gained = self.record('first-native-gain')
        self.assertEqual((gained['status'], gained['decision_metric']), ('keep', 'val_bpb'))
        self.make_run('second-native', 0.99, metric=native, runtime_changes=runtime,
                      suites=dict(qa_bpb=0.99, math_bpb=0.99, code_bpb=99))
        second = self.record('second-native')
        self.assertEqual((second['status'], second['decision_metric']), ('keep', native))

    def test_keeps_the_existing_baseline_when_only_native_evaluation_is_added(self):
        self.make_run('old-baseline', 0.974083, baseline=True)
        self.record('old-baseline')
        self.make_run('duplicate-native', 0.97, baseline=True, metric=measurement.OLMES_METRIC)
        with self.assertRaises(ValueError):
            self.record('duplicate-native')

    def test_requires_a_complete_native_evaluation_and_common_val_for_the_first_candidate(self):
        self.make_run('old-baseline', 0.974083, baseline=True)
        self.record('old-baseline')
        for identifier, val, suites in (
            ('missing-reference', math.nan, dict(qa_bpb=0.9, math_bpb=0.9, code_bpb=90)),
            ('missing-suite', 0.9, dict(qa_bpb=0.9, math_bpb=0.9)),
        ):
            self.make_run(identifier, val, metric=measurement.OLMES_METRIC, suites=suites)
            self.assertEqual(self.record(identifier)['status'], 'crash')

    def test_preserves_the_reference_val_when_native_evaluation_fails(self):
        self.make_run('old-baseline', 0.974083, baseline=True)
        self.record('old-baseline')
        _, directory = self.make_run('native-failure', 0.95, metric=measurement.OLMES_METRIC)
        (directory / 'run.log').write_text('reference_val_bpb: 0.95\nNative evaluation failed.\n')
        rental.save(dict(id='native-failure', phase='failed', exit_code=1))
        failed = self.record('native-failure')
        self.assertEqual((failed['status'], failed['val_bpb']), ('crash', 0.95))
        self.assertEqual(self.rows()[-1][1:4], ['0.000000'] * 3)

    def test_recognizes_the_same_native_task_data_in_a_different_execution_order(self):
        tasks = [dict(alias=name, task_hash={'hash': name}, dataset_fingerprints={'test': 'fixed'},
                      examples=100, requests_sha256='first tokenizer payload') for name in ('qa', 'math')]
        first = measurement.native_datasets_identity(tasks)
        changed_order = [dict(entry, requests_sha256='another tokenizer payload') for entry in reversed(tasks)]
        self.assertEqual(measurement.native_datasets_identity(changed_order), first)
        changed_data = [dict(entry, dataset_fingerprints={'test': 'different'}) for entry in tasks]
        self.assertNotEqual(measurement.native_datasets_identity(changed_data), first)

    def test_requires_one_correct_answer_measurement_for_every_original_example(self):
        native = measurement.OLMES_METRIC
        self.make_run('native-baseline', 1.0, baseline=True, metric=native)
        self.record('native-baseline')
        _, directory = self.make_run('missing-example', 0.9, metric=native)
        path = directory / 'olmes-summary.json'
        summary = json.loads(path.read_text())
        summary['computed_requests'] -= 1
        path.write_text(json.dumps(summary))
        self.assertEqual(self.record('missing-example')['status'], 'crash')


if __name__ == '__main__':
    unittest.main()
