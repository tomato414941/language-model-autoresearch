import contextlib
import csv
import io
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
                 measurement_source=None, gpu_name='H100 SXM', runtime_changes=None):
        job = dict(id=identifier, commit='a' * 40, baseline=baseline, description='fixture source URL',
                   gpu_name=gpu_name, phase='evaluated', exit_code=0)
        snapshot = rental.job_path(identifier).parent
        source = snapshot / 'source'
        source.mkdir(parents=True)
        files = {'train.py': 'MODEL_WIDTH = 64\n', 'program.md': rules,
                 'prepare.py': measurement_source or SOURCE,
                 'pyproject.toml': '[project]\nname = "fixture"\n', 'uv.lock': 'fixed dependencies\n'}
        for name, content in files.items():
            (source / name).write_text(content)
        job['source_sha256'] = {name: rental.digest(source / name) for name in rental.SOURCES}
        with tarfile.open(snapshot / 'source.tar.gz', 'w:gz') as archive:
            for name in rental.SOURCES:
                archive.add(source / name, arcname=name)
        job['source_archive_sha256'] = rental.digest(snapshot / 'source.tar.gz')
        runtime = dict.fromkeys(measurement.RUNTIME_FIELDS, 'fixed runtime')
        runtime.update(runtime_changes or {})
        (snapshot / 'runtime.json').write_text(json.dumps(runtime))
        job['runtime_manifest_sha256'] = rental.digest(snapshot / 'runtime.json')
        rental.save(job)
        directory = self.root / 'results/runs' / identifier
        directory.mkdir(parents=True)
        (directory / 'run.log').write_text(
            f'val_bpb: {bpb:.17g}\npeak_vram_mb: 45056.0\n'
            'training_seconds: 1800.2\ntotal_seconds: 2500.0\n')
        return job, directory

    def record(self, identifier):
        with contextlib.redirect_stdout(io.StringIO()):
            return record_run.record_run(identifier)

    def rows(self):
        with (self.root / 'results.tsv').open(newline='') as source:
            return list(csv.reader(source, delimiter='\t'))

    def test_records_one_complete_baseline_and_exports_round_trip_bpb_in_five_columns(self):
        self.make_run('baseline', 0.974083, baseline=True)
        first = self.record('baseline')
        self.assertEqual(first, self.record('baseline'))
        self.assertEqual(first['status'], 'keep')
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], ['commit', 'val_bpb', 'memory_gb', 'status', 'description'])
        self.assertEqual(float(rows[1][1]), first['val_bpb'])
        self.assertEqual(len(rows[1]), 5)

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
        self.assertEqual(self.rows()[-1][3], 'keep')


if __name__ == '__main__':
    unittest.main()
