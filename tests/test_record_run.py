import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare
from tools import record_run, rental


class RecordedDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labels = [task['label'] for task in prepare.verify_core_assets()]

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

    def make_run(self, identifier, core, bpb, baseline=False):
        job = dict(id=identifier, commit='a' * 40, baseline=baseline, description='fixture source URL',
                   source_sha256=dict.fromkeys(rental.FRAME_SOURCES, 'frozen'),
                   gpu_name='H100 SXM', phase='evaluated', exit_code=0)
        rental.job_path(identifier).parent.mkdir(parents=True)
        rental.save(job)
        directory = self.root / 'results/runs' / identifier
        directory.mkdir(parents=True)
        result = dict(results=dict.fromkeys(self.labels, 0.5),
                      centered_results=dict.fromkeys(self.labels, core), core_metric=core)
        prepare.save_core_results(result, directory)
        (directory / 'run.log').write_text(
            f'core_metric: {core:.17g}\nval_bpb: {bpb:.6f}\npeak_vram_mb: 45056.0\n'
            'training_seconds: 1800.2\ntotal_seconds: 2500.0\n')
        return job, directory

    def record(self, identifier):
        with contextlib.redirect_stdout(io.StringIO()):
            return record_run.record_run(identifier)

    def test_records_one_complete_baseline_with_full_precision_and_exports_real_tsv(self):
        self.make_run('baseline', 0.12345678901234566, 1.0, baseline=True)
        first = self.record('baseline')
        second = self.record('baseline')
        self.assertEqual(first, second)
        self.assertEqual(first['status'], 'keep')
        with (self.root / 'results.tsv').open(newline='') as source:
            rows = list(csv.reader(source, delimiter='\t'))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], record_run.HEADER)
        self.assertEqual(float(rows[1][1]), first['core_metric'])

    def test_adopts_the_exact_core_boundary_and_discards_a_smaller_gain_regardless_of_reference_bpb(self):
        self.make_run('baseline', 0.12, 1.0, baseline=True)
        self.record('baseline')
        self.make_run('small-gain', 0.129, 0.5)
        self.assertEqual(self.record('small-gain')['status'], 'discard')
        self.make_run('threshold-gain', 0.13, 1.5)
        self.assertEqual(self.record('threshold-gain')['status'], 'keep')
        self.make_run('later', 0.135, 0.4)
        later = self.record('later')
        self.assertEqual(later['status'], 'discard')
        self.assertEqual(later['incumbent'], 'threshold-gain')

    def test_records_failed_attempts_when_native_task_results_are_incomplete_or_nonfinite(self):
        for identifier in ('incomplete', 'nonfinite'):
            with self.subTest(identifier=identifier):
                job, directory = self.make_run(identifier, 0.12, 1.0, baseline=True)
                result = json.loads((directory / 'core.json').read_text())
                if identifier == 'incomplete':
                    result['results'].pop(self.labels[-1])
                else:
                    result['centered_results'][self.labels[-1]] = float('nan')
                (directory / 'core.json').write_text(json.dumps(result))
                record = self.record(job['id'])
                self.assertEqual(record['status'], 'crash')
                self.assertIsNone(record['core_metric'])
                self.assertEqual(record['val_bpb'], 1.0)

    def test_adopts_a_complete_core_improvement_when_reference_bpb_is_unavailable(self):
        self.make_run('baseline', 0.12, 1.0, baseline=True)
        self.record('baseline')
        self.make_run('candidate', 0.13, float('nan'))
        record = self.record('candidate')
        self.assertEqual(record['status'], 'keep')
        self.assertIsNone(record['val_bpb'])

    def test_requires_a_successful_exit_and_intact_native_csv(self):
        for identifier, bad_exit in [('failed-exit', True), ('bad-csv', False)]:
            with self.subTest(identifier=identifier):
                job, directory = self.make_run(identifier, 0.12, 1.0, baseline=True)
                if bad_exit:
                    rental.save(dict(id=identifier, phase='failed', exit_code=1))
                else:
                    (directory / 'core.csv').write_text('incomplete\n')
                self.assertEqual(self.record(job['id'])['status'], 'crash')

    def test_requires_a_new_baseline_after_an_experiment_frame_change(self):
        self.make_run('baseline', 0.12, 1.0, baseline=True)
        self.record('baseline')
        job, _ = self.make_run('candidate', 0.15, 0.9)
        job['source_sha256']['program.md'] = 'changed frame'
        rental.save(job)
        with self.assertRaises(ValueError):
            self.record('candidate')

    def test_observes_an_unfinished_job_without_prematurely_recording_a_failure(self):
        self.make_run('baseline', 0.12, 1.0, baseline=True)
        rental.save(dict(id='baseline', phase='running', exit_code=None))
        with self.assertRaises(ValueError):
            self.record('baseline')


if __name__ == '__main__':
    unittest.main()
