import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import jobs


class JobTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        root = patch.object(jobs, 'ROOT', self.root)
        root.start()
        self.addCleanup(root.stop)
        self.job = dict(id='fixture', expected_eval_seconds=900,
                        gpu_settings=dict(training_seconds=1800, rental_seconds=3600, max_hourly_cost=5))

    def test_merges_each_saved_field_into_the_job(self):
        jobs.job_path('fixture').parent.mkdir(parents=True)
        jobs.save(dict(id='fixture', phase='evaluated', exit_code=0))
        jobs.save(dict(id='fixture', returned_at=123))
        self.assertEqual(jobs.load('fixture'), dict(id='fixture', phase='evaluated', exit_code=0, returned_at=123))

    def test_keeps_jobs_to_plain_names_inside_the_queue(self):
        for identifier in ('../outside', 'Upper', 'with space'):
            with self.subTest(identifier=identifier):
                with self.assertRaises(ValueError):
                    jobs.job_path(identifier)

    def test_reserves_setup_training_the_full_evaluation_and_collection_on_the_machine(self):
        self.assertEqual(jobs.machine_seconds(self.job),
                         jobs.SETUP_SECONDS + 1800 + 900 + jobs.COLLECTION_SECONDS)

    def test_requires_a_positive_finite_evaluation_estimate(self):
        for estimate in (0, -1, math.nan, math.inf):
            with self.subTest(estimate=estimate):
                with self.assertRaises(ValueError):
                    jobs.machine_seconds(dict(self.job, expected_eval_seconds=estimate))

    def test_requires_the_run_to_fit_its_rental_with_time_to_give_the_machine_back(self):
        jobs.check_fit(self.job)
        room = 3600 - jobs.RETURN_SECONDS - jobs.machine_seconds(self.job)
        with self.assertRaisesRegex(ValueError, 'does not fit'):
            jobs.check_fit(dict(self.job, expected_eval_seconds=900 + room))


if __name__ == '__main__':
    unittest.main()
