import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from tools import rental


class RentalLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.root_patch = patch.object(rental, 'ROOT', self.root)
        self.root_patch.start()
        self.job = dict(id='fixture', label='autoresearch-fixture', instance_id=42,
                        phase='running', creation_uncertain=False, expires_at=time.time() + 180)
        rental.job_path(self.job['id']).parent.mkdir(parents=True)
        rental.save(self.job)

    def tearDown(self):
        self.root_patch.stop()
        self.directory.cleanup()

    def test_preserves_the_completed_run_when_expiry_updates_its_deletion_state(self):
        rental.save(dict(id='fixture', phase='evaluated', exit_code=0))
        rental.save(dict(id='fixture', deleted_at=123))
        stored = rental.load('fixture')
        self.assertEqual(stored['phase'], 'evaluated')
        self.assertEqual(stored['exit_code'], 0)
        self.assertEqual(stored['deleted_at'], 123)

    def test_keeps_another_experiments_instance_untouched(self):
        with patch.object(rental, 'list_instances', return_value=[{'id': 42, 'label': 'another-project'}]):
            with patch.object(rental, 'api') as api:
                with self.assertRaises(RuntimeError):
                    rental.delete_instances(self.job)
        api.assert_not_called()

    def test_confirms_deletion_using_the_provider_after_destroying_the_owned_instance(self):
        observations = [[{'id': 42, 'label': self.job['label']}], []]
        with patch.object(rental, 'list_instances', side_effect=observations):
            with patch.object(rental, 'api', return_value={'success': True}) as api:
                rental.delete_instances(self.job)
        api.assert_called_once()
        self.assertEqual(api.call_args.args[:2], ('DELETE', 'instances/42/'))
        self.assertIn('deleted_at', rental.load('fixture'))

    def test_keeps_deletion_pending_while_the_provider_still_lists_the_instance(self):
        rows = [{'id': 42, 'label': self.job['label']}]
        with patch.object(rental, 'list_instances', return_value=rows):
            with patch.object(rental, 'api', return_value={'success': True}):
                with self.assertRaises(RuntimeError):
                    rental.delete_instances(self.job)
        self.assertNotIn('deleted_at', rental.load('fixture'))

    def test_keeps_observing_the_original_label_when_the_create_response_is_uncertain(self):
        job = {**self.job, 'instance_id': None, 'creation_uncertain': True}
        with patch.object(rental, 'list_instances', return_value=[]):
            with patch.object(rental, 'api') as api:
                with self.assertRaises(RuntimeError):
                    rental.delete_instances(job)
        api.assert_not_called()

    def test_prevents_a_second_daily_request_using_the_durable_request_history(self):
        rental.save(dict(id='fixture', creation_requested_at=1000))
        with patch.object(rental, 'recent_rentals') as recent:
            with self.assertRaises(RuntimeError):
                rental.check_daily_budget(2000)
        recent.assert_not_called()

    def test_prevents_a_new_request_when_the_provider_reports_a_recent_research_contract(self):
        with patch.object(rental, 'recent_rentals', return_value=[{'id': 99}]):
            with self.assertRaises(RuntimeError):
                rental.check_daily_budget(2000)

    def test_allows_a_fresh_request_after_a_provider_confirmed_rejection(self):
        rental.save(dict(id='fixture', creation_requested_at=1000, creation_rejected=True))
        with patch.object(rental, 'recent_rentals', return_value=[]):
            rental.check_daily_budget(2000)

    def test_rechecks_absence_after_an_acknowledged_delete_without_destroying_again(self):
        rental.save(dict(id='fixture', delete_acknowledged_at=123))
        with patch.object(rental, 'list_instances', return_value=[]), patch.object(rental, 'api') as api:
            rental.delete_instances(self.job)
        api.assert_not_called()
        self.assertIn('deleted_at', rental.load('fixture'))

    def test_cancels_a_scheduled_job_before_its_worker_has_started(self):
        job = {**self.job, 'launch_unit': 'autoresearch-start-fixture', 'phase': 'prepared'}
        rental.save(job)
        states = [subprocess.CompletedProcess([], 0, stdout='loaded\n'),
                  subprocess.CompletedProcess([], 1, stdout='not-found\n')]
        with patch.object(rental.subprocess, 'run', side_effect=states):
            with patch.object(rental, 'run') as stop:
                rental.cancel(job)
        stop.assert_called_once_with(['systemctl', '--user', 'stop', 'autoresearch-start-fixture.timer'])
        self.assertEqual(rental.load('fixture')['phase'], 'cancelled')

    def test_selects_a_single_driver_compatible_h100_sxm_from_the_provider_results(self):
        base = dict(gpu_name='H100 SXM', num_gpus=1, cuda_max_good=12.8, gpu_ram=81559)
        offers = [{**base, 'id': 1, 'num_gpus': 2, 'dph_total': 1.0},
                  {**base, 'id': 2, 'gpu_name': 'H200', 'dph_total': 1.0},
                  {**base, 'id': 3, 'cuda_max_good': 12.6, 'dph_total': 1.0},
                  {**base, 'id': 4, 'dph_total': 4.0},
                  {**base, 'id': 5, 'gpu_name': 'H100 NVL', 'dph_total': 1.0},
                  {**base, 'id': 6, 'gpu_name': 'H100 PCIE', 'dph_total': 1.0}]
        with patch.object(rental, 'api', return_value={'offers': offers}):
            self.assertEqual(rental.choose_offer()['id'], 4)

    def test_requires_an_active_independent_expiry_timer_before_proceeding(self):
        with patch.object(rental, 'run'):
            with patch.object(rental.subprocess, 'check_output', return_value='inactive\n'):
                with self.assertRaises(RuntimeError):
                    rental.arm_expiry(self.job)

    def test_starts_training_once_when_one_observation_times_out(self):
        for name in ('gpu-env.tar.gz', 'gpu-assets.tar.gz'):
            path = self.root / 'artifacts' / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'archive fixture')
        (rental.job_path('fixture').parent / 'source.tar.gz').write_bytes(b'source fixture')
        probes = iter([subprocess.TimeoutExpired('ssh', 10), 'running', 'exit 0'])
        launched = []

        def command(arguments, **kwargs):
            text = arguments[-1]
            if text.startswith('nvidia-smi'):
                return subprocess.CompletedProcess(arguments, 0, stdout='NVIDIA H100 80GB HBM3, 0\n')
            if 'nohup bash gpu_run.sh' in text:
                launched.append(text)
            if 'kill -0' in text:
                result = next(probes)
                if isinstance(result, BaseException):
                    raise result
                return subprocess.CompletedProcess(arguments, 0, stdout=result)
            if kwargs.get('stdout') is not None:
                kwargs['stdout'].write(b'fixture artifact')
            return subprocess.CompletedProcess(arguments, 0, stdout='')

        key = self.root / 'key'
        with patch.object(rental, 'ssh_connection', return_value=['ssh']):
            with patch.object(rental, 'run', side_effect=command), patch.object(rental.time, 'sleep'):
                rental.execute(self.job, {}, key)
        self.assertEqual(len(launched), 1)
        self.assertEqual(rental.load('fixture')['exit_code'], 0)


if __name__ == '__main__':
    unittest.main()
