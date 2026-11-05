import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from tools import rental


class RentalLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.root_patch = patch.object(rental, 'ROOT', self.root)
        self.root_patch.start()
        self.collection_cases = 0
        self.job = dict(id='fixture', label='autoresearch-fixture', instance_id=42,
                        phase='running', creation_uncertain=False, expires_at=time.time() + 3500,
                        expected_core_seconds=900)
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

    def test_records_completed_and_failed_attempts_after_confirming_deletion(self):
        for phase in ('evaluated', 'failed'):
            with self.subTest(phase=phase):
                rental.save(dict(id='fixture', phase=phase, creation_requested_at=1000))
                events = []

                def delete(job):
                    rental.save(dict(id=job['id'], deleted_at=1100))
                    events.append('deleted')

                def record(command):
                    self.assertEqual(rental.load('fixture')['deleted_at'], 1100)
                    self.assertEqual(command[1:], [str(self.root / 'tools/record_run.py'), 'fixture'])
                    events.append('recorded')

                with patch.object(rental, 'delete_instances', side_effect=delete):
                    with patch.object(rental, 'run', side_effect=record):
                        rental.finalize(self.job)
                self.assertEqual(events, ['deleted', 'recorded'])

    def test_reloads_the_collected_phase_before_recording_an_expiry_observation(self):
        rental.save(dict(id='fixture', creation_requested_at=1000))

        def delete(job):
            rental.save(dict(id=job['id'], phase='evaluated', exit_code=0, deleted_at=1100))

        with patch.object(rental, 'delete_instances', side_effect=delete):
            with patch.object(rental, 'run') as record:
                rental.finalize(self.job)
        record.assert_called_once()
        self.assertEqual(rental.load('fixture')['phase'], 'evaluated')

    def test_retains_the_original_deletion_failure_before_exporting_a_decision(self):
        rental.save(dict(id='fixture', phase='evaluated', creation_requested_at=1000))
        with patch.object(rental, 'delete_instances', side_effect=RuntimeError('deletion pending')):
            with patch.object(rental, 'run') as record:
                with self.assertRaisesRegex(RuntimeError, 'deletion pending'):
                    rental.finalize(self.job)
        record.assert_not_called()

    def test_leaves_unfinished_rejected_and_unrequested_jobs_for_observation(self):
        cases = [dict(phase='running', creation_requested_at=1000, creation_rejected=False),
                 dict(phase='failed', creation_requested_at=1000, creation_rejected=True),
                 dict(phase='failed', creation_requested_at=None, creation_rejected=False)]
        for fields in cases:
            with self.subTest(fields=fields):
                rental.save(dict(id='fixture', **fields))
                with patch.object(rental, 'delete_instances'):
                    with patch.object(rental, 'run') as record:
                        rental.finalize(self.job)
                record.assert_not_called()

    def test_keeps_collected_results_available_when_decision_export_fails(self):
        rental.save(dict(id='fixture', phase='evaluated', exit_code=0, creation_requested_at=1000))

        def delete(job):
            rental.save(dict(id=job['id'], deleted_at=1100))

        with patch.object(rental, 'delete_instances', side_effect=delete):
            with patch.object(rental, 'run', side_effect=subprocess.CalledProcessError(1, 'record')):
                with self.assertRaises(subprocess.CalledProcessError):
                    rental.finalize(self.job)
        stored = rental.load('fixture')
        self.assertEqual((stored['phase'], stored['exit_code'], stored['deleted_at']), ('evaluated', 0, 1100))

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

    def test_reserves_time_for_thirty_minute_training_and_the_complete_core_evaluation(self):
        with patch.object(rental.time, 'time', return_value=1000):
            with self.assertRaises(TimeoutError):
                rental.require_execution_time({**self.job, 'expires_at': 2900})
            rental.require_execution_time({**self.job, 'expires_at': 4500})

    def test_requires_a_finite_evaluation_estimate_that_fits_the_daily_rental(self):
        for estimate in (0, -1, float('nan'), float('inf'), 2000):
            with self.subTest(estimate=estimate):
                with self.assertRaises(ValueError):
                    rental.execution_budget(dict(expected_core_seconds=estimate))

    def test_ends_setup_when_transfer_uses_the_time_needed_for_the_full_evaluation(self):
        for name in ('gpu-env.tar.gz', 'gpu-assets.tar.gz'):
            path = self.root / 'artifacts' / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'archive fixture')
        (rental.job_path('fixture').parent / 'source.tar.gz').write_bytes(b'source fixture')
        now = [1000]
        job = {**self.job, 'expires_at': 4500}
        launched = []

        def command(arguments, **kwargs):
            text = arguments[-1]
            if text.startswith('nvidia-smi'):
                return subprocess.CompletedProcess(arguments, 0, stdout='NVIDIA H100 80GB HBM3, 0\n')
            if text.startswith('tar -xzf'):
                now[0] += 300
            if 'nohup bash gpu_run.sh' in text:
                launched.append(text)
            return subprocess.CompletedProcess(arguments, 0, stdout='')

        with patch.object(rental, 'ssh_connection', return_value=['ssh']):
            with patch.object(rental, 'run', side_effect=command):
                with patch.object(rental.time, 'time', side_effect=lambda: now[0]):
                    with self.assertRaises(TimeoutError):
                        rental.execute(job, {}, self.root / 'key')
        self.assertEqual(launched, [])

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
            if 'utilization.gpu' in text:
                raise subprocess.TimeoutExpired('nvidia-smi', 5)
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
        sample = json.loads((self.root / 'results/runs/fixture/gpu-observations.jsonl').read_text())
        self.assertEqual(sample['error_type'], 'TimeoutExpired')

    def collect_at_boundary(self, exit_bytes, observed=None, fail_refresh=False):
        identifier = f'collection-{self.collection_cases}'
        self.collection_cases += 1
        job = {**self.job, 'id': identifier, 'label': 'autoresearch-' + identifier, 'expires_at': 4500}
        rental.job_path(identifier).parent.mkdir(parents=True)
        rental.save(job)
        for name in ('gpu-env.tar.gz', 'gpu-assets.tar.gz'):
            path = self.root / 'artifacts' / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'archive fixture')
        (rental.job_path(identifier).parent / 'source.tar.gz').write_bytes(b'source fixture')
        now = [1000]
        launches, reads = [], {}

        def command(arguments, **kwargs):
            text = arguments[-1]
            if text.startswith('nvidia-smi'):
                fields = '0, 0' if 'utilization.gpu' in text else '0'
                return subprocess.CompletedProcess(arguments, 0, stdout='NVIDIA H100 80GB HBM3, ' + fields + '\n')
            if 'nohup bash gpu_run.sh' in text:
                launches.append(text)
            if 'kill -0' in text:
                probe = 'running' if observed is None else f'exit {observed}'
                return subprocess.CompletedProcess(arguments, 0, stdout=probe)
            if text.startswith('cd autoresearch && cat '):
                name = Path(text.split('cat ', 1)[1]).name
                reads[name] = reads.get(name, 0) + 1
                if name == 'run.exit':
                    if exit_bytes is None:
                        raise subprocess.CalledProcessError(1, arguments)
                    kwargs['stdout'].write(exit_bytes)
                elif name == 'run.log' and reads[name] == 1 and observed is None:
                    kwargs['stdout'].write(b'snapshot-before-exit\n')
                elif name == 'run.log' and fail_refresh:
                    kwargs['stdout'].write(b'incomplete-transport-chunk')
                    raise subprocess.TimeoutExpired(arguments, 5)
                else:
                    kwargs['stdout'].write(('final-' + name + '\n').encode())
            return subprocess.CompletedProcess(arguments, 0, stdout='')

        def sleep(seconds):
            now[0] = job['expires_at'] - rental.COLLECTION_SECONDS

        with patch.object(rental, 'ssh_connection', return_value=['ssh']):
            with patch.object(rental, 'run', side_effect=command):
                with patch.object(rental.time, 'time', side_effect=lambda: now[0]):
                    with patch.object(rental.time, 'sleep', side_effect=sleep):
                        rental.execute(job, {}, self.root / 'key')
        return rental.load(identifier), reads, launches

    def test_observes_completion_during_collection_and_preserves_the_final_artifact_snapshot(self):
        job, reads, launches = self.collect_at_boundary(b'0\n')
        self.assertEqual((job['phase'], job['exit_code']), ('evaluated', 0))
        self.assertTrue(job['exit_code_observed_in_collection'])
        self.assertEqual(len(launches), 1)
        self.assertEqual((self.root / 'results/runs' / job['id'] / 'run.log').read_bytes(), b'final-run.log\n')
        self.assertEqual(reads['run.log'], 2)

    def test_keeps_failed_or_unconfirmed_collected_processes_failed(self):
        for content, code in ((b'1\n', 1), (b'137\n', 137), (b'0 invalid\n', None), (b'', None), (None, None)):
            with self.subTest(content=content):
                job, reads, launches = self.collect_at_boundary(content)
                self.assertEqual((job['phase'], job['exit_code']), ('failed', code))
                self.assertEqual(len(launches), 1)

    def test_preserves_the_previous_artifact_when_a_final_snapshot_copy_times_out(self):
        job, reads, launches = self.collect_at_boundary(b'0\n', fail_refresh=True)
        self.assertEqual(job['exit_code'], 0)
        self.assertEqual((self.root / 'results/runs' / job['id'] / 'run.log').read_bytes(), b'snapshot-before-exit\n')
        self.assertEqual(len(launches), 1)

    def test_rejects_conflicting_exit_status_observations(self):
        with self.assertRaisesRegex(RuntimeError, 'Exit status'):
            self.collect_at_boundary(b'1\n', observed=0)

    def test_records_timestamped_gpu_memory_and_utilization_samples_in_order(self):
        results = [subprocess.CompletedProcess([], 0, stdout='NVIDIA H100 80GB HBM3, 45060, 96\n'),
                   subprocess.CompletedProcess([], 0, stdout='NVIDIA H100 80GB HBM3, 2900, 12\n')]
        remote = Mock(side_effect=results)
        with patch.object(rental.time, 'time', side_effect=[1000, 1030]):
            rental.record_gpu_observation(remote, self.root)
            rental.record_gpu_observation(remote, self.root)
        samples = [json.loads(line) for line in (self.root / 'gpu-observations.jsonl').read_text().splitlines()]
        self.assertEqual([sample['observed_at'] for sample in samples], [1000, 1030])
        self.assertEqual([sample['memory_mib'] for sample in samples], [45060, 2900])
        self.assertEqual([sample['utilization_percent'] for sample in samples], [96, 12])

    def test_records_unavailable_telemetry_and_accepts_the_next_valid_sample(self):
        remote = Mock(side_effect=[
            subprocess.CompletedProcess([], 0, stdout='NVIDIA H100 80GB HBM3, 45060, N/A\n'),
            subprocess.CompletedProcess([], 0, stdout='NVIDIA H100 80GB HBM3, 45060, 90\n'),
        ])
        rental.record_gpu_observation(remote, self.root)
        rental.record_gpu_observation(remote, self.root)
        samples = [json.loads(line) for line in (self.root / 'gpu-observations.jsonl').read_text().splitlines()]
        self.assertEqual(samples[0]['error_type'], 'ValueError')
        self.assertEqual(samples[1]['utilization_percent'], 90)


if __name__ == '__main__':
    unittest.main()
