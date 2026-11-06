from concurrent.futures import ThreadPoolExecutor
import ast
import json
import os
from pathlib import Path
import subprocess
import tempfile
import tarfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from tools import rental
from gpu_config import read_config


class RentalLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.hold_patch = patch.dict(rental.os.environ, {
            'AUTORESEARCH_GPU': 'h100', 'AUTORESEARCH_RENTAL_HOLD_H100': '0',
            'AUTORESEARCH_RENTAL_HOLD_RTX5090': '0', **{name: '' for name in rental.r2.VARIABLES}})
        self.hold_patch.start()
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        config = read_config()
        config.update(active_gpu='h100', rental_scope='gpu')
        self.config_path = self.root / 'fixture-gpus.json'
        self.config_path.write_text(json.dumps(config))
        self.config_patch = patch('gpu_config.CONFIG_PATH', self.config_path)
        self.config_patch.start()
        self.root_patch = patch.object(rental, 'ROOT', self.root)
        self.root_patch.start()
        for default in (patch.object(rental, 'list_instances', return_value=[]),
                        patch.object(rental.vast_backend, 'entries', return_value=[]),
                        patch.object(rental.vast_backend, 'rent',
                                     side_effect=AssertionError('Use a fixture for vendor rentals.'))):
            default.start()
            self.addCleanup(default.stop)
        self.transfer_patch = patch.object(rental, 'stream_archive', return_value={'fixture': 'transfer'})
        self.transfer = self.transfer_patch.start()
        self.collection_cases = 0
        self.job = dict(id='fixture', label='autoresearch-fixture', instance_id=42,
                        phase='running', creation_uncertain=False, expires_at=time.time() + 3500,
                        gpu_settings=dict(training_seconds=1800, rental_seconds=3600, max_hourly_cost=5),
                        expected_eval_seconds=900)
        rental.job_path(self.job['id']).parent.mkdir(parents=True)
        rental.save(self.job)

    def tearDown(self):
        self.transfer_patch.stop()
        self.root_patch.stop()
        self.directory.cleanup()
        self.hold_patch.stop()
        self.config_patch.stop()

    def test_records_preparation_time_and_data_readiness_from_the_collected_run(self):
        gpu = self.root / 'autoresearch'
        gpu.mkdir()

        def remote(command, **kwargs):
            return subprocess.run(['bash', '-c', command], cwd=self.root, check=True, **kwargs)

        for data_message in ('Data: 71/71 shards ready', 'Data: all 71 shards already downloaded'):
            with self.subTest(data=data_message):
                log = data_message + '\nDone! Ready to train.\n'
                (self.root / 'prepare.log').write_text(log)
                for name, modified, text in (
                        ('run.pid', 1000, '123\n'), ('prepare.log', 1063, log),
                        ('run.log', 5000, 'completed training log'), ('run.exit', 5001, '0\n')):
                    path = gpu / name
                    path.write_text(text)
                    os.utime(path, (modified, modified))
                rental.record_preparation_timing(remote, self.root)
                observation = json.loads((self.root / 'preparation-timing.json').read_text())
                self.assertEqual(observation['approximate_preparation_seconds'], 63)
                self.assertEqual(observation['ready_shards'], 71)
                self.assertEqual(observation['requested_shards'], 71)
                self.assertTrue(observation['prepare_log_completed_marker'])

    def test_preserves_an_unavailable_timing_record_when_the_remote_read_times_out(self):
        remote = Mock(side_effect=subprocess.TimeoutExpired('ssh', 5))
        rental.record_preparation_timing(remote, self.root)
        observation = json.loads((self.root / 'preparation-timing.json').read_text())
        self.assertEqual(observation['error_type'], 'TimeoutExpired')
        self.assertEqual(observation['files'], {})

    def test_honors_the_users_rental_hold_before_requesting_a_machine(self):
        job = {**self.job, 'phase': 'prepared'}
        with patch.dict(rental.os.environ, {'AUTORESEARCH_RENTAL_HOLD_H100': '1'}):
            with patch.object(rental, 'api') as api:
                with self.assertRaisesRegex(RuntimeError, 'on hold by user request'):
                    rental.start(job)
        api.assert_not_called()
        self.assertEqual(job['phase'], 'prepared')
        self.assertNotIn('creation_requested_at', job)

    def test_keeps_a_hold_on_one_gpu_from_blocking_the_other_gpus_preflight(self):
        job = {**self.job, 'phase': 'prepared'}
        with patch.dict(rental.os.environ, {'AUTORESEARCH_RENTAL_HOLD_RTX5090': '1'}):
            with patch.object(rental, 'digest', side_effect=RuntimeError('reached H100 preflight')):
                with self.assertRaisesRegex(RuntimeError, 'reached H100 preflight'):
                    rental.start(job)

    def test_starts_the_named_gpu_with_its_own_environment_and_lock(self):
        for identifier in ('h100', 'rtx5090'):
            job = dict(self.job, gpu_id=identifier)
            with self.subTest(gpu=identifier):
                with patch.object(rental.sys, 'argv', ['rental.py', '--gpu', identifier, 'start', 'fixture']):
                    with patch.object(rental, 'load', return_value=job):
                        with patch.object(rental, 'start') as start:
                            rental.main()
                            start.assert_called_once_with(job)
                            self.assertEqual(rental.os.environ['AUTORESEARCH_GPU'], identifier)
                self.assertTrue(rental.rental_lock(rental.get_profile(identifier)).exists())

    def test_checks_the_gpu_identity_before_starting_an_existing_job(self):
        with patch.object(rental.sys, 'argv', ['rental.py', '--gpu', 'rtx5090', 'start', 'fixture']):
            with patch.object(rental, 'start') as start, patch.object(rental.sys, 'stderr'):
                with self.assertRaises(SystemExit):
                    rental.main()
        start.assert_not_called()

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

    def test_returns_a_checked_machine_through_the_shared_tool_and_confirms_its_absence(self):
        job = dict(self.job, rental_backend='vast-rental')
        observations = [[dict(id=42, label=job['label'])], []]
        with patch.object(rental, 'list_instances', side_effect=observations), \
                patch.object(rental.vast_backend, 'command') as command:
            rental.delete_instances(job)
        command.assert_called_once_with(['return', '42'], timeout=30)
        self.assertIn('deleted_at', rental.load('fixture'))

    def test_connects_through_the_checked_route_with_the_callers_key_file(self):
        instance = dict(id=42, ssh_config='fixture.config', ssh_host='vast-42')
        command = rental.ssh_connection(instance, Path('fixture-key'))
        self.assertEqual(command, ['ssh', '-F', 'fixture.config', '-i', 'fixture-key',
                                   '-o', 'IdentitiesOnly=yes', '-o', 'IdentityAgent=none',
                                   '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'vast-42'])

    def test_keeps_observing_the_original_label_when_the_create_response_is_uncertain(self):
        job = {**self.job, 'instance_id': None, 'creation_uncertain': True}
        with patch.object(rental, 'list_instances', return_value=[]):
            with patch.object(rental, 'api') as api:
                with self.assertRaises(RuntimeError):
                    rental.delete_instances(job)
        api.assert_not_called()

    def test_prevents_another_request_while_an_attempt_has_not_finished(self):
        rental.save(dict(id='fixture', creation_requested_at=1000))
        with self.assertRaises(RuntimeError):
            rental.check_daily_budget(2000)

    def test_prevents_a_new_request_while_the_provider_reports_an_active_loop_gpu(self):
        with patch.object(rental, 'list_instances', return_value=[{'id': 99, 'label': 'autoresearch-active'}]):
            with self.assertRaises(RuntimeError):
                rental.check_daily_budget(2000)

    def test_allows_the_approved_historical_six_cent_failure_to_be_retried(self):
        rental.save(dict(id='fixture', phase='failed', creation_requested_at=1000,
                         deleted_at=1493.253, hourly_cost=0.4116666666666667))
        self.assertEqual(rental.check_daily_budget(2000), rental.Decimal('0.06'))

    def test_applies_the_daily_failed_cost_limit_to_shared_ledger_amounts_once(self):
        rental.save(dict(id='fixture', phase='failed', creation_requested_at=1000,
                         deleted_at=1500, rental_backend='vast-rental'))
        for amount, allowed in (('0.06', True), ('0.50', True), ('0.501', False)):
            with self.subTest(amount=amount):
                entries = [dict(id='rent', event='rented', instance=42, label=self.job['label'],
                                at='1970-01-01T00:16:40+00:00'),
                           dict(id='failed', event='unusable', instance=42),
                           dict(id='back', event='returned', instance=42, dollars=amount)]
                with patch.object(rental.vast_backend, 'entries', return_value=entries):
                    if allowed:
                        self.assertEqual(rental.check_daily_budget(2000), rental.Decimal(amount))
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'more than'):
                            rental.check_daily_budget(2000)

    def test_keeps_a_completed_measured_run_as_the_only_valid_run_that_day(self):
        rental.save(dict(id='fixture', phase='evaluated', exit_code=0,
                         creation_requested_at=1000, deleted_at=1500))
        with self.assertRaisesRegex(RuntimeError, 'measured run today'):
            rental.check_daily_budget(2000)

    def test_allows_the_next_utc_calendar_day_without_waiting_twenty_four_hours(self):
        rental.save(dict(id='fixture', phase='evaluated', exit_code=0,
                         creation_requested_at=86300, deleted_at=86400))
        self.assertEqual(rental.check_daily_budget(86401), 0)

    def test_ignores_the_cost_and_instances_of_other_projects(self):
        entries = [dict(id='rent', event='rented', instance=99, label='another-project',
                        at='1970-01-01T00:16:40+00:00'),
                   dict(id='failed', event='unusable', instance=99),
                   dict(id='back', event='returned', instance=99, dollars=10)]
        with patch.object(rental.vast_backend, 'entries', return_value=entries), \
                patch.object(rental, 'list_instances', return_value=[dict(id=99, label='another-project')]):
            self.assertEqual(rental.check_daily_budget(2000), 0)

    def test_counts_an_owned_live_instance_even_when_its_contract_is_more_than_a_day_old(self):
        row = dict(id=42, label='autoresearch-h100-old', start_date=0, actual_status='running')
        with patch.object(rental, 'list_instances', return_value=[row]):
            with patch.object(rental, 'api', return_value={'invoices': []}):
                self.assertEqual(rental.recent_rentals(100000), [row])

    def test_counts_only_this_loops_recent_contracts_when_other_projects_are_running(self):
        other = dict(id=90, label='another-project', start_date=0, actual_status='running')
        owned = dict(id=42, label='autoresearch-h100-history', start_date=99000)
        responses = [{'invoices': [{'instance_id': 42}]}, {'contracts': [owned, other]}]
        with patch.object(rental, 'list_instances', return_value=[other]):
            with patch.object(rental, 'api', side_effect=responses):
                self.assertEqual(rental.recent_rentals(100000), [owned])

    def test_allows_the_other_gpu_to_run_while_counting_legacy_rentals_for_the_h100(self):
        rtx = dict(id=90, label='autoresearch-rtx5090-live', start_date=0)
        legacy = dict(id=42, label='autoresearch-20261104-legacy', start_date=99000)
        responses = [{'invoices': [{'instance_id': 42}]}, {'contracts': [legacy, rtx]}]
        with patch.object(rental, 'list_instances', return_value=[rtx]):
            with patch.object(rental, 'api', side_effect=responses):
                self.assertEqual(rental.recent_rentals(100000), [legacy])
        with patch.object(rental, 'list_instances', return_value=[rtx, legacy]):
            self.assertEqual(rental.recent_rentals(100000, rental.get_profile('rtx5090')), [rtx])

    def test_binds_new_launch_and_expiry_timers_to_the_jobs_gpu(self):
        jobs = [dict(self.job, gpu_id=identifier) for identifier in ('h100', 'rtx5090')]
        for job in jobs:
            with self.subTest(gpu=job['gpu_id']):
                with patch.object(rental, 'run') as run:
                    with patch.object(rental.subprocess, 'check_output', return_value='active\n'):
                        rental.arm_expiry(job)
                command = run.call_args.args[0]
                self.assertIn('--unit=autoresearch-expire-' + job['gpu_id'] + '-fixture', command)
                self.assertEqual(command[-4:], ['--gpu', job['gpu_id'], 'expire', 'fixture'])
                with patch.object(rental, 'run') as run:
                    with patch.object(rental.subprocess, 'check_output', return_value='active\n'):
                        rental.schedule(dict(job, phase='prepared'), '2099-01-01T00:00:00Z')
                command = run.call_args.args[0]
                self.assertIn('--unit=autoresearch-start-' + job['gpu_id'] + '-fixture', command)
                self.assertEqual(command[-4:], ['--gpu', job['gpu_id'], 'start', 'fixture'])

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
        base = dict(gpu_name='H100 SXM', num_gpus=1, cuda_max_good=12.8, gpu_ram=81559,
                    verification='verified')
        offers = [{**base, 'id': 1, 'num_gpus': 2, 'dph_total': 1.0},
                  {**base, 'id': 2, 'gpu_name': 'H200', 'dph_total': 1.0},
                  {**base, 'id': 3, 'cuda_max_good': 12.6, 'dph_total': 1.0},
                  {**base, 'id': 4, 'dph_total': 4.0},
                  {**base, 'id': 5, 'gpu_name': 'H100 NVL', 'dph_total': 1.0},
                  {**base, 'id': 6, 'gpu_name': 'H100 PCIE', 'dph_total': 1.0}]
        with patch.object(rental, 'api', return_value={'offers': offers}):
            self.assertEqual(rental.choose_offer()['id'], 4)

    def test_selects_verified_machines_from_the_provider_results(self):
        base = dict(gpu_name='H100 SXM', num_gpus=1, cuda_max_good=12.8, gpu_ram=81559)
        offers = [dict(base, id=index, dph_total=index, verification=state)
                  for index, state in enumerate(('deverified', 'unverified', None, 'verified'), 1)]
        with patch.object(rental, 'api', return_value={'offers': offers}) as api:
            self.assertEqual(rental.choose_offer()['id'], 4)
        self.assertEqual(api.call_args.args[2]['verified'], {'eq': True})

    def test_keeps_recorded_physical_machine_failures_out_of_future_rentals(self):
        profile = rental.get_profile('rtx5090')
        base = dict(gpu_name=profile.gpu_name, num_gpus=1, cuda_max_good=12.8,
                    gpu_ram=32000, verification='verified')
        offers = [dict(base, id=1, machine_id=10, dph_total=0.50),
                  dict(base, id=2, machine_id=20, dph_total=0.60),
                  dict(base, id=3, machine_id=30, dph_total=0.70)]
        path = self.root / 'results/operations/machine-failures.json'
        path.parent.mkdir(parents=True)
        recent = dict(gpu_id='rtx5090', machine_id=10, reported_at=1000000 - 60)
        for second, expected in (
                (dict(recent, machine_id=20), 3),
                (dict(recent, machine_id=20, gpu_id='h100'), 3),
                (dict(recent, machine_id=20, reported_at=1000000 - 8 * 86400), 3)):
            with self.subTest(second=second):
                path.write_text(json.dumps([recent, second]))
                with patch.object(rental.time, 'time', return_value=1000000), \
                        patch.object(rental, 'api', return_value={'offers': offers}) as api:
                    self.assertEqual(rental.choose_offer(profile)['id'], expected)
                excluded = api.call_args.args[2]['machine_id']['notin']
                self.assertIn(10, excluded)

    def test_selects_the_machine_after_a_code_failure_but_not_a_confirmed_machine_failure(self):
        base = dict(gpu_name='H100 SXM', num_gpus=1, cuda_max_good=12.8,
                    gpu_ram=81559, verification='verified')
        offers = [dict(base, id=1, machine_id=10, dph_total=1),
                  dict(base, id=2, machine_id=20, dph_total=2)]
        for identifier, machine, kind in (('machine-failure', 10, 'machine'), ('code-failure', 20, 'code')):
            rental.job_path(identifier).parent.mkdir()
            rental.save(dict(id=identifier, gpu_id='h100', creation_requested_at=1000,
                             provider_machine_id=machine, failure_kind=kind, error_type='RuntimeError'))
        with patch.object(rental, 'api', return_value={'offers': offers}):
            self.assertEqual(rental.choose_offer()['id'], 2)

    def test_retries_vendor_rejected_machines_before_starting_the_single_run(self):
        _, manifest, job = self.rtx_runtime()
        directory = rental.job_path('fixture').parent
        manifest['container_image'] = 'fixture-image'
        (directory / 'runtime.json').write_text(json.dumps(manifest))
        job.update(phase='prepared', instance_id=None, max_hourly_cost=1,
                   source_archive_sha256=rental.digest(directory / 'source.tar.gz'),
                   runtime_manifest_sha256=rental.digest(directory / 'runtime.json'))
        rental.save(job)
        rejected = dict(rented=None, offers=2, failed=1)
        accepted = dict(rented=42, machine=70, price=0.5,
                        ssh_config='fixture.config', ssh_host='vast-42')
        attempts = [dict(unusable=41, machine=71, stage='check'), rejected]
        with patch.object(rental, 'check_daily_budget'), \
                patch.object(rental, 'validate_environment', return_value='fixture-public-key'), \
                patch.object(rental, 'arm_expiry'), \
                patch.object(rental.vast_backend, 'rent', side_effect=[
                    (rejected, attempts), (accepted, [accepted])]) as vendor, \
                patch.object(rental, 'execute') as execute, \
                patch.object(rental, 'finalize') as finalize, \
                patch.dict(rental.os.environ, {'VAST_SSH_KEY': 'fixture-private-key'}):
            rental.start(job)
        stored = rental.load('fixture')
        self.assertEqual(vendor.call_count, 2)
        self.assertEqual(stored['selection_attempts'][0]['unusable'], 41)
        self.assertEqual(stored['instance_id'], 42)
        self.assertEqual(stored['provider_machine_id'], 70)
        execute.assert_called_once()
        finalize.assert_called_once()

    def test_records_machine_failures_when_the_gpu_preflight_fails(self):
        for response in (subprocess.CompletedProcess([], 0, stdout='NVIDIA H200, 0\n'),
                         subprocess.CalledProcessError(1, ['nvidia-smi'])):
            with self.subTest(response=response):
                job = dict(self.job, failure_kind='unknown')
                rental.save(job)

                def command(*args, **kwargs):
                    if isinstance(response, Exception):
                        raise response
                    return response

                with patch.object(rental, 'ssh_connection', return_value=['ssh']), \
                        patch.object(rental, 'run', side_effect=command):
                    with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                        rental.execute(job, {}, self.root / 'key')
                self.assertEqual(rental.load('fixture')['failure_kind'], 'machine')

    def test_selects_a_lower_total_cost_for_the_prepared_deployment(self):
        profile = rental.get_profile('rtx5090')
        base = dict(gpu_name=profile.gpu_name, num_gpus=1, cuda_max_good=12.8,
                    gpu_ram=32000, inet_up_cost=0.01, verification='verified')
        offers = [dict(base, id=1, dph_total=0.50, inet_down_cost=0.10),
                  dict(base, id=2, dph_total=0.55, inet_down_cost=0.01)]
        with patch.object(rental, 'api', return_value={'offers': offers}):
            chosen = rental.choose_offer(profile, download_bytes=4 * 1024 ** 3)
        self.assertEqual(chosen['id'], 2)

    def test_uses_the_prepared_lease_and_hourly_limit_for_transfer_savings(self):
        profile = rental.get_profile('rtx5090')
        base = dict(gpu_name=profile.gpu_name, num_gpus=1, cuda_max_good=12.8,
                    gpu_ram=32000, inet_up_cost=0.01, verification='verified')
        offers = [dict(base, id=1, dph_total=0.50, inet_down_cost=0.05),
                  dict(base, id=2, dph_total=0.60, inet_down_cost=0.01)]
        with patch.object(rental, 'api', return_value={'offers': offers}):
            self.assertEqual(rental.choose_offer(profile, download_bytes=4 * 1024 ** 3)['id'], 1)
            settings = dict(rental_seconds=3600, max_hourly_cost=1)
            self.assertEqual(rental.choose_offer(profile, download_bytes=4 * 1024 ** 3,
                                                 settings=settings)['id'], 2)
            settings['max_hourly_cost'] = 0.55
            self.assertEqual(rental.choose_offer(profile, download_bytes=4 * 1024 ** 3,
                                                 settings=settings)['id'], 1)

    def test_preserves_the_hourly_choice_when_bandwidth_prices_are_unknown(self):
        for price in (None, -1, float('nan'), float('inf'), 'unknown', True):
            with self.subTest(price=price):
                offers = [dict(id=1, dph_total=0.50, inet_down_cost=price, inet_up_cost=0.01),
                          dict(id=2, dph_total=0.55, inet_down_cost=0, inet_up_cost=0)]
                self.assertEqual(rental.lowest_cost_offer(offers, 4 * 1024 ** 3, 6300, 1)['id'], 1)

    def test_keeps_upload_costs_from_overturning_known_download_savings(self):
        offers = [dict(id=1, dph_total=0.50, inet_down_cost=0.10, inet_up_cost=0.01),
                  dict(id=2, dph_total=0.55, inet_down_cost=0, inet_up_cost=0.02)]
        self.assertEqual(rental.lowest_cost_offer(offers, 4 * 1024 ** 3, 6300, 1)['id'], 1)
        offers[1]['inet_up_cost'] = 0.01
        self.assertEqual(rental.lowest_cost_offer(offers, 0, 6300, 1)['id'], 1)

    def test_requires_an_active_independent_expiry_timer_before_proceeding(self):
        with patch.object(rental, 'run'):
            with patch.object(rental.subprocess, 'check_output', return_value='inactive\n'):
                with self.assertRaises(RuntimeError):
                    rental.arm_expiry(self.job)

    def test_reserves_time_for_thirty_minute_training_and_bpb_evaluation(self):
        with patch.object(rental.time, 'time', return_value=1000):
            with self.assertRaises(TimeoutError):
                rental.require_execution_time({**self.job, 'expires_at': 2900})
            rental.require_execution_time({**self.job, 'expires_at': 4500})

    def test_requires_a_finite_evaluation_estimate_that_fits_the_daily_rental(self):
        for estimate in (0, -1, float('nan'), float('inf'), 2000):
            with self.subTest(estimate=estimate):
                with self.assertRaises(ValueError):
                    rental.execution_budget(dict(expected_eval_seconds=estimate))

    def test_snapshots_current_sources_with_the_prepared_runtime(self):
        artifacts = self.root / 'gpus/h100/artifacts'
        artifacts.mkdir(parents=True)
        manifest = {'gpu_id': 'h100', 'tokenizer': {'fixture': 'tokenizer'}, 'sources': {}}
        for name, field in (('gpu-env.tar.gz', 'gpu_env_sha256'),
                            ('gpu-assets.tar.gz', 'gpu_assets_sha256')):
            (artifacts / name).write_bytes(name.encode())
            manifest[field] = rental.digest(artifacts / name)
        (artifacts / 'runtime.json').write_text(json.dumps(manifest))
        for name in rental.source_files():
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text('current ' + name)
        (self.root / 'tools').mkdir()
        (self.root / 'tools/gpu_run.sh').write_text('fixture')
        with patch('tools.package_runtime.verify_tokenizer', return_value=manifest['tokenizer']):
            with patch('builtins.print'):
                rental.prepare_job('a' * 40, False, 'fixture')
        directories = [path for path in (self.root / 'queue').iterdir() if path.name != 'fixture']
        self.assertEqual(len(directories), 1)
        directory = directories[0]
        job = json.loads((directory / 'job.json').read_text())
        runtime = json.loads((directory / 'runtime.json').read_text())
        self.assertEqual(runtime['sources'], job['source_sha256'])
        with tarfile.open(directory / 'source.tar.gz') as archive:
            self.assertEqual(archive.extractfile('prepare.py').read(), b'current prepare.py')

    def test_snapshots_the_native_evaluator_with_every_imported_local_module(self):
        artifacts = self.root / 'gpus/h100/artifacts'
        artifacts.mkdir(parents=True)
        manifest = dict(gpu_id='h100', tokenizer={'fixture': 'tokenizer'}, sources={},
                        evaluation_metric='olmes_bpb')
        for name, field in (('gpu-env.tar.gz', 'gpu_env_sha256'),
                            ('gpu-assets.tar.gz', 'gpu_assets_sha256')):
            (artifacts / name).write_bytes(name.encode())
            manifest[field] = rental.digest(artifacts / name)
        (artifacts / 'runtime.json').write_text(json.dumps(manifest))
        repository = Path(__file__).resolve().parents[1]
        for name in rental.source_files() + rental.OLMES_SOURCES:
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if name.startswith('evaluation/'):
                destination.write_bytes((repository / name).read_bytes())
            else:
                destination.write_text('fixture ' + name)
        (self.root / 'tools').mkdir()
        (self.root / 'tools/gpu_run.sh').write_text('fixture')
        with patch('tools.package_runtime.verify_tokenizer', return_value=manifest['tokenizer']), \
                patch('builtins.print'):
            rental.prepare_job('a' * 40, False, 'fixture')
        directory = next(path for path in (self.root / 'queue').iterdir() if path.name != 'fixture')
        with tarfile.open(directory / 'source.tar.gz') as archive:
            names = set(archive.getnames())
            for name in names:
                if name.startswith('evaluation/') and name.endswith('.py'):
                    for node in ast.walk(ast.parse(archive.extractfile(name).read())):
                        if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('evaluation.'):
                            module = node.module.replace('.', '/') + '.py'
                            self.assertIn(module, names, f'{name} imports {module}')

    def test_rents_the_prepared_snapshot_while_the_working_sources_advance(self):
        directory = rental.job_path('fixture').parent
        source = directory / 'source.tar.gz'
        source.write_bytes(b'prepared experiment')
        manifest = {'container_image': 'fixture-image'}
        artifacts = self.root / 'artifacts'
        artifacts.mkdir()
        for name, field in (('gpu-env.tar.gz', 'gpu_env_sha256'),
                            ('gpu-assets.tar.gz', 'gpu_assets_sha256')):
            (artifacts / name).write_bytes(name.encode())
            manifest[field] = rental.digest(artifacts / name)
        runtime = directory / 'runtime.json'
        runtime.write_text(json.dumps(manifest))
        (artifacts / 'runtime.json').write_text('{"container_image":"later-image"}')
        for name in rental.source_files():
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text('later ' + name)
        job = {**self.job, 'phase': 'prepared', 'source_archive_sha256': rental.digest(source),
               'runtime_manifest_sha256': rental.digest(runtime), 'max_hourly_cost': 5.0}
        rental.save(job)
        accepted = dict(rented=42, machine=70, price=4.0,
                        ssh_config='fixture.config', ssh_host='vast-42')
        with patch.object(rental, 'check_daily_budget'), \
                patch.object(rental, 'validate_environment', return_value='fixture-public-key'), \
                patch.object(rental.vast_backend, 'rent', return_value=(accepted, [accepted])) as vendor, \
                patch.object(rental, 'arm_expiry'), \
                patch.object(rental, 'ssh_connection', return_value=['ssh']), \
                patch.object(rental, 'run'), \
                patch.object(rental, 'execute') as execute, \
                patch.object(rental, 'finalize') as finalize, \
                patch.dict(rental.os.environ, {'VAST_SSH_KEY': 'fixture-private-key'}):
            rental.start(job)
        self.assertEqual(vendor.call_args.args[1]['container_image'], 'fixture-image')
        self.assertEqual(rental.load('fixture')['provider_machine_id'], 70)
        self.assertEqual(rental.load('fixture')['deployment_bytes'],
                         source.stat().st_size + sum((artifacts / name).stat().st_size
                                                     for name in rental.deployment_names(manifest)))
        execute.assert_called_once()
        finalize.assert_called_once()

    def test_reserves_one_checked_rental_across_concurrent_controllers(self):
        artifacts = self.root / 'artifacts'
        artifacts.mkdir()
        manifest = dict(container_image='fixture-image')
        for name, field in (('gpu-env.tar.gz', 'gpu_env_sha256'), ('gpu-assets.tar.gz', 'gpu_assets_sha256')):
            (artifacts / name).write_bytes(b'fixture archive')
            manifest[field] = rental.digest(artifacts / name)
        jobs = []
        for identifier in ('first-request', 'second-request'):
            directory = rental.job_path(identifier).parent
            directory.mkdir()
            (directory / 'source.tar.gz').write_bytes(b'fixture source')
            (directory / 'runtime.json').write_text(json.dumps(manifest))
            job = dict(self.job, id=identifier, phase='prepared', instance_id=None,
                       label='autoresearch-' + identifier, max_hourly_cost=5,
                       source_archive_sha256=rental.digest(directory / 'source.tar.gz'),
                       runtime_manifest_sha256=rental.digest(directory / 'runtime.json'))
            rental.save(job)
            jobs.append(job)
        accepted = dict(rented=42, machine=70, price=4.0,
                        ssh_config='fixture.config', ssh_host='vast-42')
        with patch.object(rental, 'list_instances', return_value=[]), \
                patch.object(rental, 'validate_environment', return_value='fixture-public-key'), \
                patch.object(rental.vast_backend, 'rent', return_value=(accepted, [accepted])) as vendor, \
                patch.object(rental, 'arm_expiry'), \
                patch.object(rental, 'ssh_connection', return_value=['ssh']), \
                patch.object(rental, 'run'), patch.object(rental, 'execute'), patch.object(rental, 'finalize'), \
                patch.dict(rental.os.environ, {'VAST_SSH_KEY':'fixture-private-key'}):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(rental.start, job) for job in jobs]
                outcomes = []
                for future in futures:
                    try:
                        future.result(timeout=10)
                        outcomes.append('created')
                    except RuntimeError as error:
                        self.assertIn('not finished', str(error))
                        outcomes.append('daily limit')
        self.assertCountEqual(outcomes, ['created', 'daily limit'])
        self.assertEqual(vendor.call_count, 1)

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
            if 'nohup bash gpu_run.sh' in text:
                launched.append(text)
            return subprocess.CompletedProcess(arguments, 0, stdout='')

        with patch.object(rental, 'ssh_connection', return_value=['ssh']):
            with patch.object(rental, 'run', side_effect=command):
                with patch.object(rental.time, 'time', side_effect=lambda: now[0]):
                    self.transfer.side_effect = lambda *args, **kwargs: now.__setitem__(0, now[0] + 300)
                    with self.assertRaises(TimeoutError):
                        rental.execute(job, {}, self.root / 'key')
        self.assertEqual(launched, [])

    def observe_one_launch(self, *, encoding_cache=False):
        for name in ('gpu-env.tar.gz', 'gpu-assets.tar.gz'):
            path = self.root / 'artifacts' / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'archive fixture')
        (rental.job_path('fixture').parent / 'source.tar.gz').write_bytes(b'source fixture')
        if encoding_cache:
            manifest = dict(gpu_env_sha256=rental.digest(self.root / 'artifacts/gpu-env.tar.gz'),
                            gpu_assets_sha256=rental.digest(self.root / 'artifacts/gpu-assets.tar.gz'),
                            encoding_cache={'asset_path': rental.encoding_cache_asset_path('h100')})
            (rental.job_path('fixture').parent / 'runtime.json').write_text(json.dumps(manifest))
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
        return launched

    def test_starts_training_once_when_one_observation_times_out(self):
        self.observe_one_launch()

    def test_launches_the_native_encoding_cache_from_the_jobs_frozen_runtime(self):
        launched = self.observe_one_launch(encoding_cache=True)
        self.assertIn('OLMES_ENCODING_CACHE=' + rental.encoding_cache_asset_path('h100'), launched[0])

    def test_deploys_a_verified_xz_runtime_with_its_separate_bootstrap_before_one_training_launch(self):
        self.deploy_xz_runtime('python-xz')

    def test_deploys_the_parallel_bootstrap_decoder_before_one_training_launch(self):
        commands = self.deploy_xz_runtime('python-xz-mt')
        self.assertTrue(any('.bootstrap/runtime_unpack.py extract' in text for text in commands))

    def deploy_xz_runtime(self, transport_format):
        artifacts = self.root / 'artifacts'
        artifacts.mkdir()
        for name in ('gpu-bootstrap.tar.gz', 'gpu-env.tar.xz', 'gpu-assets.tar.gz'):
            (artifacts / name).write_bytes(b'archive fixture')
        snapshot = rental.job_path('fixture').parent
        (snapshot / 'source.tar.gz').write_bytes(b'source fixture')
        runtime_hash = rental.digest(artifacts / 'gpu-env.tar.xz')
        (snapshot / 'runtime.json').write_text(json.dumps(dict(
            python_version='3.12.16', gpu_env_sha256=runtime_hash,
            gpu_assets_sha256=rental.digest(artifacts / 'gpu-assets.tar.gz'),
            transport=dict(format=transport_format, canonical_gpu_env_sha256=runtime_hash,
                           gpu_env_xz_sha256=runtime_hash,
                           bootstrap_sha256=rental.digest(artifacts / 'gpu-bootstrap.tar.gz')))))
        commands, uploads = [], []

        def command(arguments, **kwargs):
            text = arguments[-1]
            commands.append(text)
            if text.startswith('nvidia-smi'):
                return subprocess.CompletedProcess(arguments, 0, stdout='NVIDIA H100 80GB HBM3, 0, 0\n')
            if 'kill -0' in text:
                return subprocess.CompletedProcess(arguments, 0, stdout='exit 0')
            if kwargs.get('stdout') is not None:
                kwargs['stdout'].write(b'fixture artifact')
            return subprocess.CompletedProcess(arguments, 0, stdout='')

        with patch.object(rental, 'ssh_connection', return_value=['ssh']), \
                patch.object(rental, 'run', side_effect=command):
            rental.execute(self.job, {}, self.root / 'key')
        for call in self.transfer.call_args_list:
            uploads.append(Path(call.args[1]).name)
            commands.append(call.args[2])
        self.assertEqual(uploads, ['gpu-bootstrap.tar.gz', 'gpu-env.tar.xz', 'gpu-assets.tar.gz', 'source.tar.gz'])
        self.assertTrue(any('.bootstrap/.python/' in text for text in commands))
        self.assertEqual(sum('nohup bash gpu_run.sh' in text for text in commands), 1)
        self.assertEqual(rental.load('fixture')['phase'], 'evaluated')
        return commands

    def r2_runtime(self):
        artifacts = self.root / 'artifacts'
        artifacts.mkdir()
        (artifacts / 'gpu-bootstrap.tar.gz').write_bytes(b'bootstrap fixture')
        snapshot = rental.job_path('fixture').parent
        (snapshot / 'source.tar.gz').write_bytes(b'source fixture')
        manifest = dict(python_version='3.12.16', gpu_env_sha256='a' * 64, gpu_assets_sha256='b' * 64,
                        transport=dict(format='python-xz-mt', canonical_gpu_env_sha256='a' * 64,
                                       gpu_env_xz_sha256='c' * 64,
                                       bootstrap_sha256=rental.digest(artifacts / 'gpu-bootstrap.tar.gz')))
        (snapshot / 'runtime.json').write_text(json.dumps(manifest))
        return artifacts, manifest

    def rtx_runtime(self, delivery='ssh'):
        config = json.loads(self.config_path.read_text())
        values = config['gpus']['rtx5090']
        if delivery is None:
            values.pop('archive_delivery', None)
        else:
            values['archive_delivery'] = delivery
        self.config_path.write_text(json.dumps(config))
        profile = rental.get_profile('rtx5090')
        artifacts = profile.artifacts_directory(self.root)
        artifacts.mkdir(parents=True)
        for name in ('gpu-bootstrap.tar.gz', 'gpu-env.tar.gz', 'gpu-env.tar.xz', 'gpu-assets.tar.gz'):
            (artifacts / name).write_bytes(name.encode())
        manifest = dict(gpu_id=profile.identifier, python_version='3.12.16',
                        gpu_env_sha256=rental.digest(artifacts / 'gpu-env.tar.gz'),
                        gpu_assets_sha256=rental.digest(artifacts / 'gpu-assets.tar.gz'))
        manifest['transport'] = dict(format='python-xz-mt',
            canonical_gpu_env_sha256=manifest['gpu_env_sha256'],
            gpu_env_xz_sha256=rental.digest(artifacts / 'gpu-env.tar.xz'),
            bootstrap_sha256=rental.digest(artifacts / 'gpu-bootstrap.tar.gz'))
        snapshot = rental.job_path('fixture').parent
        (snapshot / 'runtime.json').write_text(json.dumps(manifest))
        (snapshot / 'source.tar.gz').write_bytes(b'source fixture')
        job = dict(self.job, gpu_id=profile.identifier, gpu_name=profile.gpu_name,
                   gpu_settings=profile.settings(), expires_at=time.time() + 6200,
                   expected_eval_seconds=1800)
        rental.save(job)
        return artifacts, manifest, job

    def test_validates_the_ssh_profiles_local_archives_when_an_object_store_is_configured(self):
        artifacts, manifest, _ = self.rtx_runtime()
        environment = dict(R2_ENDPOINT='https://store.example', R2_BUCKET='runtime',
                           R2_ACCESS_KEY_ID='key', R2_SECRET_ACCESS_KEY='secret')
        with patch.dict(rental.os.environ, environment), \
                patch.object(rental.r2, 'stored', return_value=15):
            rental.verify_archives(manifest, artifacts, 'invalid local runtime')
            (artifacts / 'gpu-env.tar.xz').write_bytes(b'changed local runtime')
            with self.assertRaisesRegex(RuntimeError, 'invalid local runtime'):
                rental.verify_archives(manifest, artifacts, 'invalid local runtime')

    def test_delivers_the_ssh_profiles_archives_over_ssh_with_global_object_store_settings(self):
        _, _, job = self.rtx_runtime()
        commands = []

        def command(arguments, **kwargs):
            text = arguments[-1]
            commands.append(text)
            if text.startswith('nvidia-smi'):
                return subprocess.CompletedProcess(arguments, 0, stdout='NVIDIA GeForce RTX 5090, 0, 0\n')
            if 'kill -0' in text:
                return subprocess.CompletedProcess(arguments, 0, stdout='exit 0')
            if kwargs.get('stdout') is not None:
                kwargs['stdout'].write(b'fixture artifact')
            return subprocess.CompletedProcess(arguments, 0, stdout='')

        environment = dict(R2_ENDPOINT='https://store.example', R2_BUCKET='runtime',
                           R2_ACCESS_KEY_ID='key', R2_SECRET_ACCESS_KEY='secret')
        with patch.dict(rental.os.environ, environment), \
                patch.object(rental.r2, 'stored', return_value=15), \
                patch.object(rental.r2, 'fetch', return_value={'source': 'r2'}), \
                patch.object(rental, 'ssh_connection', return_value=['ssh']), \
                patch.object(rental, 'run', side_effect=command):
            rental.execute(job, {}, self.root / 'key')
        uploads = [Path(call.args[1]).name for call in self.transfer.call_args_list]
        self.assertEqual(uploads, ['gpu-bootstrap.tar.gz', 'gpu-env.tar.xz', 'gpu-assets.tar.gz', 'source.tar.gz'])
        self.assertEqual(sum('nohup bash gpu_run.sh' in text for text in commands), 1)
        self.assertEqual(rental.load('fixture')['phase'], 'evaluated')

    def test_fetches_the_rtx_runtime_from_r2_with_default_delivery(self):
        _, _, job = self.rtx_runtime(delivery=None)
        transfers = []

        def command(arguments, **kwargs):
            text = arguments[-1]
            if text.startswith('nvidia-smi'):
                return subprocess.CompletedProcess(arguments, 0, stdout='NVIDIA GeForce RTX 5090, 0, 0\n')
            if 'kill -0' in text:
                return subprocess.CompletedProcess(arguments, 0, stdout='exit 0')
            if kwargs.get('stdout') is not None:
                kwargs['stdout'].write(b'fixture artifact')
            return subprocess.CompletedProcess(arguments, 0, stdout='')

        def upload(connection, path, *args, **kwargs):
            transfers.append(('ssh', Path(path).name))
            return dict(archive=Path(path).name, source='ssh')

        def fetch(connection, store, name, *args, **kwargs):
            transfers.append(('r2', name))
            return dict(archive=name, source='r2')

        self.transfer.side_effect = upload
        environment = dict(R2_ENDPOINT='https://store.example', R2_BUCKET='runtime',
                           R2_ACCESS_KEY_ID='key', R2_SECRET_ACCESS_KEY='secret')
        with patch.dict(rental.os.environ, environment), \
                patch.object(rental.r2, 'stored', return_value=15), \
                patch.object(rental.r2, 'fetch', side_effect=fetch), \
                patch.object(rental, 'ssh_connection', return_value=['ssh']), \
                patch.object(rental, 'run', side_effect=command):
            rental.execute(job, {}, self.root / 'key')
        self.assertEqual(transfers, [('ssh', 'gpu-bootstrap.tar.gz'), ('r2', 'gpu-env.tar.xz'),
                                    ('r2', 'gpu-assets.tar.gz'), ('ssh', 'source.tar.gz')])
        self.assertEqual(rental.load('fixture')['phase'], 'evaluated')

    def test_fetches_the_large_archives_from_r2_after_installing_its_fetcher(self):
        self.r2_runtime()
        events = []

        def command(arguments, **kwargs):
            text = arguments[-1]
            events.append(text)
            if text.startswith('nvidia-smi'):
                return subprocess.CompletedProcess(arguments, 0, stdout='NVIDIA H100 80GB HBM3, 0, 0\n')
            if 'kill -0' in text:
                return subprocess.CompletedProcess(arguments, 0, stdout='exit 0')
            if kwargs.get('stdout') is not None:
                kwargs['stdout'].write(b'fixture artifact')
            return subprocess.CompletedProcess(arguments, 0, stdout='')

        def fetch(connection, store, name, sha256, size, unpack, interpreter, **kwargs):
            events.append('r2 ' + name)
            self.assertEqual(interpreter, '.bootstrap/.python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12')
            self.assertLessEqual(kwargs['expires'], 3500)
            return dict(archive=name, sha256=sha256, bytes=size, source='r2')

        self.transfer.side_effect = lambda connection, path, *args, **kwargs: events.append('ssh ' + Path(path).name)
        environment = dict(R2_ENDPOINT='https://store.example', R2_BUCKET='runtime',
                           R2_ACCESS_KEY_ID='key', R2_SECRET_ACCESS_KEY='secret')
        with patch.dict(rental.os.environ, environment), \
                patch.object(rental.r2, 'stored', return_value=15), \
                patch.object(rental.r2, 'fetch', side_effect=fetch), \
                patch.object(rental, 'ssh_connection', return_value=['ssh']), \
                patch.object(rental, 'run', side_effect=command):
            rental.execute(self.job, {}, self.root / 'key')
        transfers = [event for event in events if event.startswith(('ssh ', 'r2 ', 'cat > '))]
        self.assertEqual(transfers, ['ssh gpu-bootstrap.tar.gz', 'cat > autoresearch/.bootstrap/r2_fetch.py',
                                     'r2 gpu-env.tar.xz', 'r2 gpu-assets.tar.gz', 'ssh source.tar.gz'])
        self.assertEqual(sum('nohup bash gpu_run.sh' in text for text in events), 1)
        self.assertEqual(rental.load('fixture')['phase'], 'evaluated')

    def test_uploads_only_the_archives_that_r2_lacks_and_keeps_the_bootstrap_local(self):
        artifacts, manifest = self.r2_runtime()
        (artifacts / 'gpu-assets.tar.gz').write_bytes(b'assets fixture')
        environment = dict(R2_ENDPOINT='https://store.example', R2_BUCKET='runtime',
                           R2_ACCESS_KEY_ID='key', R2_SECRET_ACCESS_KEY='secret')
        held = {'gpu-env.tar.gz': 10, 'gpu-env.tar.xz': 5}
        with patch.dict(rental.os.environ, environment), \
                patch.object(rental.r2, 'stored', side_effect=lambda store, name, sha256: held.get(name)), \
                patch.object(rental.r2, 'upload') as upload:
            sizes = rental.verify_archives(manifest, artifacts, 'missing')
            self.assertEqual(sizes, {'gpu-env.tar.gz': 10, 'gpu-env.tar.xz': 5,
                                    'gpu-bootstrap.tar.gz': (artifacts / 'gpu-bootstrap.tar.gz').stat().st_size,
                                    'gpu-assets.tar.gz': 14})
            self.assertEqual([call.args[2:] for call in upload.call_args_list], [('b' * 64, 'gpu-assets.tar.gz')])
            (artifacts / 'gpu-assets.tar.gz').unlink()
            with self.assertRaisesRegex(RuntimeError, 'missing'):
                rental.verify_archives(manifest, artifacts, 'missing')
            (artifacts / 'gpu-bootstrap.tar.gz').write_bytes(b'changed bootstrap')
            held['gpu-assets.tar.gz'] = 14
            with self.assertRaisesRegex(RuntimeError, 'missing'):
                rental.verify_archives(manifest, artifacts, 'missing')

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

    def test_records_gpu_clocks_power_limit_and_temperature_with_utilization(self):
        remote = Mock(return_value=subprocess.CompletedProcess([], 0,
            stdout='NVIDIA H100 80GB HBM3, 45060, 96, 1980, 1980, 2619, 420.5, 700, 61\n'))
        rental.record_gpu_observation(remote, self.root)
        sample = json.loads((self.root / 'gpu-observations.jsonl').read_text())
        expected = dict(memory_mib=45060, utilization_percent=96, sm_clock_mhz=1980,
                        max_sm_clock_mhz=1980, memory_clock_mhz=2619,
                        power_draw_w=420.5, power_limit_w=700, temperature_c=61)
        for field, value in expected.items():
            self.assertEqual(sample[field], value)

    def test_preserves_basic_measurements_when_optional_sensors_are_unavailable(self):
        remote = Mock(return_value=subprocess.CompletedProcess([], 0,
            stdout='NVIDIA H100 80GB HBM3, 45060, 96, N/A, N/A, nan, inf, -1, N/A\n'))
        rental.record_gpu_observation(remote, self.root)
        sample = json.loads((self.root / 'gpu-observations.jsonl').read_text())
        self.assertEqual((sample['memory_mib'], sample['utilization_percent']), (45060, 96))
        for field in ('sm_clock_mhz', 'max_sm_clock_mhz', 'memory_clock_mhz',
                      'power_draw_w', 'power_limit_w', 'temperature_c'):
            self.assertIsNone(sample[field])

    def test_recovers_basic_telemetry_within_the_same_sample_budget_after_an_unsupported_query(self):
        remote = Mock(side_effect=[subprocess.CalledProcessError(1, 'nvidia-smi'),
                                  subprocess.CompletedProcess([], 0, stdout='NVIDIA H100 80GB HBM3, 45060, 96\n')])
        with patch.object(rental.time, 'monotonic', side_effect=[100, 102]):
            rental.record_gpu_observation(remote, self.root)
        sample = json.loads((self.root / 'gpu-observations.jsonl').read_text())
        self.assertEqual(sample['extended_error_type'], 'CalledProcessError')
        self.assertEqual((sample['memory_mib'], sample['utilization_percent']), (45060, 96))
        self.assertIsNone(sample['power_draw_w'])
        self.assertEqual(remote.call_args.kwargs['timeout'], 3)

    def test_keeps_a_sample_timeout_when_its_fallback_budget_is_exhausted(self):
        remote = Mock(side_effect=subprocess.TimeoutExpired('nvidia-smi', 5))
        with patch.object(rental.time, 'monotonic', side_effect=[100, 105]):
            rental.record_gpu_observation(remote, self.root)
        sample = json.loads((self.root / 'gpu-observations.jsonl').read_text())
        self.assertEqual(sample['error_type'], 'TimeoutExpired')
        remote.assert_called_once()


if __name__ == '__main__':
    unittest.main()
