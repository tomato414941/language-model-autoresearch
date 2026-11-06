from concurrent.futures import ThreadPoolExecutor
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import tarfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from tools import rental
from gpu_config import read_config


def rented(instance, label, at='1970-01-01T00:16:40+00:00'):
    return dict(id=f'rented-{instance}', event='rented', instance=instance, label=label, at=at)


def returned(instance, dollars):
    return dict(id=f'returned-{instance}', event='returned', instance=instance, dollars=dollars)


class RentalLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.hold_patch = patch.dict(rental.os.environ, {
            'AUTORESEARCH_GPU': 'h100', 'AUTORESEARCH_RENTAL_HOLD_H100': '0',
            'AUTORESEARCH_RENTAL_HOLD_RTX5090': '0', **{name: '' for name in rental.r2.VARIABLES}})
        self.hold_patch.start()
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        config = read_config()
        config.update(active_gpu='h100')
        self.config_path = self.root / 'fixture-gpus.json'
        self.config_path.write_text(json.dumps(config))
        self.config_patch = patch('gpu_config.CONFIG_PATH', self.config_path)
        self.config_patch.start()
        self.root_patch = patch.object(rental, 'ROOT', self.root)
        self.root_patch.start()
        for default in (patch.object(rental.vast_backend, 'entries', return_value=[]),
                        patch.object(rental.vast_backend, 'command',
                                     side_effect=AssertionError('Use a fixture for vast-rental commands.')),
                        patch.object(rental.vast_backend, 'rent',
                                     side_effect=AssertionError('Use a fixture for vendor rentals.'))):
            default.start()
            self.addCleanup(default.stop)
        self.transfer_patch = patch.object(rental, 'stream_archive', return_value={'fixture': 'transfer'})
        self.transfer = self.transfer_patch.start()
        self.collection_cases = 0
        self.job = dict(id='fixture', label='autoresearch-fixture', instance_id=42,
                        phase='running', expires_at=time.time() + 3500,
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

    def prepared(self, identifier='fixture'):
        """A prepared job whose snapshot and runtime are on disk."""
        directory = rental.job_path(identifier).parent
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = self.root / 'artifacts'
        artifacts.mkdir(exist_ok=True)
        manifest = dict(container_image='fixture-image')
        for name, field in (('gpu-env.tar.gz', 'gpu_env_sha256'), ('gpu-assets.tar.gz', 'gpu_assets_sha256')):
            (artifacts / name).write_bytes(b'fixture archive')
            manifest[field] = rental.digest(artifacts / name)
        (directory / 'source.tar.gz').write_bytes(b'fixture source ' + identifier.encode())
        (directory / 'runtime.json').write_text(json.dumps(manifest))
        job = dict(self.job, id=identifier, label='autoresearch-' + identifier, phase='prepared', instance_id=None,
                   max_hourly_cost=5, source_archive_sha256=rental.digest(directory / 'source.tar.gz'),
                   runtime_manifest_sha256=rental.digest(directory / 'runtime.json'))
        rental.save(job)
        return job

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

    def test_keeps_a_hold_on_one_gpu_from_blocking_the_other_gpus_preflight(self):
        job = {**self.job, 'phase': 'prepared'}
        with patch.dict(rental.os.environ, {'AUTORESEARCH_RENTAL_HOLD_RTX5090': '1'}):
            with patch.object(rental, 'digest', side_effect=RuntimeError('reached H100 preflight')):
                with self.assertRaisesRegex(RuntimeError, 'reached H100 preflight'):
                    rental.start(job)

    def test_checks_the_gpu_identity_before_starting_an_existing_job(self):
        with patch.object(rental.sys, 'argv', ['rental.py', '--gpu', 'rtx5090', 'start', 'fixture']):
            with patch.object(rental, 'start') as start, patch.object(rental.sys, 'stderr'):
                with self.assertRaises(SystemExit):
                    rental.main()
        start.assert_not_called()

    def test_honors_the_users_rental_hold_before_requesting_a_machine(self):
        job = self.prepared()
        with patch.dict(rental.os.environ, {'AUTORESEARCH_RENTAL_HOLD_H100': '1'}), \
                patch.object(rental, 'run') as launch:
            with self.assertRaisesRegex(RuntimeError, 'on hold by user request'):
                rental.start(job)
            with self.assertRaisesRegex(RuntimeError, 'on hold by user request'):
                rental.run_job(job)
        launch.assert_not_called()
        self.assertEqual(rental.load('fixture')['phase'], 'prepared')

    def test_rents_only_the_gpu_selected_in_gpus_json(self):
        job = dict(self.prepared(), gpu_id='rtx5090', gpu_name='RTX 5090')
        with self.assertRaisesRegex(RuntimeError, 'selected in gpus.json'):
            rental.check_ready(job)

    def test_starts_the_named_gpu_with_its_own_environment(self):
        for identifier in ('h100', 'rtx5090'):
            job = dict(self.job, gpu_id=identifier)
            with self.subTest(gpu=identifier):
                with patch.object(rental.sys, 'argv', ['rental.py', '--gpu', identifier, 'start', 'fixture']):
                    with patch.object(rental, 'load', return_value=job):
                        with patch.object(rental, 'start') as start:
                            rental.main()
                            start.assert_called_once_with(job)
                            self.assertEqual(rental.os.environ['AUTORESEARCH_GPU'], identifier)

    def test_runs_a_checked_job_in_the_background_with_its_environment_and_log(self):
        job = self.prepared()
        with patch.object(rental, 'validate_environment'), patch.object(rental, 'run') as launch, \
                patch('builtins.print') as output:
            rental.start(job)
        command = launch.call_args.args[0]
        self.assertEqual(command[:3], ['systemd-run', '--user', '--unit=autoresearch-run-fixture'])
        self.assertIn('--property=EnvironmentFile=' + str(self.root / '.env'), command)
        self.assertIn('--property=StandardOutput=append:' + str(self.root / 'logs/fixture.log'), command)
        self.assertEqual(command[-4:], [rental.sys.executable, str(self.root / 'tools/rental.py'), 'run', 'fixture'])
        self.assertEqual(json.loads(output.call_args.args[0])['log'], 'logs/fixture.log')
        self.assertEqual(rental.load('fixture')['phase'], 'prepared')

    def test_starts_nothing_once_the_days_failed_attempts_cost_more_than_the_limit(self):
        job = self.prepared()
        entries = [rented(41, 'autoresearch-earlier'), returned(41, 0.51)]
        with patch.object(rental, 'validate_environment'), patch.object(rental, 'run') as launch, \
                patch.object(rental.vast_backend, 'entries', return_value=entries), \
                patch.object(rental.time, 'time', return_value=2000):
            with self.assertRaisesRegex(RuntimeError, 'more than'):
                rental.start(job)
        launch.assert_not_called()

    def test_refuses_to_start_a_job_twice(self):
        with self.assertRaisesRegex(RuntimeError, 'already been started'):
            rental.check_ready(dict(self.prepared(), phase='running'))

    def test_keeps_the_completed_run_when_its_return_is_recorded(self):
        rental.save(dict(id='fixture', phase='evaluated', exit_code=0))
        rental.save(dict(id='fixture', deleted_at=123))
        stored = rental.load('fixture')
        self.assertEqual(stored['phase'], 'evaluated')
        self.assertEqual(stored['exit_code'], 0)
        self.assertEqual(stored['deleted_at'], 123)

    def test_notes_the_machine_then_returns_it_then_records_the_result(self):
        for phase in ('evaluated', 'failed'):
            with self.subTest(phase=phase):
                rental.save(dict(id='fixture', phase=phase, instance_id=42))
                events = []

                def command(arguments, **kwargs):
                    events.append(arguments[0])

                def give_back(job):
                    rental.save(dict(id=job['id'], deleted_at=1100))
                    events.append('returned')

                def record(command):
                    self.assertEqual(rental.load('fixture')['deleted_at'], 1100)
                    self.assertEqual(command[1:], [str(self.root / 'tools/record_run.py'), 'fixture'])
                    events.append('recorded')

                with patch.object(rental.vast_backend, 'command', side_effect=command), \
                        patch.object(rental, 'return_machines', side_effect=give_back), \
                        patch.object(rental, 'run', side_effect=record):
                    rental.finalize(rental.load('fixture'))
                self.assertEqual(events, ['note', 'returned', 'recorded'])

    def test_returns_the_machine_even_when_its_note_fails(self):
        with patch.object(rental.vast_backend, 'command', side_effect=subprocess.TimeoutExpired('note', 30)), \
                patch.object(rental, 'return_machines') as give_back, patch.object(rental, 'run'):
            rental.finalize(dict(self.job, phase='evaluated'))
        give_back.assert_called_once()

    def test_reloads_the_collected_phase_before_recording(self):
        rental.save(dict(id='fixture', phase='running'))

        def give_back(job):
            rental.save(dict(id=job['id'], phase='evaluated', exit_code=0, deleted_at=1100))

        with patch.object(rental.vast_backend, 'command'), \
                patch.object(rental, 'return_machines', side_effect=give_back), \
                patch.object(rental, 'run') as record:
            rental.finalize(self.job)
        record.assert_called_once()
        self.assertEqual(rental.load('fixture')['phase'], 'evaluated')

    def test_records_nothing_while_a_machine_has_not_been_returned(self):
        rental.save(dict(id='fixture', phase='evaluated'))
        with patch.object(rental.vast_backend, 'command'), \
                patch.object(rental, 'return_machines', side_effect=RuntimeError('not returned')), \
                patch.object(rental, 'run') as record:
            with self.assertRaisesRegex(RuntimeError, 'not returned'):
                rental.finalize(self.job)
        record.assert_not_called()

    def test_records_only_finished_runs_that_had_a_machine(self):
        for fields in (dict(phase='running', instance_id=42), dict(phase='failed', instance_id=None)):
            with self.subTest(fields=fields):
                rental.save(dict(id='fixture', **fields))
                with patch.object(rental.vast_backend, 'command'), patch.object(rental, 'return_machines'), \
                        patch.object(rental, 'run') as record:
                    rental.finalize(rental.load('fixture'))
                record.assert_not_called()

    def test_keeps_collected_results_available_when_decision_export_fails(self):
        rental.save(dict(id='fixture', phase='evaluated', exit_code=0))

        def give_back(job):
            rental.save(dict(id=job['id'], deleted_at=1100))

        with patch.object(rental.vast_backend, 'command'), \
                patch.object(rental, 'return_machines', side_effect=give_back), \
                patch.object(rental, 'run', side_effect=subprocess.CalledProcessError(1, 'record')):
            with self.assertRaises(subprocess.CalledProcessError):
                rental.finalize(self.job)
        stored = rental.load('fixture')
        self.assertEqual((stored['phase'], stored['exit_code'], stored['deleted_at']), ('evaluated', 0, 1100))

    def test_returns_every_machine_rented_under_the_jobs_label_through_vast_rental(self):
        ledger = [rented(41, self.job['label']), returned(41, 0.02), rented(42, self.job['label']),
                  rented(43, 'another-project')]

        def command(arguments, **kwargs):
            ledger.append(returned(int(arguments[1]), 0.5))

        with patch.object(rental.vast_backend, 'entries', side_effect=lambda: list(ledger)), \
                patch.object(rental.vast_backend, 'command', side_effect=command) as give_back:
            self.assertEqual(rental.return_machines(self.job), [41, 42])
        give_back.assert_called_once_with(['return', '42'], timeout=120)
        self.assertIn('deleted_at', rental.load('fixture'))

    def test_keeps_the_return_pending_while_the_ledger_lacks_it(self):
        with patch.object(rental.vast_backend, 'entries', return_value=[rented(42, self.job['label'])]), \
                patch.object(rental.vast_backend, 'command'):
            with self.assertRaisesRegex(RuntimeError, 'returned'):
                rental.return_machines(self.job)
        self.assertNotIn('deleted_at', rental.load('fixture'))

    def test_connects_through_the_checked_route_with_the_callers_key_file(self):
        job = dict(id='fixture', ssh_config='fixture.config', ssh_host='vast-42')
        command = rental.ssh_command(job, Path('fixture-key'))
        self.assertEqual(command, ['ssh', '-F', 'fixture.config', '-i', 'fixture-key',
                                   '-o', 'IdentitiesOnly=yes', '-o', 'IdentityAgent=none',
                                   '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'vast-42'])

    def test_rents_nothing_while_a_machine_of_this_loop_is_out(self):
        for entries in ([rented(42, 'autoresearch-h100-earlier')],
                        [rented(42, 'autoresearch-rtx5090-earlier', at='1969-12-30T00:00:00+00:00')]):
            with self.subTest(entries=entries):
                with patch.object(rental.vast_backend, 'entries', return_value=entries):
                    with self.assertRaisesRegex(RuntimeError, 'not been returned'):
                        rental.failed_cost_today(2000)

    def test_applies_the_daily_failed_cost_limit_to_the_ledgers_amounts(self):
        for dollars, allowed in ((0.06, True), (0.5, True), (0.501, False)):
            with self.subTest(dollars=dollars):
                entries = [rented(41, 'autoresearch-h100-first'), returned(41, dollars / 2),
                           rented(42, 'autoresearch-rtx5090-second'), returned(42, dollars / 2)]
                with patch.object(rental.vast_backend, 'entries', return_value=entries):
                    if allowed:
                        self.assertEqual(rental.failed_cost_today(2000),
                                         rental.Decimal(str(dollars / 2)) * 2)
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'more than'):
                            rental.failed_cost_today(2000)

    def test_allows_one_measured_run_a_day(self):
        rental.save(dict(id='fixture', label='autoresearch-fixture', phase='evaluated', exit_code=0))
        entries = [rented(42, 'autoresearch-fixture'), returned(42, 1.0)]
        with patch.object(rental.vast_backend, 'entries', return_value=entries):
            with self.assertRaisesRegex(RuntimeError, 'measured a run today'):
                rental.failed_cost_today(2000)

    def test_counts_a_run_that_its_record_calls_a_crash_as_a_failed_attempt(self):
        rental.save(dict(id='fixture', label='autoresearch-fixture', phase='evaluated', exit_code=0))
        decisions = self.root / 'results/decisions.json'
        decisions.parent.mkdir(parents=True)
        decisions.write_text(json.dumps([dict(id='fixture', status='crash')]))
        entries = [rented(42, 'autoresearch-fixture'), returned(42, 0.2)]
        with patch.object(rental.vast_backend, 'entries', return_value=entries):
            self.assertEqual(rental.failed_cost_today(2000), rental.Decimal('0.2'))

    def test_allows_the_next_utc_calendar_day_without_waiting_twenty_four_hours(self):
        rental.save(dict(id='fixture', label='autoresearch-fixture', phase='evaluated', exit_code=0))
        entries = [rented(42, 'autoresearch-fixture', at='1970-01-01T23:58:20+00:00'), returned(42, 1.0)]
        with patch.object(rental.vast_backend, 'entries', return_value=entries):
            self.assertEqual(rental.failed_cost_today(86401), 0)

    def test_ignores_the_cost_and_machines_of_other_projects(self):
        entries = [rented(99, 'another-project'), rented(98, 'another-project'), returned(98, 10)]
        with patch.object(rental.vast_backend, 'entries', return_value=entries):
            self.assertEqual(rental.failed_cost_today(2000), 0)

    def test_rejects_a_returned_loop_machine_without_a_valid_cost(self):
        for dollars in (None, -1, float('nan'), True):
            with self.subTest(dollars=dollars):
                entries = [rented(42, 'autoresearch-fixture'), returned(42, dollars)]
                with patch.object(rental.vast_backend, 'entries', return_value=entries):
                    with self.assertRaisesRegex(RuntimeError, 'no valid cost'):
                        rental.failed_cost_today(2000)

    def run_prepared(self, job, rent, execute=None):
        """Run a prepared job with fixtures for vast-rental, the ledger allowance, the machine and finalize."""
        with patch.object(rental, 'failed_cost_today', return_value=rental.Decimal('0')), \
                patch.object(rental, 'validate_environment', return_value='fixture-public-key'), \
                patch.object(rental.vast_backend, 'rent', side_effect=rent) as vendor, \
                patch.object(rental, 'execute', side_effect=execute) as executed, \
                patch.object(rental, 'finalize') as finalize, \
                patch.dict(rental.os.environ, {'VAST_SSH_KEY': 'fixture-private-key'}):
            try:
                rental.run_job(job)
            finally:
                self.vendor, self.executed, self.finalized = vendor, executed, finalize

    def test_retries_vendor_rejected_machines_before_starting_the_single_run(self):
        job = self.prepared()
        rejected = dict(rented=None, offers=2, failed=1)
        accepted = dict(rented=42, machine=70, price=0.5, ssh_config='fixture.config', ssh_host='vast-42')
        attempts = [dict(unusable=41, machine=71, stage='check'), rejected]
        self.run_prepared(job, [(rejected, attempts), (accepted, [accepted])])
        stored = rental.load('fixture')
        self.assertEqual(self.vendor.call_count, 2)
        self.assertEqual(stored['selection_attempts'][0]['unusable'], 41)
        self.assertEqual((stored['instance_id'], stored['provider_machine_id']), (42, 70))
        self.executed.assert_called_once()
        self.assertEqual(self.executed.call_args.args[1][-1], 'vast-42')
        self.finalized.assert_called_once()

    def test_keeps_the_job_startable_when_no_machine_ran_it(self):
        job = self.prepared()
        rejected = dict(rented=None, offers=1, failed=1)
        with self.assertRaisesRegex(RuntimeError, 'no available eligible machine'):
            self.run_prepared(job, [(rejected, [dict(unusable=41)]), (dict(rented=None, offers=0, failed=0), [])])
        self.executed.assert_not_called()
        self.finalized.assert_called_once()
        stored = rental.load('fixture')
        self.assertEqual(stored['phase'], 'prepared')
        self.assertEqual(stored['selection_attempts'], [dict(unusable=41)])
        accepted = dict(rented=42, machine=70, price=0.5, ssh_config='fixture.config', ssh_host='vast-42')
        self.run_prepared(stored, [(accepted, [accepted])])
        self.executed.assert_called_once()
        self.assertEqual(rental.load('fixture')['instance_id'], 42)

    def test_rents_the_prepared_snapshot_while_the_working_sources_advance(self):
        job = self.prepared()
        directory = rental.job_path('fixture').parent
        (self.root / 'artifacts/runtime.json').write_text('{"container_image":"later-image"}')
        for name in rental.source_files():
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text('later ' + name)
        accepted = dict(rented=42, machine=70, price=4.0, ssh_config='fixture.config', ssh_host='vast-42')
        self.run_prepared(job, [(accepted, [accepted])])
        self.assertEqual(self.vendor.call_args.args[1]['container_image'], 'fixture-image')
        manifest = json.loads((directory / 'runtime.json').read_text())
        self.assertEqual(rental.load('fixture')['deployment_bytes'],
                         (directory / 'source.tar.gz').stat().st_size
                         + sum((self.root / 'artifacts' / name).stat().st_size
                               for name in rental.deployment_names(manifest)))
        self.executed.assert_called_once()

    def test_preserves_the_full_transfer_failure_before_returning_the_machine(self):
        job = self.prepared()
        accepted = dict(rented=42, machine=70, price=0.5, ssh_config='fixture.config', ssh_host='vast-42')
        error = subprocess.CalledProcessError(255, 'SSH archive transfer',
                                             output=b'fixture output\n', stderr=b'fixture connection closed\n')
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_prepared(job, [(accepted, [accepted])], execute=error)
        details = (rental.job_path('fixture').parent / 'failure.log').read_text()
        self.assertIn('fixture output', details)
        self.assertIn('fixture connection closed', details)
        self.assertEqual(rental.load('fixture')['phase'], 'failed')
        self.finalized.assert_called_once()

    def test_returns_the_machine_when_the_service_is_stopped(self):
        job = self.prepared()
        accepted = dict(rented=42, machine=70, price=0.5, ssh_config='fixture.config', ssh_host='vast-42')
        previous = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, previous)

        def stopped(job, connection):
            rental.stop(signal.SIGTERM, None)

        with self.assertRaises(SystemExit):
            self.run_prepared(job, [(accepted, [accepted])], execute=stopped)
        self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_IGN)
        self.assertEqual(rental.load('fixture')['phase'], 'failed')
        self.finalized.assert_called_once()

    def test_runs_one_job_of_this_loop_at_a_time(self):
        first, second = self.prepared('first-request'), self.prepared('second-request')
        accepted = dict(rented=42, machine=70, price=4.0, ssh_config='fixture.config', ssh_host='vast-42')
        entered, release = threading.Event(), threading.Event()

        def busy(job, connection):
            entered.set()
            release.wait(10)

        with patch.object(rental, 'failed_cost_today', return_value=rental.Decimal('0')), \
                patch.object(rental, 'validate_environment', return_value='fixture-public-key'), \
                patch.object(rental.vast_backend, 'rent', return_value=(accepted, [accepted])) as vendor, \
                patch.object(rental, 'execute', side_effect=busy), patch.object(rental, 'finalize'), \
                patch.dict(rental.os.environ, {'VAST_SSH_KEY': 'fixture-private-key'}):
            with ThreadPoolExecutor(max_workers=1) as pool:
                running = pool.submit(rental.run_job, first)
                self.assertTrue(entered.wait(10))
                with self.assertRaisesRegex(RuntimeError, 'in progress'):
                    rental.run_job(second)
                with self.assertRaisesRegex(RuntimeError, 'in progress'):
                    rental.finish(second)
                release.set()
                running.result(timeout=10)
        self.assertEqual(vendor.call_count, 1)
        self.assertEqual(rental.load('second-request')['phase'], 'prepared')

    def test_finishes_a_run_that_ended_early_as_a_failure(self):
        rental.save(dict(id='fixture', phase='running'))
        with patch.object(rental, 'finalize') as finalize:
            rental.finish(rental.load('fixture'))
        stored = rental.load('fixture')
        self.assertEqual((stored['phase'], stored['error_type']), ('failed', 'RunEnded'))
        finalize.assert_called_once()
        self.assertEqual(finalize.call_args.args[0]['phase'], 'failed')

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

                with patch.object(rental, 'run', side_effect=command):
                    with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                        rental.execute(job, ['ssh'])
                self.assertEqual(rental.load('fixture')['failure_kind'], 'machine')

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

        def transfer(*args, **kwargs):
            now[0] += 300
            return dict(fixture='transfer')

        with patch.object(rental, 'run', side_effect=command):
            with patch.object(rental.time, 'time', side_effect=lambda: now[0]):
                self.transfer.side_effect = transfer
                with self.assertRaises(TimeoutError):
                    rental.execute(job, ['ssh'])
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

        with patch.object(rental, 'run', side_effect=command), patch.object(rental.time, 'sleep'):
            rental.execute(self.job, ['ssh'])
        self.assertEqual(len(launched), 1)
        self.assertEqual(rental.load('fixture')['exit_code'], 0)
        sample = json.loads((self.root / 'results/runs/fixture/gpu-observations.jsonl').read_text())
        self.assertEqual(sample['error_type'], 'TimeoutExpired')
        return launched

    def test_starts_training_once_when_one_observation_times_out(self):
        self.observe_one_launch()

    def test_completes_on_a_reachable_machine_after_a_verified_transfers_cleanup_fails(self):
        self.transfer.return_value = dict(cleanup_error=dict(returncode=255, stderr='fixture closed'))
        self.observe_one_launch()
        self.assertEqual(rental.load('fixture')['phase'], 'evaluated')

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

        with patch.object(rental, 'run', side_effect=command):
            rental.execute(self.job, ['ssh'])
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
                patch.object(rental, 'run', side_effect=command):
            rental.execute(job, ['ssh'])
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
                patch.object(rental, 'run', side_effect=command):
            rental.execute(job, ['ssh'])
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

        def transfer(connection, path, *args, **kwargs):
            events.append('ssh ' + Path(path).name)
            return dict(fixture='transfer')

        self.transfer.side_effect = transfer
        environment = dict(R2_ENDPOINT='https://store.example', R2_BUCKET='runtime',
                           R2_ACCESS_KEY_ID='key', R2_SECRET_ACCESS_KEY='secret')
        with patch.dict(rental.os.environ, environment), \
                patch.object(rental.r2, 'stored', return_value=15), \
                patch.object(rental.r2, 'fetch', side_effect=fetch), \
                patch.object(rental, 'run', side_effect=command):
            rental.execute(self.job, ['ssh'])
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

        with patch.object(rental, 'run', side_effect=command):
            with patch.object(rental.time, 'time', side_effect=lambda: now[0]):
                with patch.object(rental.time, 'sleep', side_effect=sleep):
                    rental.execute(job, ['ssh'])
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
