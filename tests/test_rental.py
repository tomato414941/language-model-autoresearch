from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import gpu_config
from tools import delivery, jobs, rental

R2 = dict(R2_ENDPOINT='https://store.example', R2_BUCKET='runtime', R2_ACCESS_KEY_ID='key',
          R2_SECRET_ACCESS_KEY='secret')
MACHINE = dict(rented=42, machine=70, price=0.5, ssh_config='fixture.config', ssh_host='vast-42',
               deadline='2026-11-07T06:00:00+00:00')


def rented(instance, label, at='1970-01-01T00:16:40+00:00'):
    return dict(id=f'rented-{instance}', event='rented', instance=instance, label=label, at=at)


def returned(instance, dollars):
    return dict(id=f'returned-{instance}', event='returned', instance=instance, dollars=dollars)


def completed(text=''):
    return subprocess.CompletedProcess([], 0, stdout=text)


class RentalTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        config = gpu_config.read_config()
        config['active_gpu'] = 'h100'
        (self.root / 'gpus.json').write_text(json.dumps(config))
        environment = {'AUTORESEARCH_RENTAL_HOLD_H100': '0', 'AUTORESEARCH_RENTAL_HOLD_RTX5090': '0',
                       'VAST_SSH_KEY': 'fixture-private-key', **R2}
        patchers = [patch.object(jobs, 'ROOT', self.root),
                    patch.object(gpu_config, 'CONFIG_PATH', self.root / 'gpus.json'),
                    patch.dict(os.environ, environment),
                    patch.object(rental.vast_backend, 'entries', return_value=[]),
                    patch.object(rental.shutil, 'disk_usage', return_value=Mock(free=100 * 10 ** 9)),
                    patch.object(rental.subprocess, 'check_output', return_value=b'ssh-ed25519 AAAA fixture')]
        for name in ('rent', 'give_back', 'note', 'reject'):
            patchers.append(patch.object(rental.vast_backend, name,
                                         side_effect=AssertionError(f'Use a fixture for vast-rental {name}.')))
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop('AUTORESEARCH_GPU', None)

    def prepared(self, identifier='fixture', **changes):
        """A prepared job of the H100 track, with its snapshot and runtime on disk."""
        profile = gpu_config.get_profile('h100')
        artifacts = profile.artifacts_directory(self.root)
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / 'gpu-bootstrap.tar.gz').write_bytes(b'bootstrap fixture')
        directory = jobs.job_path(identifier).parent
        directory.mkdir(parents=True, exist_ok=True)
        manifest = dict(container_image='fixture-image', python_version='3.12.16', gpu_env_sha256='a' * 64,
                        gpu_assets_sha256='b' * 64,
                        transport=dict(format='python-xz-mt', canonical_gpu_env_sha256='a' * 64,
                                       gpu_env_xz_sha256='c' * 64,
                                       bootstrap_sha256=jobs.digest(artifacts / 'gpu-bootstrap.tar.gz')))
        (directory / 'source.tar.gz').write_bytes(b'fixture source ' + identifier.encode())
        (directory / 'runtime.json').write_text(json.dumps(manifest))
        job = dict(id=identifier, label='autoresearch-h100-' + identifier, gpu_id='h100', gpu_name=profile.gpu_name,
                   gpu_settings=profile.settings(), phase='prepared', expected_eval_seconds=900,
                   metric='olmes_bpb', max_hourly_cost=5,
                   source_archive_sha256=jobs.digest(directory / 'source.tar.gz'),
                   runtime_manifest_sha256=jobs.digest(directory / 'runtime.json'))
        job.update(changes)
        jobs.save(job)
        return job

    def running(self, deadline=5000, **changes):
        """A prepared job that a machine holds until the deadline."""
        return self.prepared(**dict(dict(phase='starting', instance_id=42, deadline=deadline,
                                         ssh_config='fixture.config', ssh_host='vast-42'), **changes))

    # Today's allowance

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
                        self.assertEqual(rental.failed_cost_today(2000), Decimal(str(dollars / 2)) * 2)
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'more than'):
                            rental.failed_cost_today(2000)

    def test_allows_one_measured_run_a_day(self):
        job = self.prepared(phase='evaluated', exit_code=0)
        with patch.object(rental.vast_backend, 'entries', return_value=[rented(42, job['label']), returned(42, 1.0)]):
            with self.assertRaisesRegex(RuntimeError, 'measured a run today'):
                rental.failed_cost_today(2000)

    def test_counts_a_run_that_its_record_calls_a_crash_as_a_failed_attempt(self):
        job = self.prepared(phase='evaluated', exit_code=0)
        ledger = self.root / 'gpus/h100/results/decisions.json'
        ledger.parent.mkdir(parents=True)
        ledger.write_text(json.dumps([dict(id=job['id'], status='crash')]))
        with patch.object(rental.vast_backend, 'entries', return_value=[rented(42, job['label']), returned(42, 0.2)]):
            self.assertEqual(rental.failed_cost_today(2000), Decimal('0.2'))

    def test_allows_the_next_utc_calendar_day_without_waiting_twenty_four_hours(self):
        job = self.prepared(phase='evaluated', exit_code=0)
        entries = [rented(42, job['label'], at='1970-01-01T23:58:20+00:00'), returned(42, 1.0)]
        with patch.object(rental.vast_backend, 'entries', return_value=entries):
            self.assertEqual(rental.failed_cost_today(86401), 0)

    def test_ignores_the_cost_and_machines_of_other_projects(self):
        entries = [rented(99, 'another-project'), rented(98, 'another-project'), returned(98, 10)]
        with patch.object(rental.vast_backend, 'entries', return_value=entries):
            self.assertEqual(rental.failed_cost_today(2000), 0)

    def test_rejects_a_returned_loop_machine_without_a_valid_cost(self):
        for dollars in (None, -1, float('nan'), True):
            with self.subTest(dollars=dollars):
                entries = [rented(42, 'autoresearch-h100-fixture'), returned(42, dollars)]
                with patch.object(rental.vast_backend, 'entries', return_value=entries):
                    with self.assertRaisesRegex(RuntimeError, 'no valid cost'):
                        rental.failed_cost_today(2000)

    # Checks before renting

    def test_honors_the_users_rental_hold_before_requesting_a_machine(self):
        job = self.prepared()
        with patch.dict(os.environ, {'AUTORESEARCH_RENTAL_HOLD_H100': '1'}), patch.object(rental, 'run') as launch:
            with self.assertRaisesRegex(RuntimeError, 'on hold by user request'):
                rental.start(job)
            with self.assertRaisesRegex(RuntimeError, 'on hold by user request'):
                rental.run_job(job)
        launch.assert_not_called()
        self.assertEqual(jobs.load('fixture')['phase'], 'prepared')

    def test_keeps_a_hold_on_one_gpu_from_holding_the_other(self):
        with patch.dict(os.environ, {'AUTORESEARCH_RENTAL_HOLD_RTX5090': '1'}):
            self.assertEqual(rental.check_ready(self.prepared()), 0)

    def test_rents_only_the_gpu_selected_in_gpus_json(self):
        profile = gpu_config.get_profile('rtx5090')
        job = self.prepared(gpu_id='rtx5090', gpu_name=profile.gpu_name, gpu_settings=profile.settings())
        with self.assertRaisesRegex(RuntimeError, 'selected in gpus.json'):
            rental.check_ready(job)

    def test_refuses_to_start_a_job_twice(self):
        with self.assertRaisesRegex(RuntimeError, 'already been started'):
            rental.check_ready(self.prepared(phase='running'))

    def test_refuses_a_job_whose_snapshot_changed(self):
        job = self.prepared()
        (jobs.job_path('fixture').parent / 'source.tar.gz').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'sources changed'):
            rental.check_ready(job)

    def test_requires_r2_since_large_files_never_travel_over_ssh(self):
        with patch.dict(os.environ, {name: '' for name in R2}):
            with self.assertRaisesRegex(RuntimeError, 'R2'):
                rental.check_ready(self.prepared())

    def test_requires_room_for_the_results_beyond_the_space_kept_free(self):
        with patch.object(rental.shutil, 'disk_usage', return_value=Mock(free=6 * 10 ** 9)):
            with self.assertRaisesRegex(RuntimeError, 'GB free'):
                rental.check_ready(self.prepared())

    def test_requires_the_run_to_fit_inside_its_rental(self):
        with self.assertRaisesRegex(ValueError, 'does not fit'):
            rental.check_ready(self.prepared(expected_eval_seconds=1500))

    # Starting

    def test_runs_a_checked_job_in_the_background_with_its_environment_and_log(self):
        job = self.prepared()
        with patch.object(rental, 'run') as launch, patch('builtins.print') as output:
            rental.start(job)
        command = launch.call_args.args[0]
        self.assertEqual(command[:3], ['systemd-run', '--user', '--unit=autoresearch-run-fixture'])
        self.assertIn('--property=EnvironmentFile=' + str(self.root / '.env'), command)
        self.assertIn('--property=StandardOutput=append:' + str(self.root / 'logs/fixture.log'), command)
        self.assertEqual(command[-4:], [rental.sys.executable, str(self.root / 'tools/rental.py'), 'run', 'fixture'])
        self.assertEqual(json.loads(output.call_args.args[0])['log'], 'logs/fixture.log')
        self.assertEqual(jobs.load('fixture')['phase'], 'prepared')

    def test_starts_nothing_once_the_days_failed_attempts_cost_more_than_the_limit(self):
        job = self.prepared()
        entries = [rented(41, 'autoresearch-h100-earlier'), returned(41, 0.51)]
        with patch.object(rental, 'run') as launch, \
                patch.object(rental.vast_backend, 'entries', return_value=entries), \
                patch.object(rental.time, 'time', return_value=2000):
            with self.assertRaisesRegex(RuntimeError, 'more than'):
                rental.start(job)
        launch.assert_not_called()

    def test_runs_the_named_job_on_its_own_gpu(self):
        job = self.prepared()
        with patch.object(rental.sys, 'argv', ['rental.py', 'start', 'fixture']), \
                patch.object(rental, 'start') as start:
            rental.main()
        start.assert_called_once_with(job)
        self.assertEqual(os.environ['AUTORESEARCH_GPU'], 'h100')

    # Running

    def run_prepared(self, job, rent, execute=None):
        """Run a prepared job with fixtures for the allowance, the archives, vast-rental, the machine and finalize."""
        with patch.object(rental, 'failed_cost_today', return_value=Decimal('0.1')), \
                patch.object(delivery, 'verify_archives'), \
                patch.object(rental.vast_backend, 'rent', side_effect=rent) as vendor, \
                patch.object(rental, 'execute', side_effect=execute) as executed, \
                patch.object(rental, 'finalize') as finalize:
            try:
                rental.run_job(job)
            finally:
                self.vendor, self.executed, self.finalized = vendor, executed, finalize

    def test_rents_within_the_days_remaining_failure_budget_and_runs_on_the_kept_machine(self):
        self.run_prepared(self.prepared(), [MACHINE])
        arguments = self.vendor.call_args.args
        self.assertEqual((arguments[1], arguments[5]), ('fixture-image', Decimal('0.40')))
        stored = jobs.load('fixture')
        self.assertEqual((stored['instance_id'], stored['provider_machine_id'], stored['phase']), (42, 70, 'starting'))
        self.assertEqual(stored['deadline'], rental.dt.datetime.fromisoformat(MACHINE['deadline']).timestamp())
        self.assertEqual(self.executed.call_args.args[1][-1], 'vast-42')
        self.finalized.assert_called_once()

    def test_keeps_the_job_startable_when_no_machine_ran_it(self):
        job = self.prepared()
        with self.assertRaisesRegex(RuntimeError, 'kept no machine'):
            self.run_prepared(job, [None])
        self.executed.assert_not_called()
        self.finalized.assert_called_once()
        stored = jobs.load('fixture')
        self.assertEqual(stored['phase'], 'prepared')
        self.run_prepared(stored, [MACHINE])
        self.assertEqual(jobs.load('fixture')['instance_id'], 42)

    def test_rents_the_prepared_snapshot_while_the_working_runtime_advances(self):
        job = self.prepared()
        later = self.root / 'gpus/h100/artifacts/runtime.json'
        later.write_text('{"container_image": "later-image"}')
        self.run_prepared(job, [MACHINE])
        self.assertEqual(self.vendor.call_args.args[1], 'fixture-image')

    def test_preserves_the_full_failure_before_returning_the_machine(self):
        error = subprocess.CalledProcessError(255, 'SSH archive transfer', output=b'fixture output\n',
                                              stderr=b'fixture connection closed\n')
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_prepared(self.prepared(), [MACHINE], execute=error)
        details = (jobs.job_path('fixture').parent / 'failure.log').read_text()
        self.assertIn('fixture output', details)
        self.assertIn('fixture connection closed', details)
        stored = jobs.load('fixture')
        self.assertEqual((stored['phase'], stored['error_type']), ('failed', 'CalledProcessError'))
        self.finalized.assert_called_once()

    def test_returns_the_machine_when_the_service_is_stopped(self):
        previous = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, previous)

        def stopped(job, connection):
            rental.stop(signal.SIGTERM, None)

        with self.assertRaises(SystemExit):
            self.run_prepared(self.prepared(), [MACHINE], execute=stopped)
        self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_IGN)
        self.assertEqual(jobs.load('fixture')['phase'], 'failed')
        self.finalized.assert_called_once()

    def test_runs_one_job_of_this_loop_at_a_time(self):
        first, second = self.prepared('first-request'), self.prepared('second-request')
        entered, release = threading.Event(), threading.Event()

        def busy(job, connection):
            entered.set()
            release.wait(10)

        with patch.object(rental, 'failed_cost_today', return_value=Decimal('0')), \
                patch.object(delivery, 'verify_archives'), \
                patch.object(rental.vast_backend, 'rent', return_value=MACHINE) as vendor, \
                patch.object(rental, 'execute', side_effect=busy), patch.object(rental, 'finalize'):
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
        self.assertEqual(jobs.load('second-request')['phase'], 'prepared')

    # Giving back and recording

    def test_notes_the_machine_then_returns_it_then_records_the_result(self):
        for phase in ('evaluated', 'failed'):
            with self.subTest(phase=phase):
                job = self.running(phase=phase)
                events = []

                def record(command):
                    self.assertIn('returned_at', jobs.load('fixture'))
                    self.assertEqual(command[1:], [str(self.root / 'tools/record_run.py'), 'fixture'])
                    events.append('recorded')

                with patch.object(rental.vast_backend, 'note', side_effect=lambda *args: events.append('note')), \
                        patch.object(rental.vast_backend, 'give_back', side_effect=lambda label: events.append(label)), \
                        patch.object(rental, 'run', side_effect=record):
                    rental.finalize(job)
                self.assertEqual(events, ['note', job['label'], 'recorded'])

    def test_rejects_a_machine_that_failed_the_work_before_giving_it_back(self):
        job = self.running(phase='failed', failure_kind='machine', error_detail='no route to the machine')
        events = []
        with patch.object(rental.vast_backend, 'reject', side_effect=lambda *args: events.append(('reject', *args))), \
                patch.object(rental.vast_backend, 'note'), \
                patch.object(rental.vast_backend, 'give_back', side_effect=lambda label: events.append('back')), \
                patch.object(rental, 'run'):
            rental.finalize(job)
        self.assertEqual(events, [('reject', 42, 'autoresearch fixture: no route to the machine'), 'back'])

    def test_gives_the_machine_back_even_when_its_note_fails(self):
        with patch.object(rental.vast_backend, 'note', side_effect=subprocess.TimeoutExpired('note', 30)), \
                patch.object(rental.vast_backend, 'give_back') as give_back, patch.object(rental, 'run'):
            rental.finalize(self.running(phase='evaluated'))
        give_back.assert_called_once_with('autoresearch-h100-fixture')

    def test_records_nothing_while_a_machine_has_not_gone_back(self):
        failure = subprocess.CalledProcessError(1, 'vast-rental return')
        with patch.object(rental.vast_backend, 'note'), \
                patch.object(rental.vast_backend, 'give_back', side_effect=failure), \
                patch.object(rental, 'run') as record:
            with self.assertRaises(subprocess.CalledProcessError):
                rental.finalize(self.running(phase='evaluated'))
        record.assert_not_called()
        self.assertNotIn('returned_at', jobs.load('fixture'))

    def test_records_only_finished_runs_that_had_a_machine(self):
        for job in (self.running(phase='running'), self.prepared(phase='failed', instance_id=None)):
            with self.subTest(phase=job['phase']):
                with patch.object(rental.vast_backend, 'note'), patch.object(rental.vast_backend, 'give_back'), \
                        patch.object(rental, 'run') as record:
                    rental.finalize(job)
                record.assert_not_called()

    def test_keeps_the_collected_run_when_recording_fails(self):
        job = self.running(phase='evaluated', exit_code=0)
        with patch.object(rental.vast_backend, 'note'), patch.object(rental.vast_backend, 'give_back'), \
                patch.object(rental, 'run', side_effect=subprocess.CalledProcessError(1, 'record')):
            with self.assertRaises(subprocess.CalledProcessError):
                rental.finalize(job)
        stored = jobs.load('fixture')
        self.assertEqual((stored['phase'], stored['exit_code']), ('evaluated', 0))
        self.assertIn('returned_at', stored)

    def test_finishes_a_run_that_ended_early_as_a_failure(self):
        self.running(phase='running')
        with patch.object(rental, 'finalize') as finalize:
            rental.finish(jobs.load('fixture'))
        stored = jobs.load('fixture')
        self.assertEqual((stored['phase'], stored['error_type']), ('failed', 'RunEnded'))
        self.assertEqual(finalize.call_args.args[0]['phase'], 'failed')

    def test_connects_through_the_checked_route_with_the_callers_key_file(self):
        command = rental.ssh_command(dict(ssh_config='fixture.config', ssh_host='vast-42'), Path('fixture-key'))
        self.assertEqual(command, ['ssh', '-F', 'fixture.config', '-i', 'fixture-key',
                                   '-o', 'IdentitiesOnly=yes', '-o', 'IdentityAgent=none',
                                   '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'vast-42'])

    # On the machine

    # The time a run needs on the machine once its runtime is there.
    NEEDED = jobs.SETUP_SECONDS + 1800 + 900 + jobs.COLLECTION_SECONDS

    def executing(self, job, probes=('exit 0',), gpu='NVIDIA H100 80GB HBM3, 0', collected=b'0\n',
                  problems=None, send=None, lose_launch=False, now=1000):
        """Run the job on a fake machine whose clock moves only while the run waits: the transfers, the GPU check,
        the launch, the probes, after which the process keeps running, and the collection, as given."""
        launched, samples, clock = [], [], [now]
        probes = list(probes)

        def command(arguments, **kwargs):
            text = arguments[-1]
            if text.startswith('nvidia-smi'):
                if isinstance(gpu, BaseException):
                    raise gpu
                return completed(gpu + '\n')
            if 'nohup bash gpu_run.sh' in text:
                launched.append(text)
                if lose_launch:
                    raise subprocess.TimeoutExpired('ssh', 30)
            if text == rental.PROBE:
                result = probes.pop(0) if probes else 'running'
                if isinstance(result, BaseException):
                    raise result
                return completed(result)
            return completed()

        def sleep(seconds):
            clock[0] += max(seconds, 1)

        def collect(job, connection, destination, until):
            if collected is not None:
                (destination / 'run.exit').write_bytes(collected)
            return dict(problems or {}, **({} if collected is not None else {'run.exit': 'absent'}))

        with patch.object(rental, 'run', side_effect=command), \
                patch.object(delivery, 'send', side_effect=send), \
                patch.object(rental, 'collect', side_effect=collect), \
                patch.object(rental.telemetry, 'record_gpu_sample', side_effect=lambda *args: samples.append(1)), \
                patch.object(rental.telemetry, 'record_preparation_time'), \
                patch.object(rental.time, 'time', side_effect=lambda: clock[0]), \
                patch.object(rental.time, 'sleep', side_effect=sleep):
            rental.execute(job, ['ssh'])
        return launched, samples

    def test_launches_the_training_once_and_watches_the_same_process(self):
        launched, samples = self.executing(self.running(), probes=[subprocess.TimeoutExpired('ssh', 10),
                                                                   'running', 'exit 0'])
        self.assertEqual(len(launched), 1)
        self.assertEqual(len(samples), 1)
        stored = jobs.load('fixture')
        self.assertEqual((stored['phase'], stored['exit_code'], stored['collection_errors']), ('evaluated', 0, {}))

    def test_watches_the_one_process_after_a_lost_launch_acknowledgement(self):
        launched, _ = self.executing(self.running(), lose_launch=True)
        self.assertEqual(len(launched), 1)
        self.assertEqual(jobs.load('fixture')['phase'], 'evaluated')

    def test_launches_the_native_encoding_cache_from_the_jobs_frozen_runtime(self):
        job = self.running()
        path = jobs.job_path('fixture').parent / 'runtime.json'
        manifest = json.loads(path.read_text())
        manifest['encoding_cache'] = {'asset_path': rental.encoding_cache_asset_path('h100')}
        path.write_text(json.dumps(manifest))
        launched, _ = self.executing(job)
        self.assertIn('OLMES_ENCODING_CACHE=' + rental.encoding_cache_asset_path('h100'), launched[0])

    def test_counts_a_machine_that_cannot_be_reached_before_the_work_as_its_failure(self):
        for failure, kind in ((subprocess.CalledProcessError(255, 'ssh'), 'machine'),
                              (subprocess.TimeoutExpired('ssh', 30), 'machine'),
                              (RuntimeError('A runtime archive is missing from R2.'), None)):
            with self.subTest(failure=failure):
                job = self.running()
                with self.assertRaises(type(failure)):
                    self.executing(job, send=failure)
                self.assertEqual(job.get('failure_kind'), kind)

    def test_counts_a_gpu_that_is_missing_busy_or_unreadable_before_the_work_as_the_machines_failure(self):
        for gpu in ('NVIDIA H100 80GB HBM3, 2048', 'NVIDIA H200, 0', subprocess.CalledProcessError(9, 'nvidia-smi')):
            with self.subTest(gpu=gpu):
                job = self.running()
                with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                    launched, _ = self.executing(job, gpu=gpu)
                self.assertEqual(job['failure_kind'], 'machine')

    def test_ends_setup_when_the_transfers_take_the_time_the_run_needs(self):
        with self.assertRaises(TimeoutError):
            self.executing(self.running(deadline=1000 + jobs.RETURN_SECONDS + self.NEEDED - 1))
        self.assertEqual(jobs.load('fixture')['phase'], 'starting')

    def test_takes_the_exit_status_from_the_collected_results_when_watching_missed_it(self):
        self.executing(self.running(deadline=1000 + jobs.RETURN_SECONDS + self.NEEDED), probes=[])
        stored = jobs.load('fixture')
        self.assertEqual((stored['phase'], stored['exit_code']), ('evaluated', 0))
        self.assertTrue(stored['exit_code_observed_in_collection'])

    def test_keeps_failed_or_unconfirmed_processes_failed(self):
        for collected, code in ((b'1\n', 1), (b'137\n', 137), (b'0 invalid\n', None), (None, None)):
            with self.subTest(collected=collected):
                self.executing(self.running(deadline=1000 + jobs.RETURN_SECONDS + self.NEEDED), probes=[],
                               collected=collected)
                stored = jobs.load('fixture')
                self.assertEqual((stored['phase'], stored['exit_code']), ('failed', code))

    def test_rejects_conflicting_exit_status_observations(self):
        with self.assertRaisesRegex(RuntimeError, 'exit status changed'):
            self.executing(self.running(), probes=['exit 0'], collected=b'1\n')

    def test_stops_watching_a_process_that_vanished_without_an_exit_status(self):
        with self.assertRaisesRegex(RuntimeError, 'gone'):
            self.executing(self.running(), probes=['missing'])

    # Collecting

    def collecting(self, sends, failures=(), now=1000):
        """Collect the job's results with the machine sending what each call of `sends` gives, and the downloads of
        the names in `failures` failing, each once per listed time."""
        job = self.running()
        destination = self.root / 'collected'
        destination.mkdir(exist_ok=True)
        failures, calls, deleted = list(failures), [], []

        def send(connection, config, keys, python, *, timeout, expires):
            calls.append((list(keys), timeout))
            have = sends.pop(0)
            return ({name: dict(bytes=1, sha256='0' * 64) for name in keys if name in have},
                    [name for name in keys if name not in have])

        def download(config, key, path, sha256, size, until):
            if path.name in failures:
                failures.remove(path.name)
                raise ValueError('The object differs from what the machine sent.')
            path.write_text('collected ' + path.name)

        with patch.object(rental.r2, 'send', side_effect=send), \
                patch.object(rental.r2, 'download', side_effect=download), \
                patch.object(rental.r2, 'delete', side_effect=lambda config, key: deleted.append(key)), \
                patch.object(rental.time, 'time', return_value=now), patch('builtins.print') as output:
            problems = rental.collect(job, ['ssh'], destination, until=now + 100)
        return problems, calls, deleted, destination, '\n'.join(str(call.args[0]) for call in output.call_args_list)

    def test_collects_each_result_through_r2_and_deletes_the_copies_that_arrived(self):
        everything = set(rental.RESULTS) - {'olmes.log'}
        problems, calls, deleted, destination, _ = self.collecting([everything])
        self.assertEqual(problems, {'olmes.log': 'absent'})
        self.assertEqual(calls[0][0][0], 'run.exit')
        self.assertEqual(calls[0][1], 100)
        self.assertEqual(sorted(deleted), sorted(f'results/fixture/{name}' for name in everything))
        self.assertEqual((destination / 'final-model.pt').read_text(), 'collected final-model.pt')

    def test_sends_again_a_file_that_did_not_arrive_whole(self):
        problems, calls, deleted, _, _ = self.collecting([set(rental.RESULTS), {'final-model.pt'}],
                                                         failures=['final-model.pt'])
        self.assertEqual(problems, {})
        self.assertEqual(calls[1][0], ['final-model.pt'])
        self.assertIn('results/fixture/final-model.pt', deleted)

    def test_keeps_in_r2_a_file_that_never_arrived_and_says_where(self):
        problems, calls, deleted, _, output = self.collecting([set(rental.RESULTS)] * 3,
                                                              failures=['final-model.pt'] * 3)
        self.assertEqual(problems, {'final-model.pt': 'ValueError'})
        self.assertEqual(len(calls), 3)
        self.assertNotIn('results/fixture/final-model.pt', deleted)
        self.assertIn('R2 keeps it as results/fixture/final-model.pt', output)


if __name__ == '__main__':
    unittest.main()
