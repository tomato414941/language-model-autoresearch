import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from gpu_config import get_profile
from tools import vast_backend


class VastBackendTests(unittest.TestCase):
    def test_passes_the_key_file_and_api_key_through_the_child_environment(self):
        with patch.dict(os.environ, {'PATH': '/usr/bin', 'VAST_API_KEY': 'fixture-api-key'}, clear=True), \
                patch.object(vast_backend, 'executable', return_value=Path('fixture-vast-rental')), \
                patch.object(vast_backend.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run:
            vast_backend.command(['rent', 'RTX 5090'], key=Path('fixture-key'))
        self.assertEqual(run.call_args.args[0], ['fixture-vast-rental', 'rent', 'RTX 5090'])
        environment = run.call_args.kwargs['env']
        self.assertEqual(environment['VAST_SSH_KEY_FILE'], 'fixture-key')
        self.assertEqual(environment['VAST_API_KEY'], 'fixture-api-key')

    def test_receives_the_connection_config_and_keeps_the_full_result_stream(self):
        accepted = dict(rented=42, price=0.5, machine=70,
                        ssh_config='fixture.config', ssh_host='vast-42')
        lines = [dict(unusable=41, stage='check'), accepted]
        result = subprocess.CompletedProcess([], 0, stdout='\n'.join(map(json.dumps, lines)) + '\n', stderr='')
        job = dict(label='autoresearch-fixture', max_hourly_cost=1,
                   gpu_settings=dict(rental_seconds=6300))
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(vast_backend, 'command', return_value=result) as command:
            returned, rows = vast_backend.rent(job, {'container_image': 'fixture-image'}, get_profile('rtx5090'),
                                                Path('fixture-key'), {'cuda_max_good': {'gte': 12.8}},
                                                300, Path(temporary))
            self.assertEqual(returned, accepted)
            self.assertEqual(rows, lines)
            self.assertEqual((Path(temporary) / 'vast-rental.log').read_text(), result.stdout)
        arguments = command.call_args.args[0]
        self.assertEqual(arguments[arguments.index('--hours') + 1], '1.75')
        self.assertEqual(arguments[arguments.index('--max-price') + 1], '1')
        self.assertEqual(arguments[arguments.index('--tries') + 1], '1')

    def test_returns_the_failed_health_attempt_for_the_callers_next_budget_check(self):
        response = dict(rented=None, failed=1, offers=4)
        result = subprocess.CompletedProcess([], 1, stdout=json.dumps(response) + '\n', stderr='')
        job = dict(label='autoresearch-fixture', max_hourly_cost=1,
                   gpu_settings=dict(rental_seconds=6300))
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(vast_backend, 'command', return_value=result):
            returned, rows = vast_backend.rent(job, {'container_image': 'fixture-image'}, get_profile('rtx5090'),
                                                Path('fixture-key'), {}, 300, Path(temporary))
        self.assertEqual(returned['failed'], 1)
        self.assertEqual(rows, [response])

    def test_keeps_partial_output_when_the_health_check_times_out(self):
        error = subprocess.TimeoutExpired('fixture', 300, output=b'{"unusable":41}\n', stderr=b'timed out\n')
        job = dict(label='autoresearch-fixture', max_hourly_cost=1,
                   gpu_settings=dict(rental_seconds=6300))
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(vast_backend, 'command', side_effect=error):
            with self.assertRaises(subprocess.TimeoutExpired):
                vast_backend.rent(job, {'container_image': 'fixture-image'}, get_profile('rtx5090'),
                                  Path('fixture-key'), {}, 300, Path(temporary))
            self.assertEqual((Path(temporary) / 'vast-rental.log').read_text(),
                             '{"unusable":41}\ntimed out\n')

    def test_reads_shared_events_once_and_accepts_an_unfinished_last_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'data').mkdir()
            event = dict(id='one', event='returned', instance=42, dollars=0.06)
            (root / 'data/ledger.jsonl').write_text(json.dumps(event) + '\n' + json.dumps(event) + '\n{')
            with patch.object(vast_backend, 'executable', return_value=root / 'vast_rental.py'):
                self.assertEqual(vast_backend.entries(), [event])


if __name__ == '__main__':
    unittest.main()
