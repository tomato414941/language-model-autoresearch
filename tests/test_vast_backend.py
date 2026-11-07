import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from gpu_config import get_profile
from tools import vast_backend

KEPT = dict(rented=42, price=0.5, machine=70, ssh_config='fixture.config', ssh_host='vast-42',
            deadline='2026-11-07T06:00:00+00:00', failed=1, failed_dollars=0.02)
JOB = dict(label='autoresearch-fixture', max_hourly_cost=1, gpu_settings=dict(rental_seconds=6300))


class VastBackendTests(unittest.TestCase):
    def rent(self, result):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(vast_backend, 'command', side_effect=[result]) as command:
            kept = vast_backend.rent(JOB, 'fixture-image', get_profile('rtx5090'), Path('fixture-key'),
                                     {'cuda_max_good': {'gte': 12.8}}, 0.45, Path(temporary))
            log = (Path(temporary) / 'vast-rental.log').read_text()
        return kept, command.call_args, log

    def test_passes_the_key_file_through_the_child_environment(self):
        with patch.dict(os.environ, {'PATH': '/usr/bin'}, clear=True), \
                patch.object(vast_backend, 'executable', return_value=Path('fixture-vast-rental')), \
                patch.object(vast_backend.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run:
            vast_backend.command(['rent', 'RTX 5090'], key=Path('fixture-key'))
        self.assertEqual(run.call_args.args[0], ['fixture-vast-rental', 'rent', 'RTX 5090'])
        self.assertEqual(run.call_args.kwargs['env']['VAST_SSH_KEY_FILE'], 'fixture-key')

    def test_rents_within_the_failure_budget_and_gives_the_kept_machine_with_its_deadline(self):
        lines = [dict(unusable=41, stage='check'), KEPT]
        output = '\n'.join(map(json.dumps, lines)) + '\n'
        kept, call, log = self.rent(subprocess.CompletedProcess([], 0, stdout=output, stderr=''))
        self.assertEqual(kept, KEPT)
        self.assertEqual(log, output)
        arguments = call.args[0]
        for flag, value in (('--hours', '1.75'), ('--max-price', '1'), ('--tries', str(vast_backend.TRIES)),
                            ('--failure-budget', '0.45'), ('--label', 'autoresearch-fixture')):
            self.assertEqual(arguments[arguments.index(flag) + 1], value)
        self.assertEqual(call.kwargs['timeout'], vast_backend.RENT_SECONDS)

    def test_gives_no_machine_when_none_passed_within_the_tries_and_budget(self):
        output = json.dumps(dict(rented=None, offers=4, failed=2, failed_dollars=0.6)) + '\n'
        kept, _, _ = self.rent(subprocess.CompletedProcess([], 1, stdout=output, stderr=''))
        self.assertIsNone(kept)

    def test_fails_when_vast_rental_fails_or_keeps_a_machine_it_cannot_describe(self):
        for result in (subprocess.CompletedProcess([], 1, stdout='', stderr='rent needs the watcher\n'),
                       subprocess.CompletedProcess([], 2, stdout=json.dumps(dict(unusable=41)) + '\n', stderr=''),
                       subprocess.CompletedProcess([], 0, stdout=json.dumps(dict(KEPT, deadline=None)) + '\n',
                                                   stderr='')):
            with self.subTest(result=result):
                with self.assertRaises(RuntimeError):
                    self.rent(result)

    def test_keeps_partial_output_when_renting_times_out(self):
        error = subprocess.TimeoutExpired('fixture', 300, output=b'{"unusable":41}\n', stderr=b'timed out\n')
        with tempfile.TemporaryDirectory() as temporary, patch.object(vast_backend, 'command', side_effect=error):
            with self.assertRaises(subprocess.TimeoutExpired):
                vast_backend.rent(JOB, 'fixture-image', get_profile('rtx5090'), Path('fixture-key'), {}, 0.5,
                                  Path(temporary))
            self.assertEqual((Path(temporary) / 'vast-rental.log').read_text(), '{"unusable":41}\ntimed out\n')

    def test_gives_back_every_machine_of_a_label_and_rejects_and_notes_one(self):
        with patch.object(vast_backend, 'command') as command:
            vast_backend.give_back('autoresearch-fixture')
            vast_backend.reject(42, 'unreachable')
            vast_backend.note(42, 'autoresearch fixture', {'phase': 'failed'})
        self.assertEqual([call.args[0] for call in command.call_args_list],
                         [['return', '--label', 'autoresearch-fixture'], ['reject', '42', 'unreachable'],
                          ['note', '42', 'autoresearch fixture', '--values', '{"phase": "failed"}']])
        self.assertEqual(command.call_args_list[0].kwargs['timeout'], vast_backend.RETURN_SECONDS)

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
