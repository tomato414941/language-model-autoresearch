import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools import telemetry


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

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
                telemetry.record_preparation_time(remote, self.root)
                observation = json.loads((self.root / 'preparation-timing.json').read_text())
                self.assertEqual(observation['approximate_preparation_seconds'], 63)
                self.assertEqual(observation['ready_shards'], 71)
                self.assertEqual(observation['requested_shards'], 71)
                self.assertTrue(observation['prepare_log_completed_marker'])

    def test_preserves_an_unavailable_timing_record_when_the_remote_read_times_out(self):
        remote = Mock(side_effect=subprocess.TimeoutExpired('ssh', 5))
        telemetry.record_preparation_time(remote, self.root)
        observation = json.loads((self.root / 'preparation-timing.json').read_text())
        self.assertEqual(observation['error_type'], 'TimeoutExpired')
        self.assertEqual(observation['files'], {})

    def test_records_timestamped_gpu_memory_and_utilization_samples_in_order(self):
        results = [subprocess.CompletedProcess([], 0, stdout='NVIDIA H100 80GB HBM3, 45060, 96\n'),
                   subprocess.CompletedProcess([], 0, stdout='NVIDIA H100 80GB HBM3, 2900, 12\n')]
        remote = Mock(side_effect=results)
        with patch.object(telemetry.time, 'time', side_effect=[1000, 1030]):
            telemetry.record_gpu_sample(remote, self.root)
            telemetry.record_gpu_sample(remote, self.root)
        samples = [json.loads(line) for line in (self.root / 'gpu-observations.jsonl').read_text().splitlines()]
        self.assertEqual([sample['observed_at'] for sample in samples], [1000, 1030])
        self.assertEqual([sample['memory_mib'] for sample in samples], [45060, 2900])
        self.assertEqual([sample['utilization_percent'] for sample in samples], [96, 12])

    def test_records_unavailable_telemetry_and_accepts_the_next_valid_sample(self):
        remote = Mock(side_effect=[
            subprocess.CompletedProcess([], 0, stdout='NVIDIA H100 80GB HBM3, 45060, N/A\n'),
            subprocess.CompletedProcess([], 0, stdout='NVIDIA H100 80GB HBM3, 45060, 90\n'),
        ])
        telemetry.record_gpu_sample(remote, self.root)
        telemetry.record_gpu_sample(remote, self.root)
        samples = [json.loads(line) for line in (self.root / 'gpu-observations.jsonl').read_text().splitlines()]
        self.assertEqual(samples[0]['error_type'], 'ValueError')
        self.assertEqual(samples[1]['utilization_percent'], 90)

    def test_records_gpu_clocks_power_limit_and_temperature_with_utilization(self):
        remote = Mock(return_value=subprocess.CompletedProcess([], 0,
            stdout='NVIDIA H100 80GB HBM3, 45060, 96, 1980, 1980, 2619, 420.5, 700, 61\n'))
        telemetry.record_gpu_sample(remote, self.root)
        sample = json.loads((self.root / 'gpu-observations.jsonl').read_text())
        expected = dict(memory_mib=45060, utilization_percent=96, sm_clock_mhz=1980,
                        max_sm_clock_mhz=1980, memory_clock_mhz=2619,
                        power_draw_w=420.5, power_limit_w=700, temperature_c=61)
        for field, value in expected.items():
            self.assertEqual(sample[field], value)

    def test_preserves_basic_measurements_when_optional_sensors_are_unavailable(self):
        remote = Mock(return_value=subprocess.CompletedProcess([], 0,
            stdout='NVIDIA H100 80GB HBM3, 45060, 96, N/A, N/A, nan, inf, -1, N/A\n'))
        telemetry.record_gpu_sample(remote, self.root)
        sample = json.loads((self.root / 'gpu-observations.jsonl').read_text())
        self.assertEqual((sample['memory_mib'], sample['utilization_percent']), (45060, 96))
        for field in ('sm_clock_mhz', 'max_sm_clock_mhz', 'memory_clock_mhz',
                      'power_draw_w', 'power_limit_w', 'temperature_c'):
            self.assertIsNone(sample[field])

    def test_recovers_basic_telemetry_within_the_same_sample_budget_after_an_unsupported_query(self):
        remote = Mock(side_effect=[subprocess.CalledProcessError(1, 'nvidia-smi'),
                                  subprocess.CompletedProcess([], 0, stdout='NVIDIA H100 80GB HBM3, 45060, 96\n')])
        with patch.object(telemetry.time, 'monotonic', side_effect=[100, 102]):
            telemetry.record_gpu_sample(remote, self.root)
        sample = json.loads((self.root / 'gpu-observations.jsonl').read_text())
        self.assertEqual(sample['extended_error_type'], 'CalledProcessError')
        self.assertEqual((sample['memory_mib'], sample['utilization_percent']), (45060, 96))
        self.assertIsNone(sample['power_draw_w'])
        self.assertEqual(remote.call_args.kwargs['timeout'], 3)

    def test_keeps_a_sample_timeout_when_its_fallback_budget_is_exhausted(self):
        remote = Mock(side_effect=subprocess.TimeoutExpired('nvidia-smi', 5))
        with patch.object(telemetry.time, 'monotonic', side_effect=[100, 105]):
            telemetry.record_gpu_sample(remote, self.root)
        sample = json.loads((self.root / 'gpu-observations.jsonl').read_text())
        self.assertEqual(sample['error_type'], 'TimeoutExpired')
        remote.assert_called_once()


if __name__ == '__main__':
    unittest.main()
