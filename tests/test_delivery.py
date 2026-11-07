import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gpu_config
from tools import delivery, jobs

R2 = dict(R2_ENDPOINT='https://store.example', R2_BUCKET='runtime', R2_ACCESS_KEY_ID='key',
          R2_SECRET_ACCESS_KEY='secret')


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for patcher in (patch.object(jobs, 'ROOT', self.root), patch.dict(os.environ, R2)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.artifacts = gpu_config.get_profile('h100').artifacts_directory(self.root)
        self.artifacts.mkdir(parents=True)
        for name in ('gpu-bootstrap.tar.gz', 'gpu-env.tar.gz', 'gpu-env.tar.xz', 'gpu-assets.tar.gz'):
            (self.artifacts / name).write_bytes(name.encode())
        digest = lambda name: jobs.digest(self.artifacts / name)
        self.manifest = dict(python_version='3.12.16', gpu_env_sha256=digest('gpu-env.tar.gz'),
                             gpu_assets_sha256=digest('gpu-assets.tar.gz'),
                             transport=dict(format='python-xz-mt', canonical_gpu_env_sha256=digest('gpu-env.tar.gz'),
                                            gpu_env_xz_sha256=digest('gpu-env.tar.xz'),
                                            bootstrap_sha256=digest('gpu-bootstrap.tar.gz')))
        directory = jobs.job_path('fixture').parent
        directory.mkdir(parents=True)
        (directory / 'runtime.json').write_text(json.dumps(self.manifest))
        (directory / 'source.tar.gz').write_bytes(b'source fixture')
        self.job = dict(id='fixture', gpu_id='h100', gpu_name='H100 SXM', deadline=5000,
                        source_archive_sha256=jobs.digest(directory / 'source.tar.gz'))
        jobs.save(self.job)

    def test_uploads_to_r2_only_the_archives_that_it_lacks_and_keeps_the_bootstrap_here(self):
        held = {'gpu-env.tar.gz': 10, 'gpu-env.tar.xz': 5}
        with patch.object(delivery.r2, 'stored', side_effect=lambda config, name, sha256: held.get(name)), \
                patch.object(delivery.r2, 'upload') as upload:
            delivery.verify_archives(self.manifest, self.artifacts, 'missing')
            self.assertEqual([call.args[2:] for call in upload.call_args_list],
                             [(self.manifest['gpu_assets_sha256'], 'gpu-assets.tar.gz')])
            (self.artifacts / 'gpu-assets.tar.gz').unlink()
            with self.assertRaisesRegex(RuntimeError, 'missing'):
                delivery.verify_archives(self.manifest, self.artifacts, 'missing')
            held['gpu-assets.tar.gz'] = 14
            (self.artifacts / 'gpu-bootstrap.tar.gz').write_bytes(b'changed bootstrap')
            with self.assertRaisesRegex(RuntimeError, 'missing'):
                delivery.verify_archives(self.manifest, self.artifacts, 'missing')

    def test_deploys_only_a_runtime_compressed_for_the_parallel_decoder(self):
        manifest = dict(self.manifest, transport=dict(self.manifest['transport'], format='python-xz'))
        with self.assertRaises(ValueError):
            delivery.verify_archives(manifest, self.artifacts, 'missing')

    def test_needs_r2_for_the_large_archives(self):
        with patch.dict(os.environ, {name: '' for name in R2}):
            with self.assertRaisesRegex(RuntimeError, 'R2'):
                delivery.verify_archives(self.manifest, self.artifacts, 'missing')

    def send(self, now=1000, until=2000, transfer=None):
        events, remote_commands = [], []

        def stream(connection, path, unpack, *, sha256, timeout):
            events.append(('ssh', Path(path).name, unpack, timeout))
            return transfer or dict(archive=Path(path).name, source='ssh')

        def fetch(connection, config, name, sha256, size, unpack, python, *, timeout, expires):
            events.append(('r2', name, unpack, timeout))
            self.assertEqual(python, '.bootstrap/.python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12')
            self.assertEqual(expires, 5000 - now)
            return dict(archive=name, source='r2')

        def remote(command, **kwargs):
            remote_commands.append(command)

        with patch.object(delivery, 'stream_archive', side_effect=stream), \
                patch.object(delivery.r2, 'fetch', side_effect=fetch), \
                patch.object(delivery.r2, 'stored', return_value=15), \
                patch.object(delivery.time, 'time', return_value=now):
            delivery.send(self.job, ['ssh'], remote, until)
        return events, remote_commands

    def test_sends_the_bootstrap_and_sources_over_ssh_and_has_r2_serve_the_large_archives(self):
        events, commands = self.send()
        self.assertEqual([event[:2] for event in events],
                         [('ssh', 'gpu-bootstrap.tar.gz'), ('r2', 'gpu-env.tar.xz'), ('r2', 'gpu-assets.tar.gz'),
                          ('ssh', 'source.tar.gz')])
        self.assertIn('.bootstrap/runtime_unpack.py extract', events[1][2])
        self.assertEqual([event[3] for event in events], [1000] * 4)
        self.assertEqual(commands[:2], ['mkdir -p autoresearch/.bootstrap',
                                        'cat > autoresearch/.bootstrap/r2_transfer.py'])
        self.assertEqual([receipt['archive'] for receipt in jobs.load('fixture')['archive_transfers']],
                         ['gpu-bootstrap.tar.gz', 'gpu-env.tar.xz', 'gpu-assets.tar.gz', 'source.tar.gz'])

    def test_goes_on_only_while_the_machine_answers_after_a_cleanup_lost_its_connection(self):
        _, commands = self.send(transfer=dict(archive='fixture', cleanup_error=dict(returncode=255)))
        self.assertEqual(commands.count('true'), 2)

    def test_stops_sending_once_the_transfers_took_the_time_the_run_needs(self):
        with self.assertRaises(TimeoutError):
            self.send(now=2000, until=2000)


if __name__ == '__main__':
    unittest.main()
