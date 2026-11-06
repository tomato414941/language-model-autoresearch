import gzip
import hashlib
import io
import lzma
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import threading
import tempfile
import unittest

from tools.parallel_ssh import stream_archive


class ParallelArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.connection = ['bash', '-c', 'cd "$1" && exec bash -c "$2"', '_', str(self.root)]
        self.payload = bytes(range(256)) * 4096
        self.path = self.root / 'archive.gz'
        self.path.write_bytes(gzip.compress(self.payload, compresslevel=0))
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def tearDown(self):
        self.temporary.cleanup()

    def transfer(self, **kwargs):
        options = dict(sha256=self.sha256, timeout=10, workers=4, chunk_bytes=65536)
        options.update(kwargs)
        return stream_archive(self.connection, self.path, 'gzip -d > restored', **options)

    def test_restores_every_byte_from_ordered_verified_parallel_chunks(self):
        receipt = self.transfer()
        self.assertEqual((self.root / 'restored').read_bytes(), self.payload)
        self.assertEqual(receipt['sha256'], self.sha256)
        self.assertEqual(receipt['bytes'], self.path.stat().st_size)
        self.assertEqual(receipt['payload_connections'], 4)
        self.assertGreater(receipt['chunks'], 4)

    def test_supports_a_single_small_chunk_with_the_same_extraction_protocol(self):
        receipt = self.transfer(chunk_bytes=self.path.stat().st_size + 1)
        self.assertEqual((self.root / 'restored').read_bytes(), self.payload)
        self.assertEqual(receipt['chunks'], 1)

    def test_keeps_the_verified_extraction_and_records_a_failed_temporary_directory_cleanup(self):
        wrapper = self.root / 'cleanup.sh'
        wrapper.write_text(
            '#!/bin/bash\nset -e\ncd ' + shlex.quote(str(self.root)) + '\n'
            'if [[ "$1" == "rm -rf autoresearch/.transfer-"* ]]; then\n'
            '  printf "fixture connection closed\\n" >&2\n  exit 255\n'
            'else\n  exec bash -c "$1"\nfi\n')
        self.connection = ['bash', str(wrapper)]
        receipt = self.transfer()
        self.assertEqual((self.root / 'restored').read_bytes(), self.payload)
        self.assertEqual(receipt['sha256'], self.sha256)
        self.assertEqual(receipt['cleanup_error']['returncode'], 255)
        self.assertIn('fixture connection closed', receipt['cleanup_error']['stderr'])

    def test_limits_remote_temporary_bytes_while_the_extractor_is_slow(self):
        stopped = threading.Event()
        maximum = [0]

        def sample():
            while not stopped.is_set():
                size = 0
                for directory in (self.root / 'autoresearch').glob('.transfer-*'):
                    try:
                        paths = list(directory.iterdir())
                    except FileNotFoundError:
                        continue
                    for path in paths:
                        if path.name[:8].isdigit():
                            try:
                                size += path.stat().st_size
                            except FileNotFoundError:
                                pass
                maximum[0] = max(maximum[0], size)
                stopped.wait(.001)

        sampler = threading.Thread(target=sample)
        sampler.start()
        try:
            stream_archive(self.connection, self.path, 'sleep 0.5; gzip -d > restored',
                           sha256=self.sha256, timeout=10, workers=4, chunk_bytes=65536)
        finally:
            stopped.set()
            sampler.join()
        self.assertGreater(maximum[0], 0)
        self.assertLessEqual(maximum[0], 2 * 4 * 65536)
        self.assertEqual((self.root / 'restored').read_bytes(), self.payload)

    def test_streams_xz_chunks_into_the_packaged_parallel_runtime_decoder(self):
        content = io.BytesIO()
        with tarfile.open(fileobj=content, mode='w') as archive:
            member = tarfile.TarInfo('runtime-file')
            member.size = len(self.payload)
            archive.addfile(member, io.BytesIO(self.payload))
        self.path.write_bytes(lzma.compress(content.getvalue()))
        decoder = Path(__file__).resolve().parents[1] / 'tools/unpack_runtime.py'
        target = self.root / 'runtime'
        target.mkdir()
        unpack = 'cd ' + shlex.quote(str(target)) + ' && ' + shlex.join([sys.executable, str(decoder), 'extract'])
        stream_archive(self.connection, self.path, unpack, sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                       timeout=10, workers=4, chunk_bytes=64)
        self.assertEqual((target / 'runtime-file').read_bytes(), self.payload)

    def test_reports_a_changed_source_before_accepting_the_transfer(self):
        with self.assertRaisesRegex(RuntimeError, 'frozen SHA-256'):
            self.transfer(sha256='0' * 64)

    def test_rejects_a_corrupted_received_chunk_before_publishing_it(self):
        wrapper = self.root / 'receiver.sh'
        wrapper.write_text(
            '#!/bin/bash\nset -e\ncd ' + shlex.quote(str(self.root)) + '\n'
            'if [[ "$1" == "set -e; cat > "* ]]; then\n'
            '  (printf x; cat) | bash -c "$1"\nelse\n  exec bash -c "$1"\nfi\n')
        self.connection = ['bash', str(wrapper)]
        with self.assertRaises(subprocess.CalledProcessError):
            self.transfer()

    def test_propagates_an_extractor_failure_and_stops_the_transfer(self):
        with self.assertRaises((subprocess.CalledProcessError, RuntimeError)):
            stream_archive(self.connection, self.path, 'exit 7', sha256=self.sha256,
                           timeout=10, workers=4, chunk_bytes=65536)

    def test_enforces_the_transfer_deadline(self):
        wrapper = self.root / 'slow.sh'
        wrapper.write_text('#!/bin/bash\nsleep 0.2\nexec bash -c "$1"\n')
        self.connection = ['bash', str(wrapper)]
        with self.assertRaises(subprocess.TimeoutExpired):
            self.transfer(timeout=0.01)


if __name__ == '__main__':
    unittest.main()
