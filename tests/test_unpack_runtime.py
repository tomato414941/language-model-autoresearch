import io
import lzma
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from tools.unpack_runtime import Reader


class ParallelRuntimeTests(unittest.TestCase):
    def decode(self, payload):
        with Reader(io.BytesIO(payload)) as raw, io.BufferedReader(raw) as decoded:
            result = decoded.read()
            self.assertTrue(raw.ended)
            return result

    def test_reads_complete_empty_and_concatenated_streams(self):
        for content in (b'', b'hello', bytes(range(256))*257):
            with self.subTest(bytes=len(content)):
                self.assertEqual(self.decode(lzma.compress(content)), content)
        payload = lzma.compress(b'first')+b'\0'*4+lzma.compress(b'second')
        self.assertEqual(self.decode(payload), b'firstsecond')

    def test_detects_truncation_corruption_and_trailing_garbage(self):
        payload = lzma.compress(bytes(range(256))*1024)
        corrupt = bytearray(payload)
        corrupt[len(corrupt)//2] ^= 1
        for damaged in (b'', b'bad input', payload[:-1], payload[:-12], bytes(corrupt), payload+b'garbage'):
            with self.subTest(bytes=len(damaged)), self.assertRaises(ValueError):
                self.decode(damaged)

    def test_extracts_a_complete_tar_and_checks_the_xz_footer(self):
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode='w') as output:
            entry = tarfile.TarInfo('payload')
            entry.mode, entry.size = 0o751, 5
            output.addfile(entry, io.BytesIO(b'hello'))
        payload = lzma.compress(archive.getvalue())
        command = [sys.executable, str(Path(__file__).resolve().parents[1]/'tools/unpack_runtime.py'), 'extract']
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(command, input=payload, cwd=directory, check=True)
            restored = Path(directory)/'payload'
            self.assertEqual(restored.read_bytes(), b'hello')
            self.assertEqual(restored.stat().st_mode & 0o777, 0o751)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(command, input=payload[:-1], cwd=directory, capture_output=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
