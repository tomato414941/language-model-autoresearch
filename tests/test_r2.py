import hashlib
import http.server
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.parse

from tools import r2


class RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b''
    failures = []
    requests = []

    def do_GET(self):
        type(self).requests.append(self.headers.get('Range'))
        if type(self).failures:
            self.send_error(type(self).failures.pop(0))
            return
        start, end = (int(value) for value in self.headers['Range'].split('=')[1].split('-'))
        body = type(self).payload[start:end + 1]
        self.send_response(206)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class R2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        RangeHandler.payload = bytes(range(256)) * 4099
        RangeHandler.failures = []
        RangeHandler.requests = []
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), RangeHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = 'http://127.0.0.1:%d/runtime/archive?X-Amz-Signature=secret' % self.server.server_port
        self.sha256 = hashlib.sha256(RangeHandler.payload).hexdigest()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.temporary.cleanup()

    def run_fetcher(self, **settings):
        values = dict(url=self.url, bytes=len(RangeHandler.payload), sha256=self.sha256,
                      workers=3, chunk_bytes=50000)
        values.update(settings)
        return subprocess.run([sys.executable, str(r2.FETCHER)], input=json.dumps(values).encode(),
                              capture_output=True, timeout=60)

    def test_signs_presigned_urls_like_the_aws_reference_example(self):
        url = r2.presigned_url('examplebucket.s3.amazonaws.com', '/test.txt', 'AKIAIOSFODNN7EXAMPLE',
                               'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY', 86400, '20130524T000000Z',
                               region='us-east-1')
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        self.assertEqual(query['X-Amz-Signature'],
                         'aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404')

    def test_signs_header_requests_like_the_aws_reference_example(self):
        signature = r2.signature(
            'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY', '20130524T000000Z', 'GET', '/test.txt', {},
            {'host': 'examplebucket.s3.amazonaws.com', 'range': 'bytes=0-9',
             'x-amz-content-sha256': hashlib.sha256(b'').hexdigest(), 'x-amz-date': '20130524T000000Z'},
            hashlib.sha256(b'').hexdigest(), region='us-east-1')
        self.assertEqual(signature, 'f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41')

    def test_uses_r2_only_when_every_variable_is_set(self):
        with patch.dict(r2.os.environ, {name: '' for name in r2.VARIABLES}):
            self.assertIsNone(r2.settings())
        with patch.dict(r2.os.environ, {**{name: '' for name in r2.VARIABLES}, 'R2_BUCKET': 'runtime'}):
            with self.assertRaises(RuntimeError):
                r2.settings()
        values = dict(R2_ENDPOINT='https://store.example', R2_BUCKET='runtime',
                      R2_ACCESS_KEY_ID='key', R2_SECRET_ACCESS_KEY='secret')
        with patch.dict(r2.os.environ, values):
            self.assertEqual(r2.settings()['host'], 'store.example')

    def test_writes_every_byte_in_order_from_parallel_ranges(self):
        result = self.run_fetcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, RangeHandler.payload)
        receipt = json.loads(result.stderr.decode().split(r2.RECEIPT_PREFIX, 1)[1])
        self.assertEqual((receipt['bytes'], receipt['sha256'], receipt['connections']),
                         (len(RangeHandler.payload), self.sha256, 3))

    def test_retries_a_temporarily_unavailable_range(self):
        RangeHandler.failures = [503]
        result = self.run_fetcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, RangeHandler.payload)

    def test_rejects_bytes_that_differ_from_the_frozen_digest(self):
        result = self.run_fetcher(sha256='0' * 64)
        self.assertEqual(result.returncode, 3)
        self.assertNotIn(r2.RECEIPT_PREFIX.encode(), result.stderr)

    def test_reports_a_refused_url_without_revealing_it(self):
        RangeHandler.failures = [403] * 20
        result = self.run_fetcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'HTTP Error 403', result.stderr)
        self.assertNotIn(b'X-Amz-Signature', result.stderr)
        self.assertNotIn(b'127.0.0.1', result.stderr)

    def test_streams_from_the_rental_into_its_extractor_with_settings_on_standard_input(self):
        bootstrap = self.root / 'autoresearch/.bootstrap'
        bootstrap.mkdir(parents=True)
        (bootstrap / 'r2_fetch.py').write_bytes(r2.FETCHER.read_bytes())
        (bootstrap / 'python').symlink_to(sys.executable)
        connection = ['bash', '-c', 'cd "$1" && exec bash -c "$2"', '_', str(self.root)]
        config = dict(host='store.example', bucket='runtime', key_id='key', secret='secret')
        with patch.object(r2, 'presign', return_value=self.url):
            receipt = r2.fetch(connection, config, 'gpu-env.tar.xz', self.sha256, len(RangeHandler.payload),
                               'cat > restored', '.bootstrap/python', timeout=60, expires=600)
        self.assertEqual((self.root / 'restored').read_bytes(), RangeHandler.payload)
        self.assertEqual((receipt['source'], receipt['bytes']), ('r2', len(RangeHandler.payload)))

    def test_fails_the_transfer_when_the_extractor_rejects_the_stream(self):
        bootstrap = self.root / 'autoresearch/.bootstrap'
        bootstrap.mkdir(parents=True)
        (bootstrap / 'r2_fetch.py').write_bytes(r2.FETCHER.read_bytes())
        (bootstrap / 'python').symlink_to(sys.executable)
        connection = ['bash', '-c', 'cd "$1" && exec bash -c "$2"', '_', str(self.root)]
        config = dict(host='store.example', bucket='runtime', key_id='key', secret='secret')
        with patch.object(r2, 'presign', return_value=self.url), self.assertRaises(RuntimeError):
            r2.fetch(connection, config, 'gpu-env.tar.xz', self.sha256, len(RangeHandler.payload),
                     'head -c 10 > /dev/null; exit 2', '.bootstrap/python', timeout=60, expires=600)


if __name__ == '__main__':
    unittest.main()
