"""Move files between a rented machine and R2 through presigned URLs, so that no R2 key ever reaches the machine.

`fetch` writes one archive to standard output in order, verifying every byte; its settings arrive as JSON on standard
input: url, bytes, sha256, workers, chunk_bytes. `send` puts each named file of the current folder to its own URL; its
settings arrive as JSON on standard input: files, a URL for each name. Both print a receipt on standard error. The
machine runs this with the standalone deployment interpreter, so it uses only the standard library.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request

RECEIPT_PREFIX = 'r2-receipt '
SYSTEM_CERTIFICATES = '/etc/ssl/certs/ca-certificates.crt'
ATTEMPTS = 5
# The most that one PUT may carry.
SEND_BYTES = 5 * 1000 ** 3


def context():
    certificates = SYSTEM_CERTIFICATES if os.path.exists(SYSTEM_CERTIFICATES) else None
    return ssl.create_default_context(cafile=certificates)


def hidden(error, urls):
    """The error's text with every presigned URL hidden."""
    detail = type(error).__name__ + ': ' + str(error)
    for url in urls:
        detail = detail.replace(url, '<presigned-url>')
    return detail.split('?X-Amz-')[0][:300]


def fetch_range(url, start, end, context, stopped):
    error = None
    for attempt in range(ATTEMPTS):
        if stopped.is_set():
            raise RuntimeError('The fetch stopped after another range failed.')
        try:
            message = urllib.request.Request(url, headers={'Range': 'bytes=%d-%d' % (start, end)})
            with urllib.request.urlopen(message, timeout=60, context=context) as response:
                if response.status != 206:
                    raise ConnectionError('The store ignored the byte range.')
                payload = response.read()
            if len(payload) != end - start + 1:
                raise ConnectionError('The store returned a short byte range.')
            return payload
        except urllib.error.HTTPError as failure:
            if failure.code < 500 and failure.code != 429:
                raise
            error = failure
        except (urllib.error.URLError, OSError) as failure:
            error = failure
        time.sleep(min(8, 2 ** attempt))
    raise error


def fetch(settings):
    url, size, sha256 = settings['url'], int(settings['bytes']), settings['sha256']
    workers, chunk = int(settings.get('workers', 8)), int(settings.get('chunk_bytes', 32 << 20))
    if size <= 0 or workers <= 0 or chunk <= 0:
        raise ValueError('Use a nonempty archive and positive fetch limits.')
    secure = context()
    count = -(-size // chunk)
    started = time.monotonic()
    checksum = hashlib.sha256()
    written = 0
    output = sys.stdout.buffer
    stopped = threading.Event()
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        pending = {}
        submitted = 0
        for index in range(count):
            # Keep at most two ranges per connection in memory while the extractor catches up.
            while submitted < min(count, index + 2 * workers):
                start = submitted * chunk
                pending[submitted] = pool.submit(fetch_range, url, start, min(size, start + chunk) - 1,
                                                 secure, stopped)
                submitted += 1
            payload = pending.pop(index).result()
            checksum.update(payload)
            output.write(payload)
            written += len(payload)
        output.flush()
    except BaseException as error:
        stopped.set()
        pool.shutdown(wait=False, cancel_futures=True)
        print('The archive fetch failed: ' + hidden(error, [url]), file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)
    pool.shutdown(wait=True)
    if written != size or checksum.hexdigest() != sha256:
        print('The fetched archive differs from its frozen SHA-256 digest.', file=sys.stderr)
        return 3
    print(RECEIPT_PREFIX + json.dumps(dict(bytes=written, sha256=sha256, connections=min(workers, count),
                                           seconds=round(time.monotonic() - started, 3))), file=sys.stderr)
    return 0


def put(url, name, size, secure):
    """Put the file to its URL, trying again after a failure on the way or on R2's side."""
    error = None
    for attempt in range(ATTEMPTS):
        try:
            with open(name, 'rb') as source:
                message = urllib.request.Request(url, data=source, method='PUT', headers={
                    'Content-Length': str(size), 'Content-Type': 'application/octet-stream'})
                with urllib.request.urlopen(message, timeout=300, context=secure) as response:
                    response.read()
            return
        except urllib.error.HTTPError as failure:
            if failure.code < 500 and failure.code != 429:
                raise
            error = failure
        except (urllib.error.URLError, OSError) as failure:
            error = failure
        time.sleep(min(8, 2 ** attempt))
    raise error


def send(settings):
    files = settings['files']
    secure = context()
    started = time.monotonic()
    sent, absent = {}, []
    try:
        for name, url in files.items():
            if not os.path.isfile(name):
                absent.append(name)
                continue
            size = os.path.getsize(name)
            if size > SEND_BYTES:
                raise ValueError(f'{name} holds more than one upload may carry.')
            checksum = hashlib.sha256()
            with open(name, 'rb') as source:
                for block in iter(lambda: source.read(1 << 20), b''):
                    checksum.update(block)
            put(url, name, size, secure)
            sent[name] = dict(bytes=size, sha256=checksum.hexdigest())
    except BaseException as error:
        print('The upload failed: ' + hidden(error, files.values()), file=sys.stderr)
        return 1
    print(RECEIPT_PREFIX + json.dumps(dict(sent=sent, absent=absent, seconds=round(time.monotonic() - started, 3))),
          file=sys.stderr)
    return 0


if __name__ == '__main__':
    modes = {'fetch': fetch, 'send': send}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit('Usage: r2_transfer.py fetch|send < settings.json')
    sys.exit(modes[sys.argv[1]](json.load(sys.stdin)))
