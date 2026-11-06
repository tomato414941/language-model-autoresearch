"""Write one archive from a presigned URL to standard output in order, verifying every byte.

The rental runs this with the standalone deployment interpreter, so it uses only the standard
library. Its settings arrive as JSON on standard input: url, bytes, sha256, workers, chunk_bytes.
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

RECEIPT_PREFIX = 'r2-fetch-receipt '
SYSTEM_CERTIFICATES = '/etc/ssl/certs/ca-certificates.crt'
ATTEMPTS = 5


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


def main():
    settings = json.load(sys.stdin)
    url, size, sha256 = settings['url'], int(settings['bytes']), settings['sha256']
    workers, chunk = int(settings.get('workers', 8)), int(settings.get('chunk_bytes', 32 << 20))
    if size <= 0 or workers <= 0 or chunk <= 0:
        raise ValueError('Use a nonempty archive and positive fetch limits.')
    certificates = SYSTEM_CERTIFICATES if os.path.exists(SYSTEM_CERTIFICATES) else None
    context = ssl.create_default_context(cafile=certificates)
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
                                                 context, stopped)
                submitted += 1
            payload = pending.pop(index).result()
            checksum.update(payload)
            output.write(payload)
            written += len(payload)
        output.flush()
    except BaseException as error:
        stopped.set()
        pool.shutdown(wait=False, cancel_futures=True)
        detail = (type(error).__name__ + ': ' + str(error)).replace(url, '<presigned-url>')
        print('The archive fetch failed: ' + detail.split('?X-Amz-')[0][:300], file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)
    pool.shutdown(wait=True)
    if written != size or checksum.hexdigest() != sha256:
        print('The fetched archive differs from its frozen SHA-256 digest.', file=sys.stderr)
        return 3
    print(RECEIPT_PREFIX + json.dumps(dict(bytes=written, sha256=sha256, connections=min(workers, count),
                                           seconds=round(time.monotonic() - started, 3))), file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
