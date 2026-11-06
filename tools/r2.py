"""Keep verified runtime archives in Cloudflare R2 and let a rental fetch them directly."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from xml.etree import ElementTree
from xml.sax.saxutils import escape

FETCHER = Path(__file__).resolve().parent / 'r2_fetch.py'
VARIABLES = ('R2_ENDPOINT', 'R2_BUCKET', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY')
PART_BYTES = 128 * 1024 * 1024
UPLOAD_WORKERS = 4
FETCH_WORKERS = 8
FETCH_CHUNK_BYTES = 32 * 1024 * 1024
RECEIPT_PREFIX = 'r2-fetch-receipt '


def settings():
    """Return the R2 settings, or None when every archive still travels over SSH."""
    values = [os.environ.get(name, '').strip() for name in VARIABLES]
    if not any(values):
        return None
    if not all(values):
        raise RuntimeError('Set every R2 variable or none of them.')
    endpoint = urllib.parse.urlsplit(values[0])
    if endpoint.scheme != 'https' or not endpoint.hostname or endpoint.path not in ('', '/'):
        raise RuntimeError('R2_ENDPOINT must be an HTTPS origin.')
    return dict(host=endpoint.netloc, bucket=values[1], key_id=values[2], secret=values[3])


def object_key(sha256, name):
    if not re.fullmatch('[0-9a-f]{64}', sha256 or '') or not re.fullmatch(r'[A-Za-z0-9._-]+', name):
        raise ValueError('An R2 archive needs a frozen SHA-256 digest and a plain file name.')
    return 'runtime/' + sha256 + '/' + name


def quote(text, safe='-_.~'):
    return urllib.parse.quote(text, safe=safe)


def canonical_query(params):
    return '&'.join(quote(name) + '=' + quote(value) for name, value in sorted(params.items()))


def signature(secret, stamp, method, path, params, headers, payload_hash, region='auto'):
    """Sign one request with AWS Signature Version 4, which R2 accepts for its S3 API."""
    names = sorted(headers)
    canonical = '\n'.join([method, quote(path, safe='/-_.~'), canonical_query(params),
                           ''.join(name + ':' + str(headers[name]).strip() + '\n' for name in names),
                           ';'.join(names), payload_hash])
    scope = stamp[:8] + '/' + region + '/s3/aws4_request'
    text = '\n'.join(['AWS4-HMAC-SHA256', stamp, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    key = ('AWS4' + secret).encode()
    for part in (stamp[:8], region, 's3', 'aws4_request'):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    return hmac.new(key, text.encode(), hashlib.sha256).hexdigest()


def presigned_url(host, path, key_id, secret, expires, stamp, region='auto'):
    params = {'X-Amz-Algorithm': 'AWS4-HMAC-SHA256',
              'X-Amz-Credential': key_id + '/' + stamp[:8] + '/' + region + '/s3/aws4_request',
              'X-Amz-Date': stamp, 'X-Amz-Expires': str(int(expires)), 'X-Amz-SignedHeaders': 'host'}
    params['X-Amz-Signature'] = signature(secret, stamp, 'GET', path, params, {'host': host},
                                          'UNSIGNED-PAYLOAD', region)
    return 'https://' + host + quote(path, safe='/-_.~') + '?' + canonical_query(params)


def presign(config, key, expires):
    """Let a rental read one object until its lease ends, without holding any R2 key."""
    if not 1 <= expires <= 604800:
        raise ValueError('A presigned URL lasts between one second and seven days.')
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    return presigned_url(config['host'], '/' + config['bucket'] + '/' + key,
                         config['key_id'], config['secret'], expires, stamp)


def request(config, method, key, params=None, headers=None, body=None, timeout=120):
    params = params or {}
    payload = body if body is not None else b''
    payload_hash = hashlib.sha256(payload).hexdigest()
    path = '/' + config['bucket'] + '/' + key
    for attempt in range(4):
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        signed = {name.lower(): value for name, value in (headers or {}).items()}
        signed.update({'host': config['host'], 'x-amz-content-sha256': payload_hash, 'x-amz-date': stamp})
        authorization = ('AWS4-HMAC-SHA256 Credential=' + config['key_id'] + '/' + stamp[:8]
                         + '/auto/s3/aws4_request, SignedHeaders=' + ';'.join(sorted(signed))
                         + ', Signature=' + signature(config['secret'], stamp, method, path, params,
                                                      signed, payload_hash))
        sent = {name: value for name, value in signed.items() if name != 'host'}
        sent.update({'authorization': authorization, 'content-type': 'application/octet-stream'})
        url = 'https://' + config['host'] + quote(path, safe='/-_.~')
        if params:
            url += '?' + canonical_query(params)
        message = urllib.request.Request(url, method=method, headers=sent,
                                         data=payload if method in ('PUT', 'POST') else None)
        try:
            with urllib.request.urlopen(message, timeout=timeout) as response:
                return response.headers, response.read()
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)


def stored(config, name, sha256):
    """Return the stored size when R2 already holds this exact archive, otherwise None."""
    try:
        headers, _ = request(config, 'HEAD', object_key(sha256, name), timeout=30)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    size = int(headers.get('Content-Length', '0'))
    return size if size > 0 and headers.get('x-amz-meta-sha256') == sha256 else None


def upload(config, path, sha256, name):
    """Upload one archive under its digest and confirm that R2 reports the same object."""
    path = Path(path)
    key = object_key(sha256, name)
    size = path.stat().st_size
    metadata = {'x-amz-meta-sha256': sha256}
    if size <= PART_BYTES:
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha256:
            raise RuntimeError('The archive differs from its frozen SHA-256 digest.')
        request(config, 'PUT', key, headers=metadata, body=payload, timeout=600)
    else:
        _, created = request(config, 'POST', key, params={'uploads': ''}, headers=metadata)
        upload_id = next(element.text for element in ElementTree.fromstring(created).iter()
                         if element.tag.endswith('UploadId'))
        try:
            def part(number, payload):
                headers, _ = request(config, 'PUT', key, body=payload, timeout=600,
                                     params={'partNumber': str(number), 'uploadId': upload_id})
                return number, headers['ETag']

            checksum, pending, parts = hashlib.sha256(), [], []
            with path.open('rb') as source, ThreadPoolExecutor(UPLOAD_WORKERS) as pool:
                for number in range(1, -(-size // PART_BYTES) + 1):
                    if len(pending) >= UPLOAD_WORKERS:
                        parts.append(pending.pop(0).result())
                    payload = source.read(PART_BYTES)
                    checksum.update(payload)
                    pending.append(pool.submit(part, number, payload))
                parts.extend(future.result() for future in pending)
                if source.read(1) or checksum.hexdigest() != sha256:
                    raise RuntimeError('The archive differs from its frozen SHA-256 digest.')
            body = ('<CompleteMultipartUpload>' + ''.join(
                '<Part><PartNumber>%d</PartNumber><ETag>%s</ETag></Part>' % (number, escape(etag))
                for number, etag in parts) + '</CompleteMultipartUpload>').encode()
            _, completed = request(config, 'POST', key, params={'uploadId': upload_id}, body=body, timeout=600)
            if b'<Error>' in completed:
                raise RuntimeError('R2 did not complete the multipart upload.')
        except BaseException:
            try:
                request(config, 'DELETE', key, params={'uploadId': upload_id}, timeout=30)
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            raise
    if stored(config, name, sha256) != size:
        raise RuntimeError('R2 does not report the uploaded archive with its digest.')
    return size


def fetch(connection, config, name, sha256, size, unpack, interpreter, *, timeout, expires,
          workers=FETCH_WORKERS, chunk_bytes=FETCH_CHUNK_BYTES):
    """Have the rental download one archive with parallel ranges straight into its extractor."""
    # The presigned URL travels on standard input, so it never appears in a command line or log.
    settings_text = json.dumps(dict(url=presign(config, object_key(sha256, name), expires), bytes=size,
                                    sha256=sha256, workers=workers, chunk_bytes=chunk_bytes))
    command = ('set -euo pipefail; ' + shlex.quote('autoresearch/' + interpreter)
               + ' autoresearch/.bootstrap/r2_fetch.py | (' + unpack + ')')
    started = time.monotonic()
    try:
        result = subprocess.run(connection + ['bash -c ' + shlex.quote(command)], input=settings_text.encode(),
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise TimeoutError('The R2 archive fetch exceeded its time budget.') from None
    lines = result.stderr.decode(errors='replace').splitlines()
    receipts = [json.loads(line[len(RECEIPT_PREFIX):]) for line in lines if line.startswith(RECEIPT_PREFIX)]
    if result.returncode or len(receipts) != 1:
        detail = next((line for line in reversed(lines) if line.strip()), 'no diagnostics')
        raise RuntimeError('The R2 fetch of ' + name + ' failed: ' + detail[:300])
    receipt = receipts[0]
    if receipt.get('sha256') != sha256 or receipt.get('bytes') != size:
        raise RuntimeError('The rental reported a different archive than the frozen digest.')
    return dict(archive=name, sha256=sha256, bytes=size, source='r2', connections=receipt['connections'],
                elapsed_seconds=time.monotonic() - started)
