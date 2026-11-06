"""Stream a verified archive over independent SSH connections into its extractor."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import math
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time
import uuid

WORKERS = 4
CHUNK_BYTES = 64 * 1024 * 1024


def stream_archive(connection, path, unpack, *, sha256, timeout, workers=WORKERS, chunk_bytes=CHUNK_BYTES):
    """Bound memory, verify each received chunk, and overlap transport with extraction."""
    path = Path(path)
    size = path.stat().st_size
    if not re.fullmatch('[0-9a-f]{64}', sha256):
        raise ValueError('An archive must have a frozen SHA-256 digest.')
    if size <= 0 or workers < 1 or chunk_bytes < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Use a nonempty archive and positive finite transfer limits.')
    count = math.ceil(size / chunk_bytes)
    started = time.monotonic()
    deadline = started + timeout
    directory = 'autoresearch/.transfer-' + uuid.uuid4().hex
    stopped = threading.Event()
    lock = threading.Lock()
    processes = set()

    def invoke(command, payload=None):
        remaining = deadline - time.monotonic()
        if stopped.is_set() or remaining <= 0:
            raise TimeoutError('The archive transfer budget is exhausted.')
        process = subprocess.Popen(connection + [command],
                                   stdin=subprocess.PIPE if payload is not None else subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with lock:
            processes.add(process)
            if stopped.is_set():
                process.kill()
        try:
            try:
                output, errors = process.communicate(input=payload, timeout=remaining)
            except BaseException:
                process.kill()
                process.communicate()
                raise
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, 'SSH archive transfer', output, errors)
        finally:
            with lock:
                processes.discard(process)

    def upload(index, payload):
        # Delay acknowledgements to cap queued payloads at twice the worker count.
        name = directory + '/' + f'{index:08d}'
        temporary = name + '.tmp'
        checksum = hashlib.sha256(payload).hexdigest()
        command = ('set -e; cat > ' + temporary + '; '
                   + f'test "$(wc -c < {temporary})" -eq {len(payload)}; '
                   + 'printf ' + shlex.quote(checksum + '  ' + temporary + '\n')
                   + ' | sha256sum -c - >/dev/null; mv ' + temporary + ' ' + name + '; '
                   + 'until test ' + str(index) + ' -lt $(( $(cat ' + directory + '/consumed) + '
                   + str(workers) + ' )); do test ! -e ' + directory + '/abort || exit 125; sleep 0.05; done')
        invoke(command, payload)

    invoke('mkdir -p ' + directory + '; printf "0\\n" > ' + directory + '/consumed')
    # The receiver consumes ordered, verified chunks as soon as they arrive.
    receiver = (
        'set -euo pipefail; end=$((SECONDS + ' + str(math.ceil(timeout)) + ')); '
        + '(for ((i=0; i<' + str(count) + '; i++)); do '
        + 'printf -v part "' + directory + '/%08d" "$i"; '
        + 'until test -f "$part"; do '
        + 'test ! -e ' + directory + '/abort || exit 125; '
        + 'test "$SECONDS" -lt "$end" || exit 124; sleep 0.05; done; '
        + 'cat "$part"; rm "$part"; printf "%s\\n" "$((i+1))" > ' + directory + '/consumed.tmp; '
        + 'mv ' + directory + '/consumed.tmp ' + directory + '/consumed; done) | (' + unpack + ')'
    )
    pool = ThreadPoolExecutor(max_workers=workers + 1)
    received = pool.submit(invoke, 'bash -c ' + shlex.quote(receiver))
    pending = []
    checksum = hashlib.sha256()
    sent = 0

    def uploaded(future):
        candidates = [future, received, *pending]
        while True:
            done, _ = wait(candidates, return_when=FIRST_COMPLETED)
            for completed in done:
                completed.result()
            if future.done():
                return
            candidates = [candidate for candidate in candidates if not candidate.done()]

    try:
        with path.open('rb') as source:
            for index in range(count):
                if len(pending) >= workers:
                    uploaded(pending.pop(0))
                if received.done():
                    received.result()
                    raise RuntimeError('The extractor ended before the archive transfer.')
                payload = source.read(chunk_bytes)
                if not payload:
                    raise RuntimeError('The archive shrank during its transfer.')
                checksum.update(payload)
                sent += len(payload)
                pending.append(pool.submit(upload, index, payload))
            if source.read(1):
                raise RuntimeError('The archive grew during its transfer.')
        if sent != size or checksum.hexdigest() != sha256:
            raise RuntimeError('The archive differs from its frozen SHA-256 digest.')
        for future in pending:
            uploaded(future)
        received.result()
        invoke('rm -rf ' + directory)
    except BaseException:
        stopped.set()
        with lock:
            for process in processes:
                process.kill()
        # A disconnected receiver also has a bounded deadline; signal it to stop waiting now.
        try:
            subprocess.run(connection + ['touch ' + directory + '/abort'],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=5, check=False)
        except (OSError, subprocess.SubprocessError):
            pass
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return dict(archive=path.name, sha256=sha256, bytes=sent, chunks=count,
                payload_connections=min(workers, count), elapsed_seconds=time.monotonic() - started)
