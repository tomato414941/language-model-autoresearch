"""Observations of a run that never change its course: GPU samples while it trains, and its preparation time."""

import csv
import json
import math
import re
import subprocess
import time


def record_gpu_sample(remote, destination):
    """Append one sample of the GPU's memory, utilization, clocks, power and temperature; a sample that cannot be
    taken is recorded as such."""
    observation = dict(observed_at=time.time())
    sample_deadline = time.monotonic() + 5
    try:
        try:
            result = remote(
                'nvidia-smi --query-gpu=name,memory.used,utilization.gpu,clocks.sm,clocks.max.sm,'
                'clocks.mem,power.draw,power.limit,temperature.gpu --format=csv,noheader,nounits',
                timeout=5, capture_output=True, text=True,
            )
        except (subprocess.SubprocessError, OSError) as error:
            remaining = sample_deadline - time.monotonic()
            if remaining <= 0:
                raise
            observation['extended_error_type'] = type(error).__name__
            result = remote(
                'nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader,nounits',
                timeout=remaining, capture_output=True, text=True,
            )
        rows = list(csv.reader(result.stdout.splitlines()))
        if len(rows) != 1 or len(rows[0]) not in (3, 9):
            raise ValueError('Expected one GPU observation.')
        name, memory, utilization = rows[0][:3]
        memory, utilization = float(memory), float(utilization)
        if not (math.isfinite(memory) and memory >= 0 and 0 <= utilization <= 100):
            raise ValueError('Invalid GPU observation.')
        observation.update(gpu_name=name.strip(), memory_mib=memory, utilization_percent=utilization)
        optional_fields = ('sm_clock_mhz', 'max_sm_clock_mhz', 'memory_clock_mhz',
                           'power_draw_w', 'power_limit_w', 'temperature_c')
        for index, field in enumerate(optional_fields, start=3):
            try:
                value = float(rows[0][index])
                value = value if math.isfinite(value) and value >= 0 else None
            except (IndexError, ValueError):
                value = None
            observation[field] = value
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        # Optional telemetry never changes the training lifecycle or decision.
        observation['error_type'] = type(error).__name__
    try:
        with (destination / 'gpu-observations.jsonl').open('a') as output:
            output.write(json.dumps(observation, allow_nan=False) + '\n')
    except OSError:
        print('The GPU observation could not be saved; retaining the original run.', flush=True)


def record_preparation_time(remote, destination):
    """Record roughly how long the machine took to prepare the run, from its files' modification times."""
    observation = dict(observed_at=time.time(), files={},
                       time_resolution_seconds=1,
                       limitation='GPU file modification times approximate wrapper and preparation time, '
                       'including Python/uv startup, data checks/downloads and tokenizer preparation. '
                       'This is not a measurement of network transfer alone. The training log mtime '
                       'does not identify the start of training.')
    try:
        result = remote(
            'cd autoresearch && for timing_file in run.pid prepare.log run.log run.exit; do '
            'if test -f "$timing_file"; then stat --printf=\'%n\\t%Y\\t%s\\n\' "$timing_file"; fi; done',
            timeout=5, capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            name, modified, size = line.split('\t')
            if name not in ('run.pid', 'prepare.log', 'run.log', 'run.exit'):
                raise ValueError('Unexpected preparation timing file.')
            modified, size = int(modified), int(size)
            if modified < 0 or size < 0 or name in observation['files']:
                raise ValueError('Invalid preparation file metadata.')
            observation['files'][name] = dict(mtime_seconds=modified, bytes=size)
        try:
            with (destination / 'prepare.log').open('rb') as source:
                source.seek(0, 2)
                source.seek(max(0, source.tell() - 4096))
                tail = source.read()
        except OSError:
            tail = b''
        observation['prepare_log_completed_marker'] = b'Done! Ready to train.' in tail
        ready = re.search(rb'Data: (\d+)/(\d+) shards ready', tail)
        cached = re.search(rb'Data: all (\d+) shards already downloaded', tail)
        if ready:
            observation['ready_shards'], observation['requested_shards'] = map(int, ready.groups())
        elif cached:
            observation['ready_shards'] = observation['requested_shards'] = int(cached.group(1))
        files = observation['files']
        if observation['prepare_log_completed_marker'] and all(name in files for name in ('run.pid', 'prepare.log')):
            elapsed = files['prepare.log']['mtime_seconds'] - files['run.pid']['mtime_seconds']
            if elapsed < 0:
                raise ValueError('The preparation file times are out of order.')
            observation['approximate_preparation_seconds'] = elapsed
    except (subprocess.SubprocessError, OSError, TimeoutError, ValueError, TypeError) as error:
        # Optional timing never changes the training lifecycle or decision.
        observation['error_type'] = type(error).__name__
    try:
        (destination / 'preparation-timing.json').write_text(json.dumps(observation, indent=2, allow_nan=False) + '\n')
    except OSError:
        print('The preparation timing could not be saved; retaining the original run.', flush=True)
