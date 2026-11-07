"""The jobs of this loop: where each one lives, how it is saved, and the time a run of it needs on its machine."""

import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gpu_config import profile_for_job

SETUP_SECONDS = 180  # Unpacking the data and compiling, after the transfer and before the training.
COLLECTION_SECONDS = 120  # Bringing every result file back, the checkpoint included.
RETURN_SECONDS = 60  # Giving the machine back before vast-rental's own deadline for it.


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def job_path(identifier):
    if not re.fullmatch(r'[a-z0-9-]+', identifier):
        raise ValueError('Invalid experiment identifier.')
    return ROOT / 'queue' / identifier / 'job.json'


def save(job):
    """Merge the fields into the job's saved state."""
    path = job_path(job['id'])
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = json.loads(path.read_text()) if path.exists() else {}
        current.update(job)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(current, indent=2) + '\n')
        temporary.replace(path)


def load(identifier):
    return json.loads(job_path(identifier).read_text())


def training_seconds(job):
    return job['gpu_settings']['training_seconds']


def rental_seconds(job):
    return job['gpu_settings']['rental_seconds']


def machine_seconds(job):
    """The time a run needs on its machine once the runtime is there: setup, training, the full evaluation and
    collection."""
    evaluation = float(job['expected_eval_seconds'])
    if not math.isfinite(evaluation) or evaluation <= 0:
        raise ValueError('Reserve a positive finite duration for the evaluation.')
    return SETUP_SECONDS + training_seconds(job) + evaluation + COLLECTION_SECONDS


def check_fit(job):
    """Raise unless a run of the job fits in its rental with time left to give the machine back."""
    if machine_seconds(job) + RETURN_SECONDS >= rental_seconds(job):
        raise ValueError('The run does not fit inside its rental.')


def artifact_directory(job):
    return profile_for_job(job).artifacts_directory(ROOT)


def results_directory(job):
    return profile_for_job(job).results_directory(ROOT)
