"""Prepare target-preserving candidate sets on CPU for experimental training losses."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class CandidateSet:
    token_ids: np.ndarray
    target_positions: np.ndarray
    distinct_targets: int
    fallback: bool
    next_offset: int


def _validate(targets, vocabulary, width, stride):
    if vocabulary <= 0 or not 0 < width <= vocabulary:
        raise ValueError("The candidate width must lie within a positive vocabulary.")
    if math.gcd(stride, vocabulary) != 1:
        raise ValueError("The negative stride must be coprime with the vocabulary.")
    targets = np.asarray(targets)
    if not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("Targets must be integer token IDs.")
    valid = targets != -1
    if np.any(targets[valid] < 0) or np.any(targets[valid] >= vocabulary):
        raise ValueError("Targets must be vocabulary IDs or the ignored-target value -1.")
    return targets, valid


def build_candidates(targets, vocabulary, width, offset=0, stride=20011):
    """Include every true target, fill with rotating negatives, or use the full vocabulary."""
    targets, valid = _validate(targets, vocabulary, width, stride)
    marked = np.zeros(vocabulary, dtype=bool)
    marked[targets[valid]] = True
    count = int(np.count_nonzero(marked))
    fallback = count > width
    next_offset = offset % vocabulary
    if fallback or width == vocabulary:
        candidates = np.arange(vocabulary, dtype=np.int64)
    else:
        sweep = (np.arange(vocabulary, dtype=np.int64) * stride) % vocabulary
        candidates, next_offset = _fill_candidates(
            marked, width - count, next_offset, np.concatenate((sweep, sweep)))
    positions = np.full(vocabulary, -1, dtype=np.int64)
    positions[candidates] = np.arange(len(candidates))
    mapped = np.full(targets.shape, -1, dtype=np.int64)
    mapped[valid] = positions[targets[valid]]
    if np.any(mapped[valid] < 0):
        raise RuntimeError("Every true target must be present in the candidate set.")
    return CandidateSet(candidates, mapped, count, fallback, next_offset)


def _fill_candidates(marked, needed, offset, doubled_sweep):
    """Fill one target bitmap in the declared circular negative order."""
    if needed:
        sweep = doubled_sweep[offset:offset + len(marked)]
        eligible_positions = np.flatnonzero(~marked[sweep])[:needed]
        marked[sweep[eligible_positions]] = True
        offset = (offset + int(eligible_positions[-1]) + 1) % len(marked)
    return np.flatnonzero(marked), offset


@dataclass
class CandidateBatch:
    token_ids: np.ndarray | None
    target_positions: np.ndarray | None
    distinct_targets: np.ndarray
    fallback: bool
    next_offset: int


def build_candidate_batch(targets, vocabulary, width, chunk_size, offset=0, stride=20011):
    """Prepare uniform-width chunks, using the original full loss for an overflowing batch."""
    targets, valid = _validate(targets, vocabulary, width, stride)
    if chunk_size <= 0 or targets.size == 0:
        raise ValueError("A batch needs targets and a positive head chunk size.")
    flat = targets.reshape(-1)
    flat_valid = valid.reshape(-1)
    starts = range(0, flat.size, chunk_size)
    marked = np.zeros((len(starts), vocabulary), dtype=bool)
    for index, start in enumerate(starts):
        chunk = flat[start:start + chunk_size]
        chunk_valid = flat_valid[start:start + chunk_size]
        marked[index, chunk[chunk_valid]] = True
    counts = np.count_nonzero(marked, axis=1).astype(np.int64)
    fallback = bool(np.any(counts > width))
    if fallback or width == vocabulary:
        return CandidateBatch(None, None, counts, fallback, offset % vocabulary)
    sweep = (np.arange(vocabulary, dtype=np.int64) * stride) % vocabulary
    doubled_sweep = np.concatenate((sweep, sweep))
    offset %= vocabulary
    ids = np.empty((len(starts), width), dtype=np.int64)
    positions = np.full(flat.shape, -1, dtype=np.int64)
    lookup = np.full(vocabulary, -1, dtype=np.int64)
    for index, start in enumerate(starts):
        candidates, offset = _fill_candidates(marked[index], width - int(counts[index]),
                                              offset, doubled_sweep)
        ids[index] = candidates
        lookup.fill(-1)
        lookup[candidates] = np.arange(width)
        chunk = flat[start:start + chunk_size]
        chunk_valid = flat_valid[start:start + chunk_size]
        positions[start:start + chunk_size][chunk_valid] = lookup[chunk[chunk_valid]]
    if np.any(positions[flat_valid] < 0):
        raise RuntimeError("Every true target must be present in the candidate set.")
    return CandidateBatch(ids, positions.reshape(targets.shape), counts, False, offset)
