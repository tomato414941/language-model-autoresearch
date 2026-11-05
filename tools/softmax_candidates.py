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


def build_candidates(targets, vocabulary, width, offset=0, stride=20011):
    """Include every true target, fill with rotating negatives, or use the full vocabulary."""
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
    positives = np.unique(targets[valid])
    fallback = len(positives) > width
    next_offset = offset % vocabulary
    if fallback or width == vocabulary:
        candidates = np.arange(vocabulary, dtype=np.int64)
    else:
        marked = np.zeros(vocabulary, dtype=bool)
        marked[positives] = True
        needed = width - len(positives)
        if needed:
            order = ((np.arange(vocabulary, dtype=np.int64) + next_offset) % vocabulary)
            sweep = (order * stride) % vocabulary
            eligible_positions = np.flatnonzero(~marked[sweep])[:needed]
            negatives = sweep[eligible_positions]
            next_offset = (next_offset + int(eligible_positions[-1]) + 1) % vocabulary
            candidates = np.sort(np.concatenate((positives, negatives))).astype(np.int64)
        else:
            candidates = positives.astype(np.int64, copy=True)
    positions = np.full(vocabulary, -1, dtype=np.int64)
    positions[candidates] = np.arange(len(candidates))
    mapped = np.full(targets.shape, -1, dtype=np.int64)
    mapped[valid] = positions[targets[valid]]
    if np.any(mapped[valid] < 0):
        raise RuntimeError("Every true target must be present in the candidate set.")
    return CandidateSet(candidates, mapped, len(positives), fallback, next_offset)
