"""Label-independent waveform sampling built on the immutable K10 ID streams."""

from __future__ import annotations

import hashlib
import random
from collections import deque
from collections.abc import Iterator

from experiments.k10_v2_pilot import Draw, paired_batches
from experiments.musicfm_key import MIN_SAMPLES, WINDOW_SAMPLES


def seed(purpose: str) -> int:
    return int.from_bytes(
        hashlib.sha256(("musicfm-waveform-v1:" + purpose).encode()).digest(), "big"
    )


def batches(bp: list[str], fma: list[str], updates: int = 8280) -> Iterator[list[Draw]]:
    """Shuffle domain positions within each update, preserving each ID subsequence."""
    if type(updates) is not int or updates <= 0:
        raise ValueError("A positive integer update count is required")
    for step, original in enumerate(paired_batches(bp, fma, updates=updates)):
        positions = [draw.domain for draw in original]
        streams = {
            domain: deque(d for d in original if d.domain == domain)
            for domain in ("beatport", "fmak")
        }
        random.Random(seed(f"interleave:13:{step}")).shuffle(positions)
        yield [streams[domain].popleft() for domain in positions]


def crop(total_samples: int, draw: Draw) -> tuple[int, int]:
    """Uniform legal sample start over the complete song, independent of I/O order."""
    if type(total_samples) is not int or total_samples <= MIN_SAMPLES:
        raise ValueError("Actual waveform length must exceed 1024 samples")
    length = min(WINDOW_SAMPLES, total_samples)
    rng = random.Random(seed(f"crop:{draw.augmentation_seed}"))
    return rng.randrange(total_samples - length + 1), length
