"""Prospective whole-song evaluation primitives; no model fitting or scoring entrypoint.

MusicFM uses local attention context per waveform window. Duration-weighted
window logits cover the complete song but are not whole-song attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

SAMPLE_RATE = 24_000
WINDOW_SAMPLES = 12 * SAMPLE_RATE
MIN_SAMPLES = 1_024  # Original reflected STFT padding needs length > 1024.
CLASSES = 24


@dataclass(frozen=True)
class SongWindow:
    """Input context and its disjoint real-song aggregation support, in samples."""

    start: int
    stop: int
    support_start: int
    support_stop: int

    @property
    def weight(self) -> int:
        return self.support_stop - self.support_start


def song_windows(samples: int) -> tuple[SongWindow, ...]:
    """Cover all samples without padding, dropped tails or shortened songs.

    Full windows are disjoint 12-second slices. A tail longer than 1024 samples
    uses its own actual input. A shorter tail borrows the final 12 seconds as
    context, while contributing only its newly covered samples as weight.
    """
    if type(samples) is not int or samples <= MIN_SAMPLES:
        raise ValueError("A song needs an integer sample count greater than 1024")
    if samples < WINDOW_SAMPLES:
        return (SongWindow(0, samples, 0, samples),)
    count, tail = divmod(samples, WINDOW_SAMPLES)
    windows = [
        SongWindow(
            i * WINDOW_SAMPLES,
            (i + 1) * WINDOW_SAMPLES,
            i * WINDOW_SAMPLES,
            (i + 1) * WINDOW_SAMPLES,
        )
        for i in range(count)
    ]
    if tail:
        support_start = count * WINDOW_SAMPLES
        start = support_start if tail > MIN_SAMPLES else samples - WINDOW_SAMPLES
        windows.append(SongWindow(start, samples, support_start, samples))
    return tuple(windows)


class SongLogits:
    """Ordered streaming FP64 logit mean with complete-support verification.

    The caller supplies unnormalized 24-way logits in the shared Drone key
    order. No softmax, confidence weighting or best-window selection is used.
    """

    def __init__(self, samples: int) -> None:
        self.windows = song_windows(samples)
        self.samples = samples
        self._next = 0
        self._weighted: NDArray[np.float64] = np.zeros(CLASSES, dtype=np.float64)
        self._support = 0

    def add(self, window: SongWindow, logits: ArrayLike) -> None:
        if self._next >= len(self.windows) or window != self.windows[self._next]:
            raise ValueError("Expected the next fixed song window exactly once")
        values = np.asarray(logits, dtype=np.float64)
        if values.shape != (CLASSES,) or not np.isfinite(values).all():
            raise ValueError("Expected 24 finite unnormalized logits")
        candidate = self._weighted + values * window.weight
        if not np.isfinite(candidate).all():
            raise ValueError("Duration-weighted logits overflowed")
        self._weighted = candidate
        self._support += window.weight
        self._next += 1

    def finish(self) -> NDArray[np.float64]:
        if self._next != len(self.windows) or self._support != self.samples:
            raise ValueError("Cannot predict before the complete song is covered")
        return self._weighted / self.samples
