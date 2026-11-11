"""Explicit process units and repeated-prediction timing, independent of CLIs."""

from __future__ import annotations

import os
import platform
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def thread_environment() -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in THREAD_VARIABLES}


def cpu_affinity() -> list[int] | None:
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


def runtime_environment() -> dict[str, object]:
    import onnxruntime as ort

    return {
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "cpu_affinity": cpu_affinity(),
        "thread_environment": thread_environment(),
    }


def peak_rss_kib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024 if sys.platform == "darwin" else float(value)


def peak_rss_mib() -> float:
    return peak_rss_kib() / 1024


@dataclass(frozen=True)
class PredictionTiming:
    seconds: tuple[float, ...]
    logits: NDArray[np.float32]

    @property
    def median_seconds(self) -> float:
        return float(np.median(self.seconds))


def time_prediction(
    predict: Callable[[NDArray[np.float32]], np.ndarray],
    features: NDArray[np.float32],
    repetitions: int = 20,
) -> PredictionTiming:
    if repetitions < 1:
        raise ValueError("Require at least one timing repetition")
    predict(features)  # Exactly one warmup, matching the fixed benchmark protocol.
    elapsed = []
    logits = np.empty((1, 24), dtype=np.float32)
    for _ in range(repetitions):
        start = time.perf_counter()
        logits = predict(features)
        elapsed.append(time.perf_counter() - start)
    if logits.shape != (1, 24) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ValueError("Invalid timed prediction: expected finite FP32 [1,24] logits")
    return PredictionTiming(tuple(elapsed), logits)
