"""Frozen 24 audio-only candidate descriptors; no labels or fitted transforms."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from drone.keys import key_relation

EPSILON = 1e-8
CHUNK_FRAMES = 2048
FEATURE_NAMES = (
    *(f"relative_chroma_{i}" for i in range(12)),
    "bass_tonic",
    "bass_third",
    "bass_fifth",
    *(f"quarter_contrast_{i}" for i in range(4)),
    "ending_bass_tonic",
    "candidate_minor",
    "halfbin_phase_proxy",
    "relative_margin_mean",
    "relative_margin_std",
)
FEATURE_SPEC = "candidate-tonal24-cqt24-linear-frame-l1-v1"


def relative_key(candidate: int) -> int:
    if not 0 <= candidate < 24:
        raise ValueError("Invalid candidate key")
    return next(other for other in range(24) if key_relation(other, candidate) == "relative")


def quarter_bounds(frames: int) -> list[tuple[int, int]]:
    if frames < 8:
        raise ValueError("Need at least8 frames")
    quotient, remainder = divmod(frames, 4)
    bounds, start = [], 0
    for quarter in range(4):
        end = start + quotient + int(quarter < remainder)
        bounds.append((start, end))
        start = end
    return bounds


def chroma(magnitude: np.ndarray) -> np.ndarray:
    smoothed = 0.5 * magnitude[::2] + 0.25 * (
        np.roll(magnitude, 1, axis=0)[::2] + np.roll(magnitude, -1, axis=0)[::2]
    )
    return smoothed / (smoothed.sum(axis=0, keepdims=True) + EPSILON)


def contrast(profile: np.ndarray, candidate: int) -> float:
    tonic, minor = divmod(candidate, 2)
    scale = (0, 2, 3, 5, 7, 8, 10) if minor else (0, 2, 4, 5, 7, 9, 11)
    off_scale = [(tonic + interval) % 12 for interval in range(12) if interval not in scale]
    triad = [(tonic + interval) % 12 for interval in (0, 3 if minor else 4, 7)]
    return float(profile[triad].mean() - profile[off_scale].mean())


def candidate_features(
    core: np.ndarray,
    *,
    check: Callable[[], None] | None = None,
    chunk_frames: int = CHUNK_FRAMES,
) -> np.ndarray:
    """Return rows in canonical tonic*2+minor order, never consulting a target."""
    if core.ndim != 2 or core.shape[0] != 192 or core.shape[1] < 8 or chunk_frames < 1:
        raise ValueError("Need192 core CQT bins, time>=8 and positive chunk size")
    frames = core.shape[1]
    bounds = quarter_bounds(frames)
    ending_start = frames - (frames + 7) // 8
    whole, bass, ending = (np.zeros(12, dtype=np.float64) for _ in range(3))
    quarters = np.zeros((4, 12), dtype=np.float64)
    phase = 0.0
    for start in range(0, frames, chunk_frames):
        if check is not None:
            check()
        end = min(frames, start + chunk_frames)
        block = np.asarray(core[:, start:end], dtype=np.float32)
        if not np.isfinite(block).all() or (block < 0).any():
            raise ValueError("Invalid nonnegative finite log1p CQT")
        with np.errstate(over="ignore"):
            magnitude = np.expm1(block).reshape(8, 24, end - start)
        if not np.isfinite(magnitude).all():
            raise ValueError("Magnitude reconstruction overflow")
        full = magnitude.sum(axis=0)
        pc, low = chroma(full), chroma(magnitude[:3].sum(axis=0))
        whole += pc.sum(axis=1, dtype=np.float64)
        bass += low.sum(axis=1, dtype=np.float64)
        phase += float(
            ((full[::2].sum(0) - full[1::2].sum(0)) / (full.sum(0) + EPSILON)).sum(dtype=np.float64)
        )
        for quarter, (first, last) in enumerate(bounds):
            a, b = max(first, start), min(last, end)
            if a < b:
                quarters[quarter] += pc[:, a - start : b - start].sum(axis=1, dtype=np.float64)
        a = max(ending_start, start)
        if a < end:
            ending += low[:, a - start :].sum(axis=1, dtype=np.float64)
    whole /= frames
    bass /= frames
    ending /= frames - ending_start
    quarters /= np.array([last - first for first, last in bounds])[:, None]
    result = np.empty((24, 24), dtype=np.float32)
    for candidate in range(24):
        tonic, minor = divmod(candidate, 2)
        q = np.array([contrast(profile, candidate) for profile in quarters])
        relative = np.array([contrast(profile, relative_key(candidate)) for profile in quarters])
        margin = q - relative
        result[candidate] = [
            *np.roll(whole, -tonic),
            *(bass[(tonic + interval) % 12] for interval in (0, 3 if minor else 4, 7)),
            *q,
            ending[tonic],
            minor,
            phase / frames,
            margin.mean(),
            margin.std(),
        ]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite candidate descriptors")
    return result


def grouped_targets(refs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if (
        refs.ndim != 1
        or not len(refs)
        or refs.dtype.kind not in "iu"
        or ((refs < 0) | (refs >= 24)).any()
    ):
        raise ValueError("Need one canonical integer reference per track")
    targets = (np.arange(24)[None, :] == refs[:, None]).astype(np.float32).reshape(-1)
    groups = np.repeat(np.arange(len(refs), dtype=np.int64), 24)
    return targets, groups
