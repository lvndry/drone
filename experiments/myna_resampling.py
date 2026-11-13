"""Declared playback-rate augmentation: pitch and tempo change; preserve all source content."""

from __future__ import annotations

import importlib
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def geometry(samples: int, shift: int) -> dict[str, int | float | bool]:
    if samples < 1 or shift not in range(-6, 7):
        raise ValueError("Invalid declared playback-rate view")
    down = round(16000 * 2 ** (shift / 12))
    output = (samples * 16000 + down - 1) // down
    frames = output // 512 + 1
    return {
        "input_samples": samples,
        "output_samples": output,
        "up": 16000,
        "down": down,
        "actual_playback_rate": down / 16000,
        "pitch_rounding_error_cents": 1200 * math.log2(down / 16000) - shift * 100,
        "mel_frames": frames,
        "complete_windows": frames // 196,
        "discarded_frames": frames % 196,
        "length_fixed": False,
        "normalized": False,
    }


def shift_playback(signal: Tensor, shift: int) -> tuple[Tensor, dict[str, int | float | bool]]:
    import numpy as np
    import torch

    if signal.ndim != 1 or signal.device.type != "cpu" or signal.dtype != torch.float32:
        raise ValueError("Expected mono CPU float32 PCM")
    if not torch.isfinite(signal).all():
        raise ValueError("Nonfinite playback-rate input")
    metadata = geometry(signal.numel(), shift)
    if shift == 0:
        return signal.clone(), metadata
    resample_poly = importlib.import_module("scipy.signal").resample_poly
    array = resample_poly(
        signal.numpy(),
        16000,
        metadata["down"],
        window=("kaiser", 5.0),
        padtype="constant",
    )
    if array.dtype != np.float32 or len(array) != metadata["output_samples"]:
        raise ValueError("Playback-rate dtype/length contract differs")
    result = torch.from_numpy(np.ascontiguousarray(array))
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite playback-rate output")
    return result, metadata
