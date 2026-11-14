"""Bounded mono offline R3 adapter for the pinned RubberBand 3.3 C ABI."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def shift_r3(
    signal: Tensor, shift: int, library: str
) -> tuple[Tensor, dict[str, int | float | bool]]:
    """Shift a complete mono waveform without normalization or implicit tail repair."""
    import numpy as np
    import torch

    if signal.ndim != 1 or signal.device.type != "cpu" or signal.dtype != torch.float32:
        raise ValueError("R3 input must be mono CPU float32")
    samples = signal.numel()
    if not 0 < samples < 2**32 or shift not in range(-6, 7) or not torch.isfinite(signal).all():
        raise ValueError("Invalid declared pitch input")
    if shift == 0:
        return signal.clone(), {
            "identity_bypass": True,
            "input_samples": samples,
            "output_samples": samples,
        }

    rubberband_api = ctypes.CDLL(library)
    pointer = ctypes.POINTER(ctypes.c_float)
    channels = ctypes.POINTER(pointer)
    state_type = ctypes.c_void_p
    rubberband_api.rubberband_new.argtypes = [
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
    ]
    rubberband_api.rubberband_new.restype = state_type
    rubberband_api.rubberband_delete.argtypes = [state_type]
    rubberband_api.rubberband_delete.restype = None
    rubberband_api.rubberband_get_engine_version.argtypes = [state_type]
    rubberband_api.rubberband_get_engine_version.restype = ctypes.c_int
    rubberband_api.rubberband_get_process_size_limit.argtypes = [state_type]
    rubberband_api.rubberband_get_process_size_limit.restype = ctypes.c_uint
    for name in ("rubberband_set_expected_input_duration", "rubberband_set_max_process_size"):
        function = getattr(rubberband_api, name)
        function.argtypes = [state_type, ctypes.c_uint]
        function.restype = None
    for name in ("rubberband_study", "rubberband_process"):
        function = getattr(rubberband_api, name)
        function.argtypes = [state_type, channels, ctypes.c_uint, ctypes.c_int]
        function.restype = None
    rubberband_api.rubberband_available.argtypes = [state_type]
    rubberband_api.rubberband_available.restype = ctypes.c_int
    rubberband_api.rubberband_retrieve.argtypes = [state_type, channels, ctypes.c_uint]
    rubberband_api.rubberband_retrieve.restype = ctypes.c_uint

    # Offline is zero. EngineFiner selects R3; ThreadingNever bounds worker CPUs.
    # PitchHighQuality is deliberately omitted: upstream documents it as ignored offline.
    options = 0x20000000 | 0x00010000
    state = rubberband_api.rubberband_new(16000, 1, options, 1.0, 2 ** (shift / 12))
    if not state:
        raise RuntimeError("RubberBand allocation failed")
    array = np.ascontiguousarray(signal.numpy())
    outputs = []
    output_samples = 0
    finished = False
    block_size = 4096
    try:
        engine = rubberband_api.rubberband_get_engine_version(state)
        if engine != 3 or block_size > rubberband_api.rubberband_get_process_size_limit(state):
            raise ValueError("Unexpected engine or process-size limit")
        rubberband_api.rubberband_set_expected_input_duration(state, samples)
        rubberband_api.rubberband_set_max_process_size(state, block_size)
        for start in range(0, samples, block_size):
            block = array[start : start + block_size]
            pointers = (pointer * 1)(block.ctypes.data_as(pointer))
            rubberband_api.rubberband_study(
                state, pointers, block.size, int(start + block.size == samples)
            )
        for start in range(0, samples, block_size):
            if finished:
                raise RuntimeError("R3 declared completion before final input")
            block = array[start : start + block_size]
            pointers = (pointer * 1)(block.ctypes.data_as(pointer))
            final = start + block.size == samples
            rubberband_api.rubberband_process(state, pointers, block.size, int(final))
            while True:
                available = rubberband_api.rubberband_available(state)
                if available == -1:
                    finished = True
                    break
                if available < -1:
                    raise RuntimeError("Invalid R3 availability")
                if available == 0:
                    if final:
                        raise RuntimeError("R3 final input did not drain synchronously")
                    break
                count = min(available, block_size)
                output = np.empty(count, dtype=np.float32)
                pointers = (pointer * 1)(output.ctypes.data_as(pointer))
                received = rubberband_api.rubberband_retrieve(state, pointers, count)
                if received != count:
                    raise RuntimeError("R3 retrieve count differs from availability")
                output_samples += received
                if output_samples > samples + 16000:
                    raise RuntimeError("R3 output exceeded duration safety bound")
                outputs.append(output)
        if not finished or not outputs:
            raise RuntimeError("Incomplete R3 output")
        result = torch.from_numpy(np.concatenate(outputs))
        if not torch.isfinite(result).all():
            raise RuntimeError("Nonfinite R3 output")
        return result, {
            "engine": engine,
            "input_samples": samples,
            "output_samples": output_samples,
            "process_block_samples": block_size,
            "time_ratio": 1.0,
            "pitch_ratio": 2 ** (shift / 12),
            "options": options,
            "drained": finished,
            "normalized": False,
        }
    finally:
        rubberband_api.rubberband_delete(state)
