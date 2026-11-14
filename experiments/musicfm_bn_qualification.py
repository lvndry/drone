"""Bounded qualification and timing admission for musicfm-key-pilot/BN_PROTOCOL.md.

No loader, data access, scores, runner, guardian or execution authority. The caller
must bind the sealed fitting panel and historical timing provenance. The supplied
watch() must provide external resource supervision; cooperative clock checks here
cannot interrupt a blocked reader, transfer or forward.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import torch
from torch import Tensor, nn

from experiments.musicfm_bn import Clip, WeightedPopulation, recording_moments, recording_weight
from experiments.musicfm_bn_state import (
    BN_SITES,
    Watch,
    capture_bn_prefix,
    capture_bn_trace,
    expected_temporal_support,
    snapshot_state,
    validate_bn_inventory,
    verify_state_change,
)

Clock = Callable[[], float]


def _seconds(value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("Timing must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError("Timing must be positive" if positive else "Timing cannot be negative")
    return float(value)


def qualification_clips(clips: Sequence[Clip]) -> tuple[tuple[Clip, ...], tuple[Clip, ...]]:
    """Choose from the caller's sealed rank-ordered 256-BP then 256-FMA panel.

    Membership/rank provenance must be verified by the caller against its manifest.
    Length ties choose the first manifest row, without any score-dependent choice.
    """
    clips = tuple(clips)
    if len(clips) != 512 or len({clip.identity for clip in clips}) != 512:
        raise ValueError("Qualification requires the exact unique 512-recording panel")
    for index, clip in enumerate(clips):
        domain = "beatport" if index < 256 else "fmak"
        if (
            clip.domain != domain
            or not isinstance(clip.identity, str)
            or not clip.identity
            or not isinstance(clip.weight, Fraction)
            or clip.weight != recording_weight(domain)
            or any(type(value) is not int for value in (clip.samples, clip.start, clip.length))
            or not 0 <= clip.start <= clip.samples - clip.length
        ):
            raise ValueError("Panel order, exact weights, identity or crop bounds changed")
        expected_temporal_support(clip.length)
    correctness = list(
        dict.fromkeys((clips[0], clips[256], min(clips, key=lambda clip: clip.length)))
    )
    timing = list(clips[:8] + clips[256:264])
    longest = max(clips, key=lambda clip: clip.length)
    if longest not in timing:
        timing.append(longest)
    return tuple(correctness), tuple(timing)


class _Budget:
    """One shared, non-resetting 300-second cooperative qualification budget."""

    def __init__(self, clock: Clock, watch: Watch) -> None:
        self.clock = clock
        self.watch = watch
        self.started = _seconds(clock())
        self.last = self.started

    def now(self) -> float:
        current = _seconds(self.clock())
        if current < self.last:
            raise ValueError("Qualification clock moved backwards")
        self.last = current
        if current - self.started > 300:
            raise TimeoutError("Combined qualification exceeded 300 seconds")
        return current

    def check(self) -> None:
        self.now()
        self.watch()
        self.now()


def _read_crop(
    clip: Clip,
    read: Callable[[Clip], Tensor],
    hashes: dict[str, str],
    budget: _Budget,
) -> Tensor:
    budget.check()
    waveform = read(clip)
    budget.check()
    if (
        not isinstance(waveform, Tensor)
        or waveform.device.type != "cpu"
        or waveform.dtype != torch.float32
        or waveform.shape != (1, clip.length)
        or waveform.requires_grad
        or not bool(torch.isfinite(waveform).all())
    ):
        raise ValueError("Reader must supply a finite actual-length N1 FP32 CPU crop")
    digest = hashlib.sha256(waveform.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    if hashes.setdefault(clip.identity, digest) != digest:
        raise ValueError("Repeated qualification crop bytes changed")
    budget.check()
    return waveform


def _difference(reference: Tensor, candidate: Tensor) -> float:
    for value in (reference, candidate):
        if (
            value.device.type != "cpu"
            or value.dtype != torch.float32
            or value.requires_grad
            or value.numel() == 0
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("Qualification captures must be finite owned CPU FP32 tensors")
    if reference.shape != candidate.shape:
        raise ValueError("Full-trace and stopped-prefix shapes differ")
    difference = float((reference.double() - candidate.double()).abs().max().item())
    if difference > 1e-5:
        raise ValueError("Full-trace and stopped-prefix error exceeds 1e-5")
    return difference


@dataclass(frozen=True)
class Qualification:
    correctness_ids: tuple[str, ...]
    timing_ids: tuple[str, ...]
    full_traces: int
    correctness_prefixes: int
    timing_prefixes: int
    max_absolute_difference: float
    site_max_seconds: tuple[float, ...]
    elapsed_seconds: float
    crop_sha256: tuple[tuple[str, str], ...]


def qualify(
    model: nn.Module,
    clips: Sequence[Clip],
    read: Callable[[Clip], Tensor],
    watch: Watch,
    *,
    clock: Clock = time.monotonic,
) -> Qualification:
    """Check hooks and measure prefixes without changing registered model state.

    Only disposable eval/FP32 models are valid. No provisional moments are applied.
    Failure produces no successful qualification; discard any suspect model copy.
    """
    budget = _Budget(clock, watch)
    correctness, timing = qualification_clips(clips)
    validate_bn_inventory(model)
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    devices = {value.device for value in tensors}
    if (
        len(devices) != 1
        or any(module.training for module in model.modules())
        or any(
            value.is_complex() or (value.is_floating_point() and value.dtype != torch.float32)
            for value in tensors
        )
        or torch.is_autocast_enabled("cpu")
        or torch.is_autocast_enabled("cuda")
    ):
        raise ValueError("Qualification requires a one-device all-eval FP32 model without autocast")
    device = next(iter(devices))
    before = snapshot_state(model, budget.check)
    hashes: dict[str, str] = {}
    maximum_difference = 0.0
    with torch.inference_mode():
        for clip in correctness:
            waveform = _read_crop(clip, read, hashes, budget)
            budget.check()
            trace = capture_bn_trace(
                model, waveform.to(device=device), budget.check, waveform_samples=clip.length
            )
            if tuple(row.site for row in trace) != tuple(site.name for site in BN_SITES):
                raise ValueError("Qualification full trace has a different BN call order")
            for site, reference in zip(BN_SITES, trace, strict=True):
                waveform = _read_crop(clip, read, hashes, budget)
                budget.check()
                candidate = capture_bn_prefix(
                    model,
                    waveform.to(device=device),
                    site.name,
                    budget.check,
                    waveform_samples=clip.length,
                )
                maximum_difference = max(
                    maximum_difference, _difference(reference.value, candidate)
                )
                budget.check()
            verify_state_change(before, snapshot_state(model, budget.check))
            del trace
        maxima = []
        for site in BN_SITES:
            maximum = 0.0
            population = WeightedPopulation()
            for index, clip in enumerate(timing):
                began = budget.now()
                waveform = _read_crop(clip, read, hashes, budget)
                budget.check()
                value = capture_bn_prefix(
                    model,
                    waveform.to(device=device),
                    site.name,
                    budget.check,
                    waveform_samples=clip.length,
                )
                population.add(recording_moments(value.numpy()), clip.weight)
                if index == len(timing) - 1:
                    # Include finishing/storage conversion cost in the final measured clip.
                    moments = population.finish()
                    with np.errstate(over="raise", invalid="raise"):
                        moments.mean.astype(np.float32)
                        moments.variance.astype(np.float32)
                budget.check()
                maximum = max(maximum, _seconds(budget.now() - began, positive=True))
            maxima.append(maximum)
    verify_state_change(before, snapshot_state(model, budget.check))
    budget.check()
    return Qualification(
        tuple(clip.identity for clip in correctness),
        tuple(clip.identity for clip in timing),
        len(correctness),
        len(correctness) * 18,
        len(timing) * 18,
        maximum_difference,
        tuple(maxima),
        budget.now() - budget.started,
        tuple(sorted(hashes.items())),
    )


@dataclass(frozen=True)
class TimingAdmission:
    fits_budget: bool
    required_seconds: float
    remaining_seconds: float
    projected_calibration_seconds: float


def timing_admission(
    qualification: Qualification,
    *,
    remaining_wall_seconds: float,
    historical_complete_seconds: float,
    measured_overhead_seconds: float,
) -> TimingAdmission:
    """Apply the fixed expression once; timing evidence is not launch permission.

    historical_complete_seconds comes from the separately verified terminal receipt:
    fresh baseline B1 plus candidate B1/B16, i.e. 2*B1+B16 before this safety factor.
    Terminal/source extraction belongs to musicfm_bn_terminal, not this helper.
    """
    if (
        not 2 <= len(qualification.correctness_ids) <= 3
        or not 16 <= len(qualification.timing_ids) <= 17
        or qualification.full_traces != len(qualification.correctness_ids)
        or qualification.correctness_prefixes != 18 * qualification.full_traces
        or qualification.timing_prefixes != 18 * len(qualification.timing_ids)
        or len(qualification.site_max_seconds) != 18
        or not 0 <= qualification.max_absolute_difference <= 1e-5
        or not 0 <= qualification.elapsed_seconds <= 300
    ):
        raise ValueError("Incomplete or failed bounded qualification receipt")
    remaining = _seconds(remaining_wall_seconds)
    historical = _seconds(historical_complete_seconds, positive=True)
    overhead = _seconds(measured_overhead_seconds)
    try:
        projected = 512 * math.fsum(
            _seconds(value, positive=True) for value in qualification.site_max_seconds
        )
        required = 2 * projected + 2 * historical + overhead + 120
    except OverflowError as error:
        raise ValueError("Projected timing sum overflowed") from error
    _seconds(required, positive=True)
    return TimingAdmission(required <= remaining, required, remaining, projected)
