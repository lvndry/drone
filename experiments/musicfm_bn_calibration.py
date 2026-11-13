"""Sequential BN population replacement on a caller-owned disposable copy.

No loader, dataset access, evaluator, entry point or execution authorization.
The future worker must separately bind the full protocol panel, original terminal
outcome, exact crop bytes, source/runtime and resource qualification. Toy callers
may supply smaller populations; that does not satisfy BN_PROTOCOL.md.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import torch
from torch import Tensor, nn

from experiments.musicfm_bn import Clip, PopulationMoments, WeightedPopulation, recording_moments
from experiments.musicfm_bn_state import (
    BN_SITES,
    Watch,
    capture_bn_prefix,
    expected_temporal_support,
    fingerprint_state_tensor,
    snapshot_state,
    validate_bn_inventory,
    verify_state_change,
)


@dataclass(frozen=True)
class Calibration:
    moments: tuple[PopulationMoments, ...]
    changed_buffers: tuple[str, ...]
    prefix_forwards: int
    crop_sha256: tuple[tuple[str, str], ...]


def calibrate_statistics(
    model: nn.Module,
    clips: Sequence[Clip],
    read: Callable[[Clip], Tensor],
    watch: Watch,
) -> Calibration:
    """Replace upstream statistics before observing each downstream population.

    Model must already be an eval/FP32 disposable copy on one device. On failure,
    discard that copy: earlier sites may have been replaced. No rollback, saving,
    scoring, original-checkpoint mutation or new-data selection occurs here.
    """
    clips = tuple(clips)
    if not 1 <= len(clips) <= 512 or len({clip.identity for clip in clips}) != len(clips):
        raise ValueError("Expected between one and 512 unique recordings")
    if (
        any(not isinstance(clip.weight, Fraction) or clip.weight <= 0 for clip in clips)
        or sum((clip.weight for clip in clips), Fraction()) != 1
    ):
        raise ValueError("Positive exact recording masses must sum to one")
    for clip in clips:
        if (
            clip.domain not in {"beatport", "fmak"}
            or not isinstance(clip.identity, str)
            or not clip.identity
            or any(type(value) is not int for value in (clip.samples, clip.start, clip.length))
            or not 0 <= clip.start <= clip.samples - clip.length
        ):
            raise ValueError("Invalid recording identity/domain or actual crop bounds")
        expected_temporal_support(clip.length)
    if any(module.training for module in model.modules()):
        raise ValueError("The disposable model must already be in eval mode")
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise ValueError("Autocast must be disabled")
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    devices = {value.device for value in tensors}
    if len(devices) != 1 or any(
        value.is_complex() or (value.is_floating_point() and value.dtype != torch.float32)
        for value in tensors
    ):
        raise ValueError("One-device FP32 registered state required")
    device = next(iter(devices))
    modules = validate_bn_inventory(model)
    before = snapshot_state(model, watch)
    before_buffers = {entry.name: entry for entry in before if entry.kind == "buffer"}
    replacements: dict[str, str] = {}
    crop_hashes: dict[str, str] = {}
    estimates = []
    forwards = 0
    with torch.inference_mode():
        for site, module in zip(BN_SITES, modules, strict=True):
            population = WeightedPopulation()
            for clip in clips:
                watch()
                waveform = read(clip)
                if (
                    not isinstance(waveform, Tensor)
                    or waveform.device.type != "cpu"
                    or waveform.dtype != torch.float32
                    or waveform.shape != (1, clip.length)
                    or waveform.requires_grad
                    or not bool(torch.isfinite(waveform).all())
                ):
                    raise ValueError("Reader must supply the exact finite actual-length CPU crop")
                digest = hashlib.sha256(
                    waveform.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                if crop_hashes.setdefault(clip.identity, digest) != digest:
                    raise ValueError("Repeated crop bytes changed between BN site passes")
                watch()
                value = capture_bn_prefix(
                    model,
                    waveform.to(device=device),
                    site.name,
                    watch,
                    waveform_samples=clip.length,
                )
                forwards += 1
                population.add(recording_moments(value.numpy()), clip.weight)
                watch()
            if population.exact_mass != 1:
                raise ValueError("Incomplete recording mass")
            moments = population.finish()
            try:
                with np.errstate(over="raise", invalid="raise"):
                    mean, variance = (
                        value.astype(np.float32) for value in (moments.mean, moments.variance)
                    )
            except FloatingPointError as error:
                raise ValueError("Calibrated buffers cannot be represented in FP32") from error
            assert module.running_mean is not None and module.running_var is not None
            module.running_mean.copy_(torch.from_numpy(mean).to(device=device))
            module.running_var.copy_(torch.from_numpy(variance).to(device=device))
            for suffix, values in (("running_mean", mean), ("running_var", variance)):
                name = site.name + "." + suffix
                replacements[name] = fingerprint_state_tensor(
                    "buffer", name, torch.from_numpy(values), before_buffers[name].persistent, watch
                ).sha256
            estimates.append(moments)
            watch()
    after = snapshot_state(model, watch)
    changed = verify_state_change(before, after, allow_bn_statistics=True)
    if any(
        entry.sha256 != replacements[entry.name]
        for entry in after
        if entry.kind == "buffer" and entry.name in replacements
    ):
        raise ValueError("Final BN buffers do not match the intended FP32 population replacement")
    return Calibration(tuple(estimates), changed, forwards, tuple(sorted(crop_hashes.items())))
