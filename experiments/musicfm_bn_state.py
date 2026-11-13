"""Synthetic-qualified hooks/state checks for musicfm-key-pilot/BN_PROTOCOL.md.

These helpers do not load models, clear rotary attributes, calibrate statistics,
score, launch a worker or implement a resource watchdog. Callers supply watch().
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

Watch = Callable[[], None]


@dataclass(frozen=True)
class BNSite:
    name: str
    channels: int
    frequency: int | None
    max_time: int


BN_SITES = tuple(
    BNSite(f"backbone.conv.conv.{block}.bn{branch}", 512, frequency, max_time)
    for block, frequency, max_time in ((0, 64, 600), (1, 32, 300))
    for branch in (1, 2, 3)
) + tuple(
    BNSite(f"backbone.conformer.layers.{layer}.conv_module.batch_norm", 1024, None, 300)
    for layer in range(12)
)
BN_STAT_BUFFERS = frozenset(
    f"{site.name}.{stat}" for site in BN_SITES for stat in ("running_mean", "running_var")
)


def expected_temporal_support(samples: int) -> tuple[int, int]:
    """Exact first/second-block time support for an actual-length fitting crop."""
    if type(samples) is not int or not 1200 <= samples <= 288000:
        raise ValueError("Expected an integer fitting crop length between 1200 and 288000")
    mel = samples // 240
    first = (mel + 1) // 2
    return first, (first + 1) // 2


def validate_bn_inventory(model: nn.Module) -> tuple[nn.BatchNorm1d | nn.BatchNorm2d, ...]:
    """Require all 18 named, distinct active BN modules, including both residual branches."""
    modules = dict(model.named_modules(remove_duplicate=False))
    actual = {
        name
        for name, module in modules.items()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }
    if actual != {site.name for site in BN_SITES}:
        raise ValueError("BN inventory differs from the exact 18-site contract")
    result: list[nn.BatchNorm1d | nn.BatchNorm2d] = []
    for site in BN_SITES:
        module = modules[site.name]
        expected = nn.BatchNorm1d if site.frequency is None else nn.BatchNorm2d
        if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)) or type(module) is not expected:
            raise ValueError("Unexpected BN type: " + site.name)
        if (
            module.num_features != site.channels
            or module.eps != 1e-5
            or module.momentum != 0.1
            or not module.affine
            or not module.track_running_stats
        ):
            raise ValueError("Unexpected BN configuration: " + site.name)
        for name in ("weight", "bias", "running_mean", "running_var"):
            value = getattr(module, name)
            if (
                not isinstance(value, Tensor)
                or value.shape != (site.channels,)
                or value.dtype != torch.float32
            ):
                raise ValueError("Missing or malformed BN tensor: " + site.name + "." + name)
        counter = module.num_batches_tracked
        if not isinstance(counter, Tensor) or counter.shape != () or counter.dtype != torch.int64:
            raise ValueError("Missing or malformed BN counter: " + site.name)
        result.append(module)
    if len({id(module) for module in result}) != len(BN_SITES):
        raise ValueError("BN sites must not alias one module")
    return tuple(result)


def _no_autocast() -> None:
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise ValueError("Autocast must be disabled")


def _eval_fp32(model: nn.Module) -> None:
    _no_autocast()
    if any(module.training for module in model.modules()):
        raise ValueError("Every module, including Dropout, must already be in eval mode")
    for value in (*model.parameters(), *model.buffers()):
        if value.is_complex() or (value.is_floating_point() and value.dtype != torch.float32):
            raise ValueError("All floating parameters and registered buffers must be FP32")


def _owned_cpu(value: Tensor, watch: Watch) -> Tensor:
    watch()
    result = value.detach().to(device="cpu", copy=True).contiguous()
    watch()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Nonfinite captured input")
    return result


def _site_input(
    site: BNSite, args: tuple[object, ...], expected_times: tuple[int, int] | None
) -> Tensor:
    _no_autocast()
    if len(args) != 1 or not isinstance(args[0], Tensor):
        raise ValueError("Expected one positional BN tensor input")
    value = args[0]
    rank = 3 if site.frequency is None else 4
    if (
        value.dtype != torch.float32
        or value.ndim != rank
        or value.shape[:2] != (1, site.channels)
        or not 2 <= value.shape[-1] <= site.max_time
        or (site.frequency is not None and value.shape[2] != site.frequency)
    ):
        raise ValueError("Unexpected N1 FP32 BN input shape/dtype: " + site.name)
    if expected_times is not None:
        expected = expected_times[0 if site.frequency == 64 else 1]
        if value.shape[-1] != expected:
            raise ValueError("BN temporal support disagrees with waveform length: " + site.name)
    return value


@dataclass(frozen=True)
class CapturedInput:
    site: str
    value: Tensor


class _PrefixComplete(BaseException):
    """Only the exact per-call instance raised by our target hook is caught."""


def _capture(
    model: nn.Module,
    model_input: Tensor,
    target: str | None,
    watch: Watch,
    waveform_samples: int | None,
) -> tuple[CapturedInput, ...]:
    _eval_fp32(model)
    modules = validate_bn_inventory(model)
    if target is not None and target not in {site.name for site in BN_SITES}:
        raise ValueError("Unknown BN target")
    if (
        not isinstance(model_input, Tensor)
        or model_input.dtype != torch.float32
        or model_input.ndim < 2
        or model_input.shape[0] != 1
        or any(size == 0 for size in model_input.shape)
    ):
        raise ValueError("Expected a nonempty N1 FP32 model input")
    expected_times = None
    if waveform_samples is not None:
        expected_times = expected_temporal_support(waveform_samples)
        if model_input.shape != (1, waveform_samples):
            raise ValueError("Declared waveform length must match the actual N1 input")
    elif model_input.ndim == 2:
        raise ValueError(
            "Rank-two waveform input requires an explicit waveform_samples declaration"
        )
    watch()
    finite_input = bool(torch.isfinite(model_input).all())
    watch()
    if not finite_input:
        raise ValueError("Nonfinite model input")
    captured: list[CapturedInput] = []
    seen = 0
    attempted_stop = False
    caught_stop = False
    stop = _PrefixComplete()

    def hook(index: int, module: nn.Module, args: tuple[object, ...]) -> None:
        nonlocal seen, attempted_stop
        watch()
        if attempted_stop:
            raise ValueError("Model swallowed the stopped-prefix exception")
        if index != seen:
            raise ValueError("BN forward call order/reuse differs from the contract")
        if module.training:
            raise ValueError("BN switched to training during forward")
        site = BN_SITES[index]
        value = _site_input(site, args, expected_times)
        seen += 1
        if target is None or site.name == target:
            captured.append(CapturedInput(site.name, _owned_cpu(value, watch)))
        elif not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite prefix input")
        if site.name == target:
            attempted_stop = True
            raise stop

    handles = []
    try:
        for index, module in enumerate(modules):
            # Bind each index now, not when the hook is later called.
            def bound_hook(
                current: nn.Module, args: tuple[object, ...], *, index: int = index
            ) -> None:
                hook(index, current, args)

            handles.append(module.register_forward_pre_hook(bound_hook))
        watch()
        try:
            with torch.inference_mode():
                try:
                    model(model_input)
                except _PrefixComplete as error:
                    if error is not stop:
                        raise
                    caught_stop = True
        finally:
            watch()
    finally:
        for handle in handles:
            handle.remove()
    _eval_fp32(model)
    if target is None:
        if seen != len(BN_SITES):
            raise ValueError("Forward did not visit every BN site")
    elif not attempted_stop or not caught_stop:
        raise ValueError("Target missing or model swallowed the stopped-prefix exception")
    return tuple(captured)


def capture_bn_trace(
    model: nn.Module,
    model_input: Tensor,
    watch: Watch,
    *,
    waveform_samples: int | None = None,
) -> tuple[CapturedInput, ...]:
    """Qualification trace; actual waveform inputs require exact declared geometry.

    Omitting waveform_samples is exclusively for non-waveform synthetic toy inputs.
    """
    return _capture(model, model_input, None, watch, waveform_samples)


def capture_bn_prefix(
    model: nn.Module,
    model_input: Tensor,
    target: str,
    watch: Watch,
    *,
    waveform_samples: int | None = None,
) -> Tensor:
    """Stop before target normalization; only synthetic non-waveforms may omit geometry."""
    return _capture(model, model_input, target, watch, waveform_samples)[0].value


@dataclass(frozen=True)
class StateEntry:
    kind: Literal["parameter", "buffer"]
    name: str
    dtype: str | None
    shape: tuple[int, ...] | None
    requires_grad: bool | None
    persistent: bool | None
    sha256: str


StateSnapshot = tuple[StateEntry, ...]


def fingerprint_state_tensor(
    kind: Literal["parameter", "buffer"],
    name: str,
    value: Tensor | None,
    persistent: bool | None,
    watch: Watch,
) -> StateEntry:
    """Hash one registered state field, including its kind/name and metadata."""
    if value is None:
        dtype, shape, requires_grad, raw = None, None, None, b""
    else:
        if value.layout != torch.strided or value.is_quantized or value.device.type == "meta":
            raise ValueError("State fingerprint requires dense, materialized, unquantized tensors")
        watch()
        copied = value.detach().to(device="cpu", copy=True).contiguous()
        watch()
        dtype, shape, requires_grad = str(value.dtype), tuple(value.shape), value.requires_grad
        raw = copied.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    header = json.dumps(
        [kind, name, dtype, shape, requires_grad, persistent],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return StateEntry(
        kind,
        name,
        dtype,
        shape,
        requires_grad,
        persistent,
        hashlib.sha256(header + b"\0" + raw).hexdigest(),
    )


def snapshot_state(model: nn.Module, watch: Watch) -> StateSnapshot:
    """Hash all registered state, including aliases, None entries and nonpersistent buffers.

    Uses canonical C-order little-endian raw bytes, retaining dtype/shape/kind and
    persistence. Ordinary attributes (including native rotary caches) are untouched.
    Full snapshots belong at qualification/pass boundaries, not every prefix call.
    """
    if sys.byteorder != "little":
        raise ValueError("Canonical raw-byte state hashing requires a little-endian host")
    entries: list[StateEntry] = []
    for prefix, module in model.named_modules(remove_duplicate=False):
        for name, value in module._parameters.items():
            full = f"{prefix}.{name}" if prefix else name
            entries.append(fingerprint_state_tensor("parameter", full, value, None, watch))
        for name, value in module._buffers.items():
            full = f"{prefix}.{name}" if prefix else name
            entries.append(
                fingerprint_state_tensor(
                    "buffer", full, value, name not in module._non_persistent_buffers_set, watch
                )
            )
    watch()
    return tuple(sorted(entries, key=lambda entry: (entry.kind, entry.name)))


def verify_state_change(
    before: StateSnapshot, after: StateSnapshot, *, allow_bn_statistics: bool = False
) -> tuple[str, ...]:
    """Require identical registrations/metadata and prohibit all undeclared byte changes.

    Calibration permits changes only in the exact 36 mean/variance buffers; it does
    not require every buffer to change numerically. All counters remain protected.
    """
    left = {(entry.kind, entry.name): entry for entry in before}
    right = {(entry.kind, entry.name): entry for entry in after}
    if len(left) != len(before) or len(right) != len(after) or left.keys() != right.keys():
        raise ValueError("Registered parameter/buffer keys changed or duplicate keys supplied")
    allowed = {("buffer", name) for name in BN_STAT_BUFFERS} if allow_bn_statistics else set()
    if not allowed.issubset(left):
        raise ValueError("Missing required BN statistics buffers")
    changed: list[str] = []
    for key, old in left.items():
        new = right[key]
        if (old.dtype, old.shape, old.requires_grad, old.persistent) != (
            new.dtype,
            new.shape,
            new.requires_grad,
            new.persistent,
        ):
            raise ValueError("Registered state metadata changed: " + old.name)
        if old.sha256 != new.sha256:
            if key not in allowed:
                raise ValueError("Protected state bytes changed: " + old.name)
            changed.append(old.name)
    return tuple(sorted(changed))
