"""Toy-network tests only: no MusicFM construction, audio, metadata or checkpoint reads."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from experiments import musicfm_bn_state as state
from experiments.musicfm_bn import Clip
from experiments.musicfm_bn_calibration import calibrate_statistics


@pytest.fixture(scope="module", autouse=True)
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def watch() -> None:
    pass


class Toy(nn.Module):
    """Cheap named BN chain; the real model is deliberately not imported."""

    scratch: Tensor
    optional: Tensor | None

    def __init__(self) -> None:
        super().__init__()
        for site in state.BN_SITES:
            parts = site.name.split(".")
            parent = self
            for name in parts[:-1]:
                if not hasattr(parent, name):
                    parent.add_module(name, nn.Module())
                parent = parent.get_submodule(name)
            kind = nn.BatchNorm1d if site.frequency is None else nn.BatchNorm2d
            parent.add_module(parts[-1], kind(site.channels))
        self.dropout = nn.Dropout(0.9)
        self.register_buffer("scratch", torch.tensor([3.0]), persistent=False)
        self.register_buffer("optional", None, persistent=False)
        self.rotary_cache = torch.tensor([9.0])  # Ordinary attribute, intentionally not registered.
        self.order = list(range(18))
        self.calls: list[int] = []
        self.swallow = False
        self.error: BaseException | None = None
        self.malformed: str | None = None
        self.halve_time = False
        self.eval()

    def forward(self, value: Tensor) -> Tensor:
        assert torch.is_inference_mode_enabled()
        assert not torch.is_autocast_enabled("cpu")
        if self.error is not None:
            raise self.error
        for index in self.order:
            if index == 3:
                value = value[:, :, :32, :]
                if self.halve_time:
                    value = value[..., ::2]
            if index == 6:
                value = value.mean(dim=2).repeat(1, 2, 1)
            if index == 0 and self.malformed is not None:
                if self.malformed == "dtype":
                    value = value.double()
                elif self.malformed == "batch":
                    value = value.repeat(2, 1, 1, 1)
                elif self.malformed == "frequency":
                    value = value[:, :, :31, :]
                elif self.malformed == "channel":
                    value = value[:, :511]
                elif self.malformed == "time":
                    value = value[..., :1]
                elif self.malformed == "nan":
                    value = value * float("nan")
                elif self.malformed == "rank":
                    value = value.mean(dim=2)
            self.calls.append(index)
            try:
                value = self.get_submodule(state.BN_SITES[index].name)(value)
            except BaseException:
                if self.swallow:
                    return value
                raise
        return self.dropout(value)


def input_tensor() -> Tensor:
    return torch.arange(3, dtype=torch.float32).reshape(1, 1, 1, 3).expand(1, 512, 64, 3).clone()


def assert_no_hooks(model: Toy) -> None:
    assert all(not module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize("samples,expected", [(1200, (3, 2)), (2400, (5, 3)), (288000, (600, 300))])
def test_exact_temporal_support(samples: int, expected: tuple[int, int]) -> None:
    assert state.expected_temporal_support(samples) == expected


@pytest.mark.parametrize("samples", [True, 1199, 288001, 1200.0])
def test_invalid_waveform_geometry(samples: int) -> None:
    with pytest.raises(ValueError, match="crop length"):
        state.expected_temporal_support(samples)


class GeometryToy(Toy):
    """Synthetic rank-two input; no frontend, audio or real model is used."""

    def forward(self, value: Tensor) -> Tensor:
        return super().forward(torch.ones((1, 512, 64, 3), dtype=value.dtype))


def test_waveform_geometry_rejects_in_bounds_wrong_pooling() -> None:
    model = GeometryToy()
    value = torch.ones((1, 1200), dtype=torch.float32)
    with pytest.raises(ValueError, match="temporal support"):
        state.capture_bn_trace(model, value, watch, waveform_samples=1200)
    assert_no_hooks(model)
    model.halve_time = True
    trace = state.capture_bn_trace(model, value, watch, waveform_samples=1200)
    assert [row.value.shape[-1] for row in trace] == [3] * 3 + [2] * 15
    for length, error in ((None, "explicit"), (1201, "actual N1")):
        with pytest.raises(ValueError, match=error):
            state.capture_bn_trace(model, value, watch, waveform_samples=length)
    assert_no_hooks(model)


def test_exact_inventory_trace_prefix_and_cpu_ownership_preserve_state() -> None:
    model, value = Toy(), input_tensor()
    before = state.snapshot_state(model, watch)
    rotary = model.rotary_cache
    trace = state.capture_bn_trace(model, value, watch)
    assert tuple(row.site for row in trace) == tuple(site.name for site in state.BN_SITES)
    for index in (0, 2, 5, 6, 17):
        model.calls.clear()
        captured = state.capture_bn_prefix(model, value, state.BN_SITES[index].name, watch)
        torch.testing.assert_close(captured, trace[index].value, atol=0, rtol=0)
        assert model.calls == list(range(index + 1))
        assert captured.device.type == "cpu" and not captured.requires_grad
        assert_no_hooks(model)
    value.zero_()
    assert trace[0].value[0, 0, 0, 2] == 2  # Captures never alias the caller's input.
    assert state.verify_state_change(before, state.snapshot_state(model, watch)) == ()
    assert model.rotary_cache is rotary and rotary.item() == 9


def test_sequential_upstream_replacement_changes_downstream_input() -> None:
    model, value = Toy(), input_tensor()
    first, second = state.BN_SITES[:2]
    before = state.capture_bn_prefix(model, value, second.name, watch)
    baseline = state.snapshot_state(model, watch)
    bn = model.get_submodule(first.name)
    assert isinstance(bn, nn.BatchNorm2d) and bn.running_mean is not None
    bn.running_mean.fill_(4.0)
    after = state.capture_bn_prefix(model, value, second.name, watch)
    expected = before - 4.0 / (1.0 + bn.eps) ** 0.5
    torch.testing.assert_close(after, expected, atol=1e-6, rtol=1e-6)
    assert state.verify_state_change(
        baseline, state.snapshot_state(model, watch), allow_bn_statistics=True
    ) == (first.name + ".running_mean",)
    assert_no_hooks(model)


@pytest.mark.parametrize(
    "order", [[1, 0] + list(range(2, 18)), [0, 0] + list(range(1, 18)), list(range(17))]
)
def test_reordered_reused_or_missing_forward_calls_fail(order: list[int]) -> None:
    model = Toy()
    model.order = order
    with pytest.raises(ValueError, match="order/reuse|every BN"):
        state.capture_bn_trace(model, input_tensor(), watch)
    assert_no_hooks(model)


def test_missing_target_and_swallowed_stop_are_detected() -> None:
    model = Toy()
    model.order = list(range(17))
    with pytest.raises(ValueError, match="Target missing"):
        state.capture_bn_prefix(model, input_tensor(), state.BN_SITES[-1].name, watch)
    model.order = list(range(18))
    model.swallow = True
    with pytest.raises(ValueError, match="swallowed"):
        state.capture_bn_prefix(model, input_tensor(), state.BN_SITES[0].name, watch)
    assert_no_hooks(model)


def test_unexpected_exceptions_and_foreign_stop_are_not_swallowed() -> None:
    model = Toy()
    for error in (RuntimeError("unexpected"), state._PrefixComplete()):
        model.error = error
        with pytest.raises(type(error)) as caught:
            state.capture_bn_trace(model, input_tensor(), watch)
        assert caught.value is error
        assert_no_hooks(model)


def test_watch_failure_cleans_hooks_and_keeps_preexisting_hooks() -> None:
    model = Toy()
    module = model.get_submodule(state.BN_SITES[0].name)
    handle = module.register_forward_pre_hook(lambda _module, _args: None)
    existing = set(module._forward_pre_hooks)
    calls = 0

    def fail() -> None:
        nonlocal calls
        calls += 1
        if calls == 6:  # After the target's CPU copy (following the model-input check).
            raise RuntimeError("watch aborted")

    with pytest.raises(RuntimeError, match="watch aborted"):
        state.capture_bn_prefix(model, input_tensor(), state.BN_SITES[0].name, fail)
    assert set(module._forward_pre_hooks) == existing
    handle.remove()
    assert_no_hooks(model)


@pytest.mark.parametrize("mutation", ["training", "dropout", "fp64", "autocast"])
def test_eval_precision_guards_reject_before_forward(mutation: str) -> None:
    model = Toy()
    if mutation == "training":
        model.train()
    elif mutation == "dropout":
        model.dropout.train()
    elif mutation == "fp64":
        model.double()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mutation == "autocast"):
        with pytest.raises(ValueError, match="eval|FP32|Autocast"):
            state.capture_bn_trace(model, input_tensor(), watch)
    assert model.calls == []
    assert_no_hooks(model)


@pytest.mark.parametrize(
    "malformed", ["dtype", "batch", "frequency", "channel", "time", "nan", "rank"]
)
def test_bn_input_guards_and_cleanup(malformed: str) -> None:
    model = Toy()
    model.malformed = malformed
    with pytest.raises(ValueError, match="shape/dtype|Nonfinite"):
        state.capture_bn_trace(model, input_tensor(), watch)
    assert_no_hooks(model)


@pytest.mark.parametrize("mutation", ["missing", "extra", "alias", "channels", "eps", "type"])
def test_inventory_mismatches_are_rejected(mutation: str) -> None:
    model = Toy()
    first = model.get_submodule(state.BN_SITES[0].name)
    parent = model.get_submodule("backbone.conv.conv.0")
    if mutation == "missing":
        delattr(parent, "bn1")
    elif mutation == "extra":
        model.add_module("extra", nn.BatchNorm1d(1))
    elif mutation == "alias":
        parent.bn2 = first
    elif mutation == "channels":
        parent.bn1 = nn.BatchNorm2d(511)
    elif mutation == "eps":
        assert isinstance(first, nn.BatchNorm2d)
        first.eps = 1e-4
    elif mutation == "type":
        parent.bn1 = nn.BatchNorm1d(512)
    with pytest.raises(ValueError, match="inventory|alias|configuration|type"):
        state.validate_bn_inventory(model)
    assert_no_hooks(model)


def test_nonfinite_model_input_is_rejected_before_forward() -> None:
    model, value = Toy(), input_tensor()
    value[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite model input"):
        state.capture_bn_trace(model, value, watch)
    assert model.calls == []
    assert_no_hooks(model)


def test_snapshot_includes_nonpersistent_none_alias_and_raw_signed_zero() -> None:
    model = Toy()
    model.register_buffer("alias", model.scratch, persistent=False)
    before = state.snapshot_state(model, watch)
    entries = {entry.name: entry for entry in before if entry.kind == "buffer"}
    assert entries["scratch"].persistent is False
    assert entries["optional"].dtype is None and entries["optional"].shape is None
    assert "alias" in entries
    model.scratch.zero_()
    positive = state.snapshot_state(model, watch)
    model.scratch.fill_(-0.0)
    negative = state.snapshot_state(model, watch)
    with pytest.raises(ValueError, match="Protected state"):
        state.verify_state_change(positive, negative, allow_bn_statistics=True)


@pytest.mark.parametrize(
    "mutation",
    ["parameter", "counter", "scratch", "new", "deleted", "persistence", "shape", "none"],
)
def test_protected_state_mutation_and_registration_changes_fail(mutation: str) -> None:
    model = Toy()
    before = state.snapshot_state(model, watch)
    first = model.get_submodule(state.BN_SITES[0].name)
    assert isinstance(first, nn.BatchNorm2d)
    with torch.no_grad():
        if mutation == "parameter":
            assert first.weight is not None
            first.weight.add_(1)
        elif mutation == "counter":
            assert first.num_batches_tracked is not None
            first.num_batches_tracked.add_(1)
        elif mutation == "scratch":
            model.scratch.add_(1)
        elif mutation == "new":
            model.register_buffer("new", torch.tensor(1.0), persistent=False)
        elif mutation == "deleted":
            delattr(model, "scratch")
        elif mutation == "persistence":
            model._non_persistent_buffers_set.remove("scratch")
        elif mutation == "shape":
            first.running_mean = torch.zeros(511)
        elif mutation == "none":
            model.optional = torch.tensor(0.0)
    with pytest.raises(ValueError, match="Protected state|keys changed|metadata changed"):
        state.verify_state_change(
            before, state.snapshot_state(model, watch), allow_bn_statistics=True
        )


def test_only_exact_36_stat_buffers_may_change_and_counters_stay_fixed() -> None:
    model = Toy()
    before = state.snapshot_state(model, watch)
    for module in state.validate_bn_inventory(model):
        assert module.running_mean is not None and module.running_var is not None
        module.running_mean.add_(1)
        module.running_var.add_(2)
    after = state.snapshot_state(model, watch)
    assert (
        set(state.verify_state_change(before, after, allow_bn_statistics=True))
        == state.BN_STAT_BUFFERS
    )
    with pytest.raises(ValueError, match="Protected state"):
        state.verify_state_change(before, after)
    assert len(state.BN_STAT_BUFFERS) == 36


class WaveToy(Toy):
    """Linear BN chain with waveform-shaped input; not an actual audio frontend."""

    def forward(self, value: Tensor) -> Tensor:
        first, second = state.expected_temporal_support(value.shape[1])
        value = value.mean().expand(1, 512, 64, first)
        for index, site in enumerate(state.BN_SITES):
            if index == 3:
                value = value[:, :, :32, :second]
            if index == 6:
                value = value.mean(dim=2).repeat(1, 2, 1)
            value = self.get_submodule(site.name)(value)
        return value


def toy_clips() -> tuple[Clip, ...]:
    # A small unit-test population is intentionally not the required 512-row
    # protocol panel. The low-level core supplies no dataset/execution authority.
    return (
        Clip("beatport", "toy-a", 1200, 0, 1200, Fraction(1, 2)),
        Clip("fmak", "toy-b", 1200, 0, 1200, Fraction(1, 2)),
    )


def toy_wave(clip: Clip) -> Tensor:
    return torch.full((1, clip.length), 2.0 if clip.identity == "toy-a" else 6.0)


def test_complete_sequential_core_matches_closed_form_and_preserves_original() -> None:
    original = WaveToy()
    before = state.snapshot_state(original, watch)
    disposable = copy.deepcopy(original)
    reads = []

    def read(clip: Clip) -> Tensor:
        reads.append(clip.identity)
        return toy_wave(clip)

    result = calibrate_statistics(disposable, toy_clips(), read, watch)
    assert result.prefix_forwards == 36
    assert reads == ["toy-a", "toy-b"] * 18
    assert len(result.moments) == 18
    torch.testing.assert_close(
        torch.from_numpy(result.moments[0].mean.copy()),
        torch.full((512,), 4.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.from_numpy(result.moments[0].variance.copy()),
        torch.full((512,), 4.0, dtype=torch.float64),
    )
    # Site two sees site one's NEW population normalization, not the old prefix.
    np.testing.assert_allclose(result.moments[1].mean, 0, atol=1e-6, rtol=0)
    np.testing.assert_allclose(result.moments[1].variance, 4 / (4 + 1e-5), atol=1e-6, rtol=0)
    assert set(result.changed_buffers) <= state.BN_STAT_BUFFERS
    assert state.verify_state_change(before, state.snapshot_state(original, watch)) == ()
    state.verify_state_change(
        before, state.snapshot_state(disposable, watch), allow_bn_statistics=True
    )
    assert_no_hooks(disposable)


def test_core_abort_leaves_only_declared_partial_statistics_on_disposable_copy() -> None:
    model = WaveToy()
    before = state.snapshot_state(model, watch)
    calls = 0

    def fail(clip: Clip) -> Tensor:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("reader failed")
        return toy_wave(clip)

    with pytest.raises(RuntimeError, match="reader failed"):
        calibrate_statistics(model, toy_clips(), fail, watch)
    changed = state.verify_state_change(
        before, state.snapshot_state(model, watch), allow_bn_statistics=True
    )
    assert set(changed) == {
        state.BN_SITES[0].name + suffix for suffix in (".running_mean", ".running_var")
    }
    assert_no_hooks(model)


@pytest.mark.parametrize("defect", ["shape", "dtype", "nonfinite", "gradient"])
def test_core_rejects_invalid_reader_before_prefix(defect: str) -> None:
    model = WaveToy()
    before = state.snapshot_state(model, watch)

    def read(clip: Clip) -> Tensor:
        value = toy_wave(clip)
        if defect == "shape":
            value = value[:, :-1]
        elif defect == "dtype":
            value = value.double()
        elif defect == "nonfinite":
            value.fill_(float("nan"))
        else:
            value.requires_grad_(True)
        return value

    with pytest.raises(ValueError, match="Reader"):
        calibrate_statistics(model, toy_clips(), read, watch)
    assert state.verify_state_change(before, state.snapshot_state(model, watch)) == ()
    assert_no_hooks(model)


def test_core_freezes_recording_sequence_before_reader_callbacks() -> None:
    clips = list(toy_clips())

    def read(clip: Clip) -> Tensor:
        if len(clips) == 2:
            clips.pop()
        return toy_wave(clip)

    result = calibrate_statistics(WaveToy(), clips, read, watch)
    assert len(clips) == 1
    assert result.prefix_forwards == 36
    assert {identity for identity, _ in result.crop_sha256} == {"toy-a", "toy-b"}


@pytest.mark.parametrize("mutation", ["crop", "allowed_buffer"])
def test_core_rejects_changed_crop_or_overwritten_bn_replacement(mutation: str) -> None:
    model = WaveToy()
    calls = 0

    def read(clip: Clip) -> Tensor:
        nonlocal calls
        calls += 1
        value = toy_wave(clip)
        if calls == 3:
            if mutation == "crop":
                value[0, 0] += 1
            else:
                first = state.validate_bn_inventory(model)[0]
                assert first.running_mean is not None
                first.running_mean.add_(1)
        return value

    with pytest.raises(ValueError, match="Repeated crop|intended FP32"):
        calibrate_statistics(model, toy_clips(), read, watch)
    assert_no_hooks(model)


@pytest.mark.parametrize("defect", ["empty", "duplicate", "mass", "bounds", "length"])
def test_core_rejects_invalid_population_without_reader_access(defect: str) -> None:
    clips = list(toy_clips())
    if defect == "empty":
        clips.clear()
    elif defect == "duplicate":
        clips[1] = clips[0]
    elif defect == "mass":
        clips[0] = replace(clips[0], weight=Fraction(1, 3))
    elif defect == "bounds":
        clips[0] = replace(clips[0], start=1)
    else:
        clips[0] = replace(clips[0], samples=1199, length=1199)

    def forbidden(_: Clip) -> Tensor:
        raise AssertionError("Invalid population must not read a crop")

    with pytest.raises(ValueError):
        calibrate_statistics(WaveToy(), clips, forbidden, watch)
