"""Synthetic qualification/admission tests; no recordings, MusicFM, GPU or server."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction

import pytest
import torch
from torch import Tensor, nn

from experiments import musicfm_bn_qualification as qualification
from experiments.musicfm_bn import Clip, FloatArray, PopulationMoments, recording_weight
from experiments.musicfm_bn_state import BN_SITES, CapturedInput, expected_temporal_support


def panel() -> list[Clip]:
    result = []
    for domain in ("beatport", "fmak"):
        for index in range(256):
            length = 1200 if (domain, index) == ("beatport", 100) else 2400
            if (domain, index) == ("fmak", 200):
                length = 2640
            result.append(
                Clip(domain, f"{domain}-{index}", length, 0, length, recording_weight(domain))
            )
    return result


class Clock:
    def __init__(self) -> None:
        self.value = 0.0
        self.watches = 0

    def now(self) -> float:
        return self.value

    def watch(self) -> None:
        self.value += 0.0001
        self.watches += 1


class Harness:
    def __init__(self) -> None:
        self.clock = Clock()
        self.model = nn.Linear(1, 1).eval()
        self.reads = 0
        self.traces = 0
        self.prefixes: list[str] = []
        self.delta = 0.0
        self.shape_error = False
        self.nonfinite = False

    def read(self, clip: Clip) -> Tensor:
        self.reads += 1
        self.clock.value += 0.01
        return torch.ones((1, clip.length), dtype=torch.float32)

    def trace(
        self, model: nn.Module, value: Tensor, watch: qualification.Watch, *, waveform_samples: int
    ) -> tuple[CapturedInput, ...]:
        assert model is self.model and value.shape == (1, waveform_samples)
        self.traces += 1
        self.clock.value += 0.01
        watch()
        return tuple(CapturedInput(site.name, torch.zeros((1, 2, 2))) for site in BN_SITES)

    def prefix(
        self,
        model: nn.Module,
        value: Tensor,
        target: str,
        watch: qualification.Watch,
        *,
        waveform_samples: int,
    ) -> Tensor:
        assert model is self.model and value.shape == (1, waveform_samples)
        self.prefixes.append(target)
        self.clock.value += 0.02
        watch()
        shape = (1, 2, 3) if self.shape_error else (1, 2, 2)
        return torch.full(shape, float("nan") if self.nonfinite else self.delta)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    result = Harness()
    monkeypatch.setattr(qualification, "validate_bn_inventory", lambda _: ())
    monkeypatch.setattr(qualification, "capture_bn_trace", result.trace)
    monkeypatch.setattr(qualification, "capture_bn_prefix", result.prefix)
    return result


def test_subset_selection_is_bounded_deduplicated_and_ties_follow_manifest() -> None:
    correctness, timing = qualification.qualification_clips(panel())
    assert [clip.identity for clip in correctness] == ["beatport-0", "fmak-0", "beatport-100"]
    assert len(timing) == 17 and timing[-1].identity == "fmak-200"
    equal = [replace(clip, samples=2400, length=2400) for clip in panel()]
    correctness, timing = qualification.qualification_clips(equal)
    assert len(correctness) == 2 and len(timing) == 16
    assert correctness[0].identity == "beatport-0"


@pytest.mark.parametrize("defect", ["missing", "duplicate", "domain", "weight", "bounds"])
def test_invalid_panel_is_rejected(defect: str) -> None:
    clips = panel()
    if defect == "missing":
        clips.pop()
    elif defect == "duplicate":
        clips[1] = clips[0]
    elif defect == "domain":
        clips[0] = replace(clips[0], domain="fmak")
    elif defect == "weight":
        clips[0] = replace(clips[0], weight=Fraction(1, 512))
    else:
        clips[0] = replace(clips[0], start=1)
    with pytest.raises(ValueError):
        qualification.qualification_clips(clips)


def test_qualification_counts_times_every_stage_and_never_updates_state(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_moments = qualification.recording_moments
    moment_calls = 0

    def moments(value: FloatArray) -> PopulationMoments:
        nonlocal moment_calls
        moment_calls += 1
        harness.clock.value += 0.03
        return real_moments(value)

    monkeypatch.setattr(qualification, "recording_moments", moments)
    before = qualification.snapshot_state(harness.model, lambda: None)
    result = qualification.qualify(
        harness.model, panel(), harness.read, harness.clock.watch, clock=harness.clock.now
    )
    assert (
        result.full_traces == 3
        and result.correctness_prefixes == 54
        and result.timing_prefixes == 306
    )
    assert harness.traces == 3 and len(harness.prefixes) == 360
    assert moment_calls == 306 and harness.reads == 363
    assert all(value >= 0.06 for value in result.site_max_seconds)
    assert result.max_absolute_difference == 0 and 0 < result.elapsed_seconds <= 300
    assert len(result.crop_sha256) == 18
    assert (
        qualification.verify_state_change(
            before, qualification.snapshot_state(harness.model, lambda: None)
        )
        == ()
    )


@pytest.mark.parametrize("defect", ["difference", "shape", "nonfinite"])
def test_correctness_failure_prevents_timing(harness: Harness, defect: str) -> None:
    harness.delta = 1.01e-5 if defect == "difference" else 0
    harness.shape_error = defect == "shape"
    harness.nonfinite = defect == "nonfinite"
    with pytest.raises(ValueError, match="1e-5|shapes|finite"):
        qualification.qualify(
            harness.model, panel(), harness.read, harness.clock.watch, clock=harness.clock.now
        )
    assert len(harness.prefixes) == 1


def test_threshold_and_changed_registered_state(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.delta = 1e-5
    prefix = harness.prefix

    def mutate(
        model: nn.Module,
        value: Tensor,
        target: str,
        watch: qualification.Watch,
        *,
        waveform_samples: int,
    ) -> Tensor:
        result = prefix(model, value, target, watch, waveform_samples=waveform_samples)
        harness.model.register_buffer("unexpected", torch.zeros(1))
        return result

    monkeypatch.setattr(qualification, "capture_bn_prefix", mutate)
    with pytest.raises(ValueError, match="keys changed"):
        qualification.qualify(
            harness.model, panel(), harness.read, harness.clock.watch, clock=harness.clock.now
        )
    assert len(harness.prefixes) == 18  # Numeric threshold passed; state guard rejected the group.


def test_repeated_crop_identity_and_frozen_input_sequence(harness: Harness) -> None:
    clips = panel()
    real_read = harness.read

    def read(clip: Clip) -> Tensor:
        clips.clear()
        value = real_read(clip)
        if harness.reads == 2:
            value.add_(1)
        return value

    with pytest.raises(ValueError, match="crop bytes changed"):
        qualification.qualify(
            harness.model, clips, read, harness.clock.watch, clock=harness.clock.now
        )
    assert harness.reads == 2 and clips == []


def test_combined_clock_budget_does_not_reset_between_phases(harness: Harness) -> None:
    real_read = harness.read

    def read(clip: Clip) -> Tensor:
        result = real_read(clip)
        if harness.reads == 57:
            harness.clock.value = 250
        elif harness.reads == 58:
            harness.clock.value += 51
        return result

    with pytest.raises(TimeoutError, match="Combined qualification"):
        qualification.qualify(
            harness.model, panel(), read, harness.clock.watch, clock=harness.clock.now
        )
    assert harness.traces == 3 and len(harness.prefixes) == 54


@pytest.mark.parametrize("defect", ["backwards", "nan", "watch"])
def test_clock_and_external_watch_failures_propagate(harness: Harness, defect: str) -> None:
    def watch() -> None:
        if defect == "watch":
            raise RuntimeError("external guardian")
        harness.clock.value = -1 if defect == "backwards" else float("nan")

    with pytest.raises((ValueError, RuntimeError)):
        qualification.qualify(harness.model, panel(), harness.read, watch, clock=harness.clock.now)
    assert harness.traces == 0


def receipt() -> qualification.Qualification:
    return qualification.Qualification(
        ("a", "b"), tuple(str(i) for i in range(16)), 2, 36, 288, 0, (0.01,) * 18, 12, ()
    )


def test_admission_applies_each_factor_once_and_accepts_exact_boundary() -> None:
    result = qualification.timing_admission(
        receipt(),
        remaining_wall_seconds=1000,
        historical_complete_seconds=10,
        measured_overhead_seconds=5,
    )
    assert result.projected_calibration_seconds == pytest.approx(92.16)
    assert result.required_seconds == pytest.approx(329.32)
    assert result.fits_budget
    for remaining, fits in (
        (result.required_seconds, True),
        (math.nextafter(result.required_seconds, 0), False),
    ):
        assert (
            qualification.timing_admission(
                receipt(),
                remaining_wall_seconds=remaining,
                historical_complete_seconds=10,
                measured_overhead_seconds=5,
            ).fits_budget
            is fits
        )


@pytest.mark.parametrize(
    "defect", ["maxima", "clock", "difference", "prefixes", "missing_history", "nan", "negative"]
)
def test_admission_fails_closed_on_missing_invalid_evidence(defect: str) -> None:
    value = receipt()
    history, overhead = 10.0, 5.0
    if defect == "maxima":
        value = replace(value, site_max_seconds=(0.01,) * 17)
    elif defect == "clock":
        value = replace(value, elapsed_seconds=300.01)
    elif defect == "difference":
        value = replace(value, max_absolute_difference=1.01e-5)
    elif defect == "prefixes":
        value = replace(value, timing_prefixes=307)
    elif defect == "missing_history":
        history = 0
    elif defect == "nan":
        overhead = float("nan")
    else:
        overhead = -1
    with pytest.raises(ValueError):
        qualification.timing_admission(
            value,
            remaining_wall_seconds=1000,
            historical_complete_seconds=history,
            measured_overhead_seconds=overhead,
        )


class ToyBNChain(nn.Module):
    """Real PyTorch hooks on tiny synthetic support; no MusicFM class/frontend."""

    def __init__(self) -> None:
        super().__init__()
        for site in BN_SITES:
            parts = site.name.split(".")
            parent = self
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    parent.add_module(part, nn.Module())
                parent = parent.get_submodule(part)
            kind = nn.BatchNorm1d if site.frequency is None else nn.BatchNorm2d
            parent.add_module(parts[-1], kind(site.channels))
        self.eval()

    def forward(self, value: Tensor) -> Tensor:
        first, second = expected_temporal_support(value.shape[-1])
        value = value.mean().expand(1, 512, 64, first)
        for index, site in enumerate(BN_SITES):
            if index == 3:
                value = value[:, :, :32, :second]
            if index == 6:
                value = value.mean(dim=2).repeat(1, 2, 1)
            value = self.get_submodule(site.name)(value)
        return value


@pytest.fixture
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_real_hook_composition_on_small_cpu_toy(one_thread: None) -> None:
    clock = Clock()
    model = ToyBNChain()
    before = qualification.snapshot_state(model, lambda: None)
    result = qualification.qualify(
        model, panel(), lambda clip: torch.ones((1, clip.length)), clock.watch, clock=clock.now
    )
    assert result.full_traces == 3 and result.timing_prefixes == 306
    assert result.max_absolute_difference == 0
    assert (
        qualification.verify_state_change(before, qualification.snapshot_state(model, lambda: None))
        == ()
    )
