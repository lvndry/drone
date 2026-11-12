"""Small CPU stand-ins only; never run the real KeyCNN or decode audio."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import experiments.k10_context_cost as cost
from drone import features
from experiments.k10_v2_pilot import Draw, paired_batches
from tools.musicfm_crop_io import save


@pytest.mark.parametrize("length", [65, 320])
@pytest.mark.parametrize("frames", [62, 104, 400])
def test_crop_exactly_replays_legacy_rng_pitch_start_and_padding(length: int, frames: int) -> None:
    a = np.random.default_rng(42).normal(size=(208, frames)).astype(np.float16)
    d = Draw("f", "fmak", 12345)
    actual, witness = cost.make_crop(a, d, length)
    rng = np.random.default_rng(d.augmentation_seed)
    shift = int(rng.integers(-4, 5))
    view = features.window(a, shift)
    start = int(rng.integers(0, frames - length + 1)) if frames >= length else 0
    expected = view[:, start : start + length].astype(np.float32)
    expected = np.pad(expected, ((0, 0), (0, max(0, length - expected.shape[1]))))
    np.testing.assert_array_equal(actual, expected)
    assert witness["shift"] == shift and witness["first"] == start
    assert witness["actual_frames"] + witness["padding_frames"] == length
    assert features.CROP_FRAMES == 320


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.mean((1, 2)).unsqueeze(1))


def test_both_shapes_commit_three_synthetic_updates_and_cover_full376(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drone.model

    torch.manual_seed(13)
    tiny = Tiny()
    expected = hashlib.sha256()
    for name, value in tiny.state_dict().items():
        expected.update(name.encode())
        expected.update(value.numpy().tobytes())
    monkeypatch.setattr(drone.model, "KeyCNN", Tiny)
    monkeypatch.setattr(cost, "INITIAL", expected.hexdigest())
    targets = []
    original = torch.nn.functional.cross_entropy

    def loss(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets.append(labels.tolist())
        return original(predictions, labels)

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", loss)
    batches = list(paired_batches(["b"], ["f"], updates=3))
    ids = [f"v{i}" for i in range(376)]
    rows: dict[str, dict[str, Any]] = {
        i: {"frames": 104 + n % 5, "ref": 999} for n, i in enumerate(ids)
    }

    def array(identity: str) -> np.ndarray:
        return np.full(
            (208, rows.get(identity, {"frames": 400})["frames"]), 0.125, dtype=np.float16
        )

    results = [cost.qualify_shape(n, batches, array, ids, rows, lambda: None) for n in (320, 65)]
    assert targets == [[i % 24 for i in range(32 * u, 32 * u + 32)] for u in range(3)] * 2
    for result in results:
        assert result["optimizer_commits"] == 3 and result["synthetic_examples"] == 96
        assert result["validation"]["tracks"] == 376
        assert result["validation"]["frames"] == sum(r["frames"] for r in rows.values())
        assert result["validation"]["finite"]
        assert not result["weights_retained"] and not result["task_scores_computed"]
        assert not result["predictions_retained"]
        assert result["initial_state_sha256"] == expected.hexdigest()

    def common(r: dict[str, Any]) -> list[tuple[str, int, int]]:
        return [(x["id"], x["augmentation_seed"], x["shift"]) for x in r["draws"]]

    assert common(results[0]) == common(results[1])


def test_guard_rejects_training_authority_without_cost_only_permission(tmp_path: Path) -> None:
    from tools.run_k10_context_cost import supervise

    config, permit = tmp_path / "config.json", tmp_path / "permit.json"
    save(config, {})
    save(
        permit,
        {
            "authorized": True,
            "allow_cost": True,
            "allow_training": True,
            "config_sha256": cost.sha(config),
        },
    )
    with pytest.raises(ValueError, match="cost-only"):
        supervise(config, permit, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_nonfinite_validation_cannot_complete_resource_qualification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import drone.model

    class BrokenEvaluation(Tiny):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = super().forward(x)
            return out if self.training else out * float("nan")

    torch.manual_seed(13)
    model = BrokenEvaluation()
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    monkeypatch.setattr(drone.model, "KeyCNN", BrokenEvaluation)
    monkeypatch.setattr(cost, "INITIAL", digest.hexdigest())
    batches = list(paired_batches(["b"], ["f"], updates=3))
    with pytest.raises(ValueError, match="nonfinite validation"):
        cost.qualify_shape(
            65,
            batches,
            lambda _: np.zeros((208, 104), np.float16),
            ["v"],
            {"v": {"frames": 104}},
            lambda: None,
        )


def test_no_extended_update_budget_or_unlisted_crop_arm() -> None:
    value = {
        "cpus": [14, 15],
        "crop_frames": [320, 65],
        "updates_per_shape": 3,
        "batch": 32,
        "seed": 13,
        "torch_threads": 2,
        "interop_threads": 1,
        "nice": 19,
        "full_task_fit_allowed": False,
        "caps": {
            "wall_seconds": 900,
            "cpu_seconds": 1200,
            "rss_bytes": 8 * 1024**3,
            "output_bytes": 1024**3,
        },
    }
    cost.contract(value)
    for field, changed in [
        ("updates_per_shape", 8280),
        ("crop_frames", [320, 128, 65]),
        ("full_task_fit_allowed", True),
    ]:
        with pytest.raises(ValueError, match="Frozen disposable"):
            cost.contract({**value, field: changed})


def test_worker_watch_checks_own_start_config_and_finite_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.run_k10_context_cost as guard

    save(
        tmp_path / "ready.json",
        {"pid": 123, "start_ticks": 456, "config_sha256": "frozen", "epoch": 97.0},
    )
    monkeypatch.setattr(guard, "alive", lambda p, s: (p, s) == (123, 456))
    monkeypatch.setattr(guard, "capacity_guard", lambda *_: None)
    clock = [100.0]
    tick = [1.0]
    monkeypatch.setattr(guard.time, "time", lambda: clock[0])
    monkeypatch.setattr(guard.time, "monotonic", lambda: tick[0])
    watch = guard.worker_watch({"deadline_epoch": 200.0}, tmp_path, 123, "frozen")
    watch()
    clock[0] = 108.0
    tick[0] = 2.0
    with pytest.raises(RuntimeError, match="fresh identity-bound"):
        watch()
