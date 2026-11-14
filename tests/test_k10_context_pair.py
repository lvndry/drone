"""Synthetic semantics only; no real model, audio, task fit or protected processes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import experiments.k10_context_pair as pair
import tools.run_k10_context_pair as guard
from tools.musicfm_crop_io import load, save, sha


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 24)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
            self.linear.bias[0] = 10

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features.mean((1, 2)).unsqueeze(1))


@pytest.fixture
def tiny_loop(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    import drone.model

    model = Tiny()
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    monkeypatch.setattr(drone.model, "KeyCNN", Tiny)
    monkeypatch.setattr(pair, "INITIAL", digest.hexdigest())
    monkeypatch.setattr(pair, "UPDATES", 2)
    monkeypatch.setattr(pair, "EVERY", 1)
    monkeypatch.setattr(pair, "EVENTS", 2)
    partitions = {
        "bp_fit": ["b"],
        "expanded_fma_fit": ["f"],
        "bp_selection_guard": [f"b{index}" for index in range(126)],
        "fma_inner_selection": [f"f{index}" for index in range(250)],
    }
    rows = {
        identity: {"ref": 0, "frames": 104, "canonical_samples": 120 * 22050}
        for identity in [
            "b",
            "f",
            *partitions["bp_selection_guard"],
            *partitions["fma_inner_selection"],
        ]
    }
    return partitions, rows


def test_fresh_arms_complete_same_stream_earliest_tie_and_selected_reload(
    tmp_path: Path,
    tiny_loop: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    partitions, rows = tiny_loop
    results = [
        pair.train_arm(
            crop_frames,
            partitions,
            rows,
            lambda _: np.full((208, 104), 0.125, np.float16),
            tmp_path / str(crop_frames),
            {"config": "same"},
            lambda: None,
        )
        for crop_frames in (320, 65)
    ]
    for result in results:
        assert result["updates"] == 2 and result["events"] == 2 and result["best_update"] == 1
        assert result["selected_reload_proof"]["strict_state_load"]
        assert (
            result["selected_reload_proof"]["scores_identical"]
            and result["selected_reload_proof"]["tracks"] == 376
        )
        assert not result["confirmation_used"] and not result["calibration_used"]
        assert (tmp_path / str(result["crop_frames"]) / "result.json").is_file()
        assert (tmp_path / str(result["crop_frames"]) / "best.pt").is_file()
        assert (tmp_path / str(result["crop_frames"]) / "final.pt").is_file()
    assert results[0]["common_draw_pitch_sha256"] == results[1]["common_draw_pitch_sha256"]
    assert results[0]["temporal_sha256"] != results[1]["temporal_sha256"]
    assert results[0]["coverage"] == results[1]["coverage"]


def test_corrupt_selected_state_blocks_completed_arm_record(
    tmp_path: Path,
    tiny_loop: tuple[dict[str, Any], dict[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = pair.checkpoint

    def corrupt(path: Path, model: torch.nn.Module, provenance: dict[str, Any], watch: Any) -> str:  # noqa: ANN401
        original(path, model, provenance, watch)
        if path.name == "best.pt":
            payload = torch.load(path, weights_only=True)
            payload["state_dict"].pop("linear.bias")
            torch.save(payload, path)
        return sha(path)

    monkeypatch.setattr(pair, "checkpoint", corrupt)
    partitions, rows = tiny_loop
    with pytest.raises(RuntimeError, match="Missing key"):
        pair.train_arm(
            65,
            partitions,
            rows,
            lambda _: np.zeros((208, 104), np.float16),
            tmp_path / "arm",
            {},
            lambda: None,
        )
    assert not (tmp_path / "arm/result.json").exists()


def arm(length: int, beatport_correct: int = 89, fmak_correct: int = 176) -> dict[str, Any]:
    def domain_metrics(tracks: int, correct: int, prefix: str) -> dict[str, Any]:
        return {
            "ids": [f"{prefix}{index}" for index in range(tracks)],
            "predictions": [0] * tracks,
            "tracks": tracks,
            "exact_correct": correct,
            "symmetric_units10": 700 if prefix == "b" else 2000,
            "asymmetric_units10": 650 if prefix == "b" else 1900,
        }

    return {
        "status": "COMPLETE_CONTEXT_ARM_NOT_PROMOTED",
        "crop_frames": length,
        "initial_state_sha256": pair.INITIAL,
        "common_draw_pitch_sha256": "same",
        "coverage": {"b": 1},
        "updates": 8280,
        "events": 120,
        "provenance": {"runtime": "same"},
        "confirmation_used": False,
        "calibration_used": False,
        "selected_reload_proof": {
            "strict_state_load": True,
            "scores_identical": True,
            "tracks": 376,
            "argmax_disagreements": 0,
        },
        "best_metrics": {
            "beatport": domain_metrics(126, beatport_correct, "b"),
            "fmak": domain_metrics(250, fmak_correct, "f"),
        },
    }


def test_gate_canonical65_five_cases_BP_both_metrics_and_historical_guard() -> None:
    baseline = arm(320)
    old = arm(320)
    assert not pair.compare(baseline, arm(65, fmak_correct=180), old)["continuation_earned"]
    assert pair.compare(baseline, arm(65, fmak_correct=181), old)["continuation_earned"]
    candidate = arm(65, fmak_correct=181)
    candidate["best_metrics"]["beatport"]["asymmetric_units10"] -= 1
    assert not pair.compare(baseline, candidate, old)["context_exposure_gate"]
    weaker = arm(320, beatport_correct=88)
    gate = pair.compare(weaker, arm(65, fmak_correct=181), old)
    assert gate["context_exposure_gate"] and not gate["historical_CPU320_noninferiority_guard"]
    assert not gate["continuation_earned"] and not gate["promotion_authorized"]
    broken = arm(65, fmak_correct=181)
    broken["common_draw_pitch_sha256"] = "wrong"
    with pytest.raises(ValueError, match="paired source/runtime"):
        pair.compare(baseline, broken, old)


def terminal_fixture(root: Path, updates: int = 8280) -> dict[str, Any]:
    root.mkdir()
    (root / "run-v1").mkdir()
    (root / "run-v1/worker").mkdir()
    config = {
        "source_sha256": {"teacher": "frozen"},
        "plan_sha256": "plan",
        "labels_sha256": "labels",
        "basis_initial_state_sha256": "initial",
    }
    save(root / "sealed-config.json", config)
    config_hash = sha(root / "sealed-config.json")
    result = {
        "status": "COMPLETE_FIXED_PILOT_NOT_RELEASE_EVIDENCE",
        "updates": updates,
        "events": 120,
        "confirmation_used": False,
        "calibration_used": False,
        "provenance": {"config_sha256": config_hash, **config, "initial_state_sha256": "initial"},
        "continuation_gate": {"continuation_earned": False},
    }
    save(root / "run-v1/worker/result.json", result)
    save(
        root / "run-v1/held-success.json",
        {"mode": "fit", "result_sha256": sha(root / "run-v1/worker/result.json")},
    )
    save(root / "run-v1/controller.json", {"state": "COMPLETE_NO_RAM_OWNERSHIP", "worker_exit": 0})
    return {
        "root": str(root),
        "config_sha256": config_hash,
        "starts": [{"pid": 1, "start_ticks": 2}],
    }


def test_teacher_verified_completion_accepts_negative_accuracy_gate_only_after_full_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(guard, "alive", lambda *_: False)
    verifier = guard.TeacherVerifier(terminal_fixture(tmp_path / "good"))
    assert verifier()["state"] == "VERIFIED_LEGITIMATE_TERMINAL"
    bad = guard.TeacherVerifier(terminal_fixture(tmp_path / "bad", 8279))
    with pytest.raises(ValueError, match="horizon/provenance"):
        bad()


def test_teacher_missing_receipt_has_finite_grace_not_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guard, "alive", lambda *_: False)
    clock = [1.0]
    monkeypatch.setattr(guard.time, "monotonic", lambda: clock[0])
    verifier = guard.TeacherVerifier(
        {"root": str(tmp_path), "config_sha256": "absent", "starts": [{"pid": 1, "start_ticks": 2}]}
    )
    assert verifier()["state"] == "PENDING_TERMINAL_RECEIPT"
    clock[0] = 12
    with pytest.raises(RuntimeError, match="without verified"):
        verifier()


def test_initial_teacher_guard_failure_keeps_unverified_receipt_without_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    now = time.time()
    config = tmp_path / "config.json"
    approval = tmp_path / "permit.json"
    save(
        config,
        {
            "source_sha256": {},
            "teacher": {},
            "not_before_epoch": now - 1,
            "deadline_epoch": now + 100,
            "caps": {"wall_seconds": 21600},
        },
    )
    save(
        approval,
        {
            "authorized": True,
            "allow_context_pair": True,
            "allow_training": True,
            "config_sha256": sha(config),
        },
    )
    monkeypatch.setattr(pair, "contract", lambda *_: None)
    monkeypatch.setattr(guard, "environment", lambda *_: None)
    monkeypatch.setattr(guard, "proc", lambda _: (1, "S"))

    def fail(self: Any) -> dict[str, Any]:  # noqa: ANN401
        raise ValueError("protected teacher failed")

    monkeypatch.setattr(guard.TeacherVerifier, "__call__", fail)
    monkeypatch.setattr(
        guard, "owned_resources", lambda *args, **kwargs: {"cpu_seconds": 0, "rss_bytes": 0}
    )
    with pytest.raises(ValueError, match="protected teacher failed"):
        guard.supervise(config, approval, tmp_path / "output")
    receipt = load(tmp_path / "output/controller.json")
    assert not receipt["complete"] and receipt["teacher_state"]["state"] == "UNVERIFIED"
    assert receipt["worker_exit"] is None


def test_clustered_uncertainty_and_paired_movements_are_descriptive() -> None:
    baseline = arm(320)
    candidate = arm(65)
    rows = {}
    groups = {}
    for domain in ("beatport", "fmak"):
        ids = baseline["best_metrics"][domain]["ids"]
        baseline["best_metrics"][domain]["predictions"] = [1] * len(ids)
        candidate["best_metrics"][domain]["predictions"] = [0] * len(ids)
        for identity in ids:
            rows[identity] = {"ref": 0, "canonical_samples": 300 * 22050}
            groups[identity] = domain + "-single-cluster"
    interval = pair.uncertainty(baseline, candidate, {identity: 0 for identity in rows}, groups)
    assert "adaptive development" in interval["scope"]
    for domain in ("beatport", "fmak"):
        assert interval["domains"][domain]["clusters"] == 1
        assert interval["domains"][domain]["metrics"]["exact"] == {"delta": 1.0, "ci95": [1.0, 1.0]}
    moves = pair.movements(baseline, candidate, rows, groups)
    assert len(moves["selected_per_ID"]) == 376
    assert moves["domains"]["fmak"]["paired_counts"] == {"repaired": 250}
    assert moves["domains"]["beatport"]["duration_strata"]["5-10min"]["tracks"] == 126
