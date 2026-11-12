"""Tiny synthetic and metadata-only tests; no real GPU/model/audio execution."""

import gzip
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import experiments.k10_context_pair as shared
import experiments.k10_resolution_fit as fit
from tests import test_k10_context_pair as context_tests
from tests.test_k10_context_pair import Tiny, arm


@pytest.fixture
def tiny_loop(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    fixture: Any = context_tests.tiny_loop
    return fixture.__wrapped__(monkeypatch)


def test_default_cpu_equals_explicit_factory_state_and_trajectory(
    tmp_path: Path, tiny_loop: tuple[dict[str, Any], dict[str, dict[str, Any]]]
) -> None:
    p, rows = tiny_loop
    raw = gzip.decompress(
        (Path(__file__).parent / "data/k10-context-training-90a318e.py.gz").read_bytes()
    )
    assert (
        hashlib.sha256(raw).hexdigest()
        == "dc7d35bc2fc3a5903b3fcce96b2801ca9cec496abd994aeb67c5378c4ec7eb72"
    )
    legacy: dict[str, Any] = {**vars(shared), "torch": torch}
    exec(compile(raw, "frozen-90a318e-numerical-functions", "exec"), legacy)
    results = []
    cases: list[tuple[str, dict[str, Any]]] = [
        ("legacy90", {}),
        ("default", {}),
        ("explicit", {"model_factory": Tiny, "expected_init": shared.INITIAL, "device": "cpu"}),
    ]
    for name, kwargs in cases:
        trainer: Any = legacy["train_arm"] if name == "legacy90" else shared.train_arm
        results.append(
            trainer(
                320,
                p,
                rows,
                lambda _: np.full((208, 104), 0.125, np.float16),
                tmp_path / name,
                {"common": True},
                lambda: None,
                **kwargs,
            )
        )
    for result in results[1:]:
        for key in results[0]:
            if key not in (
                "elapsed_seconds",
                "best_checkpoint_sha256",
                "final_checkpoint_sha256",
                "selected_reload_proof",
            ):
                assert results[0][key] == result[key], key
    for checkpoint in ("best.pt", "final.pt"):
        a, b = (
            torch.load(tmp_path / n / checkpoint, weights_only=True)
            for n in ("legacy90", "explicit")
        )
        assert a["format"] == b["format"] and a["provenance"] == b["provenance"]
        assert all(torch.equal(v, b["state_dict"][k]) for k, v in a["state_dict"].items())


def test_explicit_artifact_and_init_are_enforced(
    tmp_path: Path, tiny_loop: tuple[dict[str, Any], dict[str, dict[str, Any]]]
) -> None:
    p, rows = tiny_loop
    args = (320, p, rows, lambda _: np.zeros((208, 104), np.float16))
    with pytest.raises(ValueError, match="Fresh initialization"):
        shared.train_arm(
            *args, tmp_path / "bad", {}, lambda: None, model_factory=Tiny, expected_init="wrong"
        )
    assert not (tmp_path / "bad").exists()
    result = shared.train_arm(
        *args,
        tmp_path / "ok",
        {},
        lambda: None,
        model_factory=Tiny,
        expected_init=shared.INITIAL,
        artifact_format=fit.FORMAT,
    )
    assert result["selected_reload_proof"]["scores_identical"]
    assert torch.load(tmp_path / "ok/best.pt", weights_only=True)["format"] == fit.FORMAT


def gpu_arms() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    a, b, old = arm(320), arm(320, fm=181), arm(320)
    b["initial_state_sha256"] = fit.CANDIDATE_INITIAL
    for name, x in (("baseline", a), ("candidate", b)):
        x["status"] = fit.STATUS
        x["provenance"]["model_name"] = name
        x.update(temporal_sha256="same", bp_augmented_sha256="same", actual_frames={"beatport": 1})
    return a, b, old


def test_fresh_gpu_control_gate_and_failed_control_are_distinct() -> None:
    a, b, old = gpu_arms()
    assert fit.compare(a, b, old)["continuation_earned"]
    b["best_metrics"]["fmak"]["exact_correct"] -= 1
    assert not fit.compare(a, b, old)["resolution_gate"]
    b["best_metrics"]["fmak"]["exact_correct"] += 1
    old["best_metrics"]["beatport"]["symmetric_units10"] += 1
    gate = fit.compare(a, b, old)
    assert gate["resolution_gate"] and not gate["historical_GPU_control_noninferiority_guard"]
    assert not gate["continuation_earned"] and not gate["promotion_authorized"]


@pytest.mark.parametrize(
    "field", ["temporal_sha256", "coverage", "provenance", "initial_state_sha256"]
)
def test_pair_rejects_unmatched_runtime_crop_or_initialization(field: str) -> None:
    a, b, old = gpu_arms()
    b[field] = "different"
    with pytest.raises(ValueError):
        fit.compare(a, b, old)


def test_gpu_status_and_all_common_provenance_fields_are_required() -> None:
    a, b, old = gpu_arms()
    b["status"] = "COMPLETE_CONTEXT_ARM_NOT_PROMOTED"
    with pytest.raises(ValueError, match="Incomplete"):
        fit.compare(a, b, old)
    b["status"] = fit.STATUS
    b["provenance"]["runtime"] = "other"
    with pytest.raises(ValueError, match="common provenance"):
        fit.compare(a, b, old)


def test_gpu_arm_injects_factory_identity_without_constructing_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}
    monkeypatch.setattr(fit, "configure_gpu", lambda: None)

    def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        captured.update(args=args, kwargs=kwargs)
        return {"not_executed": True}

    monkeypatch.setattr(fit, "train_arm", capture)
    fit.arm(
        "candidate", {}, {}, lambda _: np.zeros((208, 8)), tmp_path, {"same": True}, lambda: None
    )
    assert captured["args"][0] == 320
    assert captured["args"][5] == {"same": True, "model_name": "candidate"}
    assert captured["kwargs"]["expected_init"] == fit.CANDIDATE_INITIAL
    assert captured["kwargs"]["artifact_format"] == fit.FORMAT
    assert captured["kwargs"]["result_status"] == fit.STATUS
    assert captured["kwargs"]["device"] == "cuda:0"
