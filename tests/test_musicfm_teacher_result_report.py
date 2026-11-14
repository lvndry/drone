"""Negative-result preservation and tamper rejection without audio or model loading."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from tools.musicfm_teacher_result_report import STUDY, load_evidence, verify


@pytest.fixture(scope="module")
def evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    return load_evidence()


def test_full_horizon_negative_result_and_actual_cost(
    evidence: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    result = verify(*evidence)
    assert (result["updates"], result["best_update"], result["best_event"]) == (8280, 828, 12)
    assert not result["continuation_earned"]
    assert all(not any(checks.values()) for checks in result["control_checks"].values())
    assert result["metrics"]["teacher"]["beatport"]["exact_correct"] == 83
    assert result["metrics"]["teacher"]["fmak"]["exact_correct"] == 166
    assert result["resources"]["wall_seconds"] == 31934.955821720185
    assert result["resources"]["cpu_seconds"] == 33450.7
    assert result["proof"]["max_absolute_error"] == 1.2904405592983892e-6
    assert result["proof"]["disagreements"] == 0
    assert result["prospective_complete_comparison_seconds"] == 573.8798344926909
    assert len(result["history"]) == len(result["b16_event_seconds"]) == 120


@pytest.mark.parametrize(
    "damage",
    [
        "horizon",
        "selection",
        "metric",
        "gate",
        "logit",
        "nan",
        "context",
        "prediction",
        "failure",
        "dangling-failure",
        "checkpoint",
        "unstable-checkpoint",
        "live-worker",
        "vllm-restart",
        "cpu-cap",
        "timing",
        "source",
    ],
)
def test_rejects_inconsistent_recorded_claims(
    evidence: tuple[dict[str, Any], dict[str, Any]],
    damage: str,
) -> None:
    docs, index = copy.deepcopy(evidence)
    result = docs["result.json"]
    audit = docs["audit.json"]
    proof = result["selected_teacher_composed_batch_proof"]
    if damage == "horizon":
        result["history"].pop()
    elif damage == "selection":
        result["best_update"] = 897
    elif damage == "metric":
        result["history"][0]["metrics"]["beatport"]["exact_correct"] += 1
    elif damage == "gate":
        result["continuation_gate"]["continuation_earned"] = True
    elif damage in ("logit", "nan"):
        proof["candidate_logits"][0][0] = 100.0 if damage == "logit" else float("nan")
    elif damage == "context":
        proof["per_song"][0]["samples"] -= 1
    elif damage == "prediction":
        result["best_metrics"]["beatport"]["predictions"][0] = 24
    elif damage in ("failure", "dangling-failure"):
        first = next(iter(audit["failures"].values()))
        first["exists" if damage == "failure" else "symlink"] = True
    elif damage == "checkpoint":
        audit["checkpoints"]["best.pt"]["sha256"] = "0" * 64
    elif damage == "unstable-checkpoint":
        audit["checkpoints"]["final.pt"]["identity_after"][4] += 1
    elif damage == "live-worker":
        audit["original_processes"]["1307"] = {"state": "S"}
    elif damage == "vllm-restart":
        audit["protected"][0]["actual"]["start_ticks"] += 1
    elif damage == "cpu-cap":
        docs["controller.json"]["peaks"]["cpu"] = 100001
    elif damage == "timing":
        result["history"][0]["validation"]["fmak"]["elapsed_seconds"] = float("nan")
    else:
        result["provenance"]["source_sha256"]["experiments/musicfm_teacher.py"] = "0" * 64
    with pytest.raises(ValueError):
        verify(docs, index)


def test_terminal_bundle_rejects_byte_damage(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "terminal-index.json").write_bytes((STUDY / "data/terminal-index.json").read_bytes())
    packed = bytearray((STUDY / "data/terminal-raw.json.gz").read_bytes())
    packed[-10] ^= 1
    (data / "terminal-raw.json.gz").write_bytes(packed)
    with pytest.raises(ValueError, match="compressed terminal evidence"):
        load_evidence(tmp_path)
