"""Synthetic terminal receipts: no checkpoint, process inspection or real run reads."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from experiments import musicfm_bn_terminal as terminal


def encode(value: object) -> bytes:
    return json.dumps(value, sort_keys=True).encode()


@pytest.fixture
def bundle(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    config = {
        "mode": "fit",
        "eval_batch_size": 16,
        "source_sha256": {"source": "a" * 64},
        "caps": {"wall_seconds": 1000, "rss_bytes": 10000, "cpu_seconds": 2000},
    }
    config_sha = hashlib.sha256(encode(config)).hexdigest()
    monkeypatch.setattr(terminal, "ORIGINAL_CONFIG_SHA", config_sha)
    best: dict[str, dict[str, Any]] = {
        domain: {
            "ids": [f"{domain}:{index}" for index in range(count)],
            "tracks": count,
            "exact_correct": correct,
            "symmetric_units10": 900,
        }
        for domain, count, correct in (("beatport", 126, 83), ("fmak", 250, 166))
    }
    history = []
    for update in range(69, 8281, 69):
        metrics = copy.deepcopy(best)
        if update != 828:
            metrics["beatport"]["exact_correct"] -= 1
        history.append(
            dict(
                update=update,
                metrics=metrics,
                validation={domain: {"elapsed_seconds": 3.0} for domain in best},
            )
        )
    result = dict(
        status="COMPLETE_FIXED_PILOT_NOT_RELEASE_EVIDENCE",
        updates=8280,
        events=120,
        history=history,
        provenance={"config_sha256": config_sha, "source_sha256": config["source_sha256"]},
        confirmation_used=False,
        calibration_used=False,
        best_update=828,
        best_metrics=best,
        best_checkpoint_sha256="b" * 64,
        continuation_gate={
            "continuation_earned": False,
            "confirmation_scored": False,
            "release_or_distillation_authorized": False,
            "checks_against_each_control": {"restricted": {}, "expanded": {}},
        },
        selected_teacher_composed_batch_proof={
            "passed": True,
            "tracks": 376,
            "argmax_disagreements": 0,
            "max_absolute_logit_error": 1e-6,
            "per_song": [
                {"id": identity, "seconds": 0.1}
                for domain in best.values()
                for identity in domain["ids"]
            ],
        },
    )
    return dict(
        config=config,
        result=result,
        controller={
            "state": "COMPLETE_NO_RAM_OWNERSHIP",
            "worker_started": True,
            "worker_exit": 0,
            "wall_seconds": 500,
            "peaks": {"rss": 1000, "cpu": 1000},
            "terminal_resources": {"rss_bytes": 500, "cpu_seconds": 1000},
        },
        success={"mode": "fit", "training_approved": False},
        launch={"config_sha256": config_sha, "pid": 1307, "guardian_start_ticks": 174372685},
    )


def files(bundle: dict[str, Any]) -> dict[str, bytes]:
    value = copy.deepcopy(bundle)
    value["success"]["result_sha256"] = hashlib.sha256(encode(value["result"])).hexdigest()
    return {name: encode(document) for name, document in value.items()}


def test_terminal_barrier_preserves_failed_gate_and_uses_comparable_timing(
    bundle: dict[str, Any],
) -> None:
    proof = terminal.verify_terminal_teacher(
        files(bundle), "b" * 64, lambda _: None, failure_receipts_absent=True
    )
    assert proof.best_update == 828 and proof.original_gate_earned is False
    assert proof.historical_b1_seconds == pytest.approx(37.6)
    assert proof.historical_b16_seconds == 6
    assert proof.complete_evaluation_seconds == pytest.approx(81.2)


@pytest.mark.parametrize("pid", [1294, 1307])
def test_live_original_identity_blocks_even_complete_receipts(
    bundle: dict[str, Any], pid: int
) -> None:
    with pytest.raises(ValueError, match="retirement"):
        terminal.verify_terminal_teacher(
            files(bundle),
            "b" * 64,
            lambda observed: terminal.ORIGINAL_PROCESSES[pid] if observed == pid else None,
            failure_receipts_absent=True,
        )
    # A reused numeric PID with a different start is not the original process.
    terminal.verify_terminal_teacher(
        files(bundle), "b" * 64, lambda _: 999, failure_receipts_absent=True
    )


@pytest.mark.parametrize(
    "defect",
    [
        "horizon",
        "cadence",
        "source",
        "selection",
        "checkpoint",
        "controller",
        "audit",
        "coverage",
        "timing",
        "gate",
    ],
)
def test_incomplete_or_inconsistent_original_state_cannot_start_diagnostic(
    bundle: dict[str, Any], defect: str
) -> None:
    digest = "b" * 64
    result = bundle["result"]
    if defect == "horizon":
        result["history"].pop()
    elif defect == "cadence":
        result["history"][-1]["update"] = 8279
    elif defect == "source":
        result["provenance"]["source_sha256"] = {}
    elif defect == "selection":
        result["best_update"] = 8280
    elif defect == "checkpoint":
        digest = "c" * 64
    elif defect == "controller":
        bundle["controller"]["worker_exit"] = 1
    elif defect == "audit":
        result["selected_teacher_composed_batch_proof"]["argmax_disagreements"] = 1
    elif defect == "coverage":
        result["best_metrics"]["beatport"]["ids"].pop()
        result["history"][11]["metrics"] = copy.deepcopy(result["best_metrics"])
    elif defect == "timing":
        result["selected_teacher_composed_batch_proof"]["per_song"][0]["seconds"] = float("nan")
    else:
        result["continuation_gate"]["release_or_distillation_authorized"] = True
    with pytest.raises(ValueError):
        terminal.verify_terminal_teacher(
            files(bundle), digest, lambda _: None, failure_receipts_absent=True
        )


def test_failure_receipt_or_late_cap_breach_overrides_success(bundle: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="failure receipt"):
        terminal.verify_terminal_teacher(
            files(bundle), "b" * 64, lambda _: None, failure_receipts_absent=False
        )
    bundle["controller"]["peaks"]["cpu"] = 2001
    with pytest.raises(ValueError, match="caps"):
        terminal.verify_terminal_teacher(
            files(bundle), "b" * 64, lambda _: None, failure_receipts_absent=True
        )


def test_hash_relationships_and_missing_documents_fail_closed(bundle: dict[str, Any]) -> None:
    value = files(bundle)
    value["result"] += b" "
    with pytest.raises(ValueError, match="receipts"):
        terminal.verify_terminal_teacher(
            value, "b" * 64, lambda _: None, failure_receipts_absent=True
        )
    value = files(bundle)
    value["config"] += b" "
    with pytest.raises(ValueError, match="configuration"):
        terminal.verify_terminal_teacher(
            value, "b" * 64, lambda _: None, failure_receipts_absent=True
        )
    value.pop("launch")
    with pytest.raises(ValueError, match="five"):
        terminal.verify_terminal_teacher(
            value, "b" * 64, lambda _: None, failure_receipts_absent=True
        )
