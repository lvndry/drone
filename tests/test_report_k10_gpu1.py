from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from tools import report_k10_gpu1 as report


@pytest.fixture
def evidence() -> dict[str, Any]:
    return report.check()


def test_preserved_attempts_and_bundle_verify(evidence: dict[str, Any]) -> None:
    assert set(evidence["attempts"]) == {"v1", "v2", "v3"}


@pytest.mark.parametrize("defect", ["coverage", "initialization", "backend", "cap", "updates"])
def test_semantic_receipts_reject_wrong_workload(evidence: dict[str, Any], defect: str) -> None:
    files = evidence["attempts"]["v3"]
    config = report.document(files, "sealed-config.json")
    result = report.document(files, "run-v1/candidate/result.json")
    if defect == "coverage":
        result["evaluation"]["selection376"]["frames"] -= 1
    elif defect == "initialization":
        result["initial_state_sha256"] = "0" * 64
    elif defect == "backend":
        result["backend_policy"]["global_deterministic_algorithms"] = True
    elif defect == "cap":
        result["peak_reserved_bytes"] = 3 * report.GIB
    else:
        result["update_with_transition_seconds"].pop()
    with pytest.raises(ValueError):
        report.workload(config, result, "candidate")


def test_actual_trace_rejects_overshoot_and_stale_active_sensor(evidence: dict[str, Any]) -> None:
    files = copy.deepcopy(evidence["attempts"]["v3"])
    config = report.document(files, "sealed-config.json")
    launch = report.document(files, "run-v1/candidate-launch.json")["epoch"]
    rows = [json.loads(line) for line in files["run-v1/samples.jsonl"].splitlines()]
    active = next(row for row in rows if row["epoch"] >= launch)
    old = active["sensor_completion_gap_seconds"]
    active["sensor_completion_gap_seconds"] = 0.201
    files["run-v1/samples.jsonl"] = "\n".join(json.dumps(row) for row in rows)
    with pytest.raises(ValueError, match="stale"):
        report.trace(files, config)
    active["sensor_completion_gap_seconds"] = old
    active["owned_bytes"] = 3 * report.GIB
    files["run-v1/samples.jsonl"] = "\n".join(json.dumps(row) for row in rows)
    with pytest.raises(ValueError, match="resource cap"):
        report.trace(files, config)
