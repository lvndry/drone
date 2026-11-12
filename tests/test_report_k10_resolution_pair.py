from __future__ import annotations

import copy
import json
import tarfile
from typing import Any

import pytest

from tools import report_k10_resolution_pair as report


@pytest.fixture
def arm() -> tuple[dict[str, Any], dict[str, Any]]:
    with tarfile.open(report.STUDY / "data/resolution-pair-raw.tar.gz") as archive:

        def load(name: str) -> dict[str, Any]:
            stream = archive.extractfile(name)
            assert stream is not None
            return json.loads(stream.read())

        return load("run-v1/candidate/result.json"), load("sealed-config.json")


def test_recorded_pair_reproduces_scores_intervals_and_active_cadence() -> None:
    result = report.check()
    assert result["raw_files_verified"] == 24
    assert result["pair"]["continuation_gate"]["continuation_earned"] is False
    assert result["cadence"]["candidate"]["poll_gap_seconds"] < 0.2


@pytest.mark.parametrize("defect", ["horizon", "selection", "reload", "initialization", "source"])
def test_arm_proof_rejects_semantic_changes(
    arm: tuple[dict[str, Any], dict[str, Any]], defect: str
) -> None:
    result, config = copy.deepcopy(arm)
    if defect == "horizon":
        result["history"].pop()
    elif defect == "selection":
        result["best_update"] = 8280
    elif defect == "reload":
        result["selected_reload_proof"]["argmax_disagreements"] = 1
    elif defect == "initialization":
        result["initial_state_sha256"] = "0" * 64
    else:
        result["provenance"]["source_sha256"] = {}
    with pytest.raises(ValueError):
        report.validate_arm(result, "candidate", config)
