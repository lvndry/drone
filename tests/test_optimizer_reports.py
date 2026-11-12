"""Recorded negative-study arithmetic must detect inconsistent movements or promotion."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from tools.check_optimizer_reports import STUDIES, check, validate_pair


def test_recorded_optimizer_reports_reproduce_without_models() -> None:
    result = check()
    assert result["passed"] and result["paired_matrices_verified"] == 16
    assert (
        result["sam_gate"]
        is result["ensemble_teacher_gate"]
        is result["ensemble_bp_hurdle"]
        is False
    )


def test_changed_exact_movement_is_rejected_even_with_same_total() -> None:
    report = json.loads((STUDIES / "k10-sam/data/candidate-result.json").read_text())
    movements = copy.deepcopy(report["confusion_movements"])
    matrix = movements["beatport"]["repo_symmetric_fifths"]["row_control_column_candidate"]
    matrix[0][0] -= 1
    matrix[0][1] += 1
    with pytest.raises(ValueError, match="New exact arithmetic"):
        validate_pair(report["control_metrics"], report["metrics"], movements)


def test_promotion_cannot_be_inserted_into_negative_report(tmp_path: Path) -> None:
    for study in ("k10-sam", "k10-ensemble"):
        destination = tmp_path / study
        destination.mkdir()
        shutil.copytree(STUDIES / study / "data", destination / "data")
        shutil.copyfile(STUDIES / study / "PROTOCOL.md", destination / "PROTOCOL.md")
    path = tmp_path / "k10-ensemble/data/result.json"
    report = json.loads(path.read_text())
    report["promoted"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Promotion/distillation"):
        check(tmp_path)
