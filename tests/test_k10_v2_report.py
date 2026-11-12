"""Recorded-result checks reject plausible target/prediction/role errors without inference."""

from __future__ import annotations

import copy

import pytest

from tools.check_k10_v2_report import STUDY, check, read, validate_rows


def test_frozen_pair_report() -> None:
    result = check()
    assert result["domains"]["fmak"]["exact"]["delta"] == 0.016
    assert result == read(STUDY / "data/results/paired-uncertainty.json")


@pytest.mark.parametrize("field", ["ref", "expanded_pred", "expanded_relation_sym"])
def test_per_id_target_prediction_relation_corruption(field: str) -> None:
    results = STUDY / "data/results"
    rows = copy.deepcopy(read(results / "selected-per-id-errors.json.gz"))
    if field == "expanded_relation_sym":
        rows[0][field] = "other"
    else:
        rows[0][field] = (rows[0][field] + 1) % 24
    runs = {arm: read(results / (arm + ".json.gz")) for arm in ("restricted", "expanded")}
    plan = read(STUDY / "data/canonical-eligibility-v2/plan-v2.json.gz")
    with pytest.raises(ValueError):
        validate_rows(rows, runs, plan)
