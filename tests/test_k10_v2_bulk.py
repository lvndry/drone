"""Bulk dispatch/identity guard regressions; no audio decode or models."""

from __future__ import annotations

import copy

from tools.prepare_k10_v2_bulk import canonical_audit
from tools.run_k10_v2_bulk import bulk_violation


def test_actual_canonical_conflicts_block_without_repair() -> None:
    plan = {"bp_fit": ["bp:fit"], "expanded_fma_fit": ["fm:fit"]}
    rows = [
        {"id": "bp:fit", "ref": 0, "canonical_model_input_sha256": "same"},
        {"id": "fm:inner", "ref": 1, "canonical_model_input_sha256": "same"},
    ]
    original = copy.deepcopy(rows)
    result = canonical_audit(rows, plan)
    assert not result["complete"]
    assert (
        len(result["cross_fit_selection_families"])
        == len(result["conflicting_target_families"])
        == 1
    )
    assert result["no_role_or_label_repair"] and rows == original
    rows[1] = {"id": "fm:fit", "ref": 0, "canonical_model_input_sha256": "same"}
    result = canonical_audit(rows, plan)
    assert result["complete"] and len(result["duplicate_families"]) == 1


def test_owned_output_cap_is_additional_to_existing_process_caps() -> None:
    caps = {
        "wall_seconds": 2700,
        "cpu_seconds": 5400,
        "aggregate_rss_bytes": 17179869184,
        "new_cache_bytes": 3221225472,
        "wav_scratch_bytes": 1073741824,
        "total_owned_output_bytes": 4294967296,
    }
    sample = {
        "wall_seconds": 1,
        "cpu_seconds": 1,
        "rss_bytes": 1,
        "cache_bytes": 1,
        "wav_bytes": 1,
        "total_owned_bytes": 1,
    }
    assert bulk_violation(sample, caps) is None
    assert bulk_violation({**sample, "total_owned_bytes": 4294967297}, caps) == "total_owned_bytes"
    assert bulk_violation({**sample, "cpu_seconds": 5401}, caps) == "cpu_seconds"
