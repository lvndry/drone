"""Prospective pairing/selection boundaries, without models or audio."""

from __future__ import annotations

import copy
import hashlib
from fractions import Fraction
from typing import Any

import pytest

from drone.features import FEATURE_SPEC
from drone.keys import LABEL_SCHEMA
from experiments import k10_v2_pilot as pilot


def metrics(bp: int, fma: int, sym_bp: int = 800, sym_fma: int = 1600) -> dict[str, dict[str, Any]]:
    return {
        "beatport": {
            "tracks": 126,
            "exact_correct": bp,
            "symmetric_units10": sym_bp,
            "asymmetric_units10": sym_bp,
            "ids": [f"beatport:{i}" for i in range(126)],
        },
        "fmak": {
            "tracks": 250,
            "exact_correct": fma,
            "symmetric_units10": sym_fma,
            "asymmetric_units10": sym_fma,
            "ids": [f"fmak:{i}" for i in range(250)],
        },
    }


def test_fixed_stream_pairing_and_exact_reference_mass() -> None:
    bp = [f"beatport:{i}" for i in range(1230)]
    restricted = [f"fmak:{i}" for i in range(976)]
    expanded = [f"fmak:{i}" for i in range(4035)]
    counts = {"beatport": 0, "fmak": 0}
    seen_small = set()
    seen_large = set()
    for step, (a, b) in enumerate(
        zip(pilot.paired_batches(bp, restricted), pilot.paired_batches(bp, expanded), strict=True),
        1,
    ):
        assert len(a) == len(b) == 32
        assert [d for d in a if d.domain == "beatport"] == [d for d in b if d.domain == "beatport"]
        for draw in a:
            counts[draw.domain] += 1
        seen_small.update(d.identity for d in a if d.domain == "fmak")
        seen_large.update(d.identity for d in b if d.domain == "fmak")
        assert counts["beatport"] == (step * 32 * 1230) // 2206
    assert counts == {"beatport": 147733, "fmak": 117227}
    assert len(seen_small) == 976 and len(seen_large) == 4035


def test_exact_first_equal_domain_selector_is_not_pooled() -> None:
    first, second = metrics(100, 100), metrics(70, 140)
    assert (100 + 100) < (70 + 140)  # pooled accuracy would choose second
    assert pilot.selection_key(first) > pilot.selection_key(second)
    assert pilot.selection_key(metrics(101, 100, 0, 0)) > pilot.selection_key(first)
    assert pilot.selection_key(first) == pilot.selection_key(
        copy.deepcopy(first)
    )  # earliest kept by strict >
    assert pilot.selection_key(metrics(100, 100, 801, 1600)) > pilot.selection_key(first)


def feature_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    plan = {
        "train_feature_bindings": {"beatport:a": "sha-a", "fmak:b": "sha-b"},
        "bp_fit": ["beatport:a"],
        "expanded_fma_fit": [],
        "expected_refs": {"beatport:a": 0, "fmak:b": 0},
        "label_export_sha256": "labels",
    }
    manifest = {
        "complete": True,
        "label_export_sha256": "labels",
        "feature_spec": FEATURE_SPEC,
        "label_schema": LABEL_SCHEMA,
        "canonical_identity_contract": "drone.features.decode_mp3/read_wav_mono/resample_linear:22050:float32le-v1",
        "rows": [
            {
                "id": i,
                "sha256": sha,
                "ref": 0,
                "frames": 320,
                "cache_sha256": "cache",
                "canonical_model_input_sha256": i,
            }
            for i, sha in plan["train_feature_bindings"].items()
        ],
    }
    return plan, manifest


def test_training_manifest_rejects_confirmation_and_unverified_inputs() -> None:
    plan, manifest = feature_fixture()
    pilot.validate_feature_manifest(plan, manifest)
    extra = copy.deepcopy(manifest)
    extra["rows"].append({"id": "fmak:confirmation"})
    with pytest.raises(ValueError, match="no confirmation/calibration"):
        pilot.validate_feature_manifest(plan, extra)
    wrong = copy.deepcopy(manifest)
    wrong["rows"][0]["sha256"] = "changed"
    with pytest.raises(ValueError, match="binding differs"):
        pilot.validate_feature_manifest(plan, wrong)
    wrong = copy.deepcopy(manifest)
    wrong["rows"][0].pop("canonical_model_input_sha256")
    with pytest.raises(ValueError, match="identity verification required"):
        pilot.validate_feature_manifest(plan, wrong)
    wrong = copy.deepcopy(manifest)
    wrong["rows"][1]["canonical_model_input_sha256"] = "beatport:a"
    with pytest.raises(ValueError, match="crosses fit/selection"):
        pilot.validate_feature_manifest(plan, wrong)


def test_hash_bound_training_permit() -> None:
    raw = b"frozen proposed plan"
    with pytest.raises(ValueError, match="permit required"):
        pilot.approve(raw, {})
    pilot.approve(raw, {"allow_training": True, "plan_sha256": hashlib.sha256(raw).hexdigest()})
    with pytest.raises(ValueError, match="permit required"):
        pilot.approve(
            raw + b"changed",
            {"allow_training": True, "plan_sha256": hashlib.sha256(raw).hexdigest()},
        )


def run(arm: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "device": "cuda:0",
        "status": "complete",
        "updates": 8280,
        "confirmation_used": False,
        "calibration_used": False,
        "source_sha256": {"driver": "same"},
        "runtime_versions": {"python": "same", "torch": "same", "numpy": "same"},
        "config": {"updates": 8280},
        "history": [{}] * 120,
        "plan_sha256": "same",
        "feature_manifest_sha256": "same",
        "initial_state_sha256": "same",
        "bp_trace_sha256": "same",
        "best_metrics": metrics(100, 150),
    }


def test_gate_requires_full_pairing_and_does_not_score_confirmation() -> None:
    a, b = run("restricted"), run("expanded")
    b["best_metrics"] = metrics(100, 155)
    result = pilot.pilot_gate(a, b)
    assert result["confirmation_earned"] and not result["confirmation_scored"]
    assert Fraction(str(result["fma_exact_gain"])) == Fraction(2, 100)
    b["best_metrics"] = metrics(99, 155)
    assert not pilot.pilot_gate(a, b)["confirmation_earned"]
    b["bp_trace_sha256"] = "different"
    with pytest.raises(ValueError, match="Pairing/provenance mismatch"):
        pilot.pilot_gate(a, b)
    b = run("expanded")
    b["updates"] = 69
    with pytest.raises(ValueError, match="complete fixed updates"):
        pilot.pilot_gate(a, b)


def test_gate_rejects_source_config_calibration_and_selection_order() -> None:
    for field, changed in (("source_sha256", {"driver": "changed"}), ("config", {"updates": 100})):
        a, b = run("restricted"), run("expanded")
        b[field] = changed
        with pytest.raises(ValueError, match="Pairing/provenance mismatch"):
            pilot.pilot_gate(a, b)
    a, b = run("restricted"), run("expanded")
    b["calibration_used"] = True
    with pytest.raises(ValueError, match="complete fixed updates"):
        pilot.pilot_gate(a, b)
    b = run("expanded")
    b["best_metrics"]["fmak"]["ids"].reverse()
    with pytest.raises(ValueError, match="ID ordering/counts differ"):
        pilot.pilot_gate(a, b)
    b = run("expanded")
    b["best_metrics"] = metrics(100, 155, sym_fma=1590)
    assert pilot.pilot_gate(a, b)["confirmation_earned"]  # FMA weighted change reported,not gated


def test_stale_numeric_key_encoding_is_rejected() -> None:
    plan, manifest = feature_fixture()
    manifest["rows"][0]["ref"] = 1
    with pytest.raises(ValueError, match="binding differs"):
        pilot.validate_feature_manifest(plan, manifest)


def test_preprocessing_allowlist_rejects_reserved_before_opening_audio() -> None:
    from pathlib import Path

    from tools.prepare_k10_v2_pilot import selected_rows

    plan = {
        "train_feature_bindings": {"fmak:000001": "expected"},
        "bp_fit": [],
        "expanded_fma_fit": ["fmak:000001"],
        "expected_refs": {"fmak:000001": 3},
    }
    result = selected_rows(plan, {"rows": []}, ["fmak:000001"], Path("/nonexistent/public/fma"))
    assert result[0]["path"] == "/nonexistent/public/fma/000001.mp3"
    with pytest.raises(ValueError, match="no calibration/confirmation audio"):
        selected_rows(plan, {"rows": []}, ["fmak:reserved"], Path("/private-never-open"))


def test_outer_guard_accounts_reaped_cpu_and_every_budget() -> None:
    from tools.run_k10_v2_preprocessing import proc_fields, violation

    fields = ["S"] + ["0"] * 23
    fields[2] = "123"
    fields[11] = "100"
    fields[12] = "50"
    fields[13] = "25"
    fields[14] = "25"
    fields[21] = "10"
    assert proc_fields("99 (worker name) " + " ".join(fields), 100, 4096) == (123, 2.0, 40960, "S")
    caps = {
        "wall_seconds": 900,
        "cpu_seconds": 1800,
        "aggregate_rss_bytes": 4294967296,
        "new_cache_bytes": 134217728,
        "wav_scratch_bytes": 268435456,
    }
    sample = {"wall_seconds": 1, "cpu_seconds": 2, "rss_bytes": 1, "cache_bytes": 1, "wav_bytes": 1}
    assert violation(sample, caps) is None
    for field, key in (
        ("wall_seconds", "wall_seconds"),
        ("cpu_seconds", "cpu_seconds"),
        ("rss_bytes", "aggregate_rss_bytes"),
        ("cache_bytes", "new_cache_bytes"),
        ("wav_bytes", "wav_scratch_bytes"),
    ):
        changed = {**sample, field: caps[key] + 1}
        assert violation(changed, caps) == field
