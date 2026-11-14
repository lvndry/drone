"""Canonical eligibility amendment is deterministic filtering, never resplitting/relabelling."""

from __future__ import annotations

import copy
import gzip
import json

from drone.features import FEATURE_SPEC
from drone.keys import LABEL_SCHEMA
from tools.amend_k10_v2_eligibility import derive, exclusions, uncertainty
from tools.prepare_k10_v2_pilot import CONTRACT


def fixture() -> tuple[dict, dict, dict, dict]:
    identities = ["beatport:fit", "beatport:selection", "fmak:bad1", "fmak:bad2", "fmak:keep"]
    references = [23, 23, 1, 15, 3]
    canonical = ["bp", "bp", "bad", "bad", "keep"]
    rows = [
        {
            "id": identity,
            "domain": identity.split(":")[0],
            "role": "validation" if identity == "beatport:selection" else "train",
            "sha256": identity,
            "ref": reference,
            "canonical_model_input_sha256": canonical_hash,
            "frames": 320,
            "cache_sha256": canonical_hash,
        }
        for identity, reference, canonical_hash in zip(
            identities, references, canonical, strict=True
        )
    ]
    plan = {
        "bp_fit": ["beatport:fit"],
        "bp_selection_guard": ["beatport:selection"],
        "restricted_fma_fit": ["fmak:keep"],
        "expanded_fma_fit": ["fmak:bad1", "fmak:bad2", "fmak:keep"],
        "fma_inner_selection": [],
        "train_feature_bindings": {row["id"]: row["sha256"] for row in rows},
        "expected_refs": dict(zip(identities, references, strict=True)),
        "label_export_sha256": "parent",
        "schedule": {"bp_mass": [1230, 2206]},
    }
    labels = {"labels": copy.deepcopy(plan["expected_refs"])}
    manifest = {
        "rows": rows,
        "complete": False,
        "feature_spec": FEATURE_SPEC,
        "label_schema": LABEL_SCHEMA,
        "canonical_identity_contract": CONTRACT,
        "label_export_sha256": "parent",
    }
    groups = {"fmak:bad1": "bad", "fmak:bad2": "bad", "fmak:keep": "keep"}
    return plan, labels, manifest, groups


def test_general_rules_keep_selection_and_quarantine_all_conflicting_members() -> None:
    plan, _, manifest, _ = fixture()
    rows = manifest["rows"]
    original = copy.deepcopy(rows)
    removed, _ = exclusions(rows, plan)
    assert set(removed) == {"beatport:fit", "fmak:bad1", "fmak:bad2"}
    assert rows == original and "beatport:selection" not in removed


def test_derivative_changes_only_eligibility_not_labels_roles_or_sampling() -> None:
    plan, labels, manifest, groups = fixture()
    original = copy.deepcopy((plan, labels, manifest))
    provenance = {
        "parent_plan_sha256": "p",
        "parent_labels_sha256": "l",
        "parent_manifest_sha256": "m",
        "amender_source_sha256": "s",
    }
    outputs = derive(plan, labels, manifest, groups, provenance)
    amended_plan = json.loads(gzip.decompress(outputs["plan-v2.json.gz"]))
    targets = json.loads(gzip.decompress(outputs["numeric-labels-v2.json.gz"]))
    feature = json.loads(gzip.decompress(outputs["feature-manifest-v2.json.gz"]))
    assert (
        amended_plan["bp_selection_guard"] == plan["bp_selection_guard"]
        and amended_plan["restricted_fma_fit"] == plan["restricted_fma_fit"]
    )
    assert amended_plan["schedule"] == plan["schedule"] and amended_plan["expected_refs"] == {
        "beatport:selection": 23,
        "fmak:keep": 3,
    }
    assert targets["labels"] == amended_plan["expected_refs"] and feature["complete"]
    assert {row["id"]: row["role"] for row in feature["rows"]} == {
        "beatport:selection": "validation",
        "fmak:keep": "train",
    }
    assert (plan, labels, manifest) == original


def test_evaluation_identity_merges_only_derived_uncertainty_groups() -> None:
    rows = [
        {"id": "fmak:a", "domain": "fmak", "ref": 0, "canonical_model_input_sha256": "same"},
        {"id": "fmak:b", "domain": "fmak", "ref": 0, "canonical_model_input_sha256": "same"},
    ]
    plan = {"bp_fit": [], "expanded_fma_fit": []}
    groups = {"fmak:a": "component1", "fmak:b": "component2"}
    raw, witness = uncertainty(rows, plan, groups)
    assert len(witness["evaluation_uncertainty_merges"]) == 1
    assert gzip.decompress(raw).count(b"fma:component1") >= 2
    assert groups == {"fmak:a": "component1", "fmak:b": "component2"}
