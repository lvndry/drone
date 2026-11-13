"""GTZAN eligibility, file identity, header support and admission are distinct."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.collection_catalog import CollectionCatalog, collection_bytes, load_collection
from tools.export_gtzan_collection import (
    COVERAGE,
    OUTPUT,
    PINS,
    ROOT,
    coverage,
    join_records,
    load_inputs,
    sha,
)


@pytest.fixture(scope="module")
def inputs() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str]:
    return load_inputs(ROOT)


def test_join_keeps_complete_inventory_and_frozen_cohorts(
    inputs: tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str],
) -> None:
    ledger, index, inventory, _ = inputs
    rows = join_records(ledger, index, inventory)
    assert len(rows) == 1000
    assert sum(row[1]["raw_single_key_reference"] for row in rows) == 837
    assert sum(row[1]["named_filtered_intersection"] for row in rows) == 769
    bad = next(row for row in rows if row[1]["id"] == "jazz.00054")
    assert bad[3]["sha256"] == "eb8be6097ee1f32965fb9545cb1621b4685e4b0d44c6584cf0ab97d244ad535f"
    assert not bad[1]["raw_single_key_reference"] and not bad[1]["named_filtered_intersection"]
    assert not bad[3]["member_identity_verified"] and "wave_error" in bad[3]
    assert all(not row["audio_verified"] for row in ledger)  # Historical snapshot is preserved.


@pytest.mark.parametrize(
    "failure", ["duplicate", "identity", "overlap", "admission", "frames", "cohort", "old_snapshot"]
)
def test_inconsistent_join_evidence_rejected(
    inputs: tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str], failure: str
) -> None:
    ledger, index, inventory, _ = copy.deepcopy(inputs)
    if failure == "duplicate":
        inventory["records"][1] = inventory["records"][0]
    elif failure == "identity":
        inventory["records"][0]["member_identity_verified"] = 1
    elif failure == "overlap":
        inventory["records"][0]["overlap_cleared"] = True
    elif failure == "admission":
        inventory["records"][0]["admitted_for_scoring"] = True
    elif failure == "frames":
        inventory["records"][0]["wave"]["frames"] = True
    elif failure == "cohort":
        ledger[0]["raw_single_key_reference"] = False
        ledger[0]["named_filtered_intersection"] = False
        ledger[0]["metadata_eligible"] = False
    else:
        ledger[0]["audio_verified"] = True
    with pytest.raises(ValueError):
        join_records(ledger, index, inventory)


def test_identity_matching_does_not_imply_parseable_header(
    inputs: tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str],
) -> None:
    ledger, index, inventory, _ = copy.deepcopy(inputs)
    observed = inventory["records"][0]
    observed.pop("wave")
    observed["wave_error"] = "Synthetic independent format failure"
    joined = join_records(ledger, index, inventory)
    first = next(row[3] for row in joined if row[3]["member"] == observed["member"])
    assert first["member_identity_verified"] and "wave" not in first


def test_changed_pinned_metadata_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / PINS["reference_ledger"][0]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"substituted reference history")
    with pytest.raises(ValueError, match="Pinned GTZAN metadata changed"):
        load_inputs(tmp_path)


@pytest.fixture(scope="module")
def catalog() -> CollectionCatalog:
    return load_collection(ROOT / OUTPUT)


def test_catalog_preserves_unknowns_failed_identity_and_unadmitted_roles(
    catalog: CollectionCatalog,
) -> None:
    stats = coverage(catalog)
    assert stats["tracks"] == 1000
    assert stats["recorded_identity_matches"] == stats["recorded_parseable_WAV_headers"] == 999
    assert (
        stats["recorded_identity_failures"]
        == stats["recorded_format_failures"]
        == ["gtzan:jazz.00054"]
    )
    assert stats["cohorts"]["raw_single_key"]["tracks"] == 837
    assert stats["cohorts"]["named_filtered"]["tracks"] == 769
    assert stats["cohorts"]["named_filtered"]["native_genres"]["hiphop"] == 73
    assert stats["cohorts"]["raw_single_key"]["native_genres"]["classical"] == 0
    assert stats["cohorts"]["named_filtered"]["native_genres"]["classical"] == 0
    assert stats["not_admitted_for_scoring"] == stats["overlap_not_established"] == 1000
    assert stats["recorded_source_excerpt_duration_known"] == 999
    assert (
        stats["publisher_full_recording_duration_known"]
        == stats["analyzed_audio_support_known"]
        == stats["bpm_known"]
        == 0
    )
    bad = next(row for row in catalog.tracks if row.source_track_id == "jazz.00054")
    assert (
        bad.source_audio.sha256
        == "eb8be6097ee1f32965fb9545cb1621b4685e4b0d44c6584cf0ab97d244ad535f"
    )
    assert bad.source_duration.value is None
    assert all(not row.relations and not row.frozen_groups for row in catalog.tracks)
    assert all(row.analyzed_audio.pcm_identity.sha256 is None for row in catalog.tracks)
    for source in catalog.sources:
        assert sha((ROOT / source.path).read_bytes()) == source.sha256


def test_catalog_has_no_target_prediction_or_private_path_fields(
    catalog: CollectionCatalog,
) -> None:
    raw = collection_bytes(catalog)
    assert not any(
        token in raw
        for token in (
            b'"drone_ref"',
            b'"annotation_integer"',
            b'"reference"',
            b'"target"',
            b'"prediction"',
            b'"predictions"',
            b"/tmp/",
            b"/home/",
        )
    )
    assert gzip.decompress((ROOT / OUTPUT).read_bytes()) == raw
    receipt = json.loads((ROOT / COVERAGE).read_bytes())
    assert receipt["catalog_sha256"] == sha((ROOT / OUTPUT).read_bytes())
    assert receipt["generator_sha256"] == sha(
        (ROOT / "tools/export_gtzan_collection.py").read_bytes()
    )
