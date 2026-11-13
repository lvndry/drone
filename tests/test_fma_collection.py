"""Check published source/cohort metadata against existing blinded evidence."""

from __future__ import annotations

import csv
import gzip
import io
import json
import re

import pytest

from experiments.collection_catalog import CollectionCatalog, collection_bytes, load_collection
from tools.export_fma_collection import DATA, ROOT, id_value, summary


@pytest.fixture(scope="module")
def catalog() -> CollectionCatalog:
    return load_collection(DATA / "fmak5489-collection.json.gz")


def test_all5489_source_identities_preserve_aliases_and_have_no_targets(
    catalog: CollectionCatalog,
) -> None:
    groups = json.loads(gzip.decompress((DATA / "input.json.gz").read_bytes()))
    expected = {row["identity"]: row for row in groups["tracks"]}
    assert {row.track_id for row in catalog.tracks} == set(expected)
    assert len(catalog.tracks) == 5489
    assert len({row.source_audio.sha256 for row in catalog.tracks}) == 5442
    assert all(
        row.source_audio.sha256 == expected[row.track_id]["audio_sha256"] for row in catalog.tracks
    )
    raw = collection_bytes(catalog)
    assert b'"target"' not in raw and b'"key_label"' not in raw and b'"predictions"' not in raw
    assert b"/tmp/" not in raw and b"/home/" not in raw


def test_reserved_and_excluded_support_is_unknown_not_publisher_duration(
    catalog: CollectionCatalog,
) -> None:
    actual = summary(catalog)
    assert (actual["analyzed_support_known"], actual["analyzed_support_unknown"]) == (4283, 1206)
    assert all(row.bpm.value is None for row in catalog.tracks)
    assert all(row.source_duration.value is None for row in catalog.tracks)
    assert all(row.catalog_recording_duration.value is not None for row in catalog.tracks)
    for row in catalog.tracks:
        support = row.analyzed_audio
        assert support.support_kind == "recorded_24khz_training_cache"
        if support.samples is None:
            assert support.seconds is None and support.pcm_identity.sha256 is None
            assert support.missing_reason
        else:
            assert support.sample_rate == 24000 and support.channels == 1 and support.dtype == "<f4"
            assert support.seconds == support.samples / 24000
            assert support.witness_kind == "saved_cache_metadata_not_new_decode_verification"
    assert actual["missing_artist"] == 0 and actual["missing_album"] == 30


def test_native_taxonomy_and_components_are_not_inferred(catalog: CollectionCatalog) -> None:
    aggregate = json.loads(gzip.decompress((DATA / "coverage-aggregate.json.gz").read_bytes()))
    taxonomy = {row["id"]: row["title"] for row in aggregate["fma"]["all_taxonomy_inventory"]}
    assert len(taxonomy) == 163
    assert all(
        tag.name == taxonomy[tag.tag_id] and tag.relation == "unspecified"
        for row in catalog.tracks
        for tag in row.genre_tags
    )
    actual = summary(catalog)
    assert actual["missing_native_top"] == 3065 and actual["empty_native_all_tags"] == 84
    assert actual["frozen_components"] == 1157
    rows = list(
        csv.DictReader(
            io.StringIO(gzip.decompress((DATA / "exposure-ledger.csv.gz").read_bytes()).decode())
        )
    )
    by_id = {row["id"]: row for row in rows}
    assert all(
        row.frozen_groups[0].group_id == by_id[row.track_id]["component"] for row in catalog.tracks
    )
    assert all(
        row.exposures[0].relation == by_id[row.track_id]["historical_exposure"]
        for row in catalog.tracks
    )
    assert all(row.exposures[0].subject_kind == "research_program" for row in catalog.tracks)


def test_frozen_roles_match_exact_allowlisted_plan_arrays(catalog: CollectionCatalog) -> None:
    path = ROOT / "experiments/studies/k10-v2-pilot/data/canonical-eligibility-v2/plan-v2.json.gz"
    text = gzip.decompress(path.read_bytes()).decode()
    for field, count in (
        ("expanded_fma_fit", 4033),
        ("restricted_fma_fit", 976),
        ("fma_inner_selection", 250),
        ("confirmation_reserved_container_unique", 997),
    ):
        match = re.search(r'"' + field + r'"\s*:\s*(\[[^\]]*\])', text)
        assert match is not None
        expected = set(json.loads(match[1]))
        actual = {
            row.track_id
            for row in catalog.tracks
            if any(
                item.plan_id == "canonical-eligibility-v2" and item.role == field
                for item in row.memberships
            )
        }
        assert actual == expected and len(actual) == count


def test_invalid_entity_sentinels_remain_unknown() -> None:
    assert id_value("") is None and id_value("0") is None and id_value("nan") is None
    assert id_value("123") == "123"
