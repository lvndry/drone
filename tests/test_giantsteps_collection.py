"""GS source identities, native genres and historical decode support remain distinct."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.collection_catalog import CollectionCatalog, collection_bytes, load_collection
from tools.export_giantsteps_collection import (
    COVERAGE,
    DATA,
    OUTPUT,
    ROOT,
    build,
    coverage,
    join_receipts,
)


@pytest.fixture(scope="module")
def catalog() -> CollectionCatalog:
    return load_collection(ROOT / OUTPUT)


def receipts() -> tuple[dict[str, Any], dict[str, Any]]:
    folder = ROOT / "experiments/studies/bp-gs-overlap/data"
    return (
        json.loads(gzip.decompress((folder / "reference-result-v1.json.gz").read_bytes())),
        json.loads(gzip.decompress((folder / "plan.json.gz").read_bytes())),
    )


def test_source604_identity_aliases_and_no_targets(catalog: CollectionCatalog) -> None:
    reference, plan = receipts()
    selected = join_receipts(reference, plan)
    expected = {row["id"]: row for _, row in selected}
    assert len(catalog.tracks) == 604 and {row.track_id for row in catalog.tracks} == set(expected)
    assert len({row.analyzed_audio.pcm_identity.sha256 for row in catalog.tracks}) == 602
    for row in catalog.tracks:
        assert row.source_audio.sha256 == expected[row.track_id]["container_sha256"]
        assert row.analyzed_audio.pcm_identity.sha256 == expected[row.track_id]["pcm_sha256"]
    raw = collection_bytes(catalog)
    assert not any(
        token in raw
        for token in (b'"key_label"', b'"target"', b'"predictions"', b"/tmp/", b"/home/")
    )


def test_native_genres_not_legacy_buckets_or_guessed_metadata(catalog: CollectionCatalog) -> None:
    actual = coverage(catalog)
    assert actual["native_genres"]["drum-and-bass"] == 38
    assert actual["native_genres"]["glitch-hop"] == 6
    assert actual["native_genres"]["hip-hop"] == 4
    assert all(row.bpm.value is None and row.bpm.missing_reason for row in catalog.tracks)
    assert all(not row.relations and not row.frozen_groups for row in catalog.tracks)
    assert all(
        len(row.genre_tags) == 1 and row.genre_tags[0].name == row.native_genre_top
        for row in catalog.tracks
    )


def test_saved_decode_not_source_probe_training_cache_or_fullsong(
    catalog: CollectionCatalog,
) -> None:
    reference, _ = receipts()
    expected = {row["id"]: row for row in reference["records"] if row["domain"] == "giantsteps"}
    for row in catalog.tracks:
        support = row.analyzed_audio
        assert support.support_kind == "saved_11025Hz_fingerprint_decode"
        assert support.sample_rate == 11025 and support.channels == 1 and support.dtype == "<f4"
        assert support.samples == expected[row.track_id]["frames"]
        assert support.samples is not None
        assert support.seconds == support.samples / 11025
        assert row.source_duration.value is None
        assert (
            row.catalog_recording_duration.method
            == "publisher_JAMS_available_preview_duration_not_full_song"
        )
        assert "no_new_audio_read" in support.witness_kind


def test_all604_have_model_scoped_evaluation_exposure(catalog: CollectionCatalog) -> None:
    release = json.loads(
        (ROOT / "experiments/studies/bp-gs-overlap/data/release-metadata.json").read_bytes()
    )
    assert all(len(row.exposures) == 1 for row in catalog.tracks)
    assert all(
        row.exposures[0].checkpoint_sha256 == release["checkpoint_sha256"] for row in catalog.tracks
    )
    assert all(
        row.exposures[0].relation
        == "published_full604_evaluation_development_exposed_direct_fit_not_established"
        for row in catalog.tracks
    )
    assert all(row.exposures[0].subject_id == "drone-v1" for row in catalog.tracks)


@pytest.mark.parametrize("mutation", ["duplicate", "hash", "incomplete", "frames", "rate"])
def test_receipt_join_rejects_false_identity_or_support(mutation: str) -> None:
    reference, plan = receipts()
    row = next(row for row in reference["records"] if row["domain"] == "giantsteps")
    if mutation == "duplicate":
        reference["records"].append(copy.deepcopy(row))
    elif mutation == "hash":
        row["pcm_sha256"] = "0" * 64
    elif mutation == "incomplete":
        row["status"] = "incomplete"
    elif mutation == "frames":
        row["frames"] += 100
    else:
        reference["contract"]["sample_rate"] = 24000
    with pytest.raises(ValueError):
        join_receipts(reference, plan)


def test_external_mapping_required_and_tamper_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="External resolver"):
        build(ROOT, {})
    fake = tmp_path / "annotations.tar.gz"
    fake.write_bytes(b"Different archive")
    with pytest.raises(ValueError, match="Metadata source changed"):
        build(ROOT, {"giantsteps_annotations": fake})


def test_committed_summary_matches_catalog_and_generator(catalog: CollectionCatalog) -> None:
    saved = json.loads((ROOT / COVERAGE).read_bytes())
    assert saved == coverage(catalog) | dict(
        catalog_sha256=hashlib.sha256((ROOT / OUTPUT).read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(
            (ROOT / "tools/export_giantsteps_collection.py").read_bytes()
        ).hexdigest(),
    )
    assert DATA.as_posix() in str(OUTPUT)
