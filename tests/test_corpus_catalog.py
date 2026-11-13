"""Metadata-only schema, provenance and adapter checks; no audio or models."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.corpus_catalog import (
    Catalog,
    Evidence,
    GenreTag,
    catalog_bytes,
    duration_band,
    load_catalog,
    validate_catalog,
    validate_sources,
)
from tools.dj_coverage import band
from tools.export_corpus_catalog import (
    DEFAULT,
    ROOT,
    export_reference_panel,
    source_record,
    verify_reference_panel,
)


def fixture() -> Catalog:
    return load_catalog(DEFAULT)


def test_committed93_catalog_is_metadata_only_and_deterministic(tmp_path: Path) -> None:
    catalog = verify_reference_panel(fixture())
    assert len(catalog.tracks) == 93
    assert {row.corpus_id for row in catalog.tracks} == {"swd-v2.1", "giantsteps"}
    assert all(row.bpm.value is None and row.bpm.missing_reason for row in catalog.tracks)
    assert all(row.duration.full_recording.value is None for row in catalog.tracks)
    assert {row.exposure.subject_model_id for row in catalog.tracks} == {"drone-v1"}
    assert all(
        row.exposure.other_model_exposure_status == "not_assessed_for_other_models"
        for row in catalog.tracks
    )
    swd = [row for row in catalog.tracks if row.corpus_id == "swd-v2.1"]
    assert len({row.grouping.work_id for row in swd}) == 24
    assert {row.grouping.performance_id for row in swd} == {"HU33", "SC06"}
    raw = catalog_bytes(catalog)
    assert b'"pred"' not in raw and b'"mirex"' not in raw and b'"score"' not in raw
    path = tmp_path / "catalog.json"
    path.write_bytes(raw)
    assert catalog_bytes(load_catalog(path)) == raw
    assert catalog_bytes(replace(catalog, tracks=tuple(reversed(catalog.tracks)))) == raw


@pytest.mark.parametrize(
    "seconds", [None, 0.1, 29.999, 30, 119.999, 120, 299.999, 300, 600, 600.001]
)
def test_existing_coverage_bands_preserved(seconds: float | None) -> None:
    assert duration_band(seconds) == band(seconds)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 0, -1])
def test_bad_measurements_rejected(value: float) -> None:
    catalog = fixture()
    row = catalog.tracks[0]
    changed = replace(row, bpm=replace(row.bpm, value=value, missing_reason=None))
    with pytest.raises(ValueError):
        validate_catalog(replace(catalog, tracks=(changed,)))


def test_missingness_has_reasons_and_known_values_do_not() -> None:
    catalog = fixture()
    row = catalog.tracks[0]
    for measurement in (replace(row.bpm, missing_reason=None), replace(row.bpm, value=120)):
        with pytest.raises(ValueError):
            validate_catalog(replace(catalog, tracks=(replace(row, bpm=measurement),)))


def test_identity_hashes_crossreferences_and_unique_rows_checked() -> None:
    catalog = fixture()
    row = catalog.tracks[0]
    invalid = [
        replace(catalog, tracks=(row, row)),
        replace(catalog, tracks=(replace(row, source_audio_sha256="bad"),)),
        replace(catalog, tracks=(replace(row, audio_id="sha256:" + "0" * 64),)),
        replace(catalog, tracks=(replace(row, key_label=replace(row.key_label, class_id=True)),)),
        replace(
            catalog,
            tracks=(replace(row, bpm=replace(row.bpm, evidence=(Evidence("missing", "/x"),))),),
        ),
    ]
    for changed in invalid:
        with pytest.raises(ValueError):
            validate_catalog(changed)
    for path in ("../outside", "/absolute", ".", "https://example.org/metadata.json"):
        changed = replace(
            catalog, sources=(replace(catalog.sources[0], path=path), *catalog.sources[1:])
        )
        with pytest.raises(ValueError):
            validate_catalog(changed)


def test_multiple_tags_keep_one_track_and_direct_ancestor_distinction() -> None:
    catalog = fixture()
    row = catalog.tracks[0]
    tags = (
        GenreTag("House", 12, "example native taxonomy", "direct", row.genre.evidence),
        GenreTag("Electronic", 1, "example native taxonomy", "ancestor", row.genre.evidence),
    )
    modified = replace(catalog, tracks=(replace(row, genre=replace(row.genre, raw_tags=tags)),))
    assert len(validate_catalog(modified).tracks) == 1
    assert len(modified.tracks[0].genre.raw_tags) == 2


def test_adapter_verification_catches_fabricated_bpm_grouping_and_fullsong() -> None:
    catalog = fixture()
    row = catalog.tracks[0]
    for changed in (
        replace(row, bpm=replace(row.bpm, value=120, missing_reason=None)),
        replace(row, grouping=replace(row.grouping, work_id="fabricated", missing_reason=None)),
        replace(
            row,
            duration=replace(
                row.duration,
                full_recording=replace(row.duration.full_recording, value=600, missing_reason=None),
            ),
        ),
    ):
        modified = replace(catalog, tracks=(changed, *catalog.tracks[1:]))
        validate_catalog(modified)
        with pytest.raises(ValueError, match="disagrees"):
            verify_reference_panel(modified)
    changed = replace(
        catalog, sources=(replace(catalog.sources[0], sha256="0" * 64), *catalog.sources[1:])
    )
    with pytest.raises(ValueError, match="source bytes changed"):
        validate_sources(changed, ROOT)


def test_adapter_runs_before_predictions_and_rejects_bad_results_join() -> None:
    path = ROOT / "experiments/studies/key-sota-4way/data/manifest.json"
    manifest = json.loads(path.read_bytes())
    source = source_record(path, "reference_manifest")
    metadata_only = export_reference_panel(manifest, (source,))
    assert metadata_only.results_source_id is None
    assert len(metadata_only.tracks) == 93
    with pytest.raises(ValueError, match="join"):
        export_reference_panel(manifest, (source,), ("unexpected",))
