"""Synthetic collection contracts; no target references, audio or models."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from experiments.collection_catalog import (
    SCHEMA,
    AudioSupport,
    CollectionCatalog,
    CollectionTrack,
    ExternalSource,
    OptionalIdentity,
    collection_bytes,
    load_collection,
    resolve_sources,
    validate_collection,
)
from experiments.corpus_catalog import Evidence, Measurement, SourceRecord


def fixture() -> CollectionCatalog:
    evidence = (Evidence("metadata", "/rows/id"),)
    absent = OptionalIdentity(None, "not_recorded", "source_hash_unavailable", evidence)
    unknown = Measurement(None, "seconds", "not_measured", "no_audio_probe", evidence)
    support = AudioSupport(
        "recorded_24khz_training_cache",
        None,
        None,
        None,
        None,
        absent,
        "not_in_selected_metadata",
        "not_known",
        "saved_metadata_only",
        "no_saved_cache_support",
        evidence,
    )
    row = CollectionTrack(
        "fmak:000001",
        "fmak",
        "000001",
        absent,
        None,
        (),
        "native_metadata",
        unknown,
        unknown,
        Measurement(None, "bpm", "not_measured", "no_tempo_source", evidence),
        support,
        (),
        ("artist_unknown",),
        (),
        (),
        (),
    )
    return CollectionCatalog(
        SCHEMA, "synthetic", (SourceRecord("metadata", "metadata.json", "a" * 64),), (), (row,)
    )


def test_absent_hash_support_and_targets_remain_absent(tmp_path: Path) -> None:
    value = fixture()
    raw = collection_bytes(value)
    assert b'"target"' not in raw and b'"key_label"' not in raw and b'"prediction"' not in raw
    path = tmp_path / "collection.json"
    path.write_bytes(raw)
    loaded = load_collection(path)
    assert loaded.tracks[0].source_audio.sha256 is None
    assert loaded.tracks[0].analyzed_audio.seconds is None
    assert collection_bytes(loaded) == raw


@pytest.mark.parametrize("bad", [True, 0, -1, float("nan")])
def test_fake_sample_support_is_rejected(bad: float) -> None:
    value = fixture()
    support = replace(
        value.tracks[0].analyzed_audio,
        samples=bad,
        sample_rate=24000,
        channels=1,
        dtype="<f4",
        missing_reason=None,
    )
    with pytest.raises(ValueError):
        validate_collection(
            replace(value, tracks=(replace(value.tracks[0], analyzed_audio=support),))
        )


def test_duplicate_rows_invalid_hash_and_missingness_rejected() -> None:
    value = fixture()
    row = value.tracks[0]
    for modified in (
        replace(value, tracks=(row, row)),
        replace(
            value,
            tracks=(
                replace(
                    row, source_audio=replace(row.source_audio, sha256="bad", missing_reason=None)
                ),
            ),
        ),
        replace(value, tracks=(replace(row, bpm=replace(row.bpm, missing_reason=None)),)),
    ):
        with pytest.raises(ValueError):
            validate_collection(modified)


def test_local_resolver_checks_declared_external_identity_and_no_embedded_paths(
    tmp_path: Path,
) -> None:
    import hashlib

    value = fixture()
    metadata = tmp_path / "metadata.json"
    external = tmp_path / "external.csv"
    external.write_text("id\n1\n")
    import json

    metadata.write_text(
        json.dumps({"external": {"sha256": hashlib.sha256(external.read_bytes()).hexdigest()}})
    )
    source = replace(value.sources[0], sha256=hashlib.sha256(metadata.read_bytes()).hexdigest())
    descriptor = ExternalSource(
        "external",
        hashlib.sha256(external.read_bytes()).hexdigest(),
        Evidence("metadata", "/external"),
    )
    value = replace(value, sources=(source,), external_sources=(descriptor,))
    assert resolve_sources(value, tmp_path, {"external": external})["external"] == external
    assert str(tmp_path).encode() not in collection_bytes(value)
    with pytest.raises(ValueError):
        resolve_sources(value, tmp_path, {})
    external.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        resolve_sources(value, tmp_path, {"external": external})


def test_streaming_encoder_keeps_exact_original_canonical_bytes() -> None:
    import json
    from dataclasses import asdict

    value = fixture()
    expected = asdict(value)
    for field in ("sources", "external_sources"):
        expected[field] = sorted(expected[field], key=lambda row: row["source_id"])
    expected["tracks"] = sorted(expected["tracks"], key=lambda row: row["track_id"])
    original = (json.dumps(expected, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    assert collection_bytes(value) == original


def test_external_hash_must_match_the_pinned_descriptor_value(tmp_path: Path) -> None:
    import hashlib
    import json

    value = fixture()
    data = tmp_path / "changed.csv"
    data.write_text("changed")
    descriptor = tmp_path / "metadata.json"
    descriptor.write_text(json.dumps({"external": {"sha256": "b" * 64}}))
    source = replace(value.sources[0], sha256=hashlib.sha256(descriptor.read_bytes()).hexdigest())
    external = ExternalSource(
        "external", hashlib.sha256(data.read_bytes()).hexdigest(), Evidence("metadata", "/external")
    )
    changed = replace(value, sources=(source,), external_sources=(external,))
    with pytest.raises(ValueError, match="bundled descriptor"):
        resolve_sources(changed, tmp_path, {"external": data})
