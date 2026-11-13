"""Regression checks for fixed-denominator retrospective development diagnostics."""

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.collection_catalog import (
    AudioSupport,
    CollectionTrack,
    FrozenGroup,
    OptionalIdentity,
)
from experiments.corpus_catalog import Evidence, GenreTag, Measurement
from tools.analyze_development_slices import (
    ROOT,
    Observation,
    analyze,
    checked_archive,
    corpus_slices,
    load_score_code,
    model_score,
    selected_plan_refs,
    slice_summary,
)


def row(number: int = 1, *, seconds: float = 120, bpm: float | None = None) -> Observation:
    evidence = (Evidence("synthetic", "/row"),)
    identity = OptionalIdentity(None, "unavailable", "synthetic", evidence)
    duration = Measurement(None, "seconds", "unavailable", "synthetic", evidence)
    support = AudioSupport(
        "synthetic", None, None, None, None, identity, "none", "none", "none", "synthetic", evidence
    )
    metadata = CollectionTrack(
        f"beatport:{number}",
        "beatport",
        str(number),
        identity,
        "House",
        (GenreTag("House", 1, "beatport_workbook_native", "direct", evidence),),
        "synthetic",
        duration,
        duration,
        Measurement(bpm, "bpm", "publisher", "unknown" if bpm is None else None, evidence),
        support,
        (),
        (),
        (),
        (FrozenGroup("proxy", "one-shared-group", evidence),),
        (),
    )
    return Observation(metadata.track_id, "beatport", 0, 0, 0, seconds, seconds, metadata)


def test_errors_and_abstentions_stay_in_scored_and_paired_denominators() -> None:
    source = row()
    rows = [
        source,
        replace(row(2), k10_pred=None, k10_status="error", musicfm_pred=14),
        replace(row(3), k10_pred=None, k10_status="abstain", musicfm_pred=0),
        replace(row(4), k10_pred=14, musicfm_pred=None, musicfm_status="error"),
    ]
    result = slice_summary(rows, "corpus", "synthetic", load_score_code(ROOT))
    k10, mfm = result["models"]["k10_expanded"], result["models"]["musicfm_pilot"]
    assert (k10["correct"], k10["total"], k10["n_error"], k10["n_abstain"]) == (1, 4, 1, 1)
    assert k10["exact"] == 0.25 and k10["mirex"] == 1.5 / 4
    assert mfm["correct"] == 2 and mfm["n_error"] == 1
    assert result["group_count"] == 1  # Four rows are not four independent units.
    assert result["paired"] == {
        "both_correct": 1,
        "both_wrong": 2,
        "musicfm_only_correct": 1,
        "k10_only_correct": 0,
        "musicfm_minus_k10_exact_count": 1,
        "musicfm_minus_k10_mirex": 0.25,
    }
    with pytest.raises(ValueError, match="status mismatch"):
        model_score([replace(source, k10_pred=None)], "k10_expanded", load_score_code(ROOT))


def test_canonical_relative_parallel_and_symmetric_fifth_conventions() -> None:
    rows = [replace(row(i + 1), k10_pred=p) for i, p in enumerate((0, 14, 10, 19, 1, 4))]
    result = model_score(rows, "k10_expanded", load_score_code(ROOT))
    assert result["correct"] == 1
    assert result["symmetric_units10"] == 25  # 1+.5+.5+.3+.2+0
    assert result["asymmetric_units10"] == 20  # Descending fifth excluded.
    assert model_score([], "k10_expanded", load_score_code(ROOT))["exact"] is None


def test_fixed_boundaries_empty_native_cells_and_unknown_metadata() -> None:
    seconds = (29, 30, 119, 120, 299, 300, 600, 601)
    bpms = (89, 90, 109, 110, 129, 130, 150, None)
    rows = [
        row(i + 1, seconds=s, bpm=b) for i, (s, b) in enumerate(zip(seconds, bpms, strict=True))
    ]
    universe = {"taxonomy:beatport_workbook_native": ["House", "Unobserved genre"]}
    cells = corpus_slices(rows, "beatport", load_score_code(ROOT), universe)
    durations = [c for c in cells if c["dimension"] == "duration_actual_evaluation"]
    assert [c["n"] for c in durations] == [1, 2, 2, 2, 1, 0]
    tempos = [c for c in cells if c["dimension"] == "bpm_publisher"]
    assert [c["n"] for c in tempos] == [1, 2, 2, 1, 1, 1]
    missing = next(c for c in cells if c["label"] == "Unobserved genre")
    assert missing["n"] == 0 and missing["models"]["musicfm_pilot"]["mirex"] is None
    api_unknown = next(
        c
        for c in cells
        if c["dimension"] == "taxonomy:beatport_api_genres" and c["label"] == "unknown"
    )
    assert api_unknown["n"] == 8  # Workbook genre cannot fill a missing API taxonomy.


def test_selected_reference_extraction_rejects_ambiguity_and_ignores_reserved_values() -> None:
    text = '{"expected_refs":{"beatport:1":3,"reserved:9":"not opened"}}'
    assert selected_plan_refs(text, ["beatport:1"]) == {"beatport:1": 3}
    for malformed in (
        '{"expected_refs":{"beatport:1":24}}',
        '{"expected_refs":{"beatport:1":3,"beatport:1":4}}',
        '{"expected_refs":{"beatport:2":3}}',
    ):
        with pytest.raises(ValueError, match="missing/repeated/invalid"):
            selected_plan_refs(malformed, ["beatport:1"])


def test_changed_pinned_source_is_rejected_before_any_join(tmp_path: Path) -> None:
    from tools.analyze_development_slices import PINS

    for name in (
        "experiments/collection_catalog.py",
        "experiments/corpus_catalog.py",
        "tools/beatport_tempo_metadata.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    path = tmp_path / PINS["expanded"][0]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not the recorded run")
    with pytest.raises(ValueError, match="Frozen input changed: expanded"):
        analyze(tmp_path)


@pytest.mark.parametrize("marker", [3, 19, 255])
def test_reproduction_checks_canonical_payload_but_keeps_archived_identity(
    tmp_path: Path, marker: int
) -> None:
    path = tmp_path / "analysis.json.gz"
    canonical = (json.dumps({"metadata": True}) + "\n").encode()
    archived = bytearray(gzip.compress(canonical, mtime=0))
    archived[9] = marker
    path.write_bytes(archived)
    assert checked_archive(path, canonical) == bytes(archived)
    with pytest.raises(ValueError, match="verified source arithmetic"):
        checked_archive(path, b'{"metadata": false}\n')


def test_selected_stream_matches_loader_and_keeps_full_genre_universe(tmp_path: Path) -> None:
    from experiments.collection_catalog import (
        SCHEMA,
        CollectionCatalog,
        collection_bytes,
        load_collection,
    )
    from experiments.corpus_catalog import SourceRecord
    from tools.analyze_development_slices import selected_collection

    one = row().metadata
    two = replace(
        row(2).metadata,
        native_genre_top="Techno",
        genre_tags=(
            GenreTag("Techno", 2, "beatport_workbook_native", "direct", one.genre_tags[0].evidence),
        ),
    )
    catalog = CollectionCatalog(
        SCHEMA, "synthetic", (SourceRecord("synthetic", "source.json", "a" * 64),), (), (one, two)
    )
    path = tmp_path / "catalog.json.gz"
    raw = collection_bytes(catalog)
    path.write_bytes(gzip.compress(raw, mtime=0))
    selected, universe = selected_collection(path, {one.track_id}, 2)
    assert selected.tracks == (load_collection(path).tracks[0],)
    assert universe["taxonomy:beatport_workbook_native"] == ["House", "Techno"]
    with pytest.raises(ValueError, match="population"):
        selected_collection(path, {one.track_id}, 3)
    for changed in (
        raw[:-8],
        raw.replace(b"    },\n    {\n", b"    }\n    {\n", 1),
        raw.replace(b"    }\n  ]\n", b"    },\n  ]\n", 1),
        raw.replace(b'  "tracks": [\n    {\n', b'  "tracks": [\n   {\n', 1),
        raw.replace(b'  "tracks": [\n', b'  "tracks": []\n', 1),
    ):
        path.write_bytes(gzip.compress(changed, mtime=0))
        with pytest.raises((ValueError, json.JSONDecodeError)):
            selected_collection(path, {one.track_id}, 2)
    path.write_bytes(gzip.compress(raw.replace(b'"schema":', b'"extra": 1, "schema":'), mtime=0))
    with pytest.raises(ValueError, match="Unexpected collection fields"):
        selected_collection(path, {one.track_id}, 2)


def test_committed_development_evidence_reproduces_same_selected_cohorts() -> None:
    from tools.analyze_development_slices import verify_analysis

    result = verify_analysis()
    assert len(result["per_track"]) == 376
    assert result["corpora"]["beatport"]["totals"]["paired"]["musicfm_only_correct"] == 13
    assert result["corpora"]["fmak"]["totals"]["paired"]["k10_only_correct"] == 26
    assert result["corpora"]["fmak"]["group_count"] == 100
