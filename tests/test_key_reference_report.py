"""Metadata-only evidence replay and fixed-denominator reporting contracts."""

from __future__ import annotations

import csv
import gzip
import io
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from tools.report_key_references import DATA, analyze, encode, seal, sha


def test_preserved_error_free_predictions_replay_without_bad_kf() -> None:
    result = analyze()
    assert result["n"] == 93
    assert result["source_counts"] == {"swd-v2.1": 48, "giantsteps": 45}
    assert result["duration_by_source"]["swd-v2.1"]["median_seconds"] == 169.1167345
    assert result["detectors"]["drone"]["overall"]["counts"]["exact"] == 63
    assert result["detectors"]["madmom"]["overall"]["counts"]["exact"] == 72
    if result["libkeyfinder_status"] == "historical_invalid":
        assert (
            result["detectors"]["libkeyfinder"]["status"]
            == "INVALID_HISTORICAL_WRAPPER_NOT_A_BASELINE"
        )
    else:
        assert result["detectors"]["libkeyfinder"]["overall"]["n_total"] == 93


@pytest.mark.parametrize(
    "name", ["bench.csv", "bench.json", "manifest.json", "historical/bench.csv.gz"]
)
def test_changed_active_or_historical_evidence_rejected(tmp_path: Path, name: str) -> None:
    shutil.copytree(DATA, tmp_path / "data")
    path = tmp_path / "data" / name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        analyze(tmp_path / "data")


def test_resealed_but_inconsistent_prediction_or_manifest_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    path = data / "manifest.json"
    value = json.loads(path.read_bytes())
    value["tracks"][0]["ref"] = (value["tracks"][0]["ref"] + 1) % 24
    path.write_bytes(encode(value))
    (data / "artifact-index.json").write_bytes(encode(seal(data, "historical_invalid")))
    with pytest.raises(ValueError, match="manifest join"):
        analyze(data)


def test_errors_and_abstentions_keep_full_scope_denominators(tmp_path: Path) -> None:
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    rows = list(csv.DictReader(io.StringIO((data / "bench.csv").read_text())))
    # Two failures on one scope must stay in that scope's and the overall denominator.
    for row, status in zip(rows[:2], ("error", "abstain"), strict=True):
        row["drone_pred"] = ""
        row["drone_status"] = status
        row["drone_rel"] = status
    for row in rows[2:]:
        row["drone_status"] = "key"
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (data / "bench.csv").write_text(stream.getvalue())
    summary = json.loads((data / "bench.json").read_bytes())
    # Remove the old scores for this deliberately synthetic result population.
    summary["detectors"]["drone-v1"] = {}
    (data / "bench.json").write_bytes(encode(summary))
    (data / "artifact-index.json").write_bytes(encode(seal(data, "historical_invalid")))
    result = analyze(data)["detectors"]["drone"]
    assert result["overall"]["n_total"] == 93
    assert result["overall"]["n_error"] == result["overall"]["n_abstain"] == 1
    assert result["by_source"]["swd-v2.1"]["n_total"] == 48
    assert result["by_source"]["swd-v2.1"]["n_valid"] == 46


def test_archive_keeps_exact_original_csv_bytes() -> None:
    index = json.loads((DATA / "historical/archive-index.json").read_bytes())
    raw = gzip.decompress((DATA / "historical/bench.csv.gz").read_bytes())
    assert sha(raw) == index["files"]["bench.csv.gz"]["decoded_sha256"]
    assert len(list(csv.DictReader(io.StringIO(raw.decode())))) == 93


def test_corrected_kf_uses_fresh_typed_rows_and_complete_cohorts() -> None:
    result = analyze()
    kf = result["detectors"]["libkeyfinder"]
    assert result["libkeyfinder_status"] == "corrected"
    assert kf["overall"]["counts"]["exact"] == 57
    assert kf["overall"]["exact"] == 0.6129 and kf["overall"]["mirex"] == 0.7366
    assert kf["overall"]["n_error"] == kf["overall"]["n_abstain"] == 0
    assert kf["by_source"]["swd-v2.1"]["n_total"] == 48
    assert kf["by_source"]["swd-v2.1"]["counts"]["exact"] == 28
    assert kf["by_source"]["giantsteps"]["n_total"] == 45
    assert kf["by_source"]["giantsteps"]["counts"]["exact"] == 29


def test_corrected_archive_tamper_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    path = data / "corrected-kf/evidence.tar.gz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Corrected KF archive changed"):
        analyze(data)


def test_resealed_wrong_fresh_prediction_rejected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    path = data / "bench.csv"
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    rows[0]["libkeyfinder_pred"] = str((int(rows[0]["libkeyfinder_pred"]) + 1) % 24)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue())
    (data / "artifact-index.json").write_bytes(encode(seal(data, "corrected")))
    with pytest.raises(ValueError, match="Published KF row differs"):
        analyze(data)


def slice_fixture(
    *, seconds: float | None = 120.0, ref: int = 0, status: str = "key", ident: str = "a"
) -> tuple[dict[str, Any], dict[str, Any]]:
    row: dict[str, Any] = {
        "track_id": ident,
        "ref": ref,
        "genre": "drum-and-bass",
        "bucket": "dubstep",
    }
    for name in ("drone", "madmom", "libkeyfinder"):
        row[name + "_pred"] = ref if status == "key" else None
        row[name + "_status"] = status
        row[name + "_rel"] = "exact" if status == "key" else status
    track: dict[str, Any] = {
        "track_id": ident,
        "corpus_id": "synthetic",
        "genre": {"raw": "drum-and-bass", "bucket": "dubstep"},
        "key_label": {"class_id": ref},
        "duration": {"analyzed": {"value": seconds}, "full_recording": {"value": None}},
        "bpm": {"value": None},
        "grouping": {"work_id": "one-repeated-work", "performance_id": ident},
    }
    return row, track


def test_slice_empty_is_na_and_failures_keep_full_denominator() -> None:
    from tools.report_key_references import descriptive_slices, slice_csv

    fixtures = [
        slice_fixture(status=status, ident=str(i))
        for i, status in enumerate(("key", "error", "abstain"))
    ]
    result = descriptive_slices([r for r, _ in fixtures], [t for _, t in fixtures])
    group = result["slices"]["analyzed_duration"]["<3m"]
    for metrics in group["detectors"].values():
        assert metrics["n_total"] == 3
        assert metrics["n_valid"] == metrics["n_error"] == metrics["n_abstain"] == 1
        assert metrics["exact"] == metrics["mirex"] == 0.3333
    empty = result["slices"]["analyzed_duration"]["8m+"]
    assert empty["n"] == 0
    assert empty["detectors"]["drone"]["exact"] is None
    assert empty["detectors"]["drone"]["mirex"] is None
    assert "NA" in slice_csv(result)
    assert result["coverage"]["bpm"] == {"known": 0, "unknown": 3}
    assert result["grouping_by_corpus"]["synthetic"]["known_work_ids"] == ["one-repeated-work"]


def test_slice_edges_raw_genre_and_key_encoding_are_fixed() -> None:
    from tools.report_key_references import DURATION_SLICES, descriptive_slices

    fixtures = [
        slice_fixture(seconds=seconds, ref=i, ident=str(i))
        for i, seconds in enumerate((179.999, 180.0, 300.0, 480.0, None))
    ]
    result = descriptive_slices([r for r, _ in fixtures], [t for _, t in fixtures])
    assert list(result["slices"]["analyzed_duration"]) == list(DURATION_SLICES)
    assert [v["n"] for v in result["slices"]["analyzed_duration"].values()] == [1] * 5
    assert list(result["slices"]["genre_original"]) == ["drum-and-bass"]
    assert list(result["slices"]["genre_legacy_group"]) == ["dubstep"]
    assert result["slices"]["key"]["C# minor"]["n"] == 1
    assert result["slices"]["key"]["B minor"]["n"] == 0
    assert result["slices"]["mode"]["major"]["n"] == 3
    assert result["slices"]["mode"]["minor"]["n"] == 2
    assert result["corpus_by_duration"]["synthetic"]["8m+"]["n"] == 1


def test_slice_catalog_must_join_all_rows_and_original_labels() -> None:
    from tools.report_key_references import descriptive_slices

    row, track = slice_fixture()
    with pytest.raises(ValueError, match="complete unique"):
        descriptive_slices([row], [])
    with pytest.raises(ValueError, match="complete unique"):
        descriptive_slices([row], [track, track])
    with pytest.raises(ValueError, match="label"):
        descriptive_slices([{**row, "ref": 1}], [track])
    with pytest.raises(ValueError, match="genre"):
        descriptive_slices([{**row, "genre": "different"}], [track])
    invalid, metadata = slice_fixture(seconds=float("nan"))
    with pytest.raises(ValueError, match="positive and finite"):
        descriptive_slices([invalid], [metadata])


def test_catalog_slices_replay_complete_panel_and_leave_historical_artifacts_unchanged() -> None:
    from tools.report_key_references import CATALOG, analyze_slices

    watched = [
        DATA / name
        for name in (
            "bench.csv",
            "bench.json",
            "manifest.json",
            "artifact-index.json",
            "metadata-analysis.json",
        )
    ]
    before = {path: sha(path.read_bytes()) for path in watched}
    result = analyze_slices()
    assert result["n"] == 93
    assert result["provenance"]["catalog_sha256"] == sha(CATALOG.read_bytes())
    assert result["coverage"]["bpm"] == {"known": 0, "unknown": 93}
    assert result["coverage"]["full_recording_duration"] == {"known": 0, "unknown": 93}
    assert [v["n"] for v in result["slices"]["analyzed_duration"].values()] == [71, 20, 2, 0, 0]
    assert result["corpus_by_duration"]["giantsteps"]["3-5m"]["n"] == 0
    assert result["corpus_by_duration"]["swd-v2.1"]["3-5m"]["n"] == 20
    assert len(result["grouping_by_corpus"]["swd-v2.1"]["known_work_ids"]) == 24
    assert (
        result["slices"]["genre_original"]["trance"]["detectors"]["drone"]["counts"].get("exact", 0)
        == 0
    )
    assert result["slices"]["genre_original"]["trance"]["n"] == 4
    assert result["slices"]["mode"]["minor"]["n"] == 64
    assert result["corpus_by_mode"]["swd-v2.1"]["major"]["n"] == 16
    assert {path: sha(path.read_bytes()) for path in watched} == before


def test_slice_rejects_structurally_valid_but_unwitnessed_bpm(tmp_path: Path) -> None:
    from dataclasses import replace

    from experiments.corpus_catalog import catalog_bytes, load_catalog
    from tools.report_key_references import CATALOG, analyze_slices

    catalog = load_catalog(CATALOG)
    row = catalog.tracks[0]
    invented = replace(
        row, bpm=replace(row.bpm, value=128.0, method="unwitnessed", missing_reason=None)
    )
    path = tmp_path / "catalog.json"
    path.write_bytes(catalog_bytes(replace(catalog, tracks=(invented, *catalog.tracks[1:]))))
    with pytest.raises(ValueError, match="source adapter"):
        analyze_slices(catalog_path=path)
