"""Metadata-only evidence replay and fixed-denominator reporting contracts."""

from __future__ import annotations

import csv
import gzip
import io
import json
import shutil
from pathlib import Path

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
