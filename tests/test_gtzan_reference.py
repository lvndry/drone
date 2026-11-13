"""Metadata parsing/mapping regressions; no audio or model dependencies."""

from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

import pytest

from tools.import_gtzan_reference import (
    DEFAULT,
    PINS,
    archive,
    encode,
    filtered_ids,
    git_tree,
    map_key,
    produce,
)


def test_all_key_mappings_and_unknown() -> None:
    assert [map_key(str(k)) for k in range(12)] == [18, 20, 22, 0, 2, 4, 6, 8, 10, 12, 14, 16]
    assert [map_key(str(k)) for k in range(12, 24)] == [19, 21, 23, 1, 3, 5, 7, 9, 11, 13, 15, 17]
    assert map_key("-1") is None


@pytest.mark.parametrize("value", ["24", "-2", "1.0", "", "00", "3 4", "C"])
def test_bad_keys_fail(value: str) -> None:
    with pytest.raises(ValueError):
        map_key(value)


def test_frozen_counts_ledger_and_no_admission() -> None:
    summary, ledger = produce(DEFAULT / "sources")
    counts = summary["counts_by_genre"]
    assert {k: sum(v.values()) for k, v in counts.items()} == {
        "raw_single_key_reference": 837,
        "in_sturm_filtered_union": 930,
        "named_filtered_intersection": 769,
        "metadata_eligible": 769,
    }
    assert counts["raw_single_key_reference"]["hiphop"] == 81
    assert counts["metadata_eligible"]["hiphop"] == 73
    assert len(ledger) == len({r["id"] for r in ledger}) == 1000
    assert all(
        not r["admitted"] and not r["audio_verified"] and not r["overlap_cleared"] for r in ledger
    )
    jazz = next(r for r in ledger if r["id"] == "jazz.00054")
    assert jazz["in_sturm_filtered_union"] and jazz["drone_ref"] is None
    assert "published_loader_corruption_flag" in jazz["exclusions"]
    assert not jazz["metadata_eligible"]
    assert all(r["exclusions"] or r["metadata_eligible"] for r in ledger)
    assert encode(produce(DEFAULT / "sources")) == encode((summary, ledger))


def test_complete_git_tree_detects_unselected_file_change() -> None:
    raw = (DEFAULT / "sources/annotation.zip").read_bytes()
    files = archive(raw, PINS["annotation"][1])
    files["README.md"] += b"changed"
    assert git_tree(files) != PINS["annotation"][1]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, value in files.items():
            z.writestr("repo/" + name, value)
    with pytest.raises(ValueError, match="Git tree"):
        archive(buf.getvalue(), PINS["annotation"][1])


def test_filtered_duplicate_or_cross_partition_alias_rejected() -> None:
    files = archive((DEFAULT / "sources/faults.zip").read_bytes(), PINS["faults"][1])
    files["valid_filtered.txt"] = (
        files["train_filtered.txt"].splitlines()[0]
        + b"\n"
        + b"\n".join(files["valid_filtered.txt"].splitlines()[1:])
    )
    with pytest.raises(ValueError, match="overlapping"):
        filtered_ids(files)


def test_input_bytes_pinned(tmp_path: Path) -> None:
    shutil.copytree(DEFAULT / "sources", tmp_path / "sources")
    target = tmp_path / "sources/annotation.zip"
    target.write_bytes(target.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="SHA mismatch"):
        produce(tmp_path / "sources")
