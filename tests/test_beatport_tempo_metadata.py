"""Missing tempo stays in the cohort; metadata never supplies key targets."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

from tools.beatport_tempo_metadata import DESTINATION, extract_rows, selected_tempo, tempo_band


def test_tempo_admission_preserves_unknown_and_quoted_nan() -> None:
    assert selected_tempo(b'{"id": 4, "bpm": nan, "title": "nan", "key": "F#"}') == (
        4,
        None,
        "missing_publisher_value",
        True,
    )
    assert selected_tempo(b'{"id": 4, "bpm": 0}')[:3] == (
        4,
        None,
        "publisher_zero_sentinel",
    )
    assert selected_tempo(b'{"id": 4, "bpm": 127.5, "key": "unusable"}')[:3] == (
        4,
        127.5,
        "publisher_reported_unverified",
    )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"id":true,"bpm":120}',
        b'{"id":4,"bpm":true}',
        b'{"id":4,"bpm":"120"}',
        b'{"id":4,"bpm":Infinity}',
        b'{"id":4,"bpm":-1}',
    ],
)
def test_invalid_tempo_is_not_silently_admitted(raw: bytes) -> None:
    with pytest.raises(ValueError):
        selected_tempo(raw)


def test_fixed_membership_and_member_identity(tmp_path: Path) -> None:
    archive = tmp_path / "metadata.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("meta/4 Track.json", '{"id":4,"bpm":120,"key":"A"}')
        stream.writestr("meta/5 Absent.json", '{"id":5,"bpm":null}')
        stream.writestr("meta/99 Outside cohort.json", "invalid outside-cohort metadata")
    rows = extract_rows(archive, {"beatport:4": "fit", "beatport:5": "fit", "beatport:6": "inner"})
    assert len(rows) == 3
    assert rows[0]["bpm"] == 120.0
    assert rows[1]["bpm_status"] == "missing_publisher_value"
    assert rows[2]["bpm_status"] == "publisher_record_missing"
    assert rows[2]["source_member_sha256"] is None
    assert all("key" not in row for row in rows)
    with zipfile.ZipFile(archive, "a") as stream:
        stream.writestr("meta/4 Duplicate.json", '{"id":4,"bpm":120}')
    with pytest.raises(ValueError, match="Duplicate"):
        extract_rows(archive, {"beatport:4": "fit"})


def test_band_boundaries_and_unknown() -> None:
    assert [tempo_band(value) for value in (89.9, 90, 110, 130, 150, None)] == [
        "<90",
        "90–<110",
        "110–<130",
        "130–<150",
        "150+",
        "unknown",
    ]


def test_committed_export_keeps_complete_frozen_membership() -> None:
    data = json.loads(gzip.decompress(DESTINATION.read_bytes()))
    tracks = data["tracks"]
    assert len({row["track_id"] for row in tracks}) == len(tracks) == 1355
    assert sum(row["bpm"] is not None for row in tracks) == 716
    assert sum(row["bpm"] is None for row in tracks) == 639
    assert sum(row["role"] == "fit" for row in tracks) == 1229
    assert sum(row["role"] == "inner_selection" for row in tracks) == 126
    assert data["audio_read"] is False and data["inference_performed"] is False


def test_check_reports_hash_of_actual_compressed_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tools import beatport_tempo_metadata as tool

    value: dict[str, object] = {"summary": {"n": 1}}

    def fixture_build(store_zip: Path, plan: Path) -> dict[str, object]:
        return value

    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    # A gzip timestamp changes the physical artifact, not its metadata content.
    compressed = gzip.compress(raw, mtime=123)
    destination = tmp_path / "tempo.json.gz"
    destination.write_bytes(compressed)
    monkeypatch.setattr(tool, "build", fixture_build)
    monkeypatch.setattr(
        sys,
        "argv",
        ["tempo", "--store-zip", "unused.zip", "--output", str(destination), "--check"],
    )
    tool.main()
    assert json.loads(capsys.readouterr().out)["sha256"] == hashlib.sha256(compressed).hexdigest()
