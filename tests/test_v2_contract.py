"""Metadata-contract correctness; no model/audio dependencies."""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import replace
from pathlib import Path

import pytest

from tools import propose_v2_contract as design


def fixture_tracks() -> list[design.Track]:
    return [
        design.Track(
            f"track:{i}",
            f"group:{i // 2}" if i < 4 else f"group:{i}",
            "fit" if i < 4 else "final",
            f"sha:{i}",
            "artist",
            "album",
            "Rock" if i % 2 else "MISSING",
            float(100 + i),
        )
        for i in range(30)
    ]


def small_panels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        design,
        "PANELS",
        (("calibration", 20, 3), ("inner_selection", 20, 3), ("development_confirmation", 20, 3)),
    )


def test_whole_components_nesting_order_and_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    small_panels(monkeypatch)
    tracks = fixture_tracks()
    roles = design.assign(tracks)
    assert roles["group:0"] == roles["group:1"] == "expanded_fit"
    for role, _, minimum in design.PANELS:
        assert sum(r == role for r in roles.values()) == minimum
    shuffled = tracks.copy()
    random.Random(73).shuffle(shuffled)
    assert design.assign(shuffled) == roles
    summary, ledger = design.proposal(tracks, {})
    assert summary["status"] == "PROPOSED_NOT_ACTIVATED"
    assert not summary["training_authorized"]
    assert summary["restricted_fma_fit"]["rows"] == 4
    assert sum(r["rows"] for r in summary["old_role_to_proposed_role"]) == 30
    assert gzip.decompress(ledger).count(b"PROPOSED_") == 30


def test_missing_components_fail_instead_of_relaxing(monkeypatch: pytest.MonkeyPatch) -> None:
    small_panels(monkeypatch)
    with pytest.raises(ValueError, match="Insufficient bounded"):
        design.assign(fixture_tracks()[:6])


def test_same_audio_cannot_cross_components(monkeypatch: pytest.MonkeyPatch) -> None:
    small_panels(monkeypatch)
    tracks = fixture_tracks()
    tracks[4] = replace(tracks[4], audio_sha256=tracks[5].audio_sha256)
    with pytest.raises(ValueError, match="identical audio crosses"):
        design.proposal(tracks, {})


def test_dedup_preserves_old_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    small_panels(monkeypatch)
    tracks = fixture_tracks()
    tracks.append(replace(tracks[0], identity="new-duplicate", old_role="final"))
    summary, ledger = design.proposal(tracks, {})
    assert (
        summary["prospective_unique_fma_fit_rows"]
        == summary["proposed_panels"]["expanded_fit"]["rows"] - 1
    )
    assert b"new-duplicate,group:0,final,PROPOSED_expanded_fit,0,0" in gzip.decompress(ledger)


def test_strict_assignment_input_rejects_labels(tmp_path: Path) -> None:
    path = tmp_path / "input.gz"
    row = dict(
        identity="a",
        component="g",
        old_role="fit",
        audio_sha256="hash",
        artist_id="1",
        album_id="1",
        genre="Rock",
        duration=100,
        target_key=7,
    )
    path.write_bytes(gzip.compress(json.dumps({"tracks": [row], "provenance": {}}).encode()))
    with pytest.raises(TypeError, match="target_key"):
        design.load_input(path)


def test_post_assignment_conflicting_labels_are_reported_not_repaired(tmp_path: Path) -> None:
    import hashlib
    import io
    import zipfile

    import numpy as np

    from tools.diagnose_v2_class_support import diagnose, write_overlay

    rows = [
        {
            "identity": "fmak:000001",
            "old_role": "fit",
            "audio_sha256": "same",
            "component": "g",
        },
        {
            "identity": "fmak:000002",
            "old_role": "quarantined_development",
            "audio_sha256": "same",
            "component": "g",
        },
    ]
    assignment = gzip.compress(json.dumps({"tracks": rows}).encode())
    ledger = gzip.compress(
        b"id,proposed_role\nfmak:000001,PROPOSED_expanded_fit\nfmak:000002,PROPOSED_expanded_fit\n"
    )
    stream = io.BytesIO()
    np.save(stream, json.dumps({"ids": ["000001", "000002"], "refs": [0, 1]}))
    prepared = tmp_path / "prepared.npz"
    with zipfile.ZipFile(prepared, "w") as archive:
        archive.writestr("metadata.npy", stream.getvalue())
        archive.writestr("features.npy", b"NOT A VALID ARRAY; must never be read")
    result = diagnose(
        prepared,
        assignment,
        ledger,
        hashlib.sha256(ledger).hexdigest(),
        hashlib.sha256(stream.getvalue()).hexdigest(),
    )
    assert result["conflicting_saved_audio_sha256_count"] == 1
    assert result["panels"]["PROPOSED_expanded_fit"]["raw_counts24"][:2] == [1, 1]
    assert result["panels"]["PROPOSED_expanded_fit"]["eligible_unique_counts24"][:2] == [0, 0]
    assert len(result["conflict_records"]) == 2
    assert result["panels"]["restricted_fit"]["eligible_unique_audio"] == 0
    assert result["panels"]["restricted_fit"]["raw_rows"] == 1
    overlay_path = tmp_path / "overlay.csv.gz"
    write_overlay(assignment, ledger, result, overlay_path)
    assert gzip.decompress(overlay_path.read_bytes()).count(b"PROPOSED_QUARANTINE") == 2
    assert result["no_replacements_or_role_reassignment"]
    assert not result["labels_used_for_assignment"]
    with pytest.raises(ValueError, match="metadata source hash changed"):
        diagnose(prepared, assignment, ledger, hashlib.sha256(ledger).hexdigest(), "changed")
    with pytest.raises(ValueError, match="witness mismatch"):
        diagnose(tmp_path / "nonexistent.npz", assignment, ledger, "changed", "unused")
