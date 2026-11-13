"""Tiny synthetic archives exercise identity, corruption and archive boundaries."""

from __future__ import annotations

import hashlib
import io
import tarfile
import wave
from pathlib import Path

import pytest

from tools import check_gtzan_audio as check


def wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 100)
    return output.getvalue()


def archive(tmp_path: Path, members: list[tuple[str, bytes]]) -> Path:
    path = tmp_path / "tiny.tar.gz"
    with tarfile.open(path, "w:gz") as stream:
        for name, raw in members:
            entry = tarfile.TarInfo(name)
            entry.size = len(raw)
            stream.addfile(entry, io.BytesIO(raw))
    return path


def test_identity_and_header_do_not_admit_for_scoring(tmp_path: Path) -> None:
    name, raw = "genres/blues/blues.00000.wav", wav()
    rows = check.inspect_members(
        archive(tmp_path, [(name, raw)]), {name: hashlib.md5(raw).hexdigest()}
    )
    assert len(rows) == 1
    assert rows[0]["member_identity_verified"]
    assert rows[0]["wave"]["frames"] == 100
    assert not rows[0]["overlap_cleared"]
    assert not rows[0]["admitted_for_scoring"]


@pytest.mark.parametrize("name", ["/absolute.wav", "genres/../escape.wav", "bad\\name.wav"])
def test_unsafe_archive_path_rejected(tmp_path: Path, name: str) -> None:
    raw = wav()
    with pytest.raises(ValueError, match="Unsafe"):
        check.inspect_members(
            archive(tmp_path, [(name, raw)]), {name: hashlib.md5(raw).hexdigest()}
        )


def test_duplicate_member_rejected(tmp_path: Path) -> None:
    name, raw = "genres/blues/blues.00000.wav", wav()
    with pytest.raises(ValueError, match="duplicate"):
        check.inspect_members(
            archive(tmp_path, [(name, raw), (name, raw)]), {name: hashlib.md5(raw).hexdigest()}
        )


def test_wrong_recording_identity_rejected(tmp_path: Path) -> None:
    name, raw = "genres/blues/blues.00000.wav", wav()
    with pytest.raises(ValueError, match="MD5 mismatch"):
        check.inspect_members(archive(tmp_path, [(name, raw)]), {name: "0" * 32})


def test_missing_recording_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Incomplete"):
        check.inspect_members(archive(tmp_path, []), {"genres/blues/blues.00000.wav": "0" * 32})


def test_only_existing_named_fault_can_be_recorded(tmp_path: Path) -> None:
    raw = b"not a wave file"
    known = "genres/jazz/jazz.00054.wav"
    rows = check.inspect_members(
        archive(tmp_path, [(known, raw)]), {known: hashlib.md5(raw).hexdigest()}
    )
    assert "known_fault" in rows[0]
    other = "genres/jazz/jazz.00053.wav"
    with pytest.raises(ValueError, match="Unexpected WAV fault"):
        check.inspect_members(
            archive(tmp_path, [(other, raw)]), {other: hashlib.md5(raw).hexdigest()}
        )


def test_truncated_pcm_is_not_silently_accepted() -> None:
    with pytest.raises(ValueError, match="Truncated"):
        check.inspect_wave(wav()[:-2])


def test_declared_oversized_member_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name, raw = "genres/blues/blues.00000.wav", wav()
    monkeypatch.setattr(check, "MEMBER_LIMIT", len(raw) - 1)
    with pytest.raises(ValueError, match="budget"):
        check.inspect_members(
            archive(tmp_path, [(name, raw)]), {name: hashlib.md5(raw).hexdigest()}
        )


def test_pinned_index_contains_complete_universe() -> None:
    expected = check.expected_members()
    assert len(expected) == 1000
    assert "genres/jazz/jazz.00054.wav" in expected


def test_packaging_metadata_does_not_skip_audio_identity(tmp_path: Path) -> None:
    name, raw = "genres/blues/blues.00000.wav", wav()
    members = [
        ("._genres", b"metadata"),
        ("genres/.DS_Store", b"finder"),
        ("genres/blues/._blues.00000.wav", b"resource fork"),
        (name, raw),
    ]
    path = archive(tmp_path, members)
    assert len(check.inspect_members(path, {name: hashlib.md5(raw).hexdigest()})) == 1
    with pytest.raises(ValueError, match="MD5 mismatch"):
        check.inspect_members(path, {name: "0" * 32})


def test_unexpected_sidecar_is_not_ignored(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unexpected archive member"):
        check.inspect_members(archive(tmp_path, [("genres/._unknown.wav", b"metadata")]), {})


def test_oversized_packaging_metadata_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Ancillary metadata budget"):
        check.inspect_members(archive(tmp_path, [("._genres", b"x" * 16385)]), {})
