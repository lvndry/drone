from __future__ import annotations

import hashlib
import io
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pytest

from drone.download import download, extract_giantsteps, extract_zip, verify


def test_download_checksums_and_atomic_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "data.zip"
    body = b"verified archive"
    checksum = "sha256:" + hashlib.sha256(body).hexdigest()
    calls: list[list[str]] = []

    def fake_curl(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        Path(command[command.index("--output") + 1]).write_bytes(body)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_curl)
    download("https://example.test/data", target, checksum)
    assert verify(target, checksum)
    download("https://example.test/data", target, checksum)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download("https://example.test/data", target, "sha256:" + "0" * 64)
    assert target.read_bytes() == body
    assert list(tmp_path.iterdir()) == [target]


def test_zip_extraction_ignores_resource_forks_and_flattens(tmp_path: Path) -> None:
    archive = tmp_path / "labels.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("keys/123 track.txt", "C minor")
        out.writestr("__MACOSX/keys/._123 track.txt", "junk")
    target = tmp_path / "labels"
    assert extract_zip(archive, target, ".txt") == 1
    assert (target / "123 track.txt").read_text() == "C minor"


def test_tar_only_extracts_annotations_and_checksums(tmp_path: Path) -> None:
    archive = tmp_path / "test.tar.gz"
    with tarfile.open(archive, "w:gz") as out:
        for name in ["repo/../../escape", "repo/audio_dl.sh", "repo/annotations/key/1.key"]:
            entry = tarfile.TarInfo(name)
            entry.size = 7
            out.addfile(entry, io.BytesIO(b"C major"))
    target = tmp_path / "dataset"
    with pytest.raises(ValueError, match="604"):
        extract_giantsteps(archive, target)
    assert (target / "annotations/key/1.key").read_text() == "C major"
    assert not (tmp_path / "escape").exists()
    assert not (target / "audio_dl.sh").exists()
