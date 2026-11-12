"""Synthetic full-waveform bytes test disk prep without running a decoder or model."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools.musicfm_crop_io import file_identity
from tools.prepare_musicfm_disk_cache import DiskPreparation


def stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DiskPreparation:
    for name in ("fit", "records", "logs"):
        (tmp_path / name).mkdir()
    result = DiskPreparation(
        {
            "cpus": [10, 11, 12, 13],
            "ffmpeg": "/frozen/ffmpeg",
            "caps": {"reference_bytes": 256 * 1024**2},
        },
        tmp_path,
    )
    result.initialize()
    monkeypatch.setattr(result, "guard", lambda *_: None)
    return result


def test_exact_full_waveform_is_readonly_and_old_invariants_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"not audio")
    samples = np.arange(1200, dtype="<f4")
    original = {
        "id": "public:one",
        "path": str(source),
        "role": "fit",
        "target": 3,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_stable_identity": file_identity(source),
        "waveform_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
        "samples": len(samples),
        "bytes": samples.nbytes,
    }
    current = stage(tmp_path, monkeypatch)
    calls = []

    def decode(command: list[str], target: Path, stderr: Path, maximum: int) -> dict[str, Any]:
        calls.append(command)
        samples.tofile(target)
        stderr.write_text("preserved warning")
        return {"child_cpu_seconds": 1.0, "exit": 0}

    monkeypatch.setattr(current, "decode", decode)
    result = current.one(original, 1)
    path = Path(result["cache"])
    assert path.read_bytes() == samples.tobytes() and not path.stat().st_mode & 0o222
    assert (
        calls[0][:3] == ["taskset", "-c", "10"] and "-ss" not in calls[0] and "-t" not in calls[0]
    )
    assert all(result[key] == value for key, value in original.items())
    assert (tmp_path / "logs/0001-full.stderr").read_text() == "preserved warning"
    assert "child_cpu_seconds" not in result["disk_preparation_execution"]


def test_wrong_canonical_decode_fails_without_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = stage(tmp_path, monkeypatch)
    path = tmp_path / "wave"
    np.zeros(1200, dtype="<f4").tofile(path)
    with pytest.raises(ValueError, match="hash"):
        current.digest(path, "a" * 64, 4800, True)
    assert path.stat().st_size == 4800


def test_nonfinite_waveform_cannot_become_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = stage(tmp_path, monkeypatch)
    path = tmp_path / "wave"
    array = np.zeros(1200, dtype="<f4")
    array[13] = np.nan
    array.tofile(path)
    with pytest.raises(ValueError, match="Nonfinite"):
        current.digest(path, hashlib.sha256(array.tobytes()).hexdigest(), array.nbytes, True)


def test_final_guard_failure_never_exposes_ready_manifest(tmp_path: Path) -> None:
    from tools.prepare_musicfm_disk_cache import publish_ready

    def failed() -> None:
        raise RuntimeError("final reserve failure")

    with pytest.raises(RuntimeError, match="reserve"):
        publish_ready(tmp_path, {"complete": True, "rows": []}, failed)
    assert not (tmp_path / "manifest.json").exists()
    assert (tmp_path / "manifest.pending.json").exists()
