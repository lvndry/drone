"""Synthetic source bytes only: four-thread order, exact command and stderr preservation."""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import experiments.musicfm_inputs as inputs
from tools.musicfm_crop_io import file_identity


def row(tmp_path: Path) -> dict[str, Any]:
    source = tmp_path / "synthetic-source"
    source.write_bytes(b"not an audio file")
    return {
        "role": "fit",
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "samples": 1232,
        "source_stable_identity": file_identity(source),
        "waveform_sha256": "a" * 64,
    }


def test_threads_preserve_order_and_losslessly_retain_all_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = row(tmp_path)
    active = peak = 0
    lock = threading.Lock()
    commands = []

    def fake(
        command: list[str],
        *,
        samples: int,
        waveform_sha256: str,
        start: int,
        length: int,
        stderr: Path,
        watch: Callable[[], None],
    ) -> np.ndarray:  # noqa: ANN401
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        assert samples == 1232 and waveform_sha256 == "a" * 64
        assert "-ss" not in command and "-t" not in command
        assert command[:2] == ["taskset", "-c"] and command[2] in ["10", "11", "12", "13"]
        commands.append(command)
        watch()
        stderr.write_bytes(b"warning\x00\xff")
        time.sleep(0.02 if start == 0 else 0.001)
        with lock:
            active -= 1
        return np.full(length, start, dtype="<f4")

    monkeypatch.setattr(inputs, "stream_crop", fake)
    output = tmp_path / "logs"
    with inputs.StreamProvider(
        {"id": metadata}, output, Path("/frozen/ffmpeg"), [10, 11, 12, 13], lambda: None
    ) as provider:
        arrays = provider.resolve(provider.submit([("id", i, 1200) for i in range(32)]))
        assert [int(a[0]) for a in arrays] == list(range(32))
    assert peak <= 4 and len(commands) == 32
    records = [
        json.loads(line)
        for path in output.glob("worker-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert sorted(r["request"] for r in records) == list(range(32))
    assert all(base64.b64decode(r["stderr_base64"]) == b"warning\x00\xff" for r in records)
    assert not list(output.glob("pending-*.stderr"))


def test_failed_decoder_keeps_raw_stderr_and_never_returns_partial_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = row(tmp_path)

    def failed(command: list[str], **kwargs: Any) -> np.ndarray:  # noqa: ANN401
        kwargs["stderr"].write_bytes(b"actual failure evidence")
        raise ValueError("full-waveform mismatch")

    monkeypatch.setattr(inputs, "stream_crop", failed)
    output = tmp_path / "logs"
    with pytest.raises(ValueError, match="full-waveform"):
        with inputs.StreamProvider(
            {"id": metadata}, output, Path("/ffmpeg"), [10, 11, 12, 13], lambda: None
        ) as provider:
            provider.resolve(provider.submit([("id", 0, 1200)] * 32))
    paths = list(output.glob("pending-*.stderr"))
    assert paths and all(p.read_bytes() == b"actual failure evidence" for p in paths)


def test_identity_overlay_cannot_relabel_or_change_numeric_reference() -> None:
    from experiments.musicfm_inputs import apply_identity_overlay

    base = {
        "id": "fmak:one",
        "path": "/public/one",
        "role": "fit",
        "target": 3,
        "sha256": "a" * 64,
        "waveform_sha256": "b" * 64,
        "samples": 1200,
        "bytes": 4800,
        "source_stable_identity": [1, 2, 13, 4, 5],
    }
    keys = ("id", "path", "role", "target", "sha256", "waveform_sha256", "samples", "bytes")
    record = {
        "invariants": {k: base[k] for k in keys},
        "source_identity_old": [1, 2, 13, 4, 5],
        "source_current": {"identity": [6, 7, 13, 4, 8], "sha256": "a" * 64, "bytes": 13},
    }
    overlay = {
        "complete": True,
        "format": "musicfm-source-physical-identity-overlay-v1",
        "canonical_manifest_sha256": "c" * 64,
        "rows": [record],
    }
    result = apply_identity_overlay({"fmak:one": base}, overlay, "c" * 64)
    assert base["source_stable_identity"] == [1, 2, 13, 4, 5]
    assert result["fmak:one"]["source_stable_identity"] == [6, 7, 13, 4, 8]
    record["invariants"]["target"] = 9
    with pytest.raises(ValueError, match="labels"):
        apply_identity_overlay({"fmak:one": base}, overlay, "c" * 64)


def test_cached_provider_preserves_exact_bits_and_checks_current_source(
    tmp_path: Path,
) -> None:
    from experiments.musicfm_inputs import CachedProvider

    source = tmp_path / "source"
    source.write_bytes(b"synthetic compressed witness")
    path = tmp_path / "cache"
    values = np.array([0.0, -0.0, 1.0, 2.0], dtype="<f4")
    values.tofile(path)
    path.chmod(0o444)
    metadata = {
        "role": "fit",
        "cache": str(path),
        "cache_identity": file_identity(path),
        "path": str(source),
        "source_stable_identity": file_identity(source),
        "samples": 4,
    }
    provider = CachedProvider({"id": metadata}, lambda: None)
    try:
        assert provider.read("id", 0, 4).tobytes() == values.tobytes()
        source.write_bytes(b"changed")
        with pytest.raises(ValueError, match="identity"):
            provider.read("id", 0, 1)
    finally:
        provider.close()
