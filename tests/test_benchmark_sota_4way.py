"""Synthetic protocol/cache/scoring regressions; no actual detector inference."""

from __future__ import annotations

import json
import subprocess
import sys
import types
import wave
from pathlib import Path
from typing import Any

import pytest

from tools import benchmark_sota_4way as bench


def kf_row(raw: int) -> dict[str, Any]:
    if raw == 24:
        return dict(
            schema=bench.KF_SCHEMA,
            raw_enum=raw,
            status="silence",
            tonic=None,
            mode=None,
            drone_class=None,
        )
    code = bench.KF_CLASSES[raw]
    return dict(
        schema=bench.KF_SCHEMA,
        raw_enum=raw,
        status="key",
        tonic=bench.TONICS[code // 2],
        mode="minor" if code % 2 else "major",
        drone_class=code,
    )


def test_failures_and_silence_keep_fixed_panel_denominator() -> None:
    rows = [
        dict(ref=0, x_pred=0, x_status="key", x_rel="exact", bucket="test", duration=120),
        dict(ref=0, x_pred=None, x_status="error", x_rel="error", bucket="test", duration=120),
        dict(ref=0, x_pred=None, x_status="abstain", x_rel="abstain", bucket="test", duration=120),
    ]
    out = bench.summarize(rows, "x", [1, 1, 1])
    assert (out["n_total"], out["n_valid"], out["n_error"], out["n_abstain"]) == (3, 1, 1, 1)
    assert out["exact"] == out["mirex"] == 0.3333
    for group in (out["by_genre"]["test"], out["by_duration"]["<3m"]):
        assert group["n"] == group["n_total"] == 3
        assert group["mirex"] == 0.3333


def wav(path: Path) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 128)


def test_decode_cache_binds_source_and_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, cache = tmp_path / "source.mp3", tmp_path / "cache"
    source.write_bytes(b"version one")
    calls = []

    def decode(src: Path, out: Path) -> None:
        calls.append(src.read_bytes())
        wav(out)

    monkeypatch.setattr(bench, "ff_decode", decode)
    first, receipt = bench.cached_decode(source, cache, dict(version=1))
    assert not receipt["cache_hit"]
    again, receipt = bench.cached_decode(source, cache, dict(version=1))
    assert first == again and receipt["cache_hit"] and len(calls) == 1
    source.write_bytes(b"version two")
    second, _ = bench.cached_decode(source, cache, dict(version=1))
    assert second != first and len(calls) == 2
    third, _ = bench.cached_decode(source, cache, dict(version=2))
    assert third != second and len(calls) == 3
    third.write_bytes(b"truncated")
    _, receipt = bench.cached_decode(source, cache, dict(version=2))
    assert not receipt["cache_hit"] and len(calls) == 4


def test_ffmpeg_partial_failure_never_publishes_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, cache = tmp_path / "source", tmp_path / "cache"
    source.write_bytes(b"source")

    def failure(src: Path, out: Path) -> None:
        out.write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, "mock ffmpeg")

    monkeypatch.setattr(bench, "ff_decode", failure)
    with pytest.raises(subprocess.CalledProcessError):
        bench.cached_decode(source, cache, {})
    assert not list(cache.iterdir())
    monkeypatch.setattr(bench, "ff_decode", lambda src, out: wav(out))
    path, receipt = bench.cached_decode(source, cache, {})
    assert path.is_file() and not receipt["cache_hit"]


def test_kf_selftest_requires_complete_enum_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli = tmp_path / "mock_cli"
    cli.write_bytes(b"binary")
    raw = "\n".join(json.dumps(kf_row(i)) for i in range(25))
    calls = []

    def output(command: list[str], **kwargs: object) -> str:
        calls.append(command)
        return raw

    monkeypatch.setattr(bench.subprocess, "check_output", output)
    proof = bench.qualify_keyfinder(cli)
    assert calls == [[str(cli), "--self-test"]]
    assert proof["binary"]["sha256"] == bench.file_sha(cli)
    monkeypatch.setattr(bench.subprocess, "check_output", lambda *a, **k: raw.splitlines()[0])
    with pytest.raises(ValueError, match="all25"):
        bench.qualify_keyfinder(cli)


@pytest.mark.parametrize(
    "raw",
    [
        "A major 18",
        json.dumps(kf_row(0) | {"drone_class": 0}),
        json.dumps(kf_row(0) | {"raw_enum": 25}),
        json.dumps(kf_row(24) | {"drone_class": 0}),
    ],
)
def test_kf_rejects_legacy_unknown_or_inconsistent_output(raw: str) -> None:
    with pytest.raises(ValueError):
        bench.keyfinder_result(raw)


def test_kf_silence_is_abstention_and_unknown_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = bench.Detectors.__new__(bench.Detectors)
    detector.kf_cli = Path("/mock/cli")
    monkeypatch.setattr(bench.subprocess, "check_output", lambda *a, **k: json.dumps(kf_row(24)))
    assert detector.kf(Path("/mock/audio.wav")) is None
    monkeypatch.setattr(
        bench.subprocess, "check_output", lambda *a, **k: json.dumps(kf_row(0) | {"raw_enum": -1})
    )
    with pytest.raises(ValueError):
        detector.kf(Path("/mock/audio.wav"))


def test_failed_madmom_init_skips_cnn_without_dsp_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = types.ModuleType("madmom.features.key")
    module.KEY_LABELS = []  # ty: ignore[missing-attribute]

    def unavailable() -> None:
        raise RuntimeError("weights unavailable")

    module.CNNKeyRecognitionProcessor = unavailable  # ty: ignore[missing-attribute]
    monkeypatch.setitem(sys.modules, "madmom.features.key", module)
    detector = bench.Detectors(cache_dir=tmp_path, include_drone=False)
    assert detector.madmom_mode is None and detector.mm_proc is None
    assert detector.provenance["madmom"]["status"] == "skipped"
    assert "weights unavailable" in detector.provenance["madmom"]["reason"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Db:maj", 2),
        ("D minor | E major", 5),
        ("C:dorian", None),
        ("F:garbage", None),
        ("G:maj7", None),
    ],
)
def test_reference_modes_are_strict(text: str, expected: int | None) -> None:
    assert bench.norm_key(text) == expected
