from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drone.datasets import Track
from drone.features import (
    BINS_PER_SEMI,
    FMIN_HZ,
    N_CORE,
    N_TOTAL,
    PAD_SEMI,
    SR,
    cqt,
    extract_cqt,
    prepare,
    window,
)


def _tone(hz: float, seconds: float = 2) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (np.sin(2 * np.pi * hz * t) * 0.5).astype(np.float32)


@pytest.mark.parametrize("n", [4, 24, 96, 150, 204])
def test_tone_lands_in_its_cqt_bin(n: int) -> None:
    freq = FMIN_HZ * 2 ** (n / 24)
    features = cqt(_tone(freq))
    got = int(features.mean(axis=1).argmax(axis=0))
    assert abs(got - n) <= 1, f"tone at bin {n} detected at column {got}"


def test_tone_is_pitch_specific_not_octave_specific() -> None:
    # 24 bins per octave: a tone one octave up must land 24 columns away.
    freq = FMIN_HZ * 2 ** (2.0 + PAD_SEMI / 12)  # C1 * 2 octaves = C3
    t = np.arange(int(SR * 2)) / SR
    features = cqt((np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32))
    profile = features.mean(axis=1)
    got = int(profile.argmax(axis=0))
    assert abs(got - 56) <= 1, f"expected column ~56 (C3), got {got}"


def test_window_slices_core_bins_and_shifts_exactly() -> None:
    rng = np.random.default_rng(0)
    features = cqt((rng.standard_normal(SR * 4) * 0.05).astype(np.float32))
    core = window(features, 0)
    assert core.shape == (N_CORE, features.shape[1])
    assert np.array_equal(
        core, features[PAD_SEMI * BINS_PER_SEMI : (PAD_SEMI * BINS_PER_SEMI) + N_CORE, :]
    )
    # +1 semitone shift moves the window 2 columns down (content sounds lower)
    shifted = window(features, 1)
    assert np.array_equal(
        shifted,
        features[(PAD_SEMI + 1) * BINS_PER_SEMI : (PAD_SEMI + 1) * BINS_PER_SEMI + N_CORE, :],
    )
    with pytest.raises(ValueError):
        window(features, PAD_SEMI + 1)


def test_prepare_pads_short_and_returns_track_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import drone.features as feats

    def stub_extract(mp3: str | Path, track_dir: str | Path) -> np.ndarray:
        return np.zeros((N_TOTAL, 10), dtype=np.float32)

    # No audio file: prepare's pad path runs on the stubbed full-length CQT.
    monkeypatch.setattr(feats, "extract_cqt", stub_extract)
    track = Track("1", "ignored", ("D", "major"))
    features, key = prepare(track, "/tmp", T=320)
    assert features.shape == (N_CORE, 320)
    assert key == ("D", "major")


def test_extract_cqt_caches_by_content(tmp_path: Path) -> None:
    import wave

    mp3 = tmp_path / "tone.mp3"
    wav = tmp_path / "tone.wav"
    samples = _tone(220.0, 1.0)
    with wave.open(str(wav), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes((samples * 32767).astype(np.int16).tobytes())
    # ffmpeg re-encodes; for the cache test we can point decode at a wav via a
    # wrapper: extract_cqt takes any ffmpeg-decodable file.
    import subprocess

    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), str(mp3)], check=True)
    first = extract_cqt(str(mp3), tmp_path / "cqt")
    # Second call must hit the content-hash cache and be byte-identical.
    second = extract_cqt(str(mp3), tmp_path / "cqt")
    assert first.shape[0] == N_TOTAL
    assert np.array_equal(first, second)
    cached = list((tmp_path / "cqt").rglob("*.npy"))
    assert len(cached) == 1


def test_short_audio_does_not_crash() -> None:
    features = cqt(np.ones(SR, dtype=np.float32))
    assert features.shape[0] == N_TOTAL
    assert np.isfinite(features).all()


def test_tiny_audio_does_not_crash() -> None:
    features = cqt(np.ones(100, dtype=np.float32))
    assert features.shape[0] == N_TOTAL
    assert np.isfinite(features).all()
