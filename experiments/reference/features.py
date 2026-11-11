"""STFT energy grouped into 33 semitone bands, A1 through F4.

This is a log-frequency spectrogram, not a constant-Q transform. Bounded
frequency translations approximate pitch shifts; they never wrap octaves.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np

SR = 22050
SEMIBASE = 55  # A1
N_SEM = 33
FFT_SIZE = 4096
HOP = 512
FEATURE_SPEC = {
    "version": 2,
    "sample_rate": SR,
    "base_hz": SEMIBASE,
    "bands": N_SEM,
    "fft_size": FFT_SIZE,
    "hop": HOP,
    "window": "hann",
    "compression": "log1p-10x-p99",
    "time_resampling": "linear-per-band",
    "normalization": "per-track-zscore",
    "augmentation": "bounded-band-shift-renormalized",
}
FEATURE_ID = hashlib.sha256(json.dumps(FEATURE_SPEC, sort_keys=True).encode()).hexdigest()[:16]


def audio_digest(path: str | Path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cqt_path(track_dir: str | Path, name: str) -> str:
    return os.path.join(track_dir, "cqt", FEATURE_ID, f"{name}.npy")


def decode_mp3(path: str | Path, out_wav: str | Path) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(SR),
                "-c:a",
                "pcm_s16le",
                str(out_wav),
            ],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Audio decoding requires ffmpeg on PATH") from exc


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read mono, uncompressed 16-bit PCM WAV produced by ffmpeg."""
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError(f"Expected mono 16-bit PCM: {path}")
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        return samples.astype(np.float32) / 32768.0, wav.getframerate()


def _audio_frames(x: np.ndarray) -> np.ndarray:
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Audio must be a finite, nonempty mono waveform")
    if len(x) < FFT_SIZE:
        x = np.pad(x, (0, FFT_SIZE - len(x)))
    return np.lib.stride_tricks.sliding_window_view(x, FFT_SIZE)[::HOP]


def _magnitude_blocks(frames: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    window = np.hanning(FFT_SIZE).astype(np.float32)
    for start in range(0, len(frames), 256):
        block = frames[start : start + 256] * window
        yield start, np.abs(np.fft.rfft(block, axis=1)).astype(np.float32)


def stft_magnitude(x: np.ndarray) -> np.ndarray:
    """Full STFT reference; band extraction uses bounded blocks directly."""
    frames = _audio_frames(x)
    result = np.empty((len(frames), FFT_SIZE // 2 + 1), dtype=np.float32)
    for start, magnitude in _magnitude_blocks(frames):
        result[start : start + len(magnitude)] = magnitude
    return result


def build_cqt(x: np.ndarray) -> np.ndarray:
    """Return (frames, bands) log-frequency energy (historical function name).

    Retain only the narrow band matrix, not a track-length full FFT matrix.
    The feature contract and floating-point aggregation order are unchanged.
    """
    frames = _audio_frames(x)
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1 / SR)
    sem_of_bin = np.rint(12 * np.log2(np.maximum(freqs, 1.0) / SEMIBASE)).astype(int)
    band_bins = [np.flatnonzero(sem_of_bin == band) for band in range(N_SEM)]
    bands = np.zeros((len(frames), N_SEM), dtype=np.float32)
    for start, magnitude in _magnitude_blocks(frames):
        for band, indices in enumerate(band_bins):
            bands[start : start + len(magnitude), band] = magnitude[:, indices].sum(axis=1)
    return np.log1p(10 * bands / max(float(np.percentile(bands, 99)), 1e-9))


def normalize(features: np.ndarray) -> np.ndarray:
    mu = features.mean(dtype=np.float64)
    sigma = features.std(dtype=np.float64)
    if sigma < 1e-8 * max(1.0, abs(mu)):
        return np.zeros_like(features)
    return ((features - mu) / sigma).astype(np.float32)


def resample_time(features: np.ndarray, frames: int) -> np.ndarray:
    """Interpolate time independently for each frequency band."""
    if frames < 1 or features.ndim != 2 or not len(features):
        raise ValueError("Expected a nonempty (time, frequency) array and positive frame count")
    positions = np.linspace(0, len(features) - 1, frames)
    source = np.arange(len(features))
    return np.column_stack(
        [np.interp(positions, source, features[:, band]) for band in range(features.shape[1])]
    ).astype(np.float32)


def shift_features(features: np.ndarray, semitones: int | np.integer) -> np.ndarray:
    """Translate absolute-frequency bands; discard edges and pad with the energy floor.

    Accepts raw or z-scored (time, frequency) features. Renormalization matches
    preparation. This is a bounded spectrogram approximation, not audio resampling.
    """
    semitones = int(semitones)
    if not semitones:
        return features.copy()
    shifted = np.full_like(features, features.min())
    if 0 < semitones < features.shape[1]:
        shifted[:, semitones:] = features[:, :-semitones]
    elif -features.shape[1] < semitones < 0:
        shifted[:, :semitones] = features[:, -semitones:]
    return normalize(shifted)


def extract(name: str, mp3: str | Path, track_dir: str | Path) -> np.ndarray:
    """Cache by audio content and feature specification, independent of track IDs."""
    out = Path(cqt_path(track_dir, audio_digest(mp3)))
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
            wav = Path(scratch) / "audio.wav"
            decode_mp3(mp3, wav)
            samples, rate = read_wav_mono(wav)
            if rate != SR:
                raise ValueError(f"Decoder returned {rate} Hz, expected {SR}")
            features = build_cqt(samples)
            pending = Path(scratch) / "features.npy"
            np.save(pending, features)
            os.replace(pending, out)
    return np.load(out, allow_pickle=False)
