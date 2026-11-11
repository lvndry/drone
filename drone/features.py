"""Constant-Q features for key detection, matching the K10 (MIREX 2026, MIT) pipeline.

A true CQT: 24 bins per octave, 8 core octaves up from C1, log1p amplitude,
hop 4096. The spectrogram is computed with a +/-4 semitone pad (208 bins)
so pitch-shift augmentation is an exact re-slice of 192 core bins; 24
bins per octave gives 2 bins per semitone, so the network sees absolute
frequency position, which carries the tonic.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from drone.datasets import Track

SR = 22050
FFT = 4096
HOP = 4096
BINS_PER_OCT = 24
N_OCT = 8
PAD_SEMI = 4
BINS_PER_SEMI = BINS_PER_OCT // 12
N_TOTAL = (N_OCT * 12 + 2 * PAD_SEMI) * BINS_PER_SEMI  # 208, with pad
N_CORE = N_OCT * BINS_PER_OCT  # 192

CROP_FRAMES = 320  # K10 training crop length (~78 s at 4.1 fps)

FEATURE_SPEC = "cqt24-8oct-c1-pad4-hop4096-v2"
FEATURE_ID = hashlib.sha256(
    json.dumps(
        {
            "spec": FEATURE_SPEC,
            "sr": SR,
            "fft": FFT,
            "hop": HOP,
            "n_oct": N_OCT,
            "bins_per_oct": BINS_PER_OCT,
            "pad_semi": PAD_SEMI,
        },
        sort_keys=True,
    ).encode()
).hexdigest()[:16]


def _note_to_hz(name: str) -> float:
    """C1 = 32.703199..., one semitone = twelfth root of two."""
    semis_from_c0 = (
        12 * int(name[-1]) + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[name[0]]
    )
    return 16.351597933828117 * 2.0 ** (semis_from_c0 / 12)


FMIN_HZ = _note_to_hz("C1") * 2.0 ** (-PAD_SEMI / 12)


def cqt(samples: np.ndarray) -> np.ndarray:
    """Constant-Q transform: (N_TOTAL bins, n_frames) log1p amplitude.

    librosa.cqt, exactly as the K10 (MIREX 2026, MIT) script: 24 bins per
    octave, 208 bins from two semitones below C1, hop 4096, default window.
    """
    import librosa

    samples = np.ascontiguousarray(samples, dtype=np.float32)
    coeffs = librosa.cqt(
        samples,
        sr=SR,
        hop_length=HOP,
        fmin=FMIN_HZ,
        n_bins=N_TOTAL,
        bins_per_octave=BINS_PER_OCT,
    )
    mag = np.abs(coeffs)
    if mag.shape[0] > N_TOTAL:
        mag = mag[:N_TOTAL]
    return np.log1p(mag).astype(np.float32)


def window(cqt: np.ndarray, shift: int = 0) -> np.ndarray:
    """Slice the N_CORE core bins, starting `shift` semitones above C1.

    A positive shift moves content down the frequency axis (the track sounds
    `shift` semitones lower), so the key label must transpose by -shift.
    """
    if not -PAD_SEMI <= shift <= PAD_SEMI:
        raise ValueError(f"shift must be within +/-{PAD_SEMI} semitones")
    lo = (PAD_SEMI + shift) * BINS_PER_SEMI
    return cqt[lo : lo + N_CORE, :]


def cqt_path(track_dir: str, name: str) -> str:
    return os.path.join(track_dir, "cqt", FEATURE_ID, f"{name}.npy")


def audio_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_mp3(path: str | Path, out_wav: str | Path) -> None:
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", path, "-ac", "1", "-ar", str(SR), out_wav],
        check=True,
    )


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM wav. Returns (float32 samples in [-1, 1], sample_rate)."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 12
    sample_rate = None
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        (chunk_size,) = struct.unpack("<I", data[pos + 4 : pos + 8])
        if chunk_id == b"fmt ":
            # audio_format, n_channels, sample_rate, byte_rate, block_align, bits
            (_, _, sample_rate, _, _, bits) = struct.unpack("<HHIIHH", data[pos + 8 : pos + 24])
            if bits != 16:
                raise ValueError(f"unexpected sample width {bits} for {path}")
        elif chunk_id == b"data":
            raw = data[pos + 8 : pos + 8 + chunk_size]
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            return samples, sample_rate  # type: ignore[return-value]  # None only if the fmt chunk is absent
        pos += 8 + chunk_size + (chunk_size & 1)
    raise ValueError(f"bad wav: {path}")


def resample_linear(x: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate:
        return x
    n_out = int(len(x) * to_rate / from_rate)
    t = np.linspace(0.0, len(x) - 1.0, n_out)
    i = np.floor(t).astype(np.int64)
    frac = (t - i).astype(np.float32)
    i = np.clip(i, 0, len(x) - 2)
    return x[i] * (1.0 - frac) + x[i + 1] * frac


def extract_cqt(mp3: str | Path, track_dir: str | Path) -> np.ndarray:
    """Decode one audio file to its full (N_TOTAL, frames) CQT, cached by
    content hash. Cached arrays are float16; load as float32."""
    out = Path(cqt_path(str(track_dir), audio_digest(mp3)))
    if out.is_file():
        features = np.load(out)
        if features.ndim == 2 and features.shape[0] == N_TOTAL and np.isfinite(features).all():
            return features.astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        wav = Path(scratch) / f"{out.name}.wav"
        decode_mp3(str(mp3), str(wav))
        samples, rate = read_wav_mono(str(wav))
        samples = resample_linear(samples, rate, SR)
        features = cqt(samples).astype(np.float16)
        pending = Path(scratch) / out.name
        np.save(pending, features)
        os.replace(pending, out)
    return features.astype(np.float32)


def prepare(
    track: Track,
    track_dir: str | Path,
    *,
    shift: int = 0,
    T: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, tuple[str, str]]:
    """Core-bins feature for one track: (N_CORE, T) or (N_CORE, frames).

    With T set: shorter tracks are zero-padded, longer ones random-cropped
    (rng) to T frames, matching the K10 training protocol.
    """
    features = window(extract_cqt(track.audio, track_dir), shift)
    if T is not None:
        if T < 1:
            raise ValueError("T must be positive")
        if features.shape[1] < T:
            features = np.pad(features, ((0, 0), (0, T - features.shape[1])))
        elif rng is not None:
            start = int(rng.integers(0, features.shape[1] - T + 1))
            features = features[:, start : start + T]
    return features, track.key
