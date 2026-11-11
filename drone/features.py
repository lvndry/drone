"""CQT features for the key detection CNN.

The spectrogram is a 33-semitone log frequency CQT (55 Hz to 880 Hz, one
semitone per column, amplitude per frame), built from a single STFT. Pitch
shifting is a circular shift of the semitone columns, which also moves the
label, so training-time augmentation is nearly free.
"""

import os
import struct
import subprocess

import numpy as np

SR = 22050
SEMIBASE = 55  # A1
N_SEM = 33
FFT_SIZE = 4096
HOP = 512

TONICS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def cqt_path(track_dir, name):
    return os.path.join(track_dir, "cqt", f"{name}.npy")


def decode_mp3(path, out_wav):
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", path, "-ac", "1", "-ar", str(SR), out_wav],
        check=True,
    )


def read_wav_mono(path):
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
            return samples, sample_rate
        pos += 8 + chunk_size + (chunk_size & 1)
    raise ValueError(f"bad wav: {path}")


def resample_linear(x, from_rate, to_rate):
    if from_rate == to_rate:
        return x
    n_out = int(len(x) * to_rate / from_rate)
    t = np.linspace(0.0, len(x) - 1.0, n_out)
    i = np.floor(t).astype(np.int64)
    frac = (t - i).astype(np.float32)
    i = np.clip(i, 0, len(x) - 2)
    return x[i] * (1.0 - frac) + x[i + 1] * frac


def stft_magnitude(x):
    # gather the strided frames into a fresh array so the window can be applied in place
    starts = np.arange(0, len(x) - FFT_SIZE + 1, HOP)
    idx = starts[:, None] + np.arange(FFT_SIZE)[None, :]
    frames = x[idx].astype(np.float32)
    window = np.hanning(FFT_SIZE).astype(np.float32)
    frames *= window
    spec = np.fft.rfft(frames, axis=1)
    return np.abs(spec).astype(np.float32)


def build_cqt(x):
    """(n_frames, n_sem) semitone-amplitude CQT. Bins per semitone are summed."""
    mag = stft_magnitude(x)
    freqs = np.arange(mag.shape[1]) * (SR / FFT_SIZE)
    # nearest semitone for every bin, A1..A7
    sem_of_bin = np.rint(np.log2(np.maximum(freqs, 1.0) / SEMIBASE)).astype(np.int64)
    valid = (sem_of_bin >= 0) & (sem_of_bin < N_SEM)
    bins = np.where(valid)[0]
    cqt = np.zeros((mag.shape[0], N_SEM), dtype=np.float32)
    for b in bins:
        cqt[:, int(sem_of_bin[b])] += mag[:, b]
    cqt = np.log1p(10.0 * cqt / max(np.percentile(cqt, 99), 1e-9))
    return cqt


def extract(name, mp3, track_dir):
    """Decode one mp3 to a CQT, cache it, return the array (n_frames, 33)."""
    out = cqt_path(track_dir, name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not os.path.exists(out):
        wav = out + ".wav"
        decode_mp3(mp3, wav)
        samples, rate = read_wav_mono(wav)
        os.remove(wav)
        x = resample_linear(samples, rate, SR)
        if len(x) < FFT_SIZE * 2:
            x = np.pad(x, (0, FFT_SIZE * 2 - len(x)))
        np.save(out, build_cqt(x))
    return np.load(out)


FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def parse_key(text):
    """'D# minor | E major' -> ('D#', 'minor'); None when unparseable.
    Flats are normalized to sharps so both datasets share the same 24 classes."""
    first = text.split("|")[0].strip()
    parts = first.split()
    if len(parts) != 2:
        return None
    tonic, mode = parts
    if mode not in ("major", "minor"):
        return None
    if tonic not in TONICS:
        tonic = FLATS.get(tonic)
        if tonic is None:
            return None
    return tonic, mode
