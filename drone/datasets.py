"""Track loading for key detection.

A track is (id, path to audio, parsed key label). The CQT is computed once
and cached by content hash; pitch-shift augmentation is a re-slice of the
core bins, which reuses the cache.
"""

from __future__ import annotations

import glob
import os
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np

from drone.features import (
    CROP_FRAMES,
    extract_cqt,
    window,
)
from drone.keys import TONICS, Key, encode_key, transpose_key


class Track:
    def __init__(self, track_id: str, audio: str, key: Key) -> None:
        self.id = track_id
        self.audio = audio
        self.key: Key = key  # (tonic, mode)
        self.class_id: int = encode_key(*key)


def parse_key(text: str) -> Key | None:
    """'D# minor | E major' -> ('D#', 'minor'); None when unparseable.

    Flats are normalized to sharps so both datasets share the same 24 classes.
    """
    first = text.split("|")[0].strip()
    parts = first.split()
    if len(parts) != 2:
        return None
    tonic, mode = parts
    if mode not in ("major", "minor"):
        return None
    if tonic in TONICS:
        return tonic, mode
    flats = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
    if tonic in flats:
        return flats[tonic], mode
    return None


def _load_dir(
    base_dir: str | Path,
    audio_glob: str,
    label_fn: Callable[[str, str], Key | None],
    recursive: bool = False,
) -> list[Track]:
    pattern = os.path.join(str(base_dir), audio_glob)
    paths = glob.glob(pattern, recursive=recursive) if recursive else glob.glob(pattern)
    tracks: list[Track] = []
    for path in sorted(paths):
        name = os.path.basename(path)
        match = re.match(r"(\d+)", name)
        if not match:
            continue
        track_id = match.group(1)
        key = label_fn(track_id, name)
        if key is None:
            continue
        tracks.append(Track(track_id, path, key))
    return tracks


def load_beatport(data_dir: str | Path, label_dir: str | Path) -> list[Track]:
    """Beatport EDM Key Dataset: audio under data/audio, one label per file."""
    labels: dict[str, Key] = {}
    for path in os.listdir(label_dir):
        if not path.endswith(".txt"):
            continue
        match = re.match(r"(\d+)", path)
        if not match:
            continue
        with open(os.path.join(label_dir, path)) as label_file:
            key = parse_key(label_file.read())
        if key is not None:
            labels[match.group(1)] = key
    audio = os.path.join(str(data_dir), "audio")
    tracks = _load_dir(
        audio, "**/*.mp3", lambda track_id, name: labels.get(track_id), recursive=True
    )
    if not tracks:
        raise ValueError(f"No tracks found under {audio}")
    return tracks


def load_giantsteps(data_dir: str | Path) -> list[Track]:
    """GiantSteps Key Dataset (the held-out test set): bench/data/key layout."""

    def label_fn(track_id: str, name: str) -> Key | None:
        # audio is <id>.LOFI.mp3, label is <id>.LOFI.key
        path = os.path.join(str(data_dir), "annotations", "key", name[: -len(".mp3")] + ".key")
        if not os.path.exists(path):
            return None
        with open(path) as label_file:
            return parse_key(label_file.read())

    tracks = _load_dir(data_dir, "audio/*.mp3", label_fn)
    if not tracks:
        raise ValueError(f"Missing GiantSteps audio/annotations under {data_dir}")
    return tracks


def sample_features(
    track: Track,
    track_dir: str | Path,
    rng: np.random.Generator,
    *,
    shift: int | None = None,
    T: int = CROP_FRAMES,
) -> tuple[np.ndarray, int]:
    """One training sample: (N_CORE, T) with an augmented class id.

    shift: semitones to raise the audio (None = uniform in +/-4 for rng).
    The label transposes by -shift, matching how window() moves the slice.
    """
    if shift is None:
        shift = int(rng.integers(-4, 5))
    cqt = extract_cqt(track.audio, track_dir)
    features = window(cqt, shift)
    if T is not None:
        if features.shape[1] < T:
            features = np.pad(features, ((0, 0), (0, T - features.shape[1])))
        else:
            start = int(rng.integers(0, features.shape[1] - T + 1))
            features = features[:, start : start + T]
    return features, transpose_key(track.class_id, -shift)


def prepare(track: Track, track_dir: str | Path, *, T: int | None = None) -> tuple[np.ndarray, Key]:
    """Inference feature for one track: (N_CORE, frames), no augmentation."""
    return window(extract_cqt(track.audio, track_dir)), track.key
