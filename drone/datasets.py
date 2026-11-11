"""Track loading for key detection.

A track is (id, path to audio, parsed key label). The CQT is computed once
and cached by content hash; pitch-shift augmentation is a re-slice of the
core bins, which reuses the cache.
"""

import os
import re

import numpy as np

from drone.features import (
    BINS_PER_SEMI,
    CROP_FRAMES,
    N_CORE,
    N_TOTAL,
    extract_cqt,
    window,
)
from drone.keys import TONICS, encode_key, transpose_key


class Track:
    def __init__(self, track_id, audio, key):
        self.id = track_id
        self.audio = audio
        self.mp3 = audio  # compatibility alias
        self.key = key  # (tonic, mode)
        self.class_id = encode_key(*key)


def parse_key(text):
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
    if tonic not in TONICS:
        tonic = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}.get(tonic)
        if tonic is None:
            return None
    return tonic, mode


def _load_dir(base_dir, audio_glob, label_fn, recursive=False):
    import glob

    pattern = os.path.join(base_dir, audio_glob)
    paths = glob.glob(pattern, recursive=recursive) if recursive else glob.glob(pattern)
    tracks = []
    for path in sorted(paths):
        name = os.path.basename(path)
        m = re.match(r"(\d+)", name)
        if not m:
            continue
        track_id = m.group(1)
        key = label_fn(track_id, name)
        if key is None:
            continue
        tracks.append(Track(track_id, path, key))
    return tracks


def load_beatport(data_dir, label_dir):
    """Beatport EDM Key Dataset: audio under data/audio, one label per file."""
    labels = {}
    for path in os.listdir(label_dir):
        if not path.endswith(".txt"):
            continue
        m = re.match(r"(\d+)", path)
        if not m:
            continue
        with open(os.path.join(label_dir, path)) as f:
            key = parse_key(f.read())
        if key is not None:
            labels[m.group(1)] = key
    audio = os.path.join(data_dir, "audio")
    tracks = _load_dir(audio, "**/*.mp3", lambda tid, name: labels.get(tid), recursive=True)
    if not tracks:
        raise ValueError(f"No tracks found under {audio}")
    return tracks


def load_giantsteps(data_dir):
    """GiantSteps Key Dataset (the held-out test set): bench/data/key layout."""

    def label_fn(track_id, name):
        # audio is <id>.LOFI.mp3, label is <id>.LOFI.key
        path = os.path.join(data_dir, "annotations", "key", name[: -len(".mp3")] + ".key")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return parse_key(f.read())

    tracks = _load_dir(data_dir, "audio/*.mp3", label_fn)
    if not tracks:
        raise ValueError(f"Missing GiantSteps audio/annotations under {data_dir}")
    return tracks


def sample_features(track, track_dir, rng, *, shift=None, T=CROP_FRAMES):
    """One training sample: (N_CORE, T) with an augmented class id.

    shift: semitones to raise the audio (None = uniform in +/-4 for rng).
    The label transposes by -shift, matching how window() moves the slice.
    """
    if shift is None:
        shift = int(rng.integers(-4, 5))
    cqt = extract_cqt(track.mp3, track_dir)
    features = window(cqt, shift)
    if T is not None:
        if features.shape[1] < T:
            features = np.pad(features, ((0, 0), (0, T - features.shape[1])))
        else:
            start = int(rng.integers(0, features.shape[1] - T + 1))
            features = features[:, start : start + T]
    return features, transpose_key(track.class_id, -shift)


def prepare(track, track_dir, *, T=None):
    """Inference feature for one track: (N_CORE, frames), no augmentation."""
    return window(extract_cqt(track.mp3, track_dir)), track.key
