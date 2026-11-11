"""Track loading for the key detection CNN.

A track is (id, path to mp3, parsed key label). The CQT is extracted once
and cached; pitch shifting is applied on the CQT, which reuses the cache.
"""

import glob
import os
import re

import numpy as np

from drone.features import TONICS, extract, parse_key


class Track:
    def __init__(self, track_id, mp3, key):
        self.id = track_id
        self.mp3 = mp3
        self.key = key  # (tonic, mode)


def load_dir(track_dir, mp3_glob, label_fn, recursive=False):
    """Load tracks from a dataset directory.

    mp3_glob: glob relative to track_dir matching audio files.
    label_fn: fn(id, filename) -> (tonic, mode) or None.
    """
    pattern = os.path.join(track_dir, mp3_glob)
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
    """Beatport 1486 set: audio under data/audio, labels are one key per file."""
    labels = {}
    for path in os.listdir(label_dir):
        if not path.endswith(".txt"):
            continue
        m = re.match(r"(\d+)", path)
        if not m:
            continue
        with open(os.path.join(label_dir, path)) as f:
            text = f.read().strip()
        key = parse_key(text)
        if key is not None:
            labels[m.group(1)] = key
    audio = os.path.join(data_dir, "audio")
    tracks = load_dir(audio, "**/*.mp3", lambda tid, name: labels.get(tid), recursive=True)
    return tracks


def load_giantsteps(data_dir):
    """GiantSteps 604 set (the test set): bench/data/key layout."""
    def label_fn(track_id, name):
        # audio is <id>.LOFI.mp3, label is <id>.LOFI.key
        path = os.path.join(data_dir, "annotations", "key", name[: -len(".mp3")] + ".key")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return parse_key(f.read())

    return load_dir(data_dir, "audio/*.mp3", label_fn)


def prepare(track, track_dir, T=300, augment=0):
    """CQT for a track, resampled in time to T frames, optional pitch shift.

    augment: semitones to shift up (negative = down), applied on the CQT
    columns; the class follows the shift. Returns (features (33, T) float32, class_id).
    """
    cqt = extract(track.id, track.mp3, track_dir)  # (n_frames, 33)
    if cqt.shape[0] < 2 * T:
        cqt = np.pad(cqt, ((0, 2 * T - cqt.shape[0]), (0, 0)))
    flat = np.linspace(0.0, cqt.shape[0] * cqt.shape[1] - 1, T * cqt.shape[1])
    i = np.floor(flat).astype(np.int64)
    frac = (flat - i).astype(np.float64)
    i = np.clip(i, 0, cqt.shape[0] * cqt.shape[1] - 2)
    flat_x = cqt.reshape(-1).astype(np.float64)
    cqt = (flat_x[i] * (1.0 - frac) + flat_x[i + 1] * frac).reshape(T, cqt.shape[1])
    if augment:
        cqt = np.roll(cqt, augment, axis=1)
    tonic_idx = TONICS.index(track.key[0])
    class_id = ((tonic_idx + augment) % 12) * 2 + (1 if track.key[1] == "minor" else 0)
    mu, sigma = cqt.mean(), cqt.std()
    features = ((cqt - mu) / (sigma + 1e-6)).astype(np.float32)
    return features, class_id
