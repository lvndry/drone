"""Track loading for the key detection CNN.

A track is (id, path to mp3, parsed key label). The log-frequency spectrogram is extracted once
and cached; bounded frequency shifting reuses the cache.
"""

import glob
import os
import re

from drone.features import extract, normalize, resample_time, shift_features
from drone.keys import encode_key, parse_key, transpose_key


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
    ids = [track.id for track in tracks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate track IDs in {track_dir}")
    return tracks


def load_beatport(data_dir, label_dir):
    """Beatport 1486 set: audio under data/audio, labels are one key per file."""
    labels = {}
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Missing Beatport labels: {label_dir}. Run bash fetch.sh.")
    for path in sorted(os.listdir(label_dir)):
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
    if not tracks:
        raise ValueError("No labelled Beatport audio found. Run bash fetch.sh.")
    missing = set(labels) - {track.id for track in tracks}
    if missing:
        raise ValueError(f"Missing {len(missing)} labelled Beatport previews. Run bash fetch.sh.")
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

    tracks = load_dir(data_dir, "audio/*.mp3", label_fn)
    expected = glob.glob(os.path.join(data_dir, "annotations", "key", "*.key"))
    if not expected or len(tracks) != len(expected):
        raise ValueError("Missing GiantSteps audio or labels. Run bash fetch.sh.")
    return tracks


def prepare(track, track_dir, T=300, augment=0):
    """Log-frequency features, resampled in time to T frames, optional bounded shift.

    augment: semitones to shift up (negative = down), applied on spectrogram
    columns; the class follows the shift. Returns (features (T, 33) float32, class_id).
    """
    features = resample_time(extract(track.id, track.mp3, track_dir), T)
    features = normalize(features)
    if augment:
        features = shift_features(features, augment)
    return features, transpose_key(encode_key(*track.key), augment)
