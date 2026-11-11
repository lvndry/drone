"""drone: a 0.33M-parameter classifier for the musical key of a track."""

from __future__ import annotations

import os

import torch

from drone.datasets import prepare, Track
from drone.features import TONICS
from drone.model import KeyCNN

_WEIGHTS = os.path.join(os.path.dirname(__file__), "weights", "drone-v1.pt")


def load_model(path: str = _WEIGHTS) -> KeyCNN:
    model = KeyCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def predict(audio_path: str, workdir: str | None = None) -> tuple[str, str, float]:
    """Return (tonic, mode, confidence) for one audio file.

    audio_path: any file ffmpeg can decode (mp3, wav, flac, ...)
    workdir: where CQT feature cache files are stored (default: ./drone-cache)
    """
    import tempfile

    workdir = workdir or os.path.join(tempfile.gettempdir(), "drone-cache")
    model = load_model()
    track = Track("track", audio_path, ("C", "major"))
    features, _ = prepare(track, workdir)
    with torch.no_grad():
        logits = model(torch.from_numpy(features[None]).transpose(1, 2))
    prob = torch.softmax(logits, 1).squeeze(0)
    conf, class_id = prob.max(0)
    class_id, conf = int(class_id), float(conf)
    return TONICS[class_id % 12], "minor" if class_id // 12 else "major", conf


if __name__ == "__main__":
    import sys

    for path in sys.argv[1:]:
        tonic, mode, conf = predict(path)
        print(f"{path}: {tonic} {mode} ({100 * conf:.1f}%)")
