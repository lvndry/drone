"""drone: a small classifier for musical key detection."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from drone.keys import decode_key
from experiments.reference.checkpoints import load_checkpoint
from experiments.reference.datasets import Track, prepare
from experiments.reference.model import KeyCNN


def default_weights():
    override = os.environ.get("DRONE_REFERENCE_WEIGHTS")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "weights" / "reference-v1.pt"


def load_model(path: str | Path | None = None) -> KeyCNN:
    return load_checkpoint(path or default_weights())[0]


class Predictor:
    """Load once per analysis worker/service; no model initialization per track.

    Thread policy belongs to the host process. The CLI uses one CPU thread;
    Euphoria should use a bounded ONNX session with worker spinning disabled.
    """

    def __init__(self, weights=None, *, device="cpu", workdir=None):
        self.model, self.checkpoint = load_checkpoint(weights or default_weights(), device)
        self.device = device
        self.workdir = workdir or os.path.join(tempfile.gettempdir(), "drone-cache")

    def predict(self, audio_path):
        features, _ = prepare(
            Track("track", audio_path, ("C", "major")),
            self.workdir,
            T=self.checkpoint["frames"],
        )
        x = torch.from_numpy(features[None]).transpose(1, 2).to(self.device)
        with torch.inference_mode():
            prob = torch.softmax(self.model(x), 1).squeeze(0)
        confidence, class_id = prob.max(0)
        return (*decode_key(int(class_id)), float(confidence))


def predict(
    audio_path: str,
    workdir: str | None = None,
    *,
    weights: str | Path | None = None,
    device="cpu",
) -> tuple[str, str, float]:
    """Return tonic, mode and uncalibrated softmax probability for one file.

    Reuse a Predictor for multiple tracks. CPU is the default; GPU use is explicit.
    """
    return Predictor(weights, device=device, workdir=workdir).predict(audio_path)
