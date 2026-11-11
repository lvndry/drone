"""drone: a small classifier for musical key detection."""

from __future__ import annotations

import os
import tempfile
from importlib.resources import files
from pathlib import Path

import torch

from drone.checkpoints import load_checkpoint
from drone.datasets import Track, prepare
from drone.keys import decode_key
from drone.model import KeyCNN


def default_weights() -> Path:
    override = os.environ.get("DRONE_WEIGHTS")
    if override:
        return Path(override)
    packaged = Path(str(files("drone").joinpath("weights", "drone-v1.pt")))
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parent.parent / "weights" / "drone-v1.pt"


def load_model(path: str | Path | None = None) -> KeyCNN:
    return load_checkpoint(path or default_weights())[0]


class Predictor:
    """Load once per analysis worker/service; no model initialization per track.

    Thread policy belongs to the host process. The CLI uses one CPU thread;
    Euphoria should use a bounded ONNX session with worker spinning disabled.
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        device: str = "cpu",
        workdir: str | None = None,
    ):
        self.model, self.checkpoint = load_checkpoint(weights or default_weights(), device)
        self.device = device
        self.workdir = workdir or os.path.join(tempfile.gettempdir(), "drone-cache")

    def predict(self, audio_path: str) -> tuple[str, str, float]:
        # Full-length feature; the net pools over time, so no fixed frame count.
        features, _ = prepare(Track("track", audio_path, ("C", "major")), self.workdir)
        x = torch.from_numpy(features[None]).to(self.device)
        with torch.inference_mode():
            prob = torch.softmax(self.model(x), 1).squeeze(0)
        confidence, class_id = prob.max(0)
        return (*decode_key(int(class_id)), float(confidence))


def predict(
    audio_path: str,
    workdir: str | None = None,
    *,
    weights: str | Path | None = None,
    device: str = "cpu",
) -> tuple[str, str, float]:
    """Return tonic, mode and uncalibrated softmax probability for one file.

    Reuse a Predictor for multiple tracks. CPU is the default; GPU use is explicit.
    """
    return Predictor(weights, device=device, workdir=workdir).predict(audio_path)
