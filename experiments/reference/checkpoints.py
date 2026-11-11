"""Versioned checkpoints; legacy weights cannot describe the corrected features."""

from pathlib import Path

import torch

from drone.keys import LABEL_SCHEMA
from experiments.reference.features import FEATURE_SPEC
from experiments.reference.model import KeyCNN


def save_checkpoint(path, model, *, frames=300, **metadata):
    payload = {
        "format_version": 1,
        "feature_spec": FEATURE_SPEC,
        "label_schema": LABEL_SCHEMA,
        "frames": frames,
        "model_config": model.config,
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No trained checkpoint at {path}. Train with `uv run python -m experiments.reference.train train`, "
            "then pass the checkpoint path or set DRONE_REFERENCE_WEIGHTS. No pretrained weights ship yet."
        )
    payload = torch.load(path, map_location=device, weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != 1
        or payload.get("feature_spec") != FEATURE_SPEC
        or payload.get("label_schema") != LABEL_SCHEMA
    ):
        raise ValueError(
            "Incompatible checkpoint. Regenerate features and retrain with this version."
        )
    if not isinstance(payload.get("frames"), int) or payload["frames"] < 148:
        raise ValueError("Invalid checkpoint frame count")
    model = KeyCNN(**payload.get("model_config", {})).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
