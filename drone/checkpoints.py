"""Versioned checkpoints; legacy weights cannot describe the K10 features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from drone.features import CROP_FRAMES, FEATURE_SPEC
from drone.keys import LABEL_SCHEMA
from drone.model import KeyCNN


def save_checkpoint(
    path: str | Path, model: KeyCNN, *, width: float = 1.0, **metadata: object
) -> None:
    payload: dict[str, Any] = {
        "format_version": 2,
        "feature_spec": FEATURE_SPEC,
        "label_schema": LABEL_SCHEMA,
        "crop": CROP_FRAMES,
        "model_config": {"width": width},
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[KeyCNN, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No trained checkpoint at {path}. Train with `uv run train.py train`, "
            "then pass the checkpoint path or set DRONE_WEIGHTS. No pretrained weights ship yet."
        )
    payload = torch.load(path, map_location=device, weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != 2
        or payload.get("feature_spec") != FEATURE_SPEC
        or payload.get("label_schema") != LABEL_SCHEMA
    ):
        raise ValueError(
            "Incompatible checkpoint. Regenerate features and retrain with this version."
        )
    model = KeyCNN(**payload.get("model_config", {})).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
