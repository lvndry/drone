"""Typed research-only model hooks and strict PitchCircle checkpoints.

The production model/checkpoint dispatch is intentionally independent of this format.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from drone.features import FEATURE_SPEC
from drone.keys import LABEL_SCHEMA
from experiments.pitch_circle_model import SPEC, PitchCircleNet

MODEL_SOURCE = Path(__file__).with_name("pitch_circle_model.py")

ARCHITECTURE = "pitch-circle-v1"
FORMAT_VERSION = 1
KIND = "drone-key-research-checkpoint"
CheckpointLoader = Callable[[Path, str], tuple[nn.Module, dict[str, Any]]]


class CheckpointWriter(Protocol):
    def __call__(self, path: Path, model: nn.Module, *, metadata: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class ResearchModelHooks:
    architecture: str
    experiment: str
    factory: Callable[[], nn.Module]
    writer: CheckpointWriter
    loader: CheckpointLoader


def validate_metadata(metadata: object) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("Research checkpoint needs metadata")
    epoch, sources = metadata.get("epoch"), metadata.get("source_sha256")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
        raise ValueError("Research checkpoint needs a positive selected epoch")
    if (
        not isinstance(sources, dict)
        or not sources
        or any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or re.fullmatch("[0-9a-f]{64}", digest) is None
            for name, digest in sources.items()
        )
    ):
        raise ValueError("Research checkpoint needs source SHA-256 identities")
    if (
        sources.get("experiments/pitch_circle_model.py")
        != hashlib.sha256(MODEL_SOURCE.read_bytes()).hexdigest()
    ):
        raise ValueError("Research checkpoint architecture source differs from its source identity")
    if (
        metadata.get("experiment") != "pitch-circle-joint-development-v1"
        or metadata.get("arm") != "joint"
        or metadata.get("seed") not in (13, 37, 73)
        or not isinstance(metadata.get("feature_manifest_sha256"), str)
        or re.fullmatch("[0-9a-f]{64}", metadata["feature_manifest_sha256"]) is None
    ):
        raise ValueError("Research checkpoint needs declared experiment/seed/data provenance")
    return metadata


def validate_state(model: nn.Module, state: object) -> dict[str, torch.Tensor]:
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Research checkpoint state keys differ from the fixed architecture")
    for key, target in expected.items():
        value = state[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != target.shape
            or value.dtype != target.dtype
        ):
            raise ValueError(f"Research checkpoint tensor shape/dtype differs: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Research checkpoint tensor is nonfinite: {key}")
        if key.endswith(("running_var", "num_batches_tracked")) and (value < 0).any():
            raise ValueError(f"Research checkpoint BN state is invalid: {key}")
    return state


def save_pitch_circle(path: Path, model: nn.Module, *, metadata: dict[str, Any]) -> None:
    if not isinstance(model, PitchCircleNet):
        raise ValueError("Pitch-circle writer accepts only its own architecture")
    metadata = validate_metadata(metadata)
    # Preserve the source model; CPU copies also bind the serialized dtype explicitly.
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if any(value.is_floating_point() and value.dtype != torch.float32 for value in state.values()):
        raise ValueError("Pitch-circle fitting checkpoints require FP32 tensors")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Cannot save nonfinite research state")
    payload = dict(
        kind=KIND,
        format_version=FORMAT_VERSION,
        architecture=ARCHITECTURE,
        feature_spec=copy.deepcopy(FEATURE_SPEC),
        label_schema=LABEL_SCHEMA,
        model_spec=copy.deepcopy(SPEC),
        state_dict=state,
        metadata=copy.deepcopy(metadata),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_pitch_circle(path: Path, device: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    keys = {
        "kind",
        "format_version",
        "architecture",
        "feature_spec",
        "label_schema",
        "model_spec",
        "state_dict",
        "metadata",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != keys
        or payload.get("kind") != KIND
        or type(payload.get("format_version")) is not int
        or payload.get("format_version") != FORMAT_VERSION
        or payload.get("architecture") != ARCHITECTURE
        or payload.get("feature_spec") != FEATURE_SPEC
        or payload.get("label_schema") != LABEL_SCHEMA
        or payload.get("model_spec") != SPEC
    ):
        raise ValueError("Incompatible research checkpoint format/architecture/feature/label/spec")
    validate_metadata(payload["metadata"])
    model = PitchCircleNet().float()
    state = validate_state(model, payload["state_dict"])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), payload


def pitch_circle_hooks() -> ResearchModelHooks:
    return ResearchModelHooks(
        ARCHITECTURE,
        "pitch-circle-joint-development-v1",
        PitchCircleNet,
        save_pitch_circle,
        load_pitch_circle,
    )
