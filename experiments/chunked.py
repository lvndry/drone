"""Research-only, Torch-free, unchanged-weight K10 CPU chunk inference."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import onnxruntime as ort

FEATURE_SPEC = "cqt24-8oct-c1-pad4-hop4096-v2"
LABEL_SCHEMA = "tonic-times-two-plus-minor-v1"
CLASS_ORDER = [
    f"{tonic} {mode}"
    for tonic in ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    for mode in ("major", "minor")
]
ATOL = 1e-5
RTOL = 1e-5


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_session(path: str | Path) -> ort.InferenceSession:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def core_windows(frames: int, core_outputs: int = 64) -> Iterator[tuple[int, int, int, int]]:
    """Yield q0, q1, input-start, input-stop; starts preserve pooling phase."""
    if frames < 8 or core_outputs < 1:
        raise ValueError("Require frames >= 8 and positive core_outputs")
    for q0 in range(0, frames // 8, core_outputs):
        q1 = min(frames // 8, q0 + core_outputs)
        yield q0, q1, max(0, 8 * q0 - 16), min(frames, 8 * q1 + 16)


def validate_input(features: np.ndarray) -> None:
    if features.dtype != np.float32 or features.ndim != 3 or features.shape[:2] != (1, 192):
        raise ValueError("Expected FP32 [1,192,T], with T >= 8")
    if features.shape[-1] < 8 or not np.isfinite(features).all():
        raise ValueError("Require finite features and T >= 8")


def compare_logits(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    if actual.shape != (1, 24) or expected.shape != (1, 24):
        raise ValueError("Expected [1,24] logits")
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    error = float(np.max(np.abs(actual.astype(np.float64) - expected)))
    ordered = np.sort(expected[0])
    return {
        "finite": finite,
        "allclose": bool(finite and np.allclose(actual, expected, atol=ATOL, rtol=RTOL)),
        "argmax_agrees": bool(finite and np.argmax(actual) == np.argmax(expected)),
        "max_abs_error": error if np.isfinite(error) else None,
        "reference_margin": float(ordered[-1] - ordered[-2]) if finite else None,
    }


class ChunkedK10:
    def __init__(self, bundle: str | Path, core_outputs: int = 64) -> None:
        if core_outputs < 1:
            raise ValueError("core_outputs must be positive")
        bundle = Path(bundle)
        self.metadata = json.loads((bundle / "contract.json").read_text())
        required = {
            "format": "k10-hidden-chunks-v1",
            "feature_spec": FEATURE_SPEC,
            "label_schema": LABEL_SCHEMA,
            "class_order": CLASS_ORDER,
            "input_bands": 192,
            "frequency_bins": 24,
            "stride": 8,
            "receptive_field": 36,
            "halo_frames": 16,
            "aggregation": "hidden-fp64-mean-cast-fp32-head-once",
        }
        if any(self.metadata.get(k) != v for k, v in required.items()):
            raise ValueError("Incompatible chunk bundle contract")
        self.channels = int(self.metadata["channels"])
        if self.channels < 1:
            raise ValueError("Invalid channel count")
        for name in ("prefix", "head"):
            if sha256(bundle / f"{name}.onnx") != self.metadata[f"{name}_sha256"]:
                raise ValueError(f"Changed {name} ONNX artifact")
        self.core_outputs = core_outputs
        self.prefix = cpu_session(bundle / "prefix.onnx")
        self.head = cpu_session(bundle / "head.onnx")

    def hidden(self, features: np.ndarray) -> NDArray[np.float32]:
        validate_input(features)
        result = np.asarray(self.prefix.run(["hidden"], {"features": features})[0])
        if result.shape != (1, self.channels, 24, features.shape[-1] // 8):
            raise ValueError("Unexpected prefix output shape")
        if result.dtype != np.float32 or not np.isfinite(result).all():
            raise ValueError("Invalid prefix output")
        return result

    def pooled_logits(self, pooled: NDArray[np.float32]) -> NDArray[np.float32]:
        if pooled.shape != (1, self.channels, 24, 1) or pooled.dtype != np.float32:
            raise ValueError("Unexpected pooled input")
        result = np.asarray(self.head.run(["logits"], {"pooled": pooled})[0])
        if result.shape != (1, 24) or not np.isfinite(result).all():
            raise ValueError("Invalid head output")
        return result

    def full_prefix_logits(self, features: np.ndarray) -> NDArray[np.float32]:
        hidden = self.hidden(features)
        pooled = hidden.mean(axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
        return self.pooled_logits(pooled)

    def predict(self, features: np.ndarray) -> NDArray[np.float32]:
        validate_input(features)
        total = np.zeros((1, self.channels, 24, 1), dtype=np.float64)
        count = 0
        for q0, q1, a, b in core_windows(features.shape[-1], self.core_outputs):
            hidden = self.hidden(np.ascontiguousarray(features[:, :, a:b]))
            retained = hidden[:, :, :, q0 - a // 8 : q1 - a // 8]
            if retained.shape[-1] != q1 - q0:
                raise ValueError("Incomplete output core")
            total += retained.sum(axis=-1, keepdims=True, dtype=np.float64)
            count += q1 - q0
            # Release the chunk before evaluating the next: no overlapping workspaces.
            del retained, hidden
        if count != features.shape[-1] // 8:
            raise ValueError("Output count mismatch")
        return self.pooled_logits((total / count).astype(np.float32))
