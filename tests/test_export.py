from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from drone.checkpoints import load_checkpoint, save_checkpoint
from drone.model import KeyCNN

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("onnx", "onnxscript", "onnxruntime")),
    reason="Install the export extra to verify ONNX deployment",
)


@pytest.mark.parametrize("width", [0.5, 0.7, 1.0])
def test_export_verifies_dynamic_batch_and_cpu_parity(tmp_path: Path, width: float) -> None:
    from drone.export import cpu_session, export_checkpoint

    torch.set_num_threads(1)
    torch.manual_seed(13)
    checkpoint = tmp_path / "model.pt"
    output = tmp_path / "model.onnx"
    save_checkpoint(checkpoint, KeyCNN(width=width), width=width)
    metadata = export_checkpoint(checkpoint, output)
    assert abs(metadata["model_config"]["width"] - width) < 1e-9
    assert metadata["onnx_parity_max_abs_error"] < 1e-5
    assert output.is_file()
    assert output.with_suffix(".json").is_file()
    session = cpu_session(output)
    restored, _ = load_checkpoint(checkpoint)
    for frames in (8, 56, 319, 645):
        features = torch.randn(1, 192, frames)
        with torch.inference_mode():
            expected = restored(features)
        actual = session.run(["logits"], {"features": features.numpy()})[0]
        torch.testing.assert_close(torch.from_numpy(actual), expected, rtol=1e-4, atol=1e-5)
    options = cpu_session(output).get_session_options()
    assert options.intra_op_num_threads == 1
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"


def test_packaged_onnx_matches_shipped_checkpoint_at_full_lengths() -> None:
    from drone.export import cpu_session

    weights = Path(__file__).resolve().parents[1] / "weights"
    model, _ = load_checkpoint(weights / "drone-v1.pt")
    session = cpu_session(weights / "drone-v1.onnx")
    shape = session.get_inputs()[0].shape
    assert isinstance(shape[0], str) and isinstance(shape[2], str)
    for frames in (8, 319, 645):
        features = torch.randn(1, 192, frames)
        with torch.inference_mode():
            expected = model(features)
        actual = session.run(["logits"], {"features": features.numpy()})[0]
        torch.testing.assert_close(torch.from_numpy(actual), expected, rtol=1e-4, atol=1e-5)
