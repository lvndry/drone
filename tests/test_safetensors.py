from __future__ import annotations

from pathlib import Path

import torch

from drone.checkpoints import load_checkpoint, save_checkpoint
from drone.features import FEATURE_SPEC
from drone.model import KeyCNN
from drone.safetensors import export_safetensors, read_safetensors


def test_safetensors_round_trip_reproduces_the_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(13)
    checkpoint = tmp_path / "model.pt"
    output = tmp_path / "model.safetensors"
    trained = KeyCNN()
    state = trained.state_dict()
    state["net.1.running_mean"] = torch.linspace(-1, 1, 16)
    state["net.1.running_var"] = torch.linspace(0.5, 2, 16)
    trained.load_state_dict(state)
    save_checkpoint(checkpoint, trained)
    metadata = export_safetensors(checkpoint, output)
    tensors, stored = read_safetensors(output)
    assert stored == metadata
    assert stored["feature_spec"] == FEATURE_SPEC
    model, _ = load_checkpoint(checkpoint)
    expected = {
        name: value for name, value in model.state_dict().items() if value.is_floating_point()
    }
    assert tensors.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(tensors[name], value), name
    restored = KeyCNN()
    restored.load_state_dict(tensors, strict=False)
    restored.eval()
    features = torch.randn(1, 192, 64)
    with torch.inference_mode():
        assert torch.equal(restored(features), model(features))
