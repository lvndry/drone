import importlib.util

import pytest
import torch

from drone.checkpoints import save_checkpoint
from drone.model import KeyCNN

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("onnx", "onnxscript", "onnxruntime")),
    reason="Install the export extra to verify ONNX deployment",
)


@pytest.mark.parametrize("width", [0.5, 0.7, 1.0])
def test_export_verifies_dynamic_batch_and_cpu_parity(tmp_path, width):
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
    options = cpu_session(output).get_session_options()
    assert options.intra_op_num_threads == 1
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"
