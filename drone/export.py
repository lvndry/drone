"""Export a checkpoint to ONNX and verify CPU output parity."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from drone.checkpoints import load_checkpoint
from drone.features import CROP_FRAMES, N_CORE

if TYPE_CHECKING:
    import onnxruntime as ort


class _TimeMean(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1, keepdim=True)


def cpu_session(path: str | Path, threads: int = 1) -> ort.InferenceSession:
    import onnxruntime as ort

    if threads < 1:
        raise ValueError("threads must be positive")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def export_checkpoint(checkpoint: str | Path, output: str | Path) -> dict[str, Any]:
    model, payload = load_checkpoint(checkpoint)
    # Three frequency pools leave 24 bins, so the final (24, 1) adaptive
    # pool only averages time. An explicit reduction prevents the exporter
    # from specializing its kernel to the 320-frame example.
    export_model = copy.deepcopy(model)
    export_model.net[-1] = _TimeMean()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(13)
    # Batch and time vary at inference; retain each recording's full duration.
    example = torch.randn(2, N_CORE, CROP_FRAMES, generator=generator)
    torch.onnx.export(
        export_model,
        (example,),
        str(output),
        dynamo=True,
        opset_version=18,
        input_names=["features"],
        output_names=["logits"],
        dynamic_shapes=(
            {0: torch.export.Dim("batch", min=1), 2: torch.export.Dim("frames", min=8)},
        ),
        external_data=False,
    )
    session = cpu_session(output)
    errors: list[float] = []
    for batch, frames in ((1, 8), (2, 56), (3, 319), (1, CROP_FRAMES), (2, 645)):
        x = torch.randn(batch, N_CORE, frames, generator=generator)
        with torch.inference_mode():
            expected = model(x).numpy()
        actual = np.asarray(session.run(["logits"], {"features": x.numpy()})[0], dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        errors.append(float(np.max(np.abs(actual - expected))))
    metadata = {k: v for k, v in payload.items() if k != "state_dict"}
    metadata.update(
        {
            "input_layout": "batch,bands,time",
            "dynamic_axes": ["batch", "time"],
            "output": "logits",
            "onnx_parity_max_abs_error": max(errors),
        }
    )
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    export_checkpoint(args.checkpoint, args.output)
    print(f"Exported and verified {args.output}")


if __name__ == "__main__":
    main()
