"""Export a checkpoint to ONNX and verify CPU output parity."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from drone.checkpoints import load_checkpoint
from drone.features import N_SEM


def cpu_session(path, threads=1):
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


def export_checkpoint(checkpoint, output):
    model, payload = load_checkpoint(checkpoint)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(13)
    # Export with batch >1 so the batch dimension remains symbolic.
    example = torch.randn(2, N_SEM, payload["frames"], generator=generator)
    torch.onnx.export(
        model,
        (example,),
        str(output),
        dynamo=True,
        opset_version=18,
        input_names=["features"],
        output_names=["logits"],
        dynamic_shapes=({0: torch.export.Dim("batch", min=1)},),
        external_data=False,
    )
    session = cpu_session(output)
    errors = []
    for batch in (1, 2, 3):
        x = torch.randn(batch, N_SEM, payload["frames"], generator=generator)
        with torch.inference_mode():
            expected = model(x).numpy()
        actual = session.run(["logits"], {"features": x.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        errors.append(float(np.max(np.abs(actual - expected))))
    metadata = {k: v for k, v in payload.items() if k != "state_dict"}
    metadata.update(
        {
            "input_layout": "batch,bands,time",
            "output": "logits",
            "onnx_parity_max_abs_error": max(errors),
        }
    )
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    export_checkpoint(args.checkpoint, args.output)
    print(f"Exported and verified {args.output}")


if __name__ == "__main__":
    main()
