"""Export a separate research K10 prefix/head bundle without changing weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from drone.checkpoints import load_checkpoint
from experiments.chunked import CLASS_ORDER, FEATURE_SPEC, LABEL_SCHEMA, sha256


class HiddenPrefix(torch.nn.Module):
    def __init__(self, net: torch.nn.Sequential) -> None:
        super().__init__()
        self.net = net[:-1]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.unsqueeze(1))


def export_bundle(checkpoint: Path, output: Path) -> dict[str, Any]:
    # Never overwrite an existing bundle or the authoritative full ONNX artifact.
    if output.exists():
        raise FileExistsError(f"Choose a new bundle directory: {output}")
    model, _ = load_checkpoint(checkpoint)
    prefix = HiddenPrefix(model.net).eval()
    head = torch.nn.Sequential(torch.nn.Flatten(1), model.head).eval()
    channels = model.head.in_features // 24
    output.mkdir(parents=True)
    torch.onnx.export(
        prefix,
        (torch.zeros(1, 192, 320),),
        str(output / "prefix.onnx"),
        dynamo=True,
        opset_version=18,
        input_names=["features"],
        output_names=["hidden"],
        dynamic_shapes=({2: torch.export.Dim("frames", min=8)},),
        external_data=False,
    )
    torch.onnx.export(
        head,
        (torch.zeros(1, channels, 24, 1),),
        str(output / "head.onnx"),
        dynamo=True,
        opset_version=18,
        input_names=["pooled"],
        output_names=["logits"],
        external_data=False,
    )
    metadata = {
        "format": "k10-hidden-chunks-v1",
        "feature_spec": FEATURE_SPEC,
        "label_schema": LABEL_SCHEMA,
        "class_order": CLASS_ORDER,
        "checkpoint_sha256": sha256(checkpoint),
        "prefix_sha256": sha256(output / "prefix.onnx"),
        "head_sha256": sha256(output / "head.onnx"),
        "channels": channels,
        "frequency_bins": 24,
        "input_bands": 192,
        "stride": 8,
        "receptive_field": 36,
        "halo_frames": 16,
        "input": "FP32 [1,192,T], T>=8",
        "hidden": "FP32 [1,C,24,floor(T/8)]",
        "aggregation": "hidden-fp64-mean-cast-fp32-head-once",
        "research_only": True,
        "validation_passed": False,
        "torch_version": torch.__version__,
    }
    (output / "contract.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(export_bundle(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
