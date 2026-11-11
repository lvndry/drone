"""Export a checkpoint's tensors as safetensors for native hosts without PyTorch.

The file holds every learned parameter and BatchNorm running statistic in
float32, under its PyTorch state-dict name, and records the feature, label and
checkpoint contracts in the header metadata. It does not change the weights:
a host that reads it must reproduce the feature front end and the network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch

from drone.checkpoints import load_checkpoint


def export_safetensors(checkpoint: str | Path, output: str | Path) -> dict[str, str]:
    checkpoint = Path(checkpoint)
    model, payload = load_checkpoint(checkpoint)
    tensors = {
        name: tensor.detach().to(torch.float32).contiguous()
        for name, tensor in model.state_dict().items()
        if tensor.is_floating_point()
    }
    metadata = {
        "format": "drone-safetensors-v1",
        "feature_spec": str(payload["feature_spec"]),
        "label_schema": str(payload["label_schema"]),
        "model_config": json.dumps(payload.get("model_config", {}), sort_keys=True),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    blobs: list[bytes] = []
    for name, tensor in tensors.items():
        blob = tensor.numpy().astype("<f4").tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)
        blobs.append(blob)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))
    return metadata


def read_safetensors(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    data = Path(path).read_bytes()
    (header_length,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + header_length])
    metadata = header.pop("__metadata__")
    start = 8 + header_length
    tensors = {}
    for name, entry in header.items():
        begin, end = entry["data_offsets"]
        values = torch.frombuffer(bytearray(data[start + begin : start + end]), dtype=torch.float32)
        tensors[name] = values.reshape(entry["shape"])
    return tensors, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    metadata = export_safetensors(args.checkpoint, args.output)
    print(f"Exported {args.output} from checkpoint {metadata['checkpoint_sha256']}")


if __name__ == "__main__":
    main()
