"""Score frozen checkpoints on FMAK development only; never score the final holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.reference.checkpoints import load_checkpoint
from experiments.reference.train import load_prepared
from experiments.small import development_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    features, meta = load_prepared(args.data, "fmak")
    partition_bytes = args.partition.read_bytes()
    split = json.loads(partition_bytes)
    expected = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    if split["metadata_sha256"] != expected:
        raise ValueError("Partition does not match the prepared corpus")
    indices = split["development_indices"]
    if set(indices) & set(split["final_holdout_indices"]):
        raise ValueError("Development and final holdout overlap")
    model, payload = load_checkpoint(args.checkpoint)
    if payload["frames"] != meta["frames"]:
        raise ValueError("Feature frame count differs from the frozen checkpoint")
    used_audio = set(payload["metadata"].get("training_audio_sha256", []))
    if used_audio & {meta["tracks"][index]["sha256"] for index in indices}:
        raise ValueError("Exact training audio overlaps external development corpus")
    result = {
        "corpus": "FMAKv2 development; final 1000 holdout not scored",
        "tracks": len(indices),
        "final_holdout_scored": False,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "partition_sha256": hashlib.sha256(partition_bytes).hexdigest(),
        "data_metadata_sha256": expected,
        "metrics": development_metrics(
            model, features[indices], [meta["refs"][index] for index in indices]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "tracks": len(indices),
                "metrics": {
                    key: value
                    for key, value in result["metrics"].items()
                    if key in ("exact", "mirex", "mirex_asymmetric", "mode_accuracy")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
