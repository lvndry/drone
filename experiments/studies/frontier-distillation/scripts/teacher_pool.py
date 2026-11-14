"""Soft key labels for a packed unlabeled pool from an ensemble averaged over pitch shifts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from environment import required_path

sys.path.insert(0, str(required_path("DRONE_SOURCE")))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from analyze import shifted_logits  # noqa: E402
from frontier_train import load_manifest, manifest_arrays  # noqa: E402
from teacher import ensemble_probabilities  # noqa: E402

from drone.checkpoints import load_checkpoint  # noqa: E402

CHUNK_ITEMS = 2000


def pool_probabilities(
    checkpoints: list[str], pool: np.ndarray, offsets: np.ndarray, lengths: np.ndarray, device: str
) -> np.ndarray:
    models = [load_checkpoint(checkpoint, device)[0] for checkpoint in checkpoints]
    output = np.zeros((len(offsets), 24), dtype=np.float32)
    for start in range(0, len(offsets), CHUNK_ITEMS):
        stop = min(start + CHUNK_ITEMS, len(offsets))
        arrays = [
            np.asarray(pool[:, offsets[item_index] : offsets[item_index] + lengths[item_index]])
            for item_index in range(start, stop)
        ]
        total = np.zeros((stop - start, 24), dtype=np.float64)
        for model in models:
            logits = torch.from_numpy(shifted_logits(model, arrays, device))
            total += logits.softmax(-1).mean(1).numpy()
        output[start:stop] = total / len(models)
        print(f"teacher {stop}/{len(offsets)}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--pool", required=True, help="pool prefix written by pack_pool.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    checkpoints = arguments.checkpoints.split(",")
    pool = np.load(f"{arguments.pool}.npy", mmap_mode="r")
    index = np.load(f"{arguments.pool}-index.npz")
    train_rows = load_manifest()["train"]
    train = ensemble_probabilities(checkpoints, manifest_arrays(train_rows), arguments.device)
    unlabeled = pool_probabilities(
        checkpoints, pool, index["offsets"], index["lengths"], arguments.device
    )
    np.savez_compressed(
        arguments.out,
        train=train,
        train_ids=np.array([row["track_id"] for row in train_rows]),
        unlabeled=unlabeled,
        unlabeled_ids=index["track_ids"],
    )
    print("unlabeled mean top probability:", round(float(unlabeled.max(-1).mean()), 3))


if __name__ == "__main__":
    main()
