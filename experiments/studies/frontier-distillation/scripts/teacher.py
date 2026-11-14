"""Write track-level soft key labels from an ensemble averaged over pitch shifts."""

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
from frontier_train import load_arrays, load_manifest, manifest_arrays  # noqa: E402

from drone.checkpoints import load_checkpoint  # noqa: E402


def ensemble_probabilities(
    checkpoints: list[str], arrays: list[np.ndarray], device: str
) -> np.ndarray:
    total = np.zeros((len(arrays), 24), dtype=np.float64)
    for checkpoint in checkpoints:
        model, _ = load_checkpoint(checkpoint, device)
        logits = torch.from_numpy(shifted_logits(model, arrays, device))
        total += logits.softmax(-1).mean(1).numpy()
    return (total / len(checkpoints)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--unlabeled", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    checkpoints = arguments.checkpoints.split(",")
    train_rows = load_manifest()["train"]
    train = ensemble_probabilities(checkpoints, manifest_arrays(train_rows), arguments.device)
    unlabeled_paths = sorted(Path(arguments.unlabeled).glob("*.npy"))
    unlabeled = ensemble_probabilities(checkpoints, load_arrays(unlabeled_paths), arguments.device)
    np.savez_compressed(
        arguments.out,
        train=train,
        train_ids=np.array([row["track_id"] for row in train_rows]),
        unlabeled=unlabeled,
        unlabeled_ids=np.array([path.stem for path in unlabeled_paths]),
    )
    references = np.array([int(row["key_label"]) for row in train_rows])
    print(
        "teacher agrees with fitting-pool labels:",
        round(100 * float((train.argmax(-1) == references).mean()), 2),
    )
    print("unlabeled mean top probability:", round(float(unlabeled.max(-1).mean()), 3))


if __name__ == "__main__":
    main()
