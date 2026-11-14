"""Flag training labels that out-of-fold predictions confidently contradict (confident learning).

A track is suspect when its shift-averaged out-of-fold prediction differs from its label
and the probability of some other class reaches that class's average self-confidence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

KEY_CLASSES = 24


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(-1, keepdims=True))
    return shifted / shifted.sum(-1, keepdims=True)


def key_relation(prediction: int, reference: int) -> str:
    if prediction == reference:
        return "exact"
    predicted_tonic, predicted_minor = divmod(prediction, 2)
    reference_tonic, reference_minor = divmod(reference, 2)
    if predicted_minor == reference_minor and (predicted_tonic - reference_tonic) % 12 in (5, 7):
        return "fifth"
    if predicted_minor != reference_minor and predicted_tonic == reference_tonic:
        return "parallel"
    if reference_minor and not predicted_minor and (predicted_tonic - reference_tonic) % 12 == 3:
        return "relative"
    if predicted_minor and not reference_minor and (reference_tonic - predicted_tonic) % 12 == 3:
        return "relative"
    return "other"


def load_held_out(run_directories: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = []
    references = []
    track_ids = []
    for directory in run_directories:
        held_out = np.load(directory / "held-out.npz")
        logits.append(held_out["logits"])
        references.append(held_out["references"])
        track_ids.append(held_out["track_ids"])
    return (
        np.concatenate(logits).astype(np.float64),
        np.concatenate(references).astype(int),
        np.concatenate(track_ids).astype(str),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-runs", nargs="+", required=True, help="run directories holding held-out.npz"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    logits, references, track_ids = load_held_out(
        [Path(directory) for directory in arguments.fold_runs]
    )
    probabilities = softmax(logits).mean(axis=1)
    predicted = probabilities.argmax(-1)
    class_thresholds = np.ones(KEY_CLASSES)
    for key_class in range(KEY_CLASSES):
        members = references == key_class
        if members.any():
            class_thresholds[key_class] = probabilities[members, key_class].mean()
    confident_elsewhere = probabilities >= class_thresholds[None, :]
    confident_elsewhere[np.arange(len(references)), references] = False
    suspect = confident_elsewhere.any(-1) & (predicted != references)
    relations = np.array(
        [
            key_relation(int(guess), int(label))
            for guess, label in zip(predicted, references, strict=True)
        ]
    )
    rows = json.loads(Path(arguments.manifest).read_text())["tracks"]
    source_by_track = {
        str(row["track_id"]): row["source"] for row in rows if row["panel"] == "train"
    }
    sources = np.array([source_by_track.get(track, "unknown") for track in track_ids])
    print(f"out-of-fold exact {100 * (predicted == references).mean():.2f}")
    for source in sorted(set(sources)):
        members = sources == source
        exact = 100 * (predicted[members] == references[members]).mean()
        print(
            f"{source}: tracks {members.sum()}, exact {exact:.2f}, suspect {suspect[members].sum()} "
            f"({100 * suspect[members].mean():.1f}%), {dict(Counter(relations[members & suspect]))}"
        )
    np.savez(
        arguments.out,
        track_ids=track_ids,
        suspect=suspect,
        predicted=predicted,
        references=references,
        probabilities=probabilities,
    )


if __name__ == "__main__":
    main()
