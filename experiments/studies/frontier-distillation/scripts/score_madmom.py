"""Score madmom KEY_CNN on GiantSteps panels and compare it, track by track, with a Drone group."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from compare import BOOTSTRAP_SAMPLES, SHIFTS, interval, is_test_half
from frontier_train import GIANTSTEPS_DIRECTORY, WORK_DIRECTORY
from rescore_gsplus import load_gsplus, load_original, parse_single_key

from drone.keys import score_mirex


def panel_references() -> dict[str, tuple[list[str], list[int]]]:
    audio_files = sorted(
        path
        for path in (GIANTSTEPS_DIRECTORY / "audio").glob("*.mp3")
        if not path.name.startswith("._")
    )
    track_ids = [path.name.split(".")[0] for path in audio_files]
    original = load_original(GIANTSTEPS_DIRECTORY / "annotations" / "key")
    gsplus, _ = load_gsplus(WORK_DIRECTORY / "gsplus-download" / "annotations" / "new_annotations")
    original_ids = [track for track in track_ids if track in original]
    gsplus_ids = [track for track in track_ids if track in gsplus]
    clean_ids = [track for track in gsplus_ids if gsplus[track]["clean"]]
    return {
        "gs_original": (original_ids, [original[track] for track in original_ids]),
        "gsplus_primary": (gsplus_ids, [gsplus[track]["primary"] for track in gsplus_ids]),
        "gsplus_clean": (clean_ids, [gsplus[track]["primary"] for track in clean_ids]),
    }


def madmom_predictions(path: Path) -> dict[str, int]:
    record = json.loads(path.read_text())
    label_classes: list[int] = []
    for label in record["labels"]:
        key_class = parse_single_key(label)
        if key_class is None:
            raise ValueError(f"unparsed madmom label: {label}")
        label_classes.append(key_class)
    return {
        track: label_classes[int(np.argmax(values))]
        for track, values in record["probabilities"].items()
    }


def drone_correctness(stack: np.ndarray, references: list[int], shift_average: bool) -> np.ndarray:
    logits = stack.mean(2) if shift_average else stack[:, :, SHIFTS.index(0)]
    ensemble = logits.mean(0)
    return (ensemble.argmax(-1) == np.array(references)).astype(float)


def paired_row(
    drone_correct: np.ndarray, madmom_correct: np.ndarray, generator: np.random.Generator
) -> dict:
    differences = drone_correct - madmom_correct
    low, high = interval(differences, generator)
    return {
        "tracks": len(differences),
        "madmom_exact": round(100 * madmom_correct.mean(), 2),
        "drone_exact": round(100 * drone_correct.mean(), 2),
        "difference": round(100 * differences.mean(), 2),
        "interval_95": [round(100 * low, 2), round(100 * high, 2)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--madmom", required=True)
    parser.add_argument("--drone", required=True, help="npz:group")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    predictions = madmom_predictions(Path(arguments.madmom))
    npz_path, group = arguments.drone.split(":")
    stored = np.load(npz_path)
    generator = np.random.default_rng(0)
    result: dict = {"drone": arguments.drone, "bootstrap_samples": BOOTSTRAP_SAMPLES, "panels": {}}
    for panel, (track_ids, references) in panel_references().items():
        madmom_classes = [predictions.get(track, -1) for track in track_ids]
        madmom_exact, madmom_mirex = score_mirex(madmom_classes, references)
        madmom_correct = (np.array(madmom_classes) == np.array(references)).astype(float)
        stack = stored[f"{group}/{panel}"]
        rows = {
            "madmom_mirex": round(100 * madmom_mirex, 2),
            "missing_predictions": sum(1 for value in madmom_classes if value == -1),
        }
        for mode, shift_average in (("ensemble", False), ("ensemble_shift_average", True)):
            drone_correct = drone_correctness(stack, references, shift_average)
            test_mask = np.array([is_test_half(track) for track in track_ids])
            rows[mode] = {
                "all": paired_row(drone_correct, madmom_correct, generator),
                "test_half": paired_row(
                    drone_correct[test_mask], madmom_correct[test_mask], generator
                ),
            }
        result["panels"][panel] = rows
    Path(arguments.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
