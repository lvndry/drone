"""Exact and weighted accuracy of every key detector on the GiantSteps panels, all with the same labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from compare import SHIFTS, is_test_half
from score_madmom import madmom_predictions, panel_references

from drone.keys import score_mirex

MISSING_PREDICTION = -1


def keyfinder_predictions(path: Path) -> dict[str, int]:
    records = json.loads(path.read_text())
    return {
        track: int(record["drone_class"])
        for track, record in records.items()
        if record.get("status") == "key" and record.get("drone_class") is not None
    }


def drone_predictions(specification: str, shift_average: bool) -> dict[str, np.ndarray]:
    npz_path, group = specification.split(":")
    stored = np.load(npz_path)
    by_panel = {}
    for panel in ("gs_original", "gsplus_primary", "gsplus_clean"):
        log_probabilities = stored[f"{group}/{panel}"]
        if shift_average:
            scores = np.exp(log_probabilities).mean(axis=(0, 2))
        else:
            scores = np.exp(log_probabilities[:, :, SHIFTS.index(0)]).mean(axis=0)
        by_panel[panel] = scores.argmax(-1)
    return by_panel


def scores(predictions: list[int], references: list[int]) -> dict:
    exact, weighted = score_mirex(predictions, references)
    return {
        "tracks": len(references),
        "exact": round(100 * exact, 2),
        "weighted": round(100 * weighted, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyfinder", required=True)
    parser.add_argument("--madmom", required=True)
    parser.add_argument(
        "--drone", action="append", required=True, help="name=npz:group[:shift-average]"
    )
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    references_by_panel = panel_references()
    lookup_systems = {
        "libkeyfinder 2.2.8": keyfinder_predictions(Path(arguments.keyfinder)),
        "madmom KEY_CNN": madmom_predictions(Path(arguments.madmom)),
    }
    array_systems = {}
    for specification in arguments.drone:
        name, source = specification.split("=", 1)
        shift_average = source.endswith(":shift-average")
        array_systems[name] = drone_predictions(
            source.removesuffix(":shift-average"), shift_average
        )
    result: dict = {}
    for panel, (track_ids, references) in references_by_panel.items():
        test_mask = np.array([is_test_half(track) for track in track_ids])
        test_references = [
            reference for reference, is_test in zip(references, test_mask, strict=True) if is_test
        ]
        result[panel] = {}
        for name, predictions in lookup_systems.items():
            classes = [predictions.get(track, MISSING_PREDICTION) for track in track_ids]
            result[panel][name] = {
                "all": scores(classes, references),
                "test_half": scores(
                    [value for value, is_test in zip(classes, test_mask, strict=True) if is_test],
                    test_references,
                ),
                "missing": sum(1 for value in classes if value == MISSING_PREDICTION),
            }
        for name, by_panel in array_systems.items():
            classes = [int(value) for value in by_panel[panel]]
            result[panel][name] = {
                "all": scores(classes, references),
                "test_half": scores(
                    [value for value, is_test in zip(classes, test_mask, strict=True) if is_test],
                    test_references,
                ),
                "missing": 0,
            }
    Path(arguments.out).write_text(json.dumps(result, indent=1))
    for panel, systems in result.items():
        print(panel)
        for name, values in systems.items():
            print(
                f"  {name:40s} all {values['all']['exact']:6.2f} / {values['all']['weighted']:6.2f}   test half {values['test_half']['exact']:6.2f}   missing {values['missing']}"
            )


if __name__ == "__main__":
    main()
