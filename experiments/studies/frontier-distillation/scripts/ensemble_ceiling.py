"""Shift-averaged ensemble accuracy for combinations of trained groups, on development and test halves."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from analyze import load_panels  # noqa: E402
from compare import is_test_half  # noqa: E402

SOURCES = {
    "shift12": "analysis-r6.npz",
    "shift7": "analysis-r6.npz",
    "poolall": "analysis-r8a.npz",
    "gen3": "analysis-r9.npz",
    "gen3w2": "analysis-r10.npz",
    "harmonic": "analysis-r11.npz",
}
COMBINATIONS = [
    ["shift12"],
    ["shift12", "shift7"],
    ["shift12", "shift7", "poolall"],
    ["shift12", "gen3"],
    ["shift12", "shift7", "poolall", "gen3", "gen3w2"],
    ["harmonic"],
    ["shift12", "harmonic"],
    ["shift12", "shift7", "harmonic"],
    ["shift12", "shift7", "poolall", "gen3", "gen3w2", "harmonic"],
]


def main() -> None:
    panels = load_panels()
    stored = {name: np.load(path) for name, path in SOURCES.items()}
    for combination in COMBINATIONS:
        cells = []
        for panel in ("valid", "holdout", "gsplus_primary", "gsplus_clean"):
            _, references, track_ids = panels[panel]
            probabilities = np.concatenate(
                [np.exp(stored[name][f"{name}/{panel}"]) for name in combination]
            )
            predictions = probabilities.mean(axis=(0, 2)).argmax(-1)
            correct = predictions == np.array(references)
            if panel.startswith("gsplus"):
                test = np.array([is_test_half(track) for track in track_ids])
                cells.append(
                    f"{panel} dev {100 * correct[~test].mean():.2f} test {100 * correct[test].mean():.2f}"
                )
            else:
                cells.append(f"{panel} {100 * correct.mean():.2f}")
        print("+".join(combination), "|", " | ".join(cells))


if __name__ == "__main__":
    main()
