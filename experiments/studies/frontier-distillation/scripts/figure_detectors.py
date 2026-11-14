"""Slope chart of exact accuracy per detector as the GiantSteps labels get cleaner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PANELS = [
    ("gs_original", "Original labels\n(604 tracks)"),
    ("gsplus_primary", "Expert labels,\nprimary key (571)"),
    ("gsplus_clean", "Expert labels,\nsingle-key tracks (457)"),
]
DETECTORS = [
    ("shift12 ensemble, shift-averaged", "This study (109k parameters)", "#2a78d6", "solid"),
    ("madmom KEY_CNN", "madmom KEY_CNN", "#1baf7a", "solid"),
    ("drone-v1", "drone-v1 (released)", "#eb6834", (0, (4, 3))),
    ("libkeyfinder 2.2.8", "libkeyfinder 2.2.8", "#eda100", "solid"),
]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#22221f"
TEXT_SECONDARY = "#5f5e57"
GRID = "#e6e5e0"
LABEL_GAP = 2.0


def label_positions(values: list[float]) -> list[float]:
    """Nudge end labels apart so lines that finish close together stay readable."""
    order = sorted(range(len(values)), key=lambda position: values[position])
    positions = list(values)
    for rank in range(1, len(order)):
        below = order[rank - 1]
        current = order[rank]
        positions[current] = max(positions[current], positions[below] + LABEL_GAP)
    return positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--out", required=True, help="output path without extension")
    arguments = parser.parse_args()
    results = json.loads(Path(arguments.baselines).read_text())
    figure, axis = plt.subplots(figsize=(8.4, 5.2), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    x_positions = list(range(len(PANELS)))
    end_values = []
    for detector, _, color, line_style in DETECTORS:
        values = [results[panel][detector]["all"]["exact"] for panel, _ in PANELS]
        end_values.append(values[-1])
        axis.plot(x_positions, values, color=color, linewidth=2, linestyle=line_style, zorder=3)
        axis.scatter(
            x_positions, values, s=64, color=color, edgecolors=SURFACE, linewidths=2, zorder=4
        )
    for (detector, label, _, _), position in zip(
        DETECTORS, label_positions(end_values), strict=True
    ):
        value = results[PANELS[-1][0]][detector]["all"]["exact"]
        axis.annotate(
            f"{label}  {value:.1f}%",
            xy=(x_positions[-1], value),
            xytext=(x_positions[-1] + 0.12, position),
            va="center",
            fontsize=9.5,
            color=TEXT_PRIMARY,
            annotation_clip=False,
        )
    axis.set_xticks(x_positions, [name for _, name in PANELS], fontsize=9.5, color=TEXT_SECONDARY)
    axis.set_xlim(-0.25, len(PANELS) - 1 + 0.1)
    axis.set_ylim(55, 92)
    axis.set_ylabel("Exact key accuracy (%)", fontsize=10, color=TEXT_SECONDARY)
    axis.tick_params(axis="y", colors=TEXT_SECONDARY, labelsize=9)
    axis.tick_params(axis="x", length=0)
    axis.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left", "bottom"):
        axis.spines[side].set_visible(False)
    axis.set_title(
        "GiantSteps key detection: the cleaner the labels, the wider the gap",
        loc="left",
        fontsize=12,
        color=TEXT_PRIMARY,
        pad=14,
    )
    figure.subplots_adjust(left=0.09, right=0.66, top=0.88, bottom=0.16)
    for extension in ("png", "svg"):
        figure.savefig(f"{arguments.out}.{extension}", dpi=200, facecolor=SURFACE)


if __name__ == "__main__":
    main()
