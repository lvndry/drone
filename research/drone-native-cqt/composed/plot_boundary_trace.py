# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.11.2"]
# ///
"""Render fixed-case numerical diagnostics from verified aggregate metadata."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from summarize_boundary_trace import summarize


def plot(root: Path) -> None:
    result = summarize(root)
    comparisons = result["complete_array_comparisons"]
    plt.rcParams["svg.hashsalt"] = "cqt-fixed-boundary-v1"
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4), layout="constrained")
    groups = [
        (
            [f"stage{i}.response" for i in range(9)],
            [f"Stage {i}" for i in range(9)],
            "CSR outputs differ at every stage",
        ),
        (
            [
                "terminal." + key
                for key in ("pre_normalization", "normalized", "magnitude", "log1p", "fp16")
            ],
            ["Stacked complex", "Normalized complex", "Magnitude", "log1p", "Final FP16"],
            "Nine differences remain in final FP16",
        ),
    ]
    for axis, (keys, labels, title) in zip(axes, groups, strict=True):
        for index, key in enumerate(keys):
            row = comparisons[key]
            percentage = 100 * row["unequal_bits"] / row["elements"]
            axis.barh(index, percentage, color="#336b8e" if index < len(keys) - 1 else "#a4582c")
            axis.text(percentage + 1.0, index, f"{row['unequal_bits']:,}", va="center", fontsize=9)
        axis.set_yticks(range(len(keys)), labels)
        axis.invert_yaxis()
        axis.set_xlim(0, 85)
        axis.set_xlabel("Elements differing bitwise (%)")
        axis.set_title(title, fontsize=11, pad=12)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=0.15)
        axis.set_axisbelow(True)
    fig.suptitle("Fixed Mac CQT case: identical CSR operands, different responses", fontsize=14)
    fig.supxlabel(
        "Full arrays, not probes. Labels show differing element counts.\n"
        "All stage inputs/resamples agree; the single FFT difference is outside CSR support.\n"
        "No model execution. The historical strict portability gate remains failed.",
        fontsize=9,
    )
    fig.savefig(root / "boundary-trace.png", dpi=180)
    fig.savefig(root / "boundary-trace.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    plot(Path(__file__).resolve().parent)
