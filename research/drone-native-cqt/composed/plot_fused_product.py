# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.11.2"]
# ///
"""Plot retained arithmetic explanations and fixed-case candidate outcome."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from summarize_fused_product import summarize


def plot(root: Path) -> None:
    candidate = summarize(root)
    raw = (root / "csr-product-rounding-results.json").read_bytes()
    provenance = json.loads((root / "csr-product-rounding-provenance.json").read_bytes())
    if hashlib.sha256(raw).hexdigest() != provenance["results_sha256"]:
        raise ValueError("Arithmetic aggregate identity changed")
    stages = json.loads(raw)["stages"]
    plt.rcParams["svg.hashsalt"] = "cqt-fused-product-v1"
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.5), layout="constrained")
    for index, stage in enumerate(stages):
        count = stage["opposite_unequal"]
        axes[0].barh(index, count, height=0.6, color="#a4582c")
        axes[0].plot(stage["fused_python_unequal"], index, "o", color="#277f67", clip_on=False)
        axes[0].text(count + 100, index, f"{count:,}", va="center", fontsize=9)
    axes[0].set_yticks(range(9), [f"Stage {i}" for i in range(9)])
    axes[0].set_ylim(10, -0.7)
    axes[0].set_xlim(0, 8000)
    axes[0].set_xlabel("Responses differing from archived Python")
    axes[0].set_title("Arithmetic replay: zero fused-path differences", fontsize=11, pad=12)
    axes[0].plot([], [], color="#a4582c", linewidth=8, label="Separate products")
    axes[0].plot([], [], "o", color="#277f67", label="Fused products: 0 at every stage")
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    labels = ["Original native\ncontrol", "Fused-product\ncandidate"]
    values = [candidate["original_differences"], candidate["candidate_differences"]]
    axes[1].bar(labels, values, color=["#a4582c", "#277f67"], width=0.55)
    for i, count in enumerate(values):
        axes[1].text(i, count + 0.3, str(count), ha="center", fontsize=14, weight="bold")
    axes[1].set_ylim(0, 11)
    axes[1].set_ylabel("Different final-core values / 76,608")
    axes[1].set_title("Compiled frontend: all nine differences removed", fontsize=11, pad=12)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    fig.suptitle("Fused complex products explain and remove one retained CQT failure", fontsize=14)
    fig.supxlabel(
        "One 74.1-second FMAK case on Apple M5; no classifier execution or accuracy result.\n"
        "Left: exact binary32 models on retained operands. Right: separately built native frontend.\n"
        "Default and historical results preserved; wider-recording and platform validation remain open.",
        fontsize=9,
    )
    fig.savefig(root / "fused-product.png", dpi=180)
    fig.savefig(root / "fused-product.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    plot(Path(__file__).resolve().parent)
