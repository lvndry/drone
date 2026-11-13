# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.11.2"]
# ///
"""Plot retained arithmetic explanations and fixed-case candidate outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from summarize_fused_product import summarize, summarize_goldens, summarize_public


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
    svg = root / "fused-product.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def plot_goldens(root: Path) -> None:
    result = summarize_goldens(root)
    rows = result["cases"]
    plt.rcParams["svg.hashsalt"] = "cqt-fused-goldens-v1"
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 6.3), layout="constrained")
    for index, row in enumerate(rows):
        count = row["original_differences"]
        axes[0].barh(index, count, color="#a4582c", height=0.6)
        axes[0].plot(0, index, "o", color="#277f67", clip_on=False)
        axes[0].text(count + 4, index, str(count), va="center", fontsize=9)
        axes[1].barh(index, row["decoded_seconds"], color="#336b8e", height=0.6)
        axes[1].text(
            row["decoded_seconds"] * 1.12,
            index,
            f"{row['decoded_seconds']:,.2f} s",
            va="center",
            fontsize=9,
        )
    axes[0].set_yticks(range(9), [row["id"] for row in rows])
    axes[0].set_xlim(0, 250)
    axes[0].set_title("All original controls preserved; candidates exact", fontsize=11, pad=12)
    axes[0].set_xlabel("Final-core values differing from archived Mac Python")
    axes[0].plot([], [], color="#a4582c", linewidth=8, label="Original separate-product control")
    axes[0].plot([], [], "o", color="#277f67", label="Fused candidate: zero in every case")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    axes[1].set_yticks(range(9), [""] * 9)
    axes[1].set_xscale("log")
    axes[1].set_xlim(1, 6000)
    axes[1].set_title("Six synthetic signals and three recordings", fontsize=11, pad=12)
    axes[1].set_xlabel("Decoded duration in seconds (log scale)")
    for axis in axes:
        axis.set_ylim(10.2, -0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("The same product rule passes all nine retained golden cases", fontsize=14)
    fig.supxlabel(
        "303 original differences → 0 candidate differences across 2,048,256 final-core values.\n"
        "The longest recording is 28 min 2 s; genre and BPM coverage are not established here.\n"
        "Frontend-only numerical validation on one Mac runtime, not key accuracy or a latency benchmark.",
        fontsize=9,
    )
    fig.savefig(root / "fused-goldens.png", dpi=180)
    svg = root / "fused-goldens.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def plot_public(root: Path) -> None:
    result = summarize_public(root)
    totals = result["totals"]
    plt.rcParams["svg.hashsalt"] = "cqt-fused-public376-v1"
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    values = [totals["original_mac_exact"], totals["candidate_mac_exact"]]
    axes[0].bar(
        ["Original native", "Fused candidate"], values, color=["#a4582c", "#277f67"], width=0.55
    )
    for index, value in enumerate(values):
        axes[0].text(index, value + 8, f"{value} / 376", ha="center", weight="bold", fontsize=13)
    axes[0].set_ylim(0, 430)
    axes[0].set_ylabel("Entire feature arrays with exact Mac reference bytes")
    axes[0].set_title("Numerical identity: all original controls preserved", fontsize=11, pad=12)
    bands = ["under_2_min", "2_to_under_5_min", "5_to_under_10_min", "10_min_or_longer"]
    # Duration bins were fixed before candidate execution; no genre inference.
    rows = result["cases"]
    counts = [
        [sum(r["source"] == source and r["duration_band"] == band for r in rows) for band in bands]
        for source in ["beatport", "fmak"]
    ]
    axes[1].bar(range(4), counts[0], label="Beatport (126)", color="#336b8e")
    axes[1].bar(range(4), counts[1], bottom=counts[0], label="FMAK (250)", color="#a58b47")
    for index, (a, b) in enumerate(zip(*counts, strict=True)):
        axes[1].text(index, a + b + 5, str(a + b), ha="center", fontsize=11)
    axes[1].set_xticks(range(4), ["<2 min", "2–<5 min", "5–<10 min", "≥10 min"])
    axes[1].set_ylim(0, 350)
    axes[1].set_ylabel("Recordings (all pass the fixed Mac reference)")
    axes[1].set_title("Decoded length coverage: 40 seconds to 28 minutes", fontsize=11, pad=12)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fused products reproduce all 376 archived Mac feature cores", fontsize=15)
    fig.supxlabel(
        "82,562,688 values checked; primary Mac identity 376/376, separate Linux identity 0/376.\n"
        "Feature identity is not classifier accuracy: the original already predicted the same key on 376/376.\n"
        "Genre, BPM and full-song provenance unknown here; no new model inference or latency benchmark.",
        fontsize=9,
    )
    fig.savefig(root / "fused-376.png", dpi=180)
    svg = root / "fused-376.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldens", action="store_true")
    parser.add_argument("--public376", action="store_true")
    args = parser.parse_args()
    function = plot_public if args.public376 else plot_goldens if args.goldens else plot
    function(Path(__file__).resolve().parent)
