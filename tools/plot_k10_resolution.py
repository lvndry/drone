"""Plot the preserved synthetic resource screen, without running either model."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from tools.plot_research import BLUE, INPUTS, ORANGE, ROOT, canvas, plt, read, save

BASE = "studies/k10-capacity/"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    provenance = read(BASE + "data/resolution-shape-audit-provenance.json")
    report = read(BASE + "data/resolution-shape-audit.json")
    source_path = ROOT / "experiments" / BASE / "data/resolution-shape-audit.py.txt.gz"
    source = source_path.read_bytes()
    if (
        INPUTS[BASE + "data/resolution-shape-audit.json"] != provenance["result_sha256"]
        or sha256(source) != provenance["source_gzip_sha256"]
        or sha256(gzip.decompress(source)) != provenance["source_sha256"]
    ):
        raise ValueError("Historical resource-screen source/evidence hash mismatch")
    INPUTS[BASE + "data/resolution-shape-audit.py.txt.gz"] = sha256(source)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "k10-resolution-v1"}
    )
    figure, axes = canvas(
        "Same parameter budget, different frequency/channel allocation",
        2,
        "Synthetic shape audit only · MACs count convolution + linear multiply-accumulates; exclude BN, ELU, pooling and reductions.\n"
        "Fewer MACs are not a measured speedup. No accuracy, ordinary-PC latency or total-memory advantage established.",
    )
    names = ("baseline", "candidate")
    labels = ["K10\n16/32/64", "Semitone grid\n16/32/32"]
    totals = [report["models"][name]["params"] for name in names]
    heads = [report["models"][name]["head_params"] for name in names]
    backbone = [total - head for total, head in zip(totals, heads, strict=True)]
    axes[0].bar(labels, backbone, color=BLUE, label="Convolution + BN")
    axes[0].bar(labels, heads, bottom=backbone, color=ORANGE, label="Linear head")
    for index, (total, body, head) in enumerate(zip(totals, backbone, heads, strict=True)):
        axes[0].text(index, body / 2, f"{body:,}", ha="center", va="center", color="white")
        axes[0].text(index, body + head / 2, f"{head:,}", ha="center", va="center", color="white")
        axes[0].text(index, total + 3000, f"{total:,}", ha="center", fontweight="bold")
    axes[0].set(title="Trainable parameter allocation", ylabel="Parameters", ylim=(0, 137000))
    axes[0].legend(frameon=False, loc="upper center", ncol=2, fontsize=9)
    cases = report["cases"]
    positions = list(range(len(cases)))
    for offset, name, color, label in (
        (-0.19, "baseline", BLUE, "K10"),
        (0.19, "candidate", ORANGE, "Semitone grid"),
    ):
        values = [row["conv_and_linear_MACs"][name] / 1e9 for row in cases]
        bars = axes[1].bar(
            [p + offset for p in positions], values, width=0.36, color=color, label=label
        )
        axes[1].bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, fontsize=8)
    axes[1].set(
        title="Analytical operations from actual shapes",
        ylabel="Billion MACs per batch-one forward",
        xticks=positions,
        xticklabels=[f"T = {row['T']:,}" for row in cases],
        ylim=(0, 3.6),
    )
    axes[1].legend(frameon=False, loc="upper left", fontsize=9)
    save(figure, "k10-capacity", "resolution-screen")
    receipt = dict(INPUTS)
    receipt["_generator_sha256"] = sha256(Path(__file__).read_bytes())
    receipt["_shared_plot_helpers_sha256"] = sha256((ROOT / "tools/plot_research.py").read_bytes())
    import matplotlib

    receipt["_matplotlib_version"] = matplotlib.__version__
    output = ROOT / "experiments" / BASE / "figures"
    for suffix in ("png", "svg"):
        receipt[f"_resolution_screen_{suffix}_sha256"] = sha256(
            (output / f"resolution-screen.{suffix}").read_bytes()
        )
    (output / "resolution-screen-inputs.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
