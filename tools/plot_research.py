"""Regenerate research PNG/SVG figures solely from checked-in measurement JSON."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
STUDIES = EXP / "studies"
BLUE, ORANGE, RED = "#2563eb", "#c2410c", "#b91c1c"
INPUTS: dict[str, str] = {}


def read(relative: str) -> dict[str, Any]:
    path = EXP / relative
    INPUTS[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_bytes())


def canvas(title: str, columns: int, note: str) -> tuple[Figure, list[Axes]]:
    figure = plt.figure(figsize=(12, 5.4), facecolor="white")
    axes = [figure.add_subplot(1, columns, i + 1) for i in range(columns)]
    figure.suptitle(title, x=0.07, y=0.96, ha="left", fontsize=17, fontweight="bold")
    figure.subplots_adjust(left=0.075, right=0.96, bottom=0.24, top=0.79, wspace=0.32)
    figure.text(0.075, 0.065, note, fontsize=9, color="#475569", linespacing=1.6)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    return figure, axes


def save(figure: Figure, study: str, name: str) -> None:
    output = STUDIES / study / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{name}.png", dpi=170, metadata={"Software": "Drone research"})
    svg = output / f"{name}.svg"
    figure.savefig(svg, metadata={"Date": None, "Creator": "Drone research"})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(figure)


def memory_curve() -> None:
    report = read("studies/k10-chunked/data/memory-curve.json")
    if report["status"] != "complete" or not report["paired_parity_passed"]:
        raise ValueError("Incomplete or non-parity memory experiment")
    figure, axes = canvas(
        "K10 memory growth with track length",
        2,
        "20 public validation tracks · one CPU core · fresh process for each track/runtime · cached features only\n"
        "RSS includes Python, ONNX Runtime and input features. Duration is a CQT-frame proxy. No decoding/CQT or native app measurement.",
    )
    for mode, label, color in (("full", "Full track", BLUE), ("chunked", "Chunked", ORANGE)):
        rows = sorted(
            (r for r in report["records"] if r["mode"] == mode), key=lambda r: r["frames"]
        )
        minutes = [r["duration_proxy_seconds"] / 60 for r in rows]
        axes[0].plot(
            minutes, [r["process_peak_mib"] for r in rows], "o-", color=color, label=label, ms=4
        )
        axes[1].plot(
            minutes,
            [r["median_seconds"] * 1000 for r in rows],
            "o-",
            color=color,
            label=label,
            ms=4,
        )
    for axis, title, unit in zip(
        axes,
        ("Peak process memory", "Median inference latency"),
        ("MiB", "Milliseconds"),
        strict=True,
    ):
        axis.set(title=title, xlabel="Track duration (minutes)", ylabel=unit, ylim=(0, None))
        axis.legend(frameon=False)
    save(figure, "k10-chunked", "k10-memory-length")


def pipeline() -> None:
    raw = read("studies/k10-cpu/data/pipeline.json")
    figure, axes = canvas(
        "K10 CPU baseline: frontend cost dominates inference",
        1,
        "20 public validation recordings · server CPU, one core · median across cases\n"
        "Each phase has a different cache condition. Separate medians are not additive. No native Euphoria or target-laptop measurement.",
    )
    times = [
        float(np.median([r["frontend_seconds"] for r in raw["records"]])) * 1000,
        float(np.median([r["pytorch_full_forward"]["p50_seconds"] for r in raw["records"]])) * 1000,
        float(np.median([r["onnx_full_forward"]["p50_seconds"] for r in raw["records"]])) * 1000,
        float(np.median([r["warm_feature_cache_plus_onnx"]["p50_seconds"] for r in raw["records"]]))
        * 1000,
    ]
    axes[0].bar(
        [
            "Decode + CQT\n(empty feature cache)",
            "Torch inference\n(features loaded)",
            "ONNX inference\n(features loaded)",
            "Feature cache + ONNX\n(warm cache)",
        ],
        times,
        color=["#64748b", BLUE, ORANGE, "#7c3aed"],
    )
    axes[0].set(
        ylabel="Milliseconds",
        ylim=(0, max(times) * 1.23),
    )
    for i, value in enumerate(times):
        axes[0].text(i, value + 10, f"{value:.1f} ms", ha="center", fontweight="bold")
    save(figure, "k10-cpu", "k10-pipeline-cost")


def chunk_summary() -> None:
    base = "studies/k10-chunked/data/v2/"
    full, chunk = read(base + "full-benchmark.json"), read(base + "chunked-benchmark.json")
    figure, axes = canvas(
        "K10 chunking: less memory with similar latency",
        2,
        "Same 20 public tracks · one CPU core · separate fresh full/chunked processes · 20 repetitions per track\n"
        "Cached-feature inference only. Median paired latency change +1.91%; all 376 validation keys agree. Native application cost is unmeasured.",
    )
    latencies = [
        full["median_track_median_seconds"] * 1000,
        chunk["median_track_median_seconds"] * 1000,
    ]
    axes[0].bar(["Full track", "Chunked"], latencies, color=[BLUE, ORANGE], width=0.55)
    axes[0].set(title="Median per-track median latency", ylabel="Milliseconds", ylim=(0, 46))
    for i, value in enumerate(latencies):
        axes[0].text(i, value + 1, f"{value:.2f} ms", ha="center", fontweight="bold")
    peaks = [full["peak_process_rss_kib"] / 1024, chunk["peak_process_rss_kib"] / 1024]
    axes[1].bar(["Full track", "Chunked"], peaks, color=[BLUE, ORANGE], width=0.55)
    axes[1].set(
        title="Paired repeated-run memory",
        ylabel="Peak process RSS (MiB)",
        ylim=(0, max(peaks) * 1.22),
    )
    for i, value in enumerate(peaks):
        axes[1].text(i, value + 20, f"{value:.0f} MiB", ha="center", fontweight="bold")
    save(figure, "k10-chunked", "k10-chunked-summary")


def multidomain() -> None:
    report = read("studies/k10-multidomain/data/summary.json")
    figure, axes = canvas(
        "More fitting data helped, but missed the confirmation gate",
        2,
        "K10 development validation · symmetric-fifth MIREX · 3 training seeds · identical recipes except fitting data\n"
        "FMAK mean gain +1.60 points, below the predeclared +2.00-point gate. These seeds do not measure uncertainty on unseen recordings.",
    )
    for axis, domain, title in zip(
        axes,
        ("beatport", "fmak"),
        ("Beatport validation (126 tracks)", "FMAK validation (250 tracks)"),
        strict=True,
    ):
        summary = report["mirex"]
        means = [summary[arm][domain]["mean"] * 100 for arm in ("control", "joint")]
        deviations = [summary[arm][domain]["sample_std"] * 100 for arm in ("control", "joint")]
        axis.errorbar([0, 1], means, yerr=deviations, fmt="o-", color=BLUE, capsize=6, linewidth=2)
        axis.set(
            xticks=[0, 1],
            xticklabels=["Beatport-only fit", "Joint fit"],
            ylabel="MIREX (%)",
            title=title,
            xlim=(-0.4, 1.4),
            ylim=(68, 80),
        )
        for i, value in enumerate(means):
            axis.annotate(
                f"{value:.2f}%",
                (i, value + deviations[i]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontweight="bold",
            )
        axis.text(
            0.5,
            0.04,
            "Error bars: sample SD across seeds",
            transform=axis.transAxes,
            ha="center",
            fontsize=9,
            color="#475569",
        )
    save(figure, "k10-multidomain", "k10-multidomain")


def ema() -> None:
    report = read("studies/k10-ema/data/ema-result.json")
    if not report["raw_reproduced"] or report["method_gate"]["qualifies_for_confirmation"]:
        raise ValueError("Figure expects the exactly reproduced, rejected EMA pilot")
    figure, axes = canvas(
        "EMA pilot: both recalibration and averaging hurt this recipe",
        2,
        "Seed 13 only · fixed checkpoint policy · identical fitting-only BN recalibration · no release-weight change\n"
        "EMA versus RAW + BN: FMAK MIREX −1.84 points, exact −2.00 points. Pilot failed; confirmation seeds were not run.",
    )
    names = ("raw", "raw_bn", "ema_bn")
    for axis, domain, title in zip(
        axes, ("beatport", "fmak"), ("Beatport validation", "FMAK validation"), strict=True
    ):
        for metric, label, color in (("mirex", "MIREX", BLUE), ("exact", "Exact accuracy", ORANGE)):
            values = [report["outputs"][n]["metrics"][domain][metric] * 100 for n in names]
            axis.plot(range(3), values, "o-", color=color, label=label, linewidth=2)
            for i, value in enumerate(values):
                axis.annotate(
                    f"{value:.2f}",
                    (i, value),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )
        axis.set(
            xticks=range(3),
            xticklabels=["RAW", "RAW + BN", "EMA + BN"],
            title=title,
            ylabel="Score (%)",
            ylim=(59, 81),
            xlim=(-0.25, 2.25),
        )
        axis.legend(frameon=False, loc="lower left")
    save(figure, "k10-ema", "k10-ema")


def paired_result(study: str, title: str, candidate_name: str) -> None:
    report = read(f"studies/{study}/data/candidate-result.json")
    if report["status"] != "complete" or not report["control_reproduced"]:
        raise ValueError("Paired result needs a completed candidate and verified control")
    figure, axes = canvas(
        title,
        2,
        "Seed 13 · same 2,206 fitting / 376 validation identities · fixed selection policy · symmetric-fifth MIREX\n"
        "Reused development data, not independent test evidence. Exact accuracy is a separate gate; no confirmation after a failed pilot.",
    )
    for axis, domain, label in zip(
        axes, ("beatport", "fmak"), ("Beatport validation", "FMAK validation"), strict=True
    ):
        for metric, metric_label, color in (
            ("mirex", "MIREX", BLUE),
            ("exact", "Exact accuracy", ORANGE),
        ):
            values = [report[key][domain][metric] * 100 for key in ("control_metrics", "metrics")]
            axis.plot([0, 1], values, "o-", color=color, label=metric_label, linewidth=2)
            for i, value in enumerate(values):
                axis.annotate(
                    f"{value:.2f}",
                    (i, value),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                )
        axis.set(
            title=label,
            ylabel="Score (%)",
            xticks=[0, 1],
            xticklabels=["Verified RAW control", candidate_name],
            xlim=(-0.3, 1.3),
            ylim=(60, 81),
        )
        axis.legend(frameon=False, loc="lower left")
    save(figure, study, "accuracy")


def capacity_history() -> None:
    report = read("studies/k10-capacity/data/histories.json")
    figure, axes = canvas(
        "K10 capacity: observed fitting and selection traces",
        2,
        "Hard cross entropy in both arms · same data/augmentation/optimizer recipe · seed 13 · selected epochs marked\n"
        "These traces describe one width comparison. They do not establish a general capacity or optimization law.",
    )
    for name, label, color in (
        ("control", "K10 width 1", BLUE),
        ("candidate", "K10 width 2", ORANGE),
    ):
        run = report[name]
        epochs = [r["epoch"] for r in run["history"]]
        axes[0].plot(epochs, [r["loss"] for r in run["history"]], color=color, label=label)
        axes[1].plot(epochs, [r["mirex"] * 100 for r in run["history"]], color=color, label=label)
        best = next(r for r in run["history"] if r["epoch"] == run["best_epoch"])
        axes[1].plot(best["epoch"], best["mirex"] * 100, "o", color=color)
        axes[1].annotate(
            f"Epoch {best['epoch']}",
            (best["epoch"], best["mirex"] * 100),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color=color,
        )
    axes[0].set(
        title="Fitting loss", xlabel="Epoch", ylabel="Mean hard cross entropy", ylim=(0, None)
    )
    axes[1].set(
        title="Combined validation criterion",
        xlabel="Epoch",
        ylabel="Symmetric-fifth MIREX (%)",
        ylim=(0, 83),
    )
    for axis in axes:
        axis.legend(frameon=False, loc="lower right")
    save(figure, "k10-capacity", "training-history")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlepad": 16,
            "svg.hashsalt": "drone-research-v1",
        }
    )
    memory_curve()
    pipeline()
    chunk_summary()
    multidomain()
    ema()
    paired_result(
        "k10-structured-targets",
        "Structured targets: fixed pilot missed the accuracy gate",
        "Structured targets",
    )
    paired_result("k10-capacity", "Wider K10: no gain under the fixed recipe", "K10 width 2")
    capacity_history()
    INPUTS["_generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    INPUTS["_uv_lock_sha256"] = hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    INPUTS["_matplotlib_version"] = matplotlib.__version__
    (STUDIES / "figure-inputs.json").write_text(json.dumps(INPUTS, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
