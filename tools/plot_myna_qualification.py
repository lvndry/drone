"""Render the recorded synthetic qualification results, without running audio models."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def derive_tables(root: Path) -> None:
    """Regenerate display tables from preserved receipts, without changing measurements."""
    rows: list[dict[str, Any]] = []
    for filename, method in [
        ("phase-vocoder-diagnostic.json", "Phase vocoder + polyphase"),
        ("rubberband-r2-diagnostic.json", "FFmpeg RubberBand R2"),
        ("rubberband-r3-diagnostic.json", "Offline RubberBand R3"),
    ]:
        receipt = json.loads((root / "data" / filename).read_text())
        for row in receipt["rows"]:
            rows.append(
                {
                    "method": method,
                    "semitones": row["semitones"],
                    "tone_440_rms_ratio": row["rms_ratio"][3],
                    "max_pure_tone_error_cents": max(
                        abs(tone["error_cents"]) for tone in row["tones"][:4]
                    ),
                    "max_chord_peak_error_cents": max(
                        abs(tone["error_cents"]) for tone in row["tones"][4:]
                    ),
                    "original_amplitude_gate": row["passes_original_amplitude_gate"],
                    "original_pitch_gate": row["passes_original_pitch_gate"],
                }
            )
    write_csv(root / "data/pitch_diagnostics.csv", rows)
    energy = json.loads((root / "data/rubberband-r3-energy-v2.json").read_text())
    rows = [
        {
            "chord": row["mode"],
            "semitones": row["semitones"],
            "r3_union_fraction": row["r3"]["union_total_power_fraction"],
            "r3_min_component_fraction": min(row["r3"]["component_total_power_fractions"]),
            "correct_union_fraction": row["analytic_correct"]["union_total_power_fraction"],
            "correct_min_component_fraction": min(
                row["analytic_correct"]["component_total_power_fractions"]
            ),
            "wrong_union_fraction": row["wrong_semitone_control"]["union_total_power_fraction"],
            "deleted_min_component_fraction": min(
                row["missing_middle_control"]["component_total_power_fractions"]
            ),
            "controls_pass": row["controls_pass"],
            "passes_v2": row["passes_v2"],
        }
        for row in energy["rows"]
    ]
    write_csv(root / "data/pitch_energy_v2.csv", rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "experiments/studies/myna-key-pilot"
    derive_tables(root)
    with (root / "data/pitch_diagnostics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    methods = list(dict.fromkeys(row["method"] for row in rows))
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    metrics = [
        ("tone_440_rms_ratio", "440 Hz tone amplitude", "Output / input RMS"),
        ("max_pure_tone_error_cents", "Worst individual-tone pitch error", "Absolute cents"),
        ("max_chord_peak_error_cents", "Worst local bass-chord peak error", "Absolute cents"),
    ]
    colors = ["#a54a28", "#46749a", "#30795c"]
    for axis, (metric, title, label) in zip(axes, metrics, strict=True):
        for method, color in zip(methods, colors, strict=True):
            selected = [row for row in rows if row["method"] == method]
            axis.plot(
                [int(row["semitones"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o",
                markersize=3,
                linewidth=1.4,
                color=color,
                label=method,
            )
        axis.set(title=title, xlabel="Requested shift (semitones)", ylabel=label)
        axis.set_xticks([-6, -3, 0, 3, 6])
        axis.grid(alpha=0.2)
        if metric != "tone_440_rms_ratio":
            axis.axhline(5, color="#777777", linestyle="--", linewidth=0.8)
        else:
            axis.axhline(0.5, color="#777777", linestyle="--", linewidth=0.8)
            axis.set_ylim(0, 1.1)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    figure.suptitle(
        "Pitch augmentation diagnostics — synthetic fixtures, not key-detection accuracy"
    )
    figure.savefig(root / "figures/pitch_diagnostics.png", dpi=180)
    plt.close(figure)
    with (root / "data/pitch_energy_v2.csv").open() as stream:
        energy_rows = list(csv.DictReader(stream))
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for axis, metric, title, limit in zip(
        axes,
        ["r3_union_fraction", "r3_min_component_fraction"],
        ["Power inside all expected semitone bands", "Weakest expected component band"],
        [0.90, 0.20],
        strict=True,
    ):
        for chord, color in [("A_major", "#46749a"), ("A_minor", "#a54a28")]:
            selected = [row for row in energy_rows if row["chord"] == chord]
            axis.plot(
                [int(row["semitones"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o",
                markersize=3,
                color=color,
                label=chord.replace("_", " "),
            )
        axis.axhline(limit, color="#777777", linestyle="--", label="Prospective v2 bound")
        axis.set(
            title=title,
            xlabel="Requested shift (semitones)",
            ylabel="Fraction of total spectral power",
        )
        axis.set_ylim(0, 1.05 if metric == "r3_union_fraction" else 0.4)
        axis.set_xticks([-6, -3, 0, 3, 6])
        axis.grid(alpha=0.2)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    figure.suptitle("Unchanged offline R3 fails the prospective energy check")
    figure.savefig(root / "figures/pitch_energy_v2.png", dpi=180)
    plt.close(figure)

    runtime = json.loads((root / "data/runtime-qualification.json").read_text())
    resources = json.loads((root / "data/runtime-resources.json").read_text())
    views = [
        item["result"]
        for item in runtime["phases"]
        if item["phase"] == "pitch_and_composed_extraction"
    ]
    groups = [[row for row in views if row["id"] == "beatport:100066"]]
    groups += [[row] for row in views if row["id"] == "fmak:082660"]
    labels = ["120 s × 13 views"] + [
        f"35.9 min, {rows[0]['semitones']:+d} st" for rows in groups[1:]
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    offset = [0.0] * len(groups)
    for key, label, color in [
        ("pitch_seconds", "Playback rate", "#a54a28"),
        ("blocked_frontend_seconds", "CPU mel frontend", "#46749a"),
        ("encoder_batch16_seconds", "GPU encoder", "#30795c"),
    ]:
        values = [sum(row[key] for row in rows) for rows in groups]
        axes[0].barh(labels, values, left=offset, label=label, color=color)
        offset = [a + b for a, b in zip(offset, values, strict=True)]
    axes[0].set(xlabel="Seconds (selected stages only)", title="Feature extraction dominates")
    axes[0].legend(loc="lower right", fontsize=8)
    memory = [
        resources["sampled_peak_host_rss_bytes"],
        runtime["peak_host_rss_bytes"],
        runtime["peak_gpu_allocated_bytes"],
        runtime["peak_gpu_reserved_bytes"],
        resources["sampled_peak_worker_nvml_bytes"],
    ]
    axes[1].barh(
        [
            "Host: sampled",
            "Host: high-water",
            "GPU: allocated",
            "GPU: reserved",
            "GPU: process sampled",
        ],
        [value / 1024**3 for value in memory],
        color=["#46749a"] * 2 + ["#30795c"] * 3,
    )
    axes[1].set(xlabel="GiB", title="Memory counters differ")
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle("Server qualification: one CPU thread + RTX 5090; no key-accuracy claim")
    figure.savefig(root / "figures/runtime_qualification.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
