"""Plot actual completed five-case original-file CPU diagnostics; never execute models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

PANEL_SHA = "e7bec614b0f99404655d49111a8a56ddb0d5e96ad267a8d807d2873a129303ba"
IDS = ("beatport:1059355", "fmak:083351", "fmak:101615", "fmak:088812", "fmak:087373")
STAGES = ("Setup", "Decode", "CQT", "ONNX", "Verification / other")
COLORS = ("#98a8b5", "#e9ac52", "#247c9c", "#6a9d68", "#d7d4cd")


def load(path: Path, evidence: dict[str, str]) -> dict[str, Any]:
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("Oversized measurement artifact")
    raw = path.read_bytes()
    evidence[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def finite(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("Expected finite nonnegative measurement")
    return float(value)


def read_run(root: Path) -> dict[str, Any]:
    evidence: dict[str, str] = {}
    result = load(root / "result.json", evidence)
    config = load(root / "config.json", evidence)
    if result.get("complete") is not True or result.get("failure") is not None:
        raise ValueError("Only completed five-case timing runs can be plotted")
    if (
        result.get("panel_sha256") != PANEL_SHA
        or result["config_sha256"] != evidence[str(root / "config.json")]
    ):
        raise ValueError("Panel/config identity mismatch")
    if (
        [r["id"] for r in config["cases"]] != list(IDS)
        or config["warm_repetitions"] != 3
        or len(result["cases"]) != 5
    ):
        raise ValueError("Fixed membership/repetition policy differs")
    context = config["build_context"]
    host = context["host_description"]
    if not isinstance(host, str) or not host.strip():
        raise ValueError("Hardware description is required")
    lane = config.get("input_lane", "canonical")
    if result.get("input_lane", lane) != lane:
        raise ValueError("Result/config lane mismatch")
    if lane not in {"canonical", "decoder_fidelity"}:
        raise ValueError("Unknown measurement lane")
    panel = load(Path(__file__).with_name("END_TO_END_PANEL.json"), evidence)
    if evidence[str(Path(__file__).with_name("END_TO_END_PANEL.json"))] != PANEL_SHA:
        raise ValueError("Changed frozen panel")
    if [r["id"] for r in panel["cases"]] != list(IDS):
        raise ValueError("Changed panel identities")
    rows = []
    for i, identity in enumerate(IDS):
        worker = load(root / f"case-{i}.json", evidence)
        receipt = load(root / f"case-{i}-resources.json", evidence)
        if (
            receipt != result["cases"][i]
            or receipt.get("complete") is not True
            or receipt.get("remaining_pids") != []
            or receipt.get("failure") is not None
        ):
            raise ValueError("Case supervision is incomplete or inconsistent")
        if (
            worker.get("complete") is not True
            or worker.get("id") != identity
            or worker.get("input_lane", "canonical") != lane
        ):
            raise ValueError("Worker identity/lane incomplete")
        runs = worker["runs"]
        if [r["repetition"] for r in runs] != [0, 1, 2, 3] or [r["kind"] for r in runs] != [
            "first_in_fresh_process",
            "warm_session",
            "warm_session",
            "warm_session",
        ]:
            raise ValueError("All four ordered repetitions required")
        member = panel["cases"][i]
        if config["cases"][i]["audio"]["sha256"] != member["audio_sha256"]:
            raise ValueError("Original audio identity differs")
        if "pcm" in runs[0]:
            samples = runs[0]["pcm"]["samples"]
            if (
                type(samples) is not int
                or samples != member["pcm_samples"]
                or any(r["pcm"] != runs[0]["pcm"] for r in runs)
            ):
                raise ValueError("Decoded input support/identity differs across repetitions")
            sample_basis = "Observed per-run PCM receipt; same frozen sample support"
        else:
            if (
                config["format"] != "drone-original-file-measurement-v1"
                or lane != "canonical"
                or any("pcm" in r for r in runs)
            ):
                raise ValueError("PCM metadata absent outside strict canonical v1")
            samples = member["pcm_samples"]
            sample_basis = "Frozen panel samples admitted by strict canonical v1 source; old receipt omits per-run PCM fields"
        setup = finite(worker["setup"]["wall_seconds"])
        cold = runs[0]
        stages = [finite(cold[name]["wall_seconds"]) for name in ("decode", "cqt", "ort")]
        residual = finite(cold["wall_seconds"]) - sum(stages)
        if residual < -1e-6 or residual + 1e-6 < finite(cold["identity_verification_wall_seconds"]):
            raise ValueError("Inconsistent stage accounting")
        for run in runs[1:]:
            for name in ("decode", "cqt", "ort"):
                finite(run[name]["wall_seconds"])
        rows.append(
            {
                "id": identity,
                "duration_seconds": samples / 22050,
                "duration_basis": sample_basis,
                "cold_stage_seconds": [setup, *stages, max(0.0, residual)],
                "cold_instrumented_seconds": setup + finite(cold["wall_seconds"]),
                "warm_instrumented_seconds": [finite(r["wall_seconds"]) for r in runs[1:]],
                "sampled_tree_peak_mib": finite(receipt["sampled_tree_peak_rss_bytes"]) / 1024**2,
                "worker_os_peak_mib": finite(worker["self_peak_rss_bytes"]) / 1024**2,
                "sample_interval_seconds": finite(receipt["sample_interval_seconds"]),
                "numerical_pass": worker["numerical_pass"],
                "unchanged_key": worker["unchanged_key"],
                "historical_reference_qualified": worker.get(
                    "historical_reference_qualified", worker["benchmark_qualified"]
                ),
                "canonical_pcm_matches": worker.get("canonical_pcm_matches"),
            }
        )
    return {
        "root": str(root),
        "host": host,
        "platform": result["platform"],
        "build_context": context,
        "input_lane": lane,
        "historical_reference_qualified": result.get(
            "historical_reference_qualified", result["benchmark_qualified"]
        ),
        "canonical_pcm_matches": result.get("canonical_pcm_matches"),
        "rows": rows,
        "evidence_sha256": evidence,
    }


def render(measurements: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.hashsalt": "drone-original-file-timing-v1",
        }
    )
    figure, axes = plt.subplots(
        3,
        len(measurements),
        figsize=(9 * len(measurements), 12),
        squeeze=False,
        sharey="row",
    )
    for column, measurement in enumerate(measurements):
        rows = measurement["rows"]
        x = list(range(5))
        labels = [f"{row['duration_seconds'] / 60:.2f} min\n{row['id']}" for row in rows]
        top, middle, bottom = (axes[r, column] for r in range(3))
        base = [0.0] * 5
        for stage, label, color in zip(range(5), STAGES, COLORS, strict=True):
            values = [row["cold_stage_seconds"][stage] for row in rows]
            top.bar(x, values, bottom=base, color=color, width=0.65, label=label)
            base = [a + b for a, b in zip(base, values, strict=True)]
        for i, total in enumerate(base):
            top.annotate(
                f"{total:.2f}s",
                (i, total),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        top.set_ylabel("Fresh process: setup + first pass (seconds)")
        top.margins(y=0.17)
        if "Apple M5" in measurement["host"] and "25769803776 bytes RAM" in measurement["host"]:
            title = "Apple M5 · 24 GiB · macOS"
            policy = "Affinity not enforced"
        elif "Threadripper PRO 9955WX" in measurement["host"] and "CPU14" in measurement["host"]:
            title = "Threadripper PRO 9955WX · Linux · one logical CPU"
            policy = "CPU14, shared server"
        else:
            raise ValueError("A verified concise hardware title is required for this host")
        strict = sum(row["numerical_pass"] is True for row in rows)
        keys = sum(row["unchanged_key"] is True for row in rows)
        top.set_title(
            f"{title}\n{policy}\nStrict logit checks: {strict}/5 · same reference keys: {keys}/5",
            fontsize=11,
            pad=12,
        )
        for repetition, marker, color, offset in zip(
            range(3),
            ("o", "s", "^"),
            ("#247c9c", "#a06545", "#6a5891"),
            (-0.15, 0, 0.15),
            strict=True,
        ):
            middle.scatter(
                [i + offset for i in x],
                [row["warm_instrumented_seconds"][repetition] for row in rows],
                label=f"Warm {repetition + 1}",
                marker=marker,
                color=color,
                s=42,
            )
        middle.set_ylabel("Warm session: each complete pass (seconds)")
        middle.set_ylim(bottom=0)
        middle.legend(loc="upper left", frameon=False, ncol=3)
        bottom.bar(
            [i - 0.17 for i in x],
            [r["sampled_tree_peak_mib"] for r in rows],
            width=0.32,
            color="#247c9c",
            label="Sampled process tree",
        )
        bottom.bar(
            [i + 0.17 for i in x],
            [r["worker_os_peak_mib"] for r in rows],
            width=0.32,
            color="#9b9e9f",
            label="Worker OS high-water",
        )
        bottom.set_ylabel("Memory across all four passes (MiB)")
        bottom.legend(loc="upper left", frameon=False)
        bottom.margins(y=0.2)
        for axis in (top, middle, bottom):
            axis.set_xticks(x, labels)
            axis.tick_params(axis="x", labelsize=8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", alpha=0.16)
            axis.set_axisbelow(True)
    # Fix limits from every host after plotting: shared axes can otherwise keep
    # an earlier host's manually bounded warm scale and clip later observations.
    maxima = [
        max(row["cold_instrumented_seconds"] for data in measurements for row in data["rows"]),
        max(
            value
            for data in measurements
            for row in data["rows"]
            for value in row["warm_instrumented_seconds"]
        ),
        max(
            max(row["sampled_tree_peak_mib"], row["worker_os_peak_mib"])
            for data in measurements
            for row in data["rows"]
        ),
    ]
    for row, maximum in enumerate(maxima):
        axes[row, 0].set_ylim(0, max(maximum * 1.2, 1e-6))
    figure.suptitle("Original-file CPU timing · five fixed recordings", fontsize=17, y=0.99)
    figure.legend(
        handles=[
            Patch(color=color, label=name) for color, name in zip(COLORS, STAGES, strict=True)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.963),
        ncol=5,
        frameon=False,
    )
    figure.text(
        0.5,
        0.012,
        "Fresh = new session; disk/CPU caches are not flushed. All three warm repetitions are shown.\nBars start after process startup and artifact admission; they include setup and timed PCM verification.\nLater hashes/reporting are excluded. Supervisor totals include polling/cleanup and are not plotted.\nSampled tree RSS and worker OS peak have different scopes; sampling can miss the true peak.\nHost/CPU policies differ: these are not controlled speedups or evidence of a memory optimization.\nHardware-specific diagnostics only: no accuracy result or ordinary-PC performance guarantee.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.09 / len(measurements),
        right=0.985,
        top=0.85,
        bottom=0.14,
        hspace=0.43,
        wspace=0.22,
    )
    output.mkdir(parents=True, exist_ok=False)
    figure.savefig(output / "original-file-timing.png", dpi=180)
    figure.savefig(output / "original-file-timing.svg", metadata={"Date": None})
    plt.close(figure)
    (output / "plotted-measurements.json").write_text(
        json.dumps(
            {
                "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "measurements": measurements,
                "accuracy_evaluated": False,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def read_summary(path: Path, expected: str) -> list[dict[str, Any]]:
    evidence: dict[str, str] = {}
    summary = load(path, evidence)
    if (
        evidence[str(path)] != expected
        or summary.get("schema") != "drone-original-file-public-timing-v1"
        or summary.get("panel_sha256") != PANEL_SHA
    ):
        raise ValueError("Public summary identity/schema mismatch")
    measurements = summary["measurements"]
    if not 1 <= len(measurements) <= 2:
        raise ValueError("One or two complete measured hosts required")
    for measurement in measurements:
        if [row["id"] for row in measurement["rows"]] != list(IDS):
            raise ValueError("Public summary membership mismatch")
        for row in measurement["rows"]:
            if len(row["cold_stage_seconds"]) != 5 or len(row["warm_instrumented_seconds"]) != 3:
                raise ValueError("Public summary repetition/stage count differs")
            for value in [
                *row["cold_stage_seconds"],
                *row["warm_instrumented_seconds"],
                row["duration_seconds"],
                row["sampled_tree_peak_mib"],
                row["worker_os_peak_mib"],
            ]:
                finite(value)
            if not math.isclose(
                sum(row["cold_stage_seconds"]),
                row["cold_instrumented_seconds"],
                rel_tol=0,
                abs_tol=1e-9,
            ):
                raise ValueError("Public stage total differs")
        measurement["public_summary_sha256"] = expected
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, action="append")
    source.add_argument("--summary", type=Path)
    parser.add_argument("--summary-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.summary:
        if not args.summary_sha256:
            parser.error("Public summary requires --summary-sha256")
        data = read_summary(args.summary, args.summary_sha256)
    else:
        if not 1 <= len(args.run) <= 2:
            parser.error("Supply one or two completed measurement directories")
        data = [read_run(path) for path in args.run]
    if args.validate_only:
        print(json.dumps({"validated_runs": len(data), "figures_written": False}))
    else:
        render(data, args.output)


if __name__ == "__main__":
    main()
