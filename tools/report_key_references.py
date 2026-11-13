"""Reproduce reference-benchmark metadata and inline figures; never runs detectors."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/studies/key-sota-4way"
DATA = STUDY / "data"
METRICS = ("n", "n_total", "n_valid", "n_error", "n_abstain", "counts", "exact", "mirex")
DETECTORS = {"drone": "drone-v1", "madmom": "madmom", "libkeyfinder": "libkeyfinder-2.2.8"}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def seal(data: Path, kf_status: str) -> dict[str, Any]:
    """Bind published artifacts, independently from their scientific interpretation."""
    if kf_status not in {"historical_invalid", "corrected"}:
        raise ValueError("Unknown libkeyfinder result status")
    return {
        "schema": "key-reference-artifact-index-v1",
        "libkeyfinder_status": kf_status,
        "files": {
            name: {
                "sha256": sha((data / name).read_bytes()),
                "bytes": len((data / name).read_bytes()),
            }
            for name in ("bench.csv", "bench.json", "manifest.json")
        },
        "historical_archive_index_sha256": sha(
            (data / "historical/archive-index.json").read_bytes()
        ),
    }


def checked(
    data: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    index = json.loads((data / "artifact-index.json").read_bytes())
    values = {}
    for name in ("bench.csv", "bench.json", "manifest.json"):
        raw = (data / name).read_bytes()
        binding = index["files"][name]
        if sha(raw) != binding["sha256"] or len(raw) != binding["bytes"]:
            raise ValueError("Published artifact changed: " + name)
        values[name] = raw
    historical = data / "historical/archive-index.json"
    if sha(historical.read_bytes()) != index["historical_archive_index_sha256"]:
        raise ValueError("Historical archive index changed")
    for name, binding in json.loads(historical.read_bytes())["files"].items():
        raw = (historical.parent / name).read_bytes()
        if sha(raw) != binding["sha256"]:
            raise ValueError("Historical evidence changed: " + name)
        decoded = gzip.decompress(raw)
        if (sha(raw), sha(decoded), len(decoded)) != (
            binding["sha256"],
            binding["decoded_sha256"],
            binding["decoded_bytes"],
        ):
            raise ValueError("Historical evidence changed: " + name)
    rows: list[dict[str, Any]] = []
    for item in csv.DictReader(io.StringIO(values["bench.csv"].decode())):
        row: dict[str, Any] = dict(item)
        row["ref"] = int(item["ref"])
        row["duration"] = float(item["duration"])
        for name in DETECTORS:
            value = item.get(f"{name}_pred", "")
            row[f"{name}_pred"] = None if value in {"", "None", "null"} else int(value)
        rows.append(row)
    return index, rows, json.loads(values["bench.json"]), json.loads(values["manifest.json"])


def analyze(data: Path = DATA) -> dict[str, Any]:
    # Share the exact harness aggregation, including fixed-denominator failures.
    from tools.benchmark_sota_4way import aggregate_rows, duration_bucket

    index, rows, summary, manifest = checked(data)
    ids = [r["track_id"] for r in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Empty or duplicate per-recording results")
    if ids != [r["track_id"] for r in manifest["tracks"]]:
        raise ValueError("Result/manifest membership or order differs")
    for row, witness in zip(rows, manifest["tracks"], strict=True):
        for field in ("source", "ref", "duration", "bucket"):
            if row[field] != witness[field]:
                raise ValueError("Per-recording manifest join differs: " + field)
    if summary["panel"]["n"] != len(rows) or manifest["n"] != len(rows):
        raise ValueError("Published denominator differs")
    result: dict[str, Any] = {
        "schema": "key-reference-metadata-analysis-v1",
        "n": len(rows),
        "artifact_index_sha256": sha((data / "artifact-index.json").read_bytes()),
        "libkeyfinder_status": index["libkeyfinder_status"],
        "source_counts": dict(Counter(r["source"] for r in rows)),
        "duration_by_source": {},
        "detectors": {},
        "inference_performed": False,
    }
    for source in sorted(result["source_counts"]):
        durations = [r["duration"] for r in rows if r["source"] == source]
        result["duration_by_source"][source] = {
            "n": len(durations),
            "min_seconds": min(durations),
            "median_seconds": statistics.median(durations),
            "max_seconds": max(durations),
        }
    for name, label in DETECTORS.items():
        if name == "libkeyfinder" and index["libkeyfinder_status"] == "historical_invalid":
            result["detectors"][name] = {"status": "INVALID_HISTORICAL_WRAPPER_NOT_A_BASELINE"}
            continue
        overall = aggregate_rows(rows, name)
        saved = summary["detectors"][label]
        # Older error-free artifacts omit new denominator/status fields. They
        # remain replayable; any stored metric must agree with the shared scorer.
        for field in METRICS:
            if field in saved and saved[field] != overall[field]:
                raise ValueError(f"Published {label} {field} differs from predictions")
        by_source = {
            source: aggregate_rows([r for r in rows if r["source"] == source], name)
            for source in sorted(result["source_counts"])
        }
        by_genre = {
            genre: aggregate_rows([r for r in rows if r["bucket"] == genre], name)
            for genre in sorted({r["bucket"] for r in rows})
        }
        by_duration = {
            bucket: aggregate_rows(
                [r for r in rows if duration_bucket(r["duration"]) == bucket], name
            )
            for bucket in ("<3m", "3-5m", "5-8m", "8m+")
            if any(duration_bucket(r["duration"]) == bucket for r in rows)
        }
        for group_name, groups in (("by_genre", by_genre), ("by_duration", by_duration)):
            for group, stored in saved.get(group_name, {}).items():
                if group not in groups:
                    raise ValueError("Published group has no input recordings")
                for field in METRICS:
                    if field not in stored:
                        continue
                    actual = groups[group][field]
                    if field in {"exact", "mirex"} and "n_total" not in stored:
                        actual = round(actual, 3)  # Historical group serializer precision.
                    if stored[field] != actual:
                        raise ValueError(f"Published {label} {group_name}/{group} {field} differs")
        result["detectors"][name] = {
            "status": "recorded",
            "overall": overall,
            "by_source": by_source,
            "by_genre": by_genre,
            "by_duration": by_duration,
        }
    return result


def plot(result: dict[str, Any], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "key-reference-comparison-v1",
        }
    )
    colors = {"drone": "#267A91", "madmom": "#A35D2C", "libkeyfinder": "#666D78"}
    labels = {"drone": "Drone v1", "madmom": "madmom KEY_CNN", "libkeyfinder": "libkeyfinder"}
    active = [name for name, row in result["detectors"].items() if row["status"] == "recorded"]
    sources = sorted(result["source_counts"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained", sharey=True)
    fig.suptitle(
        "Reference detectors on two different evaluation populations",
        fontweight="bold",
        fontsize=16,
    )
    width = 0.72 / len(active)
    for axis, metric, title in zip(
        axes, ("exact", "mirex"), ("Exact key accuracy", "Weighted key agreement"), strict=True
    ):
        for i, name in enumerate(active):
            x = [j - 0.36 + width * (i + 0.5) for j in range(len(sources))]
            values = [100 * result["detectors"][name]["by_source"][s][metric] for s in sources]
            bars = axis.bar(x, values, width=width * 0.9, color=colors[name], label=labels[name])
            axis.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=4, fontsize=10)
        axis.set(
            title=title,
            ylim=(0, 105),
            ylabel="Percent of the complete fixed cohort",
            xticks=range(len(sources)),
            xticklabels=[
                f"{'GiantSteps' if s == 'giantsteps' else 'Winterreise'}\nn = {result['source_counts'][s]}"
                for s in sources
            ],
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.supxlabel(
        "GiantSteps is development-exposed for Drone. Winterreise is one song cycle, two performances.\nNo independent SOTA ranking or confidence interval is claimed.",
        fontsize=10,
    )
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(
            destination / f"corpus-comparison.{suffix}",
            dpi=180,
            metadata={"Date": None} if suffix == "svg" else {},
        )
    plt.close(fig)
    genres = sorted(result["detectors"][active[0]]["by_genre"])
    fig, axis = plt.subplots(figsize=(10, 7), layout="constrained")
    fig.suptitle(
        "Genre slices describe these recordings, not genre populations",
        fontsize=15,
        fontweight="bold",
    )
    for i, name in enumerate(active):
        values = [100 * result["detectors"][name]["by_genre"][g]["mirex"] for g in genres]
        y = [j + (i - (len(active) - 1) / 2) * 0.18 for j in range(len(genres))]
        axis.scatter(values, y, color=colors[name], label=labels[name], s=55, zorder=3)
    axis.set(
        yticks=range(len(genres)),
        yticklabels=[
            f"{g}  (n = {result['detectors'][active[0]]['by_genre'][g]['n_total']})" for g in genres
        ],
        xlabel="Symmetric-fifths weighted agreement (% of all recordings in slice)",
        xlim=(-3, 105),
    )
    axis.invert_yaxis()
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", alpha=0.18)
    axis.legend(frameon=False, loc="lower left")
    fig.supxlabel(
        "Classical is Winterreise only; every other slice is a small development-exposed GiantSteps sample.\nDots are descriptive point estimates. No confidence intervals or independent SOTA claim.",
        fontsize=10,
    )
    for suffix in ("png", "svg"):
        fig.savefig(
            destination / f"genre-comparison.{suffix}",
            dpi=180,
            metadata={"Date": None} if suffix == "svg" else {},
        )
    plt.close(fig)
    receipt = {
        "analysis_sha256": sha(encode(result)),
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "matplotlib_version": matplotlib.__version__,
        "outputs": {
            f"{figure}.{suffix}": sha((destination / f"{figure}.{suffix}").read_bytes())
            for figure in ("corpus-comparison", "genre-comparison")
            for suffix in ("png", "svg")
        },
    }
    (destination / "corpus-comparison-inputs.json").write_bytes(encode(receipt))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--seal-index", choices=("historical_invalid", "corrected"))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.seal_index:
        (args.data / "artifact-index.json").write_bytes(encode(seal(args.data, args.seal_index)))
    result = analyze(args.data)
    raw = encode(result)
    output = args.data / "metadata-analysis.json"
    if args.write:
        output.write_bytes(raw)
    elif output.read_bytes() != raw:
        raise ValueError("Published metadata analysis differs")
    if args.plot:
        plot(result, args.data.parent / "figures")
    print(json.dumps({"n": result["n"], "sha256": sha(raw), "inference_performed": False}))


if __name__ == "__main__":
    main()
