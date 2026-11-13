"""Render source-verified development diagnostics inside the MusicFM result report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/studies/musicfm-key-pilot"
COLORS = {"k10_expanded": "#267A91", "musicfm_pilot": "#B35A38"}
NAMES = {"k10_expanded": "K10 expanded", "musicfm_pilot": "MusicFM pilot"}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encode(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def plot(result: dict[str, Any], destination: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "drone-dev-slices-v1"}
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    def save(fig: Figure, name: str) -> None:
        for extension in ("png", "svg"):
            path = destination / f"{name}.{extension}"
            fig.savefig(path, dpi=170, metadata={"Date": None} if extension == "svg" else {})
            if extension == "svg":
                path.write_text("\n".join(s.rstrip() for s in path.read_text().splitlines()) + "\n")
            outputs[path.name] = digest(path.read_bytes())
        plt.close(fig)

    def slices(domain: str, dimension: str) -> list[dict[str, Any]]:
        rows = [r for r in result["corpora"][domain]["slices"] if r["dimension"] == dimension]
        if not rows:
            raise ValueError("Missing plotted dimension: " + domain + "/" + dimension)
        return rows

    def style(axis: Axes, rows: list[dict[str, Any]], title: str) -> None:
        axis.set_yticks(
            range(len(rows)), [str(r["label"]).replace("duration_unknown", "Unknown") for r in rows]
        )
        axis.set_ylim(len(rows) - 0.5, -0.8)
        axis.set_xlim(0, 145)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_title(title, loc="left", fontweight="bold", pad=18)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(axis="x", alpha=0.15)
        axis.set_axisbelow(True)

    genre_specs = (
        ("beatport", "taxonomy:beatport_workbook_native", "Beatport · publisher workbook genres"),
        ("fmak", "native_genre_top", "FMA · native top genre, including missing"),
    )
    genre_rows = [slices(domain, dimension) for domain, dimension, _ in genre_specs]
    fig, axes = plt.subplots(
        1, 2, figsize=(17, max(6.5, 0.52 * max(map(len, genre_rows)) + 2)), layout="constrained"
    )
    for axis, rows, (_, _, title) in zip(axes, genre_rows, genre_specs, strict=True):
        style(axis, rows, title)
        axis.set_xlabel("Exact-key accuracy (%)")
        for model, offset in (("k10_expanded", -0.14), ("musicfm_pilot", 0.14)):
            for index, row in enumerate(rows):
                n = row["n"]
                metric = row["models"][model]
                if n:
                    axis.scatter(
                        100 * metric["correct"] / n, index + offset, color=COLORS[model], s=34
                    )
                axis.text(
                    104,
                    index + offset,
                    f"{metric['correct']}/{n}" if n else "NA · n=0",
                    va="center",
                    fontsize=8,
                    color=COLORS[model],
                )
        for index, row in enumerate(rows):
            axis.text(
                126,
                index,
                f"g={row['group_count']}" + (" *" if 0 < row["n"] < 10 else ""),
                va="center",
                fontsize=8,
                color="#4F5660",
            )
    fig.suptitle("Selected research checkpoints · retrospective development records", fontsize=15)
    fig.legend(
        handles=[Patch(color=COLORS[m], label=NAMES[m]) for m in COLORS],
        loc="outside lower center",
        ncol=2,
        frameon=False,
        title="Counts are exact successes / tracks; g = frozen groups. * Fewer than 10 tracks.",
    )
    save(fig, "development-native-genres")

    outcomes = (
        ("both_correct", "Both correct", "#729A81"),
        ("k10_only_correct", "K10 only correct", COLORS["k10_expanded"]),
        ("musicfm_only_correct", "MusicFM only correct", COLORS["musicfm_pilot"]),
        ("both_wrong", "Both wrong", "#D1D5DB"),
    )

    def paired(axis: Axes, rows: list[dict[str, Any]], title: str) -> None:
        style(axis, rows, title)
        axis.set_xlabel("Share of tracks in the slice (%)")
        axis.set_xlim(0, 170)
        for index, row in enumerate(rows):
            n, left = row["n"], 0.0
            if sum(row["paired"][key] for key, _, _ in outcomes) != n:
                raise ValueError("Paired outcomes do not partition the slice")
            for key, _, color in outcomes:
                count = row["paired"][key]
                width = 100 * count / n if n else 0.0
                axis.barh(index, width, left=left, height=0.66, color=color)
                if width >= 8:
                    axis.text(
                        left + width / 2,
                        index,
                        str(count),
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white"
                        if key in {"k10_only_correct", "musicfm_only_correct"}
                        else "#14251C",
                    )
                left += width
            axis.text(
                104,
                index,
                f"n={n}, g={row['group_count']}"
                + (" *" if 0 < n < 10 else "")
                + f"\nK10 / MusicFM only: {row['paired']['k10_only_correct']} / {row['paired']['musicfm_only_correct']}",
                va="center",
                fontsize=8,
            )
            if not n:
                axis.text(1, index, "NA — no records", va="center", fontsize=9, color="#69717B")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.3), layout="constrained")
    for axis, domain, title in zip(
        axes,
        ("beatport", "fmak"),
        ("Beatport · 126 development previews", "FMA · 250 development recordings"),
        strict=True,
    ):
        paired(axis, slices(domain, "duration_actual_evaluation"), title)
    fig.suptitle("Which records do both systems miss? · Actual evaluation duration", fontsize=15)
    fig.legend(
        handles=[Patch(color=color, label=label) for _, label, color in outcomes],
        loc="outside lower center",
        ncol=4,
        frameon=False,
        title="Counts inside bars; g = frozen groups. * Fewer than 10 tracks. No independent-test inference.",
    )
    save(fig, "development-duration-pairs")

    fig, axis = plt.subplots(figsize=(10.5, 6.3), layout="constrained")
    paired(
        axis,
        slices("beatport", "bpm_publisher"),
        "Beatport · publisher-reported BPM, missing values retained",
    )
    fig.suptitle("Tempo coverage is partial · 126 development records", fontsize=14)
    fig.legend(
        handles=[Patch(color=color, label=label) for _, label, color in outcomes],
        loc="outside lower center",
        ncol=2,
        frameon=False,
        title="Reported tempo is not a verified reference. All 250 FMA BPM values are unknown.",
    )
    save(fig, "development-bpm-pairs")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Render the three diagnostic figures")
    args = parser.parse_args()
    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    stop = threading.Event()

    def guard() -> None:
        while not stop.wait(0.025):
            use = resource.getrusage(resource.RUSAGE_SELF)
            rss = use.ru_maxrss if sys.platform == "darwin" else use.ru_maxrss * 1024
            if rss > 512 * 1024**2 or time.monotonic() - started > 120:
                os._exit(124)

    threading.Thread(target=guard, daemon=True).start()
    # Import after the process caps; the analyzer opens only recorded metadata.
    from tools.analyze_development_slices import DEFAULT, verify_analysis

    result = verify_analysis()
    raw = DEFAULT.read_bytes()
    path = STUDY / "data/development-slice-figures.json"
    base = {
        "analysis": str(DEFAULT.relative_to(ROOT)),
        "analysis_sha256": digest(raw),
        "generator_sha256": digest(Path(__file__).read_bytes()),
        "inference_performed": False,
    }
    receipt: dict[str, Any]
    outputs: dict[str, str]
    if args.write:
        import matplotlib

        outputs = plot(result, STUDY / "figures")
        receipt = {
            **base,
            "matplotlib": matplotlib.__version__,
            "outputs": outputs,
        }
        path.write_bytes(encode(receipt))
    else:
        receipt = json.loads(path.read_bytes())
        if any(receipt[key] != value for key, value in base.items()):
            raise ValueError("Plot source or analysis binding differs")
        outputs = receipt["outputs"]
        if not isinstance(outputs, dict) or not all(
            isinstance(name, str) and isinstance(value, str) for name, value in outputs.items()
        ):
            raise ValueError("Invalid figure hash mapping")
        for name, expected in outputs.items():
            if digest((STUDY / "figures" / name).read_bytes()) != expected:
                raise ValueError("Plot artifact differs: " + name)
    output_bytes = sum((STUDY / "figures" / name).stat().st_size for name in outputs)
    if output_bytes > 16 * 1024**2:
        raise ValueError("Aggregate figure output budget exceeded")
    stop.set()
    use = resource.getrusage(resource.RUSAGE_SELF)
    rss = use.ru_maxrss if sys.platform == "darwin" else use.ru_maxrss * 1024
    if rss > 512 * 1024**2 or time.monotonic() - started > 120:
        raise RuntimeError("Completed plot exceeded its resource budget")
    print(
        json.dumps(
            {
                "wall_seconds": time.monotonic() - started,
                "cpu_seconds": use.ru_utime + use.ru_stime,
                "os_peak_rss_bytes": use.ru_maxrss
                if sys.platform == "darwin"
                else use.ru_maxrss * 1024,
                "output_bytes": output_bytes,
            }
        )
    )


if __name__ == "__main__":
    main()
