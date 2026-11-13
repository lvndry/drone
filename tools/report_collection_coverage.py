"""Plot metadata availability from pinned target-free collections, without audio access."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.collection_catalog import CollectionCatalog, load_collection, validate_collection
from tools.dj_coverage_bp import band
from tools.dj_coverage_report import BANDS

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/studies/v2-data-contract"
INPUTS = {
    "Beatport": (
        "beatport1355-collection.json.gz",
        "24a09353dc4e653bfc62898a0ca61ee2ff4838d62c3ea07ee261a2fe4f00b11f",
        1355,
    ),
    "FMA": (
        "fmak5489-collection.json.gz",
        "a626b886b61e94a6296e6160a672be54797b62a7f8f60916c82905f2c34a73b2",
        5489,
    ),
}
FIELDS = {
    "genre_tags": "Any native genre tags",
    "native_top": "Native top / workbook genre",
    "artist": "Observed artist ID",
    "album_release": "Observed album / release ID",
    "bpm": "Admitted publisher BPM",
    "recorded_cache_duration": "Recorded 24-kHz cache duration",
    "publisher_duration": "Publisher recording duration",
    "source_duration": "Probed source duration",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def summarize(catalog: CollectionCatalog) -> dict[str, Any]:
    validate_collection(catalog)
    total = len(catalog.tracks)
    known = dict.fromkeys(FIELDS, 0)
    durations = dict.fromkeys((*BANDS, "unknown"), 0)
    unknown_reasons: Counter[str] = Counter()
    groups: dict[str, set[str]] = {}
    source_hashes = set()
    support_kinds: Counter[str] = Counter()
    genre_methods: Counter[str] = Counter()
    for row in catalog.tracks:
        present = {
            "genre_tags": bool(row.genre_tags),
            "native_top": bool(row.native_genre_top),
            "artist": any(relation.kind == "artist" for relation in row.relations),
            "album_release": any(
                relation.kind in {"album", "release"} for relation in row.relations
            ),
            "bpm": row.bpm.value is not None,
            "recorded_cache_duration": row.analyzed_audio.seconds is not None,
            "publisher_duration": row.catalog_recording_duration.value is not None,
            "source_duration": row.source_duration.value is not None,
        }
        for field, value in present.items():
            known[field] += int(value)
        seconds = row.analyzed_audio.seconds
        durations["unknown" if seconds is None else band(seconds)] += 1
        if seconds is None:
            unknown_reasons[str(row.analyzed_audio.missing_reason)] += 1
        for group in row.frozen_groups:
            groups.setdefault(group.scheme, set()).add(group.group_id)
        if row.source_audio.sha256 is not None:
            source_hashes.add(row.source_audio.sha256)
        support_kinds[row.analyzed_audio.support_kind] += 1
        genre_methods[row.genre_method] += 1
    return {
        "collection_id": catalog.collection_id,
        "rows": total,
        "availability": {
            field: {"known": count, "unknown": total - count} for field, count in known.items()
        },
        "recorded_cache_duration_bands": durations,
        "unknown_cache_reasons": dict(unknown_reasons),
        "distinct_recorded_source_hashes": len(source_hashes),
        "frozen_group_counts": {scheme: len(ids) for scheme, ids in groups.items()},
        "support_kinds": dict(support_kinds),
        "genre_methods": dict(genre_methods),
    }


def analyze(study: Path = STUDY) -> dict[str, Any]:
    results = {}
    sources = {}
    for name, (filename, expected, count) in INPUTS.items():
        path = study / "data" / filename
        if sha(path.read_bytes()) != expected:
            raise ValueError("Pinned collection changed: " + name)
        catalog = load_collection(path)
        if len(catalog.tracks) != count:
            raise ValueError("Unexpected collection size: " + name)
        results[name] = summarize(catalog)
        sources[name] = {"path": "data/" + filename, "sha256": expected}
        del catalog
        gc.collect()
    return {
        "schema": "drone.collection-coverage-report.v1",
        "collections": results,
        "sources": sources,
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "audio_read": False,
        "targets_selected": False,
        "inference_performed": False,
        "interpretation": "Metadata availability only; row counts are not independent-song counts or model accuracy.",
    }


def plot(result: dict[str, Any], destination: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "drone-collection-coverage-v1",
        }
    )
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {}
    colors = {"Beatport": "#267A91", "FMA": "#996128"}

    def save(fig: Figure, stem: str) -> None:
        for suffix in ("png", "svg"):
            path = destination / f"{stem}.{suffix}"
            fig.savefig(path, dpi=170, metadata={"Date": None} if suffix == "svg" else {})
            if suffix == "svg":
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
                )
            outputs[path.name] = sha(path.read_bytes())
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for axis, (name, summary) in zip(axes, result["collections"].items(), strict=True):
        total = summary["rows"]
        counts = [summary["availability"][field]["known"] for field in FIELDS]
        percentages = [100 * count / total for count in counts]
        y = list(range(len(FIELDS)))
        axis.barh(y, [100] * len(y), color="#ECEFF1", height=0.66)
        axis.barh(y, percentages, color=colors[name], height=0.66)
        for index, (count, percentage) in enumerate(zip(counts, percentages, strict=True)):
            axis.text(102, index, f"{count:,}/{total:,}", va="center", fontsize=9)
        axis.set(
            yticks=y,
            yticklabels=list(FIELDS.values()),
            xlim=(0, 130),
            xticks=[0, 25, 50, 75, 100],
            xlabel="Available metadata (% of collection rows)",
            title=f"{name} · {total:,} rows",
            ylim=(len(y) - 0.5, -0.5),
        )
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    fig.suptitle(
        "An organized collection makes missing metadata visible", fontsize=16, fontweight="bold"
    )
    fig.supxlabel(
        "Colored bars = recorded metadata; pale remainder = unknown or absent.\nPublisher values are not independent measurements; genre taxonomies remain source-specific.",
        fontsize=10,
    )
    save(fig, "collection-metadata-availability")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    labels = (*BANDS, "unknown")
    for axis, (name, summary) in zip(axes, result["collections"].items(), strict=True):
        counts = [summary["recorded_cache_duration_bands"][label] for label in labels]
        bars = axis.bar(range(len(labels)), counts, color=[colors[name]] * len(BANDS) + ["#92999F"])
        bars[-1].set_hatch("//")
        axis.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=4, fontsize=10)
        axis.set(
            xticks=range(len(labels)),
            xticklabels=labels,
            ylabel="Collection rows",
            title=f"{name} · recorded 24-kHz training cache",
            ylim=(0, max(counts) * 1.18),
            xlabel="Recorded samples ÷ sample rate",
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", labelrotation=15)
    fig.suptitle(
        "Duration evidence covers training audio, with reserved gaps explicit",
        fontsize=15,
        fontweight="bold",
    )
    fig.supxlabel(
        "FMA's 1,206 unknown rows are reserved or excluded IDs without support in the selected cache receipt.\nThese are not full-song length measurements or release-model inputs. Publisher durations are not substituted.",
        fontsize=10,
    )
    save(fig, "collection-recorded-duration")
    return {
        "analysis_sha256": sha(encode(result)),
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "matplotlib_version": matplotlib.__version__,
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    stop = threading.Event()
    peak_sampled = [0]

    def guard() -> None:
        while not stop.wait(0.02):
            use = resource.getrusage(resource.RUSAGE_SELF)
            # getrusage is the OS high-water mark, observed repeatedly; not instantaneous RSS.
            peak_sampled[0] = int(
                use.ru_maxrss if sys.platform == "darwin" else use.ru_maxrss * 1024
            )
            if peak_sampled[0] > 512 * 1024**2 or time.monotonic() - started > 90:
                os._exit(124)

    resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    threading.Thread(target=guard, daemon=True).start()
    result = analyze()
    path = STUDY / "data/collection-coverage-analysis.json"
    raw = encode(result)
    if args.write:
        path.write_bytes(raw)
    elif path.read_bytes() != raw:
        raise ValueError("Stored analysis differs")
    if args.plot:
        receipt = plot(result, STUDY / "figures")
        (STUDY / "data/collection-coverage-figures.json").write_bytes(encode(receipt))
    owned = [path, STUDY / "data/collection-coverage-figures.json"]
    owned.extend((STUDY / "figures").glob("collection-*.png"))
    owned.extend((STUDY / "figures").glob("collection-*.svg"))
    output_bytes = sum(file.stat().st_size for file in owned if file.exists())
    if output_bytes > 16 * 1024**2:
        raise ValueError("Aggregate output budget exceeded")
    stop.set()
    use = resource.getrusage(resource.RUSAGE_SELF)
    print(
        json.dumps(
            {
                "wall_seconds": time.monotonic() - started,
                "cpu_seconds": use.ru_utime + use.ru_stime,
                "os_peak_rss_bytes": use.ru_maxrss
                if sys.platform == "darwin"
                else use.ru_maxrss * 1024,
                "output_bytes": output_bytes,
                "inference_performed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
