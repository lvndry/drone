"""Reproduce reference-benchmark metadata and inline figures; never runs detectors."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
import tarfile
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


def corrected_evidence(data: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Verify the sealed, audio-free execution chain and retain its native predictions."""
    folder = data / "corrected-kf"
    receipt = json.loads((folder / "receipt.json").read_bytes())
    raw = (folder / "evidence.tar.gz").read_bytes()
    if sha(raw) != receipt["evidence_sha256"]:
        raise ValueError("Corrected KF archive changed")
    verifier = (folder / "verify_evidence.py.txt.gz").read_bytes()
    if (sha(verifier), sha(gzip.decompress(verifier))) != (
        receipt["original_verifier_gzip_sha256"],
        receipt["original_verifier_sha256"],
    ):
        raise ValueError("Preserved original verifier changed")
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        files = {}
        for member in archive:
            if not member.isfile():
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Missing evidence member")
            files[member.name.removeprefix("./")] = stream.read()

    def read(name: str) -> dict[str, Any]:
        return json.loads(files[name])

    plan, controller, complete = (
        read(n) for n in ("plan.json", "controller.json", "worker-complete.json")
    )
    if (
        sha(files["controller.json"]) != receipt["controller_sha256"]
        or controller["status"] != "COMPLETE"
        or complete["status"] != "COMPLETE"
        or controller["plan_sha256"] != sha(files["plan.json"])
        or complete["plan_sha256"] != controller["plan_sha256"]
        or controller["worker_complete_sha256"] != sha(files["worker-complete.json"])
        or plan["runner_sha256"] != sha(files["runner.py"])
        or plan["protocol_sha256"] != sha(files["PROTOCOL.md"])
        or len(plan["rows"]) != 93
        or len(complete["result_hashes"]) != 93
    ):
        raise ValueError("Corrected KF execution chain differs")
    if (
        controller["rows"] != 93
        or controller["pcm_equivalent"] != 93
        or complete["rows"] != 93
        or complete["pcm_equivalent"] != 93
        or controller["wall_seconds"] > 900
        or controller["cpu_seconds"] > 1200
        or controller["sampled_peak_rss_bytes"] > 1024**3
    ):
        raise ValueError("Corrected KF population or resource receipt differs")
    rows = []
    for index, planned in enumerate(plan["rows"]):
        stem = f"row-{index:03d}"
        name = stem + "-result.json"
        row = read(name)
        if (
            sha(files[name]) != complete["result_hashes"][name]
            or row["index"] != index
            or row["track_id"] != planned["track_id"]
        ):
            raise ValueError("Corrected KF row identity differs")
        eq = row["equivalence"]
        if (
            eq != read(stem + "-equivalence.json")
            or eq["pcm_equivalent"] is not True
            or eq["container_identical"] is not True
            or eq["fresh"] != eq["retained"]
            or eq["retained"] != planned["retained_audio"]
        ):
            raise ValueError("Corrected KF shared audio identity differs")
        for stage in ("decode", "keyfinder"):
            command = read(stem + "-" + stage + ".command.json")
            if row[stage] != command or command["exit_code"] != 0:
                raise ValueError("Corrected KF command failed or changed")
            for stream in ("stdout", "stderr"):
                if sha(files[stem + "-" + stage + "." + stream]) != command[stream + "_sha256"]:
                    raise ValueError("Corrected KF command output changed")
        from tools.benchmark_sota_4way import keyfinder_result

        native = keyfinder_result(files[stem + "-keyfinder.stdout"].decode())
        if native != row["prediction"]:
            raise ValueError("Corrected KF raw enum/typed prediction differs")
        row["verified_status"] = "abstain" if native["status"] == "silence" else "key"
        rows.append(row)
    return plan, rows, controller


def publish_corrected(data: Path) -> None:
    """Replace only invalid KF outputs, joining fresh execution by frozen order/ID."""
    from drone.keys import WEIGHTS, score_breakdown, score_mirex
    from tools.benchmark_sota_4way import summarize

    plan, fresh, controller = corrected_evidence(data)
    historic = data / "historical"
    rows = list(
        csv.DictReader(
            io.StringIO(gzip.decompress((historic / "bench.csv.gz").read_bytes()).decode())
        )
    )
    summary = json.loads(gzip.decompress((historic / "bench.json.gz").read_bytes()))
    manifest = json.loads(gzip.decompress((historic / "manifest.json.gz").read_bytes()))
    for name, path in (
        ("bench.csv", historic / "bench.csv.gz"),
        ("bench.json", historic / "bench.json.gz"),
        ("manifest.json", historic / "manifest.json.gz"),
    ):
        historical_sha = sha(gzip.decompress(path.read_bytes()))
        if plan["pins"]["/home/lysk-admin/drone-bench/results/" + name] != historical_sha:
            raise ValueError("Fresh execution was not bound to these historical artifacts")
    timings = []
    track_artifacts = []
    for index, (row, recorded, planned, track) in enumerate(
        zip(rows, fresh, plan["rows"], manifest["tracks"], strict=True)
    ):
        if (
            row["track_id"] != recorded["track_id"]
            or row["track_id"] != track["track_id"]
            or int(row["ref"]) != planned["ref"]
            or row["ref_raw"] != planned["ref_raw"]
        ):
            raise ValueError("Fresh execution/reference join differs")
        pred = recorded["prediction"]["drone_class"]
        row["ref"] = int(row["ref"])
        row["duration"] = float(row["duration"])
        row["libkeyfinder_pred"] = pred
        row["libkeyfinder_raw_enum"] = recorded["prediction"]["raw_enum"]
        row["libkeyfinder_status"] = recorded["verified_status"]
        row["libkeyfinder_rel"] = (
            next(iter(score_breakdown([pred], [row["ref"]])))
            if pred is not None
            else recorded["verified_status"]
        )
        row["libkeyfinder_score"] = (
            score_mirex([pred], [row["ref"]])[1] if pred is not None else 0.0
        )
        row["libkeyfinder_ms"] = round(recorded["keyfinder"]["wall_seconds"] * 1000, 1)
        timings.append(recorded["keyfinder"]["wall_seconds"] * 1000)
        for name in ("drone", "madmom"):
            row[name + "_pred"] = int(row[name + "_pred"])
            row[name + "_status"] = "key"
        binding = {
            "track_id": row["track_id"],
            "source_sha256": planned["source_sha256"],
            "canonical_wav": recorded["equivalence"]["fresh"],
            "fresh_and_retained_identical": True,
            "evidence_result": f"row-{index:03d}-result.json",
        }
        track_artifacts.append(binding)
        track["input_identity"] = binding
    summary["detectors"]["libkeyfinder-2.2.8"] = summarize(rows, "libkeyfinder", timings)
    summary["scoring"]["weights"] = dict(WEIGHTS)
    provenance = {
        "kind": "mixed_execution_record_corrected_comparator",
        "libkeyfinder": {
            "execution": "fresh_corrected_wrapper",
            "plan_pins": plan["pins"],
            "controller": controller,
            "evidence_receipt": "corrected-kf/receipt.json",
        },
        "drone": {"execution": "unchanged_historical_predictions_and_timings"},
        "madmom": {"execution": "unchanged_historical_predictions_and_timings"},
        "timing_comparison": "Separate runs/conditions; not a fair speed ranking",
        "publication_generator_sha256": sha(Path(__file__).read_bytes()),
    }
    availability = json.loads((data / "musicfm-availability.json").read_bytes())
    availability_receipt = json.loads((data / "musicfm-availability-receipt.json").read_bytes())
    summary["detectors"]["musicfm-mlp"] = {
        "status": "blocked",
        "reason": "No published checkpoint in the dated official repository snapshot; no inference adapter configured or model run",
        "published_gs": "72.19% exact / 77.62% MIREX (per repo README; not this panel)",
        "checked_revision": availability["sha"],
        "checked_utc": availability_receipt["retrieved_utc"],
        "metadata_receipt": "musicfm-availability-receipt.json",
    }
    extra = summary["detectors_extra"]
    extra["original_run_wall_clock_total_s"] = extra.pop("wall_clock_total_s")
    extra["corrected_kf_run_wall_clock_total_s"] = controller["wall_seconds"]
    extra["timings_scope"] = (
        "Mixed execution record: historical Drone/madmom timings and fresh corrected KF per-call timings; no fair speed ranking"
    )
    summary["provenance"] = provenance
    summary["track_artifacts"] = track_artifacts
    summary["generated"] = (
        "2026-11-13 corrected KF publication; original other-detector run 2026-11-12"
    )
    manifest["provenance"] = provenance
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (data / "bench.csv").write_text(stream.getvalue())
    (data / "bench.json").write_bytes(encode(summary))
    (data / "manifest.json").write_bytes(encode(manifest))


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
            for name in (
                "bench.csv",
                "bench.json",
                "manifest.json",
                "musicfm-availability.json",
                "musicfm-availability-receipt.json",
            )
        },
        "corrected_evidence_receipt_sha256": sha((data / "corrected-kf/receipt.json").read_bytes())
        if kf_status == "corrected"
        else None,
        "historical_archive_index_sha256": sha(
            (data / "historical/archive-index.json").read_bytes()
        ),
    }


def checked(
    data: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    index = json.loads((data / "artifact-index.json").read_bytes())
    values = {}
    for name in (
        "bench.csv",
        "bench.json",
        "manifest.json",
        "musicfm-availability.json",
        "musicfm-availability-receipt.json",
    ):
        raw = (data / name).read_bytes()
        binding = index["files"][name]
        if sha(raw) != binding["sha256"] or len(raw) != binding["bytes"]:
            raise ValueError("Published artifact changed: " + name)
        values[name] = raw
    availability = json.loads(values["musicfm-availability.json"])
    availability_receipt = json.loads(values["musicfm-availability-receipt.json"])
    if (
        sha(values["musicfm-availability.json"]) != availability_receipt["sha256"]
        or availability["sha"] != availability_receipt["revision"]
        or any(
            Path(row["rfilename"]).suffix in {".pt", ".pth", ".bin", ".safetensors"}
            for row in availability["siblings"]
        )
    ):
        raise ValueError("MusicFM dated availability witness differs")
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
    if index["libkeyfinder_status"] == "corrected":
        if (
            sha((data / "corrected-kf/receipt.json").read_bytes())
            != index["corrected_evidence_receipt_sha256"]
        ):
            raise ValueError("Corrected KF receipt changed")
        plan, fresh, _ = corrected_evidence(data)
        old_rows = list(
            csv.DictReader(
                io.StringIO(
                    gzip.decompress((data / "historical/bench.csv.gz").read_bytes()).decode()
                )
            )
        )
        old_summary = json.loads(gzip.decompress((data / "historical/bench.json.gz").read_bytes()))
        for name in ("drone", "madmom"):
            if summary["detectors"][DETECTORS[name]] != old_summary["detectors"][DETECTORS[name]]:
                raise ValueError("Unchanged detector summary was modified")
        for row, old, recorded, planned, track, artifacts in zip(
            rows,
            old_rows,
            fresh,
            plan["rows"],
            manifest["tracks"],
            summary["track_artifacts"],
            strict=True,
        ):
            for name in ("drone", "madmom"):
                for field in ("pred", "score", "ms"):
                    if float(row[name + "_" + field]) != float(old[name + "_" + field]):
                        raise ValueError("Unchanged historical detector row was modified")
                if row[name + "_rel"] != old[name + "_rel"]:
                    raise ValueError("Unchanged historical detector relation was modified")
            if (
                track["input_identity"] != artifacts
                or artifacts["canonical_wav"] != recorded["equivalence"]["fresh"]
                or artifacts["source_sha256"] != planned["source_sha256"]
            ):
                raise ValueError("Published shared-input provenance differs")
            if (
                row["track_id"] != recorded["track_id"]
                or row["ref"] != planned["ref"]
                or row["libkeyfinder_pred"] != recorded["prediction"]["drone_class"]
                or row["libkeyfinder_status"] != recorded["verified_status"]
                or int(row["libkeyfinder_raw_enum"]) != recorded["prediction"]["raw_enum"]
            ):
                raise ValueError("Published KF row differs from fresh evidence")
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
    for name in ("corpus-comparison.svg", "genre-comparison.svg"):
        path = destination / name
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
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


DURATION_SLICES = ("<3m", "3-5m", "5-8m", "8m+", "unknown")
TONIC_SLICES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
CATALOG = ROOT / "experiments/studies/v2-data-contract/data/reference-panel-catalog.json.gz"


def slice_metrics(rows: list[dict[str, Any]], detector: str) -> dict[str, Any]:
    """Use the existing scorer; an unobserved slice is NA, not zero accuracy."""
    from tools.benchmark_sota_4way import aggregate_rows

    result = aggregate_rows(rows, detector)
    if not rows:
        result["exact"] = result["mirex"] = None
    return result


def descriptive_slices(rows: list[dict[str, Any]], tracks: list[dict[str, Any]]) -> dict[str, Any]:
    """Fixed metadata partitions, never ranked/selected using model outcomes."""
    from tools.benchmark_sota_4way import duration_bucket

    metadata = {track["track_id"]: track for track in tracks}
    ids = [row["track_id"] for row in rows]
    if len(metadata) != len(tracks) or len(set(ids)) != len(ids) or set(metadata) != set(ids):
        raise ValueError("Slice catalog must join the complete unique prediction cohort")
    dimensions: dict[str, dict[str, list[dict[str, Any]]]] = {
        "corpus": {},
        "genre_original": {},
        "genre_legacy_group": {},
        "analyzed_duration": {key: [] for key in DURATION_SLICES},
        "key": {f"{tonic} {mode}": [] for tonic in TONIC_SLICES for mode in ("major", "minor")},
        "mode": {"major": [], "minor": []},
        "bpm_coverage": {"known": [], "unknown": []},
    }
    cross: dict[str, dict[str, list[dict[str, Any]]]] = {}
    mode_cross: dict[str, dict[str, list[dict[str, Any]]]] = {}
    missing = Counter()
    grouping: dict[str, dict[str, Any]] = {}
    for row in rows:
        track = metadata[row["track_id"]]
        if track["key_label"]["class_id"] != row["ref"]:
            raise ValueError("Slice catalog label disagrees with frozen prediction reference")
        if track["genre"]["raw"] != row["genre"] or track["genre"]["bucket"] != row["bucket"]:
            raise ValueError("Slice catalog genre disagrees with frozen metadata")
        duration = track["duration"]["analyzed"]["value"]
        if duration is not None and (not math.isfinite(duration) or duration <= 0):
            raise ValueError("Analyzed duration must be positive and finite, or explicitly unknown")
        bucket = "unknown" if duration is None else duration_bucket(duration)
        corpus = track["corpus_id"]
        raw_genre = track["genre"]["raw"] or "unknown"
        genre = track["genre"]["bucket"] or "unknown"
        mode = "minor" if row["ref"] % 2 else "major"
        key = f"{TONIC_SLICES[row['ref'] // 2]} {mode}"
        bpm = "unknown" if track["bpm"]["value"] is None else "known"
        for name, value in (
            ("corpus", corpus),
            ("genre_original", raw_genre),
            ("genre_legacy_group", genre),
            ("analyzed_duration", bucket),
            ("key", key),
            ("mode", mode),
            ("bpm_coverage", bpm),
        ):
            dimensions[name].setdefault(value, []).append(row)
        cross.setdefault(corpus, {band: [] for band in DURATION_SLICES})[bucket].append(row)
        mode_cross.setdefault(corpus, {value: [] for value in ("major", "minor")})[mode].append(row)
        for name, value in (
            ("genre_original", track["genre"]["raw"]),
            ("analyzed_duration", duration),
            ("full_recording_duration", track["duration"]["full_recording"]["value"]),
            ("bpm", track["bpm"]["value"]),
            ("work_id", track["grouping"]["work_id"]),
        ):
            if value is None or value == "":
                missing[name] += 1
        record = grouping.setdefault(
            corpus,
            {
                "n": 0,
                "known_work_ids": set(),
                "known_performance_ids": set(),
                "unknown_work_rows": 0,
            },
        )
        record["n"] += 1
        if track["grouping"]["work_id"] is None:
            record["unknown_work_rows"] += 1
        else:
            record["known_work_ids"].add(track["grouping"]["work_id"])
        if track["grouping"]["performance_id"] is not None:
            record["known_performance_ids"].add(track["grouping"]["performance_id"])
    for name in ("corpus", "genre_original", "genre_legacy_group"):
        dimensions[name] = dict(sorted(dimensions[name].items()))
    for record in grouping.values():
        record["known_work_ids"] = sorted(record["known_work_ids"])
        record["known_performance_ids"] = sorted(record["known_performance_ids"])
    return {
        "schema": "key-reference-descriptive-slices-v1",
        "n": len(rows),
        "scoring": "existing symmetric-fifths agreement; errors/abstentions zero; empty slice null",
        "duration_semantics": "canonical analyzed PCM frames / sample rate, not complete-song length",
        "duration_edges_seconds": [180, 300, 480],
        "bpm_policy": "coverage only; no BPM estimator or unbound metadata added",
        "genre_grouping_policy": {
            "label": "legacy benchmark grouping",
            "methods": sorted({track["genre"].get("method", "unspecified") for track in tracks}),
            "taxonomy_status": "not a curated modern genre taxonomy; raw tags are retained",
        },
        "uncertainty_policy": "descriptive only; no independent-recording confidence intervals or significance claims",
        "coverage": {
            name: {"known": len(rows) - missing[name], "unknown": missing[name]}
            for name in (
                "genre_original",
                "analyzed_duration",
                "full_recording_duration",
                "bpm",
                "work_id",
            )
        },
        "grouping_by_corpus": grouping,
        "slices": {
            dimension: {
                value: {
                    "n": len(members),
                    "detectors": {name: slice_metrics(members, name) for name in DETECTORS},
                }
                for value, members in groups.items()
            }
            for dimension, groups in dimensions.items()
        },
        "corpus_by_duration": {
            corpus: {
                band: {
                    "n": len(members),
                    "detectors": {name: slice_metrics(members, name) for name in DETECTORS},
                }
                for band, members in groups.items()
            }
            for corpus, groups in sorted(cross.items())
        },
        "corpus_by_mode": {
            corpus: {
                mode: {
                    "n": len(members),
                    "detectors": {name: slice_metrics(members, name) for name in DETECTORS},
                }
                for mode, members in groups.items()
            }
            for corpus, groups in sorted(mode_cross.items())
        },
        "inference_performed": False,
    }


def analyze_slices(data: Path = DATA, catalog_path: Path = CATALOG) -> dict[str, Any]:
    from dataclasses import asdict

    from experiments.corpus_catalog import load_catalog
    from tools.export_corpus_catalog import verify_reference_panel

    base = analyze(data)  # Preserve/replay the original chain; do not republish or reseal it.
    if base["libkeyfinder_status"] != "corrected":
        raise ValueError("Slice comparison requires the corrected comparator")
    catalog = load_catalog(catalog_path)
    _, rows, _, manifest = checked(data)
    verify_reference_panel(catalog, ROOT)
    sources = {source.source_id: source for source in catalog.sources}
    if catalog.results_source_id != "benchmark_results":
        raise ValueError("Slice catalog must bind the actual benchmark results relation")
    for source_id, filename in (
        ("reference_manifest", "manifest.json"),
        ("benchmark_results", "bench.csv"),
    ):
        if (ROOT / sources[source_id].path).resolve() != (data / filename).resolve():
            raise ValueError("Slice catalog must bind these actual benchmark input files")
    tracks = [asdict(track) for track in catalog.tracks]
    by_id = {track["track_id"]: track for track in tracks}
    for witness in manifest["tracks"]:
        track = by_id[witness["track_id"]]
        if track["corpus_id"] != witness["source"]:
            raise ValueError("Slice catalog corpus differs from frozen manifest")
        identity = witness["input_identity"]
        wav = identity["canonical_wav"]
        if (
            track["source_audio_sha256"] != identity["source_sha256"]
            or track["canonical_wav_sha256"] != wav["container_sha256"]
            or track["canonical_pcm_sha256"] != wav["pcm_sha256"]
        ):
            raise ValueError("Slice catalog audio identity differs from scored input")
        if (
            track["duration"]["analyzed"]["value"]
            != wav["parameters"]["frames"] / wav["parameters"]["sample_rate"]
        ):
            raise ValueError("Slice duration is not derived from the scored PCM")
    result = descriptive_slices(rows, tracks)
    result["provenance"] = {
        "catalog_sha256": sha(catalog_path.read_bytes()),
        "artifact_index_sha256": base["artifact_index_sha256"],
        "original_metadata_analysis_sha256": sha((data / "metadata-analysis.json").read_bytes()),
        "slice_generator_sha256": sha(Path(__file__).read_bytes()),
        "kind": "additive_metadata_derivative_no_resealed_execution",
    }
    return result


def slice_csv(result: dict[str, Any]) -> str:
    stream = io.StringIO()
    columns = (
        "dimension",
        "slice",
        "detector",
        "n_total",
        "n_valid",
        "n_error",
        "n_abstain",
        "n_exact",
        "exact",
        "mirex",
    )
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    groups = list(result["slices"].items())
    for crossing in ("corpus_by_duration", "corpus_by_mode"):
        groups.extend(
            (crossing + ":" + corpus, values) for corpus, values in result[crossing].items()
        )
    for dimension, slices in groups:
        for label, item in slices.items():
            for name, metrics in item["detectors"].items():
                writer.writerow(
                    {
                        "dimension": dimension,
                        "slice": label,
                        "detector": name,
                        **{
                            field: metrics[field]
                            for field in ("n_total", "n_valid", "n_error", "n_abstain")
                        },
                        "n_exact": metrics["counts"].get("exact", 0),
                        "exact": "NA" if metrics["exact"] is None else metrics["exact"],
                        "mirex": "NA" if metrics["mirex"] is None else metrics["mirex"],
                    }
                )
    return stream.getvalue()


def plot_slices(result: dict[str, Any], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "key-reference-slices-v1"}
    )
    colors = {"drone": "#267A91", "madmom": "#A35D2C", "libkeyfinder": "#666D78"}
    labels = {"drone": "Drone v1", "madmom": "madmom KEY_CNN", "libkeyfinder": "libkeyfinder"}
    destination.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    def save(fig: Figure, stem: str) -> None:
        for suffix in ("png", "svg"):
            path = destination / f"{stem}.{suffix}"
            fig.savefig(path, dpi=180, metadata={"Date": None} if suffix == "svg" else {})
            if suffix == "svg":
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
                )
            outputs[path.name] = sha(path.read_bytes())
        plt.close(fig)

    def dots(axis: Axes, groups: dict[str, Any], metric: str) -> None:
        for i, name in enumerate(DETECTORS):
            values = [
                np.nan
                if item["detectors"][name][metric] is None
                else 100 * item["detectors"][name][metric]
                for item in groups.values()
            ]
            y = [j + (i - 1) * 0.18 for j in range(len(groups))]
            axis.scatter(values, y, color=colors[name], label=labels[name], s=38, zorder=3)
        for j, item in enumerate(groups.values()):
            if item["n"] == 0:
                axis.text(
                    50,
                    j,
                    "NA — no recordings",
                    ha="center",
                    va="center",
                    color="#666D78",
                    fontsize=9,
                )
        axis.set(
            yticks=range(len(groups)),
            yticklabels=[f"{label}  (n={item['n']})" for label, item in groups.items()],
            xlim=(-3, 105),
            xlabel="Exact (%)" if metric == "exact" else "Symmetric-fifths agreement (%)",
        )
        axis.set_ylim(len(groups) - 0.5, -0.5)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(axis="x", alpha=0.15)

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5), layout="constrained", gridspec_kw={"width_ratios": [1.3, 1, 1]}
    )
    corpora = list(result["corpus_by_duration"])
    counts = np.array(
        [
            [result["corpus_by_duration"][corpus][band]["n"] for band in DURATION_SLICES]
            for corpus in corpora
        ]
    )
    axes[0].imshow(counts, cmap="Blues", vmin=0, vmax=max(1, int(counts.max())), aspect="auto")
    for (y, x), value in np.ndenumerate(counts):
        axes[0].text(
            x,
            y,
            str(value),
            ha="center",
            va="center",
            color="white" if value > counts.max() / 2 else "#263746",
            fontsize=12,
        )
    axes[0].set(
        xticks=range(len(DURATION_SLICES)),
        xticklabels=DURATION_SLICES,
        yticks=range(len(corpora)),
        yticklabels=corpora,
        title="Coverage: corpus × analyzed duration",
    )
    for axis, metric in zip(axes[1:], ("exact", "mirex"), strict=True):
        dots(axis, result["slices"]["analyzed_duration"], metric)
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False, fontsize=8
    )
    fig.suptitle(
        "Short previews and repeated classical performances leave long-track gaps",
        fontsize=15,
        fontweight="bold",
    )
    fig.supxlabel(
        "Duration is analyzed audio, not full-song length. Both ≥5-minute items are performances of one SWD work.\nPoint estimates only; corpus/repertoire and duration are confounded. Empty slices are NA.",
        fontsize=10,
    )
    save(fig, "duration-coverage-slices")

    for dimension, stem, title, height in (
        (
            "genre_original",
            "original-genre-slices",
            "Original genre tags retain detail hidden by legacy grouping",
            10,
        ),
        ("key", "key-mode-slices", "Reference-key slices are uneven and descriptive", 12),
    ):
        groups = result["slices"][dimension]
        fig, axes = plt.subplots(1, 2, figsize=(13, height), layout="constrained")
        for axis, metric in zip(axes, ("exact", "mirex"), strict=True):
            dots(axis, groups, metric)
        axes[0].legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False, fontsize=8
        )
        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.supxlabel(
            "All slice denominators are shown; errors and abstentions count as zero.\nNo confidence intervals or population-level strength/weakness claims; repeated works and corpus mix remain.",
            fontsize=10,
        )
        save(fig, stem)
    receipt = {
        "analysis_sha256": sha(encode(result)),
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "matplotlib_version": matplotlib.__version__,
        "outputs": outputs,
    }
    (destination / "slice-figure-inputs.json").write_bytes(encode(receipt))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--seal-index", choices=("historical_invalid", "corrected"))
    parser.add_argument("--publish-corrected", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--slices", action="store_true", help="Replay additive catalog-based slice analysis"
    )
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    args = parser.parse_args()
    if args.slices:
        if args.publish_corrected or args.seal_index:
            raise ValueError("Slice analysis cannot republish or reseal execution artifacts")
        result = analyze_slices(args.data, args.catalog)
        outputs = {
            "slice-analysis.json": encode(result),
            "slice-analysis.csv": slice_csv(result).encode(),
        }
        for name, raw in outputs.items():
            output = args.data / name
            if args.write:
                output.write_bytes(raw)
            elif output.read_bytes() != raw:
                raise ValueError("Published slice analysis differs: " + name)
        if args.plot:
            plot_slices(result, args.data.parent / "figures")
        print(
            json.dumps(
                {"n": result["n"], "sha256": sha(encode(result)), "inference_performed": False}
            )
        )
        return
    if args.publish_corrected:
        publish_corrected(args.data)
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
