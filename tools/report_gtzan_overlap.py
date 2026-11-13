"""Verify and report the completed GTZAN overlap audit; never run audio or models."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.bp_gs_scope import classify, outcome
from tools.import_gtzan_reference import GENRES
from tools.prepare_gtzan_overlap import OUTPUT as MANIFEST

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/studies/v2-data-contract"
DATA = STUDY / "data/gtzan-overlap"
INDEX_SHA256 = "54b4eed2ae4925f30b13991448c45b827c7aa4b985970639fc8e06e25bf178d6"
PAYLOAD_SHA256 = "b9d2ad4910a56f11f7d4380702af13b878c41e2cde30838004eb79767c8a70aa"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "recordings": len(rows),
        "complete": sum(row["complete"] for row in rows),
        "outcomes": dict(sorted(Counter(row["outcome"] for row in rows).items())),
    }


def check_rows(payload: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Reject missing/repeated membership, retry substitutions and altered decisions."""
    rows = payload["rows"]
    primary = {r["id"] for r in manifest["queries"]}
    sensitivity = {r["id"]: r for r in manifest["sensitivity_queries"]}
    ids = [row["id"] for row in rows]
    if len(rows) != 837 or len(set(ids)) != 837 or set(ids) != set(sensitivity):
        raise ValueError("Audit must retain all 837 frozen IDs exactly once")
    if ids != payload["archive_order_ids"]:
        raise ValueError("Audit order differs from frozen archive order")
    if (
        payload["admitted_for_scoring"] is not False
        or payload["classifier_outputs_used"] is not False
    ):
        raise ValueError("Overlap results do not authorize classifier scoring")
    if payload["labels_scored"] is not False or payload["rule"] != {
        "identity_anchors": 20,
        "identity_span_seconds": 8,
        "identity_chromaprint_max": 0.15,
        "review_anchors": 10,
        "review_span_seconds": 5,
    }:
        raise ValueError("Numerical decision policy changed")
    if len(payload["dictionary"]) != 7579:
        raise ValueError("Reference dictionary is incomplete")
    dictionary = {r["ref_id"]: r["id"] for r in payload["dictionary"]}
    if set(dictionary) != set(range(7579)):
        raise ValueError("Reference dictionary identities differ")
    for i, row in enumerate(rows):
        if row["primary769"] != (row["id"] in primary):
            raise ValueError("Primary membership changed")
        if row["execution"] != ("original" if i < 377 else "continuation"):
            raise ValueError("Original prefix was retried or replaced")
        if type(row["complete"]) is not bool or type(row["informative"]) is not bool:
            raise ValueError("Invalid query completion flags")
        decisions = []
        for match in row["supported_alignments"]:
            decision = classify(match, match["chromaprint_distance"])
            if decision != match["decision"] or decision == "below_screening_support":
                raise ValueError("Supported alignment decision changed")
            if match["ref_id"] not in dictionary:
                raise ValueError("Unknown reference in supported alignment")
            decisions.append(decision)
        if row["below_screening_support"] < 0 or (
            row["alignments"] != len(decisions) + row["below_screening_support"]
        ):
            raise ValueError("Alignment accounting is incomplete")
        expected = outcome(
            exact=False,
            complete=row["complete"],
            informative=row["informative"],
            decisions=decisions,
        )
        if row["outcome"] != expected:
            raise ValueError("Query outcome changed; incomplete is not a clean negative")
        if "extraction" in row:
            extracted = row["extraction"]
            if extracted["container_sha256"] != sensitivity[row["id"]]["wav_sha256"]:
                raise ValueError("Extraction belongs to different acquired audio")
            if not math.isfinite(extracted["seconds"]) or extracted["seconds"] <= 0:
                raise ValueError("Invalid analyzed duration")
    failed = [(row["id"], row["reason"]) for row in rows if not row["complete"]]
    if failed != [("reggae.00086", "vote_memory_budget")]:
        raise ValueError("Original incomplete recording must remain unresolved")
    if any(not row["complete"] for row in rows[377:]):
        raise ValueError("Continuation contains an incomplete recording")
    return rows


def summarize(payload: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    rows = check_rows(payload, manifest)
    groups = {"primary769": [r for r in rows if r["primary769"]], "sensitivity837": rows}
    result: dict[str, Any] = {
        "populations": {name: counts(group) for name, group in groups.items()},
        "genres": {
            name: {
                genre: counts([r for r in group if r["id"].split(".")[0] == genre])
                for genre in GENRES
            }
            for name, group in groups.items()
        },
        "admitted_for_scoring": False,
        "classifier_accuracy_measured": False,
        "completed_new_queries": len(rows[377:]),
        "preserved_original_attempts": len(rows[:377]),
        "review_alignment_count": sum(len(r["supported_alignments"]) for r in rows),
        "below_screening_alignment_count": sum(r["below_screening_support"] for r in rows),
    }
    aliases = manifest["release_exposure"]["known_exact_exposure_aliases"]
    exposure = {
        role: {row["ref_id"] for row in aliases if role in row["release_roles"]}
        for role in ("fitting", "selection")
    }
    result["release_exposure_review"] = {
        name: {
            role: sum(
                any(match["ref_id"] in ref_ids for match in row["supported_alignments"])
                for row in group
            )
            for role, ref_ids in exposure.items()
        }
        for name, group in groups.items()
    }
    durations = [r["extraction"]["seconds"] for r in rows if "extraction" in r]
    result["analyzed_duration_seconds"] = {
        "observed": len(durations),
        "missing": len(rows) - len(durations),
        "minimum": min(durations),
        "median": statistics.median(durations),
        "maximum": max(durations),
    }
    return result


def verify_raw(payload: dict[str, Any], path: Path) -> None:
    """Optional exhaustive decision replay from retained metadata, without fingerprints."""
    with tarfile.open(path) as archive:
        for i, row in enumerate(payload["rows"]):
            name = f"{row['execution']}/evidence/{i:06}.json.gz"
            stream = archive.extractfile(name)
            if stream is None:
                raise ValueError("Missing raw query evidence")
            raw = stream.read(row["raw_evidence_bytes"] + 1)
            if len(raw) != row["raw_evidence_bytes"] or sha(raw) != row["evidence_sha256"]:
                raise ValueError("Raw query evidence checksum differs")
            decoded = gzip.decompress(raw)
            if len(decoded) != row["raw_decoded_bytes"] or sha(decoded) != row["decoded_sha256"]:
                raise ValueError("Decoded query evidence checksum differs")
            evidence = json.loads(decoded)
            matched = evidence["match"]
            supported = []
            for match in matched["matches"]:
                if classify(match, match["chromaprint_distance"]) != match["decision"]:
                    raise ValueError("Original alignment classification differs")
                if match["decision"] != "below_screening_support":
                    supported.append(match)
            if (
                evidence["id"] != row["id"]
                or matched["complete"] != row["complete"]
                or matched.get("reason") != row["reason"]
                or supported != row["supported_alignments"]
                or len(matched["matches"]) != row["alignments"]
            ):
                raise ValueError("Projected summary differs from original query evidence")


def verify() -> tuple[dict[str, Any], dict[str, Any]]:
    raw = (DATA / "audit.json.gz").read_bytes()
    if sha(raw) != PAYLOAD_SHA256:
        raise ValueError("Published audit projection changed")
    payload = json.loads(gzip.decompress(raw))
    if sha(MANIFEST.read_bytes()) != payload["query_manifest_sha256"]:
        raise ValueError("Frozen query manifest changed")
    manifest = json.loads(gzip.decompress(MANIFEST.read_bytes()))
    index_raw = (DATA / "index.json").read_bytes()
    if sha(index_raw) != INDEX_SHA256:
        raise ValueError("Published index changed")
    index = json.loads(index_raw)
    for name, expected in index["published_artifacts"].items():
        if Path(name).name != name or sha((DATA / name).read_bytes()) != expected:
            raise ValueError("Published evidence changed: " + name)
    result = summarize(payload, manifest)
    guardian = json.loads((DATA / "continuation-controller.json").read_bytes())
    if guardian["result_sha256"] != index["private_originals"]["continuation_result"]:
        raise ValueError("Controller result binding changed")
    if guardian["status"] != "complete" or guardian["mode"] != "audit":
        raise ValueError("Continuation did not complete as an audit")
    qualification = json.loads((DATA / "qualification-summary.json").read_bytes())
    controls = qualification["rows"]
    if len(controls) != 250 or len({row["id"] for row in controls}) != 250:
        raise ValueError("Qualification control identities are incomplete")
    if not all(row["complete"] for row in controls):
        raise ValueError("Qualification contains an incomplete control")
    observed_controls = {
        "cached_positive": len(controls[:180]),
        "cached_positive_retrieved": sum(row["expected_retrieved"] for row in controls[:180]),
        "negative": len(controls[180:244]),
        "negative_identity": sum(row["identity_flagged"] for row in controls[180:244]),
        "negative_review": sum(row["review_flagged"] for row in controls[180:244]),
        "fresh_positive": len(controls[244:]),
        "fresh_positive_retrieved": sum(row["expected_retrieved"] for row in controls[244:]),
    }
    if qualification["counts"] != observed_controls:
        raise ValueError("Qualification summary differs from all 250 control outcomes")
    if qualification["counts"] != {
        "cached_positive": 180,
        "cached_positive_retrieved": 180,
        "negative": 64,
        "negative_identity": 0,
        "negative_review": 5,
        "fresh_positive": 6,
        "fresh_positive_retrieved": 6,
    }:
        raise ValueError("Qualification control counts changed")
    for name in ("original-qualification-controller.json", "qualification-controller.json"):
        controller = json.loads((DATA / name).read_bytes())
        if controller["status"] != "complete" or controller["qualification_passed"] is not True:
            raise ValueError("Qualification did not pass")
    failed = json.loads((DATA / "preparation-failure.json").read_bytes())
    if failed["status"] != "preparation_failed" or failed["continuation_launched"] is not False:
        raise ValueError("Preserved preparation failure changed")
    result["continuation_resources"] = {
        "wall_seconds": guardian["wall_seconds"],
        **guardian["peaks"],
    }
    result["qualification"] = qualification["counts"]
    return payload, result


def plot(payload: dict[str, Any], result: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "drone-gtzan-overlap"}
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={"width_ratios": [1.4, 1.4, 1]})
    styles = [
        ("no_detected_overlap", "No detected overlap", "#387D80"),
        ("review", "Needs review", "#CC8B3C"),
        ("insufficient_information", "Incomplete", "#A34C51"),
    ]
    for axis, population, title in zip(
        axes[:2],
        ["primary769", "sensitivity837"],
        ["Primary cohort · 769 excerpts", "Sensitivity cohort · 837 excerpts"],
        strict=True,
    ):
        group = result["genres"][population]
        left = [0] * len(GENRES)
        for key, _, color in styles:
            values = [group[g]["outcomes"].get(key, 0) for g in GENRES]
            axis.barh(range(len(GENRES)), values, left=left, color=color, height=0.68)
            for i, value in enumerate(values):
                if value >= 5:
                    axis.text(
                        left[i] + value / 2,
                        i,
                        str(value),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=9,
                    )
            left = [a + b for a, b in zip(left, values, strict=True)]
        for i, total in enumerate(left):
            axis.text(total + 1.5, i, str(total), va="center", fontsize=9)
        axis.set_yticks(
            range(len(GENRES)), ["Hip-hop" if g == "hiphop" else g.title() for g in GENRES]
        )
        axis.invert_yaxis()
        axis.set_xlim(0, 112)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Recordings (zero classical coverage)")
        axis.spines[["top", "right", "left"]].set_visible(False)
    durations = [r["extraction"]["seconds"] for r in payload["rows"] if "extraction" in r]
    axes[2].hist(
        durations, bins=[29.9, 30.0, 30.05, 30.2, 30.4, 30.7], color="#387D80", edgecolor="white"
    )
    axes[2].set_title("Actual analyzed audio", loc="left", fontweight="bold")
    axes[2].set_xlabel("Seconds · 836 observed; one missing")
    axes[2].set_ylabel("Recordings")
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.legend(
        handles=[Patch(color=c, label=n) for _, n, c in styles],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "GTZAN overlap audit: review flags remain unresolved",
        x=0.05,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.05,
        0.09,
        "No confirmed identities. These are fingerprint outcomes, not key accuracy or scoring clearance.\nOne incomplete reggae query is preserved. These excerpts do not establish long-song or contemporary DJ coverage.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.92))
    folder = STUDY / "figures"
    folder.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        path = folder / f"gtzan-overlap.{extension}"
        fig.savefig(path, dpi=160, metadata={"Date": None} if extension == "svg" else {})
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)
    receipt = {
        "audit_sha256": PAYLOAD_SHA256,
        "renderer_sha256": sha(Path(__file__).read_bytes()),
        "files": {
            f"gtzan-overlap.{extension}": sha((folder / f"gtzan-overlap.{extension}").read_bytes())
            for extension in ("png", "svg")
        },
    }
    (STUDY / "data/gtzan-overlap-figures.json").write_bytes(encode(receipt))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-audit",
        type=Path,
        help="Optional retained metadata tar for exhaustive alignment replay",
    )
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    payload, result = verify()
    if args.raw_audit:
        verify_raw(payload, args.raw_audit)
        result["raw_alignment_replay"] = "passed"
    if args.plot:
        plot(payload, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
