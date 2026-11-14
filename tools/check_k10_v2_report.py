"""Reproduce paired report arithmetic from frozen predictions; never load audio/models."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from drone.keys import WEIGHTS, key_relation
from experiments.k10_v2_pilot import pilot_gate

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/studies/k10-v2-pilot"
BINDINGS = (
    "plan_sha256",
    "feature_manifest_sha256",
    "initial_state_sha256",
    "bp_trace_sha256",
    "source_sha256",
    "runtime_versions",
    "device",
    "config",
)


def read(path: Path) -> Any:  # noqa: ANN401 — heterogeneous frozen JSON evidence
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def bootstrap(rows: list[dict[str, Any]], scheme: str) -> list[float]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        delta = (
            float(row["exact_change"])
            if scheme == "exact"
            else WEIGHTS[row["expanded_relation_" + scheme]]
            - WEIGHTS[row["restricted_relation_" + scheme]]
        )
        groups.setdefault(row["group"], []).append(delta)
    sums = np.array([sum(values) for values in groups.values()])
    counts = np.array([len(values) for values in groups.values()])
    draws = np.random.default_rng(20261112).integers(0, len(groups), (10000, len(groups)))
    distribution = sums[draws].sum(1) / counts[draws].sum(1)
    return np.quantile(distribution, [0.025, 0.975]).tolist()


def validate_rows(
    rows: list[dict[str, Any]],
    runs: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    require(len(rows) == len({track["id"] for track in rows}) == 376, "Wrong paired panel")
    for domain, count in (("beatport", 126), ("fmak", 250)):
        selected = [track for track in rows if track["domain"] == domain]
        require(len(selected) == count, "Wrong domain denominator")
        for arm, run in runs.items():
            metrics = run["best_metrics"][domain]
            require([track["id"] for track in selected] == metrics["ids"], "ID order differs")
            require(
                [track[arm + "_pred"] for track in selected] == metrics["predictions"],
                "Recorded predictions differ",
            )
            for row in selected:
                reference = plan["expected_refs"][row["id"]]
                prediction = row[arm + "_pred"]
                require(row["ref"] == reference, "Pinned target differs")
                require(row[arm + "_correct"] == (prediction == reference), "Exact flag differs")
                for scheme in ("sym", "asym"):
                    require(
                        row[arm + "_relation_" + scheme]
                        == key_relation(prediction, reference, symmetric=scheme == "sym"),
                        "Relation differs",
                    )
            for scheme, metric in (("sym", "mirex"), ("asym", "mirex_asymmetric")):
                score = (
                    sum(WEIGHTS[track[arm + "_relation_" + scheme]] for track in selected) / count
                )
                require(abs(score - metrics[metric]) < 1e-12, "Weighted score differs")
            require(
                sum(track[arm + "_correct"] for track in selected) == metrics["exact_correct"],
                "Exact count differs",
            )
    for row in rows:
        require(
            row["exact_change"] == int(row["expanded_correct"]) - int(row["restricted_correct"]),
            "Error movement differs",
        )


def check(study: Path = STUDY) -> dict[str, Any]:
    data = study / "data"
    results = study / "data/results"
    for name, record in read(results / "archive-index.json").items():
        raw = (data / name).read_bytes()
        decoded = gzip.decompress(raw) if name.endswith(".gz") else raw
        require(hashlib.sha256(raw).hexdigest() == record["file_sha256"], "File hash: " + name)
        require(
            hashlib.sha256(decoded).hexdigest() == record["decoded_sha256"], "Decoded hash: " + name
        )
    runs = {arm: read(results / (arm + ".json.gz")) for arm in ("restricted", "expanded")}
    old, new = runs.values()
    for name in (
        "experiments/k10_v2_pilot.py",
        "drone/features.py",
        "drone/model.py",
        "drone/keys.py",
    ):
        require(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == old["source_sha256"][name],
            "Numerical source changed: " + name,
        )
    for key in BINDINGS:
        require(old[key] == new[key], "Pair binding differs: " + key)
    for run in runs.values():
        require(run["updates"] == 8280 and len(run["history"]) == 120, "Incomplete horizon")
        best = max(
            run["history"], key=lambda entry: (entry["selection_exact"], entry["selection_mirex"])
        )
        require(best["update"] == run["best_update"], "Selector differs")
        require(not run["calibration_used"] and not run["confirmation_used"], "Reserved exposure")
    require(pilot_gate(old, new) == read(results / "gate.json"), "Gate differs")
    require(not pilot_gate(old, new)["confirmation_earned"], "Historical failed gate changed")
    rows = read(results / "selected-per-id-errors.json.gz")
    plan = read(data / "canonical-eligibility-v2/plan-v2.json.gz")
    manifest = read(data / "canonical-eligibility-v2/feature-manifest-v2.json.gz")
    require(
        hashlib.sha256((data / "canonical-eligibility-v2/plan-v2.json.gz").read_bytes()).hexdigest()
        == old["plan_sha256"],
        "Executed plan differs",
    )
    require(
        hashlib.sha256(
            gzip.decompress(
                (data / "canonical-eligibility-v2/feature-manifest-v2.json.gz").read_bytes()
            )
        ).hexdigest()
        == old["feature_manifest_sha256"],
        "Executed feature manifest differs",
    )
    validate_rows(rows, runs, plan)
    metadata = {track["id"]: track for track in manifest["rows"]}
    original = read(ROOT / "experiments/studies/v2-data-contract/data/input.json.gz")
    groups = {track["identity"]: track for track in original["tracks"]}
    house = {
        track["id"]
        for track in read(results / "house-metadata-witness.json.gz")["house_metadata_rows"]
        if track["house"]
    }
    summary = read(results / "summary.json.gz")
    for row in rows:
        samples = metadata[row["id"]]["canonical_samples"]
        band = (
            "<2min"
            if samples < 120 * 22050
            else "2–<5min"
            if samples < 300 * 22050
            else "5–10min"
            if samples <= 600 * 22050
            else ">10min"
        )
        require(
            row["canonical_seconds"] == samples / 22050 and row["duration_bin"] == band,
            "Duration source differs",
        )
        require(row["taxonomy_house_182"] == (row["id"] in house), "House metadata differs")
        if row["domain"] == "fmak":
            require(
                row["group"] == groups[row["id"]]["component"]
                and row["genre_top"] == groups[row["id"]]["genre"],
                "Group/genre source differs",
            )
    uncertainty: dict[str, Any] = {}
    for domain in ("beatport", "fmak"):
        panel = [track for track in rows if track["domain"] == domain]
        require(
            np.allclose(
                bootstrap(panel, "exact"),
                summary["domains"][domain]["paired_exact_delta_ci95"],
                atol=1e-14,
            ),
            "Exact interval differs",
        )
        uncertainty[domain] = {
            scheme: {
                "delta": sum(
                    track["exact_change"]
                    if scheme == "exact"
                    else WEIGHTS[track["expanded_relation_" + scheme]]
                    - WEIGHTS[track["restricted_relation_" + scheme]]
                    for track in panel
                )
                / len(panel),
                "ci95": bootstrap(panel, scheme),
            }
            for scheme in ("exact", "sym", "asym")
        }
        for scheme in ("sym", "asym"):
            actual = Counter(
                (track["restricted_relation_" + scheme], track["expanded_relation_" + scheme])
                for track in panel
            )
            saved = {
                (movement["restricted_relation"], movement["expanded_relation"]): movement["tracks"]
                for movement in summary["relation_movements"][domain][scheme]
            }
            require(actual == saved, "Movement matrix differs")
        for band, saved in summary["duration_domains"][domain].items():
            subset = [track for track in panel if track["duration_bin"] == band]
            require(
                len(subset) == saved["tracks"]
                and len({track["group"] for track in subset}) == saved["groups"],
                "Stratum count differs",
            )
    return {
        "method": "10000 paired group bootstrap draws; fixed seed 20261112; track-weighted",
        "interpretation": "Descriptive selected-validation intervals, not fresh test inference; "
        "BP record groups do not establish artist independence",
        "domains": uncertainty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = check()
    path = STUDY / "data/results/paired-uncertainty.json"
    if args.write:
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        require(result == read(path), "Uncertainty artifact differs")
    print(json.dumps({"status": "PASS", **result}, indent=2))


if __name__ == "__main__":
    main()
