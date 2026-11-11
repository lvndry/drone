"""Validate completed small-model experiment records and summarize training seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def summarize(screen: dict[str, Any], confirmation: dict[str, Any]) -> dict[str, Any]:
    for key in ("source_sha256", "manifest_sha256", "split_sha256", "split_policy"):
        if screen[key] != confirmation[key]:
            raise ValueError(f"Screen and confirmation differ: {key}")
    if screen["status"] != "complete" or confirmation["status"] != "complete":
        raise ValueError("Both experiment stages must complete before a final summary")
    original = {r["identity"]["recipe"]["name"]: r for r in screen["results"]}
    grouped = {}
    for run in confirmation["results"]:
        if run["status"] != "complete":
            raise ValueError("Incomplete confirmation run")
        name, seed = run["identity"]["recipe"]["name"], run["identity"]["seed"]
        grouped.setdefault(name, {})[seed] = run
        if seed == 13:
            control = original[name]
            if control["history"] != run["history"] or control["validation"] != run["validation"]:
                raise ValueError(f"Seed 13 failed to reproduce: {name}")
    rows = []
    for name, seeds in grouped.items():
        if set(seeds) != {13, 37, 73}:
            raise ValueError(f"Wrong confirmation seeds for {name}")
        scores = [seeds[s]["validation"]["mirex"] for s in (13, 37, 73)]
        rows.append(
            {
                "recipe": name,
                "parameters": seeds[13]["parameters"],
                "validation_mirex_mean": float(np.mean(scores)),
                "validation_mirex_sample_sd": float(np.std(scores, ddof=1)),
                "validation_exact_mean": float(
                    np.mean([r["validation"]["exact"] for r in seeds.values()])
                ),
                "mirex_seed13": scores[0],
                "mirex_seed37": scores[1],
                "mirex_seed73": scores[2],
                "mode_accuracy_mean": float(
                    np.mean([r["validation"]["mode_accuracy"] for r in seeds.values()])
                ),
                "total_training_seconds": sum(r["elapsed_seconds"] for r in seeds.values()),
            }
        )
    rows.sort(key=lambda r: (r["validation_mirex_mean"], r["validation_exact_mean"]), reverse=True)
    return {
        "selection": "Highest mean validation symmetric MIREX over three fixed seeds; exact tie-breaker",
        "canonical_artifact_seed": 13,
        "selected_recipe": rows[0]["recipe"],
        "confirmation": rows,
        "screening": [
            {
                "recipe": r["identity"]["recipe"]["name"],
                "mirex": r["validation"]["mirex"],
                "exact": r["validation"]["exact"],
                "best_epoch": r["best_epoch"],
                "seconds": r["elapsed_seconds"],
            }
            for r in screen["results"]
        ],
        "total_training_seconds": sum(r["elapsed_seconds"] for r in screen["results"])
        + sum(r["elapsed_seconds"] for r in confirmation["results"]),
        "manifest_sha256": screen["manifest_sha256"],
        "split_sha256": screen["split_sha256"],
        "source_sha256": screen["source_sha256"],
        "caveat": "Validation was used for selection; seed spread is not external-data uncertainty.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        json.loads(args.screen.read_text()), json.loads(args.confirmation.read_text())
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "confirmation.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["confirmation"][0]))
        writer.writeheader()
        writer.writerows(result["confirmation"])
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k in ("selected_recipe", "confirmation", "total_training_seconds")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
