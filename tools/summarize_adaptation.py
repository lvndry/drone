"""Verify and summarize the fixed three-seed adaptation confirmation."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    args = parser.parse_args()
    seeds = (13, 37, 73)
    runs = [read(args.root / f"seed{s}/seed{s}/run.json") for s in seeds]
    results = [read(args.root / f"seed{s}/result.json") for s in seeds]
    splits = [read(args.root / f"seed{s}/split.json") for s in seeds]
    pilot = read(args.pilot / "seed13/run.json")
    assert all(run["status"] == "complete" for run in runs)
    assert [run["identity"]["seed"] for run in runs] == list(seeds)
    identities = [dict(run["identity"]) for run in runs]
    for identity in identities:
        identity.pop("seed")
    assert identities[0] == identities[1] == identities[2]
    assert splits[0] == splits[1] == splits[2] == read(args.pilot / "split.json")
    assert pilot["history"] == runs[0]["history"]
    assert pilot["validation"]["predictions"] == runs[0]["validation"]["predictions"]
    assert all(not row["calibration_used"] and not row["final_holdout_used"] for row in results)
    baseline = read(args.root / "seed13/baseline.json")
    domains = {}
    for name in ("beatport", "fmak"):
        values = [row["metrics"][name]["mirex"] for row in results]
        domains[name] = {
            "baseline_mirex": baseline[name]["mirex"],
            "seed_mirex": dict(zip(map(str, seeds), values, strict=True)),
            "mirex_mean": statistics.mean(values),
            "mirex_sample_sd": statistics.stdev(values),
            "exact_mean": statistics.mean(row["metrics"][name]["exact"] for row in results),
            "delta_mirex_mean": statistics.mean(values) - baseline[name]["mirex"],
        }
    canonical = args.root / "seed13/seed13/best.pt"
    result = {
        "source_sha256": runs[0]["identity"]["source_sha256"],
        "split": splits[0],
        "seed13_reproduces_pilot": True,
        "domains": domains,
        "passes_mean_development_gate": domains["fmak"]["delta_mirex_mean"] >= 0.02
        and domains["beatport"]["delta_mirex_mean"] >= -0.02,
        "canonical_seed": 13,
        "canonical_checkpoint_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest(),
        "confirmation_elapsed_seconds": sum(run["elapsed_seconds"] for run in runs),
        "pilot_elapsed_seconds": pilot["elapsed_seconds"],
        "final_holdout_used": False,
        "calibration_used": False,
        "limitations": "Development validation selected checkpoints; SD measures seed variation. Reference is one frozen seed13 artifact, not a paired three-seed control.",
    }
    (args.root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "domains",
                    "passes_mean_development_gate",
                    "confirmation_elapsed_seconds",
                    "pilot_elapsed_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
