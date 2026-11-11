"""Summarize the predeclared three paired seeds without choosing a lucky seed."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from tools.compare_k10_multidomain import compare


def summarize(pairs):
    if sorted(pairs) != [13, 37, 73]:
        raise ValueError("Need exactly the three declared paired seeds")
    reference = pairs[13][0]
    results = {}
    for seed, (control, joint) in pairs.items():
        if control["seed"] != seed or joint["seed"] != seed:
            raise ValueError("Seed label and run disagree")
        for run in (control, joint):
            for key in ("config", "source_sha256", "feature_manifest_sha256", "validation_ids"):
                if run[key] != reference[key]:
                    raise ValueError(f"Confirmation changed {key}")
        results[str(seed)] = compare(control, joint)
    means = {}
    for arm_index, arm in enumerate(("control", "joint")):
        means[arm] = {}
        for domain in ("beatport", "fmak"):
            values = [pairs[s][arm_index]["domain_metrics"][domain]["mirex"] for s in (13, 37, 73)]
            means[arm][domain] = dict(
                mean=statistics.mean(values), sample_std=statistics.stdev(values)
            )
    delta = {
        d: means["joint"][d]["mean"] - means["control"][d]["mean"] for d in ("beatport", "fmak")
    }
    qualifies = delta["fmak"] >= 0.02 and delta["beatport"] >= -0.02
    return dict(
        seeds=results,
        mirex=means,
        delta_mean_mirex=delta,
        qualifies_for_external_freeze=qualifies,
        canonical_seed=13 if qualifies else None,
        calibration_used=False,
        final_holdout_used=False,
        uncertainty="Sample SD over training seeds; not uncertainty over new recordings",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    pairs, hashes = {}, {}
    for seed in (13, 37, 73):
        runs = []
        for arm in ("control", "joint"):
            path = args.root / f"{arm}-seed{seed}" / "run.json"
            raw = path.read_bytes()
            runs.append(json.loads(raw))
            hashes[f"{arm}-seed{seed}"] = hashlib.sha256(raw).hexdigest()
        pairs[seed] = runs
    result = summarize(pairs)
    result["run_sha256"] = hashes
    result["summary_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.root / "confirmation-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
