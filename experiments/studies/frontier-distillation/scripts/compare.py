"""Paired comparison of two recipes under the frontier decision rules.

Each recipe is three seeds. Per track, correctness is averaged over seeds (single-model
predictions at shift zero), and the per-track difference is bootstrapped over tracks.
Development evidence: FMA valid and holdout, plus the GiantSteps+ development half.
The GiantSteps+ test half is reported and never used for a decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from analyze import SHIFTS, load_panels  # noqa: E402

BOOTSTRAP_SAMPLES = 10000


def is_test_half(track_id: str) -> bool:
    return hashlib.sha256(f"gsplus-split-v1:{track_id}".encode()).digest()[0] % 2 == 1


def seed_correctness(stack: np.ndarray, references: list[int], shift_average: bool) -> np.ndarray:
    logits = stack.mean(2) if shift_average else stack[:, :, SHIFTS.index(0)]
    return (logits.argmax(-1) == np.array(references)[None, :]).mean(0)


def interval(differences: np.ndarray, generator: np.random.Generator) -> tuple[float, float]:
    samples = generator.integers(0, len(differences), (BOOTSTRAP_SAMPLES, len(differences)))
    means = differences[samples].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="npz:group")
    parser.add_argument("--candidate", required=True, help="npz:group")
    parser.add_argument("--shift-average", action="store_true")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    panels = load_panels()
    generator = np.random.default_rng(20261114)
    stacks = {}
    for role, specification in (
        ("baseline", arguments.baseline),
        ("candidate", arguments.candidate),
    ):
        path, group = specification.split(":")
        stacks[role] = np.load(path)
        stacks[role + "_group"] = group
    rows = {}
    pools: dict[str, list[np.ndarray]] = {
        "development": [],
        "fma_development": [],
        "gsplus_development": [],
    }
    for panel in ("valid", "holdout", "gsplus_primary", "gsplus_clean"):
        _, references, track_ids = panels[panel]
        baseline = seed_correctness(
            stacks["baseline"][f"{stacks['baseline_group']}/{panel}"],
            references,
            arguments.shift_average,
        )
        candidate = seed_correctness(
            stacks["candidate"][f"{stacks['candidate_group']}/{panel}"],
            references,
            arguments.shift_average,
        )
        difference = candidate - baseline
        if panel.startswith("gsplus"):
            test_mask = np.array([is_test_half(track) for track in track_ids])
            parts = {f"{panel}_development": ~test_mask, f"{panel}_test": test_mask}
        else:
            parts = {panel: np.ones(len(track_ids), dtype=bool)}
        for name, mask in parts.items():
            low, high = interval(difference[mask], generator)
            rows[name] = {
                "tracks": int(mask.sum()),
                "baseline_exact": round(100 * float(baseline[mask].mean()), 2),
                "candidate_exact": round(100 * float(candidate[mask].mean()), 2),
                "difference": round(100 * float(difference[mask].mean()), 2),
                "interval_95": [round(100 * low, 2), round(100 * high, 2)],
            }
        if panel in ("valid", "holdout"):
            pools["development"].append(difference)
            pools["fma_development"].append(difference)
        if panel == "gsplus_primary":
            pools["development"].append(difference[~test_mask])
            pools["gsplus_development"].append(difference[~test_mask])
    pooled_intervals: dict[str, tuple[float, float]] = {}
    for name, parts_list in pools.items():
        combined = np.concatenate(parts_list)
        low, high = interval(combined, generator)
        pooled_intervals[name] = (round(100 * low, 2), round(100 * high, 2))
        rows[f"pooled_{name}"] = {
            "tracks": len(combined),
            "difference": round(100 * float(combined.mean()), 2),
            "interval_95": list(pooled_intervals[name]),
        }
    development_low, development_high = pooled_intervals["development"]
    if development_low > 0:
        decision = "adopt"
    elif development_high < 0:
        decision = "reject"
    else:
        decision = "inconclusive"
    report = {
        "baseline": arguments.baseline,
        "candidate": arguments.candidate,
        "shift_average": arguments.shift_average,
        "rows": rows,
        "decision": decision,
    }
    Path(arguments.out).write_text(json.dumps(report, indent=1))
    for name, row in rows.items():
        print(f"{name:28s} {json.dumps(row)}")
    print("decision:", decision)


if __name__ == "__main__":
    main()
