"""Does mixing a second model family into the shift12 ensemble beat shift12 alone?

Each family's seeds and shifts are averaged into one probability per track; the
mix averages the two families with a weight. Correctness is compared with
shift12 track by track, with a paired bootstrap on the development pool (FMA
valid and holdout plus the GiantSteps+ development half) and on the GiantSteps+
test half, which chooses nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from compare import interval, is_test_half
from frontier_train import load_manifest
from score_madmom import panel_references

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def family_probabilities(path: str, group: str, panel: str) -> np.ndarray:
    log_probabilities = np.load(path)[f"{group}/{panel}"]
    return np.exp(log_probabilities).mean(axis=(0, 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="analysis-r6.npz:shift12")
    parser.add_argument("--candidate", required=True, help="npz:group")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    baseline_path, baseline_group = arguments.baseline.split(":")
    candidate_path, candidate_group = arguments.candidate.split(":")
    manifest = load_manifest()
    references = {
        panel: [int(row["key_label"]) for row in manifest[panel]] for panel in ("valid", "holdout")
    }
    giantsteps = panel_references()
    generator = np.random.default_rng(0)
    report: dict = {"baseline": arguments.baseline, "candidate": arguments.candidate, "weights": {}}
    for weight in WEIGHTS:
        development = []
        test = []
        rows = {}
        for panel in ("valid", "holdout", "gsplus_primary", "gsplus_clean"):
            baseline = family_probabilities(baseline_path, baseline_group, panel)
            candidate = family_probabilities(candidate_path, candidate_group, panel)
            mixed = (1 - weight) * baseline + weight * candidate
            if panel in references:
                panel_references_list = references[panel]
                test_mask = np.zeros(len(panel_references_list), dtype=bool)
            else:
                track_ids, panel_references_list = giantsteps[panel]
                test_mask = np.array([is_test_half(track) for track in track_ids])
            truth = np.array(panel_references_list)
            difference = (mixed.argmax(-1) == truth).astype(float) - (
                baseline.argmax(-1) == truth
            ).astype(float)
            rows[panel] = round(100 * float((mixed.argmax(-1) == truth).mean()), 2)
            if panel != "gsplus_clean":
                development.append(difference[~test_mask])
            if panel == "gsplus_clean":
                test.append(difference[test_mask])
        pooled = np.concatenate(development)
        test_difference = np.concatenate(test)
        low, high = interval(pooled, generator)
        test_low, test_high = interval(test_difference, generator)
        report["weights"][str(weight)] = {
            "exact": rows,
            "development_difference": round(100 * float(pooled.mean()), 2),
            "development_interval_95": [round(100 * low, 2), round(100 * high, 2)],
            "clean_test_difference": round(100 * float(test_difference.mean()), 2),
            "clean_test_interval_95": [round(100 * test_low, 2), round(100 * test_high, 2)],
        }
        print(
            f"weight {weight:.2f}: {rows} | development vs baseline {100 * pooled.mean():+.2f} [{100 * low:+.2f}, {100 * high:+.2f}] | clean test {100 * test_difference.mean():+.2f} [{100 * test_low:+.2f}, {100 * test_high:+.2f}]"
        )
    Path(arguments.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
