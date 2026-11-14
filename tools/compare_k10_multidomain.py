"""Apply the declared development gate only to a comparable completed pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def compare(control: dict[str, Any], joint: dict[str, Any]) -> dict[str, Any]:
    for run in (control, joint):
        if run["status"] != "complete":
            raise ValueError("Both arms must complete before comparison")
        if run["calibration_used"] or run["final_holdout_used"]:
            raise ValueError("The development boundary was crossed")
    if control["arm"] != "control" or joint["arm"] != "joint":
        raise ValueError("Need the declared control and joint arms")
    for key in ("seed", "config", "source_sha256", "feature_manifest_sha256", "validation_ids"):
        if control[key] != joint[key]:
            raise ValueError(f"Arms differ in {key}, beyond their fitting data")
    control_fit = set(map(str, control["fit_ids"]))
    joint_fit = set(map(str, joint["fit_ids"]))
    if not control_fit < joint_fit or not all(
        track_id.startswith("beatport:") for track_id in control_fit
    ):
        raise ValueError("The control must fit the same Beatport subset only")
    if not all(track_id.startswith("fmak:") for track_id in joint_fit - control_fit):
        raise ValueError("The candidate may add only FMAK fitting rows")
    delta: dict[str, float] = {}
    for domain in ("beatport", "fmak"):
        control_metrics = control["domain_metrics"][domain]
        joint_metrics = joint["domain_metrics"][domain]
        if (
            control_metrics["ids"] != joint_metrics["ids"]
            or control_metrics["tracks"] != joint_metrics["tracks"]
        ):
            raise ValueError("Domain validation identities differ")
        delta[domain] = float(joint_metrics["mirex"]) - float(control_metrics["mirex"])
    return {
        "seed": control["seed"],
        "delta_mirex": delta,
        "qualifies_for_confirmation": delta["fmak"] >= 0.02 and delta["beatport"] >= -0.02,
        "calibration_used": False,
        "final_holdout_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.control.read_bytes()), json.loads(args.joint.read_bytes()))
    result["run_sha256"] = {
        "control": hashlib.sha256(args.control.read_bytes()).hexdigest(),
        "joint": hashlib.sha256(args.joint.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
