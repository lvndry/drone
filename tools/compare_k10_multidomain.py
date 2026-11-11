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
    control_fit, joint_fit = set(map(str, control["fit_ids"])), set(map(str, joint["fit_ids"]))
    if not control_fit < joint_fit or not all(i.startswith("beatport:") for i in control_fit):
        raise ValueError("The control must fit the same Beatport subset only")
    if not all(i.startswith("fmak:") for i in joint_fit - control_fit):
        raise ValueError("The candidate may add only FMAK fitting rows")
    delta: dict[str, float] = {}
    for domain in ("beatport", "fmak"):
        a, b = control["domain_metrics"][domain], joint["domain_metrics"][domain]
        if a["ids"] != b["ids"] or a["tracks"] != b["tracks"]:
            raise ValueError("Domain validation identities differ")
        delta[domain] = float(b["mirex"]) - float(a["mirex"])
    return {
        "seed": control["seed"],
        "delta_mirex": delta,
        "qualifies_for_confirmation": delta["fmak"] >= 0.02 and delta["beatport"] >= -0.02,
        "calibration_used": False,
        "final_holdout_used": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--joint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = compare(json.loads(args.control.read_bytes()), json.loads(args.joint.read_bytes()))
    result["run_sha256"] = {
        "control": hashlib.sha256(args.control.read_bytes()).hexdigest(),
        "joint": hashlib.sha256(args.joint.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
