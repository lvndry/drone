"""Check published fixed-case evidence; optionally verify retained cores without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULT_SHA = "919a50bf44fc099d97353d070ee02095f94332032a099911fd3ef4fc3214978a"
PROVENANCE_SHA = "510912743857307eea257161ab9f72633ba1fa8c21f0ed8cff1d0a67a1fd5050"


def read_checked(path: Path, expected: str, maximum: int = 1024**2) -> bytes:
    if path.stat().st_size > maximum:
        raise ValueError("Artifact exceeds size bound")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Artifact identity changed")
    return raw


def summarize(root: Path = ROOT, run_root: Path | None = None) -> dict[str, Any]:
    result = json.loads(read_checked(root / "fused-product-results.json", RESULT_SHA))
    provenance = json.loads(read_checked(root / "fused-product-provenance.json", PROVENANCE_SHA))
    if provenance["results"]["sha256"] != RESULT_SHA:
        raise ValueError("Publication result identity changed")
    if not result["baseline"]["archived_native_core_preserved"]:
        raise ValueError("Original native control not reproduced")
    control = result["baseline"]["python_comparison"]
    candidate = result["candidate_python_comparison"]
    if (control["elements"], control["unequal_bits"]) != (76608, 9):
        raise ValueError("Original negative comparison changed")
    if (candidate["elements"], candidate["unequal_bits"]) != (76608, 0):
        raise ValueError("Candidate whole-core comparison changed")
    if result["candidate_core_sha256"] != result["python_core_sha256"]:
        raise ValueError("Candidate not byte-identical to reference")
    if not result["no_model_or_ort_invoked"] or not result["no_release_promotion"]:
        raise ValueError("Fixed frontend scope changed")
    if result["default_product_rule"] != "separate":
        raise ValueError("Historical default changed")
    for receipt in provenance["resources"].values():
        if (
            not receipt["complete"]
            or receipt["worker_exit_code"]
            or receipt["remaining_owned_pids"]
        ):
            raise ValueError("Incomplete supervised execution")
    verified = False
    if run_root is not None:
        for name, record in provenance["runtime_receipts"].items():
            read_checked(run_root / name, record["sha256"])
        raw_control = (run_root / "output/separate-product-core.f32").read_bytes()
        raw_candidate = (run_root / "output/fused-product-core.f32").read_bytes()
        if len(raw_control) != 76608 * 4 or len(raw_candidate) != len(raw_control):
            raise ValueError("Core geometry changed")
        # The candidate hash is the separately archived Python reference identity.
        if hashlib.sha256(raw_candidate).hexdigest() != result["python_core_sha256"]:
            raise ValueError("Retained candidate differs from archived Python core")
        coordinates = [
            {"bin": index // 399, "frame": index % 399}
            for index, (a, b) in enumerate(
                zip(struct.iter_unpack("<I", raw_control), struct.iter_unpack("<I", raw_candidate))
            )
            if a != b
        ]
        if coordinates != control["first_different_coordinates"]:
            raise ValueError("Original differing coordinates changed")
        verified = True
    return {
        "case": result["case"],
        "original_differences": control["unequal_bits"],
        "candidate_differences": candidate["unequal_bits"],
        "core_elements": candidate["elements"],
        "retained_cores_verified": verified,
        "new_accuracy_or_portability_claim": False,
        "resources": provenance["resources"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(run_root=args.run_root), indent=2))
