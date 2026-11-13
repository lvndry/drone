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


GOLDENS_RESULT_SHA = "9a77dfa043da33aef7b3e0b4b835f862ea637be37311176221037c192f05bda4"
GOLDENS_PROVENANCE_SHA = "499cdef848cda8174b679f5eb18ff43e67615d0188237623217ca9a657e4568a"
GOLDEN_IDS = [
    "silence",
    "impulses",
    "tone",
    "major",
    "minor",
    "noise",
    "fmak-110544",
    "fmak-107354",
    "fmak-087373",
]


def summarize_goldens(root: Path = ROOT, run_root: Path | None = None) -> dict[str, Any]:
    result = json.loads(read_checked(root / "fused-goldens-results.json", GOLDENS_RESULT_SHA))
    provenance = json.loads(
        read_checked(root / "fused-goldens-provenance.json", GOLDENS_PROVENANCE_SHA)
    )
    if provenance["results"]["sha256"] != GOLDENS_RESULT_SHA:
        raise ValueError("Golden publication identity changed")
    if [case["id"] for case in result["cases"]] != GOLDEN_IDS:
        raise ValueError("Frozen golden membership changed")
    if (
        not all(
            result[name]
            for name in [
                "complete",
                "all_controls_preserved",
                "all_candidate_cores_exact",
                "no_model_or_ort_invoked",
                "no_release_promotion",
            ]
        )
        or result["default_product_rule"] != "separate"
    ):
        raise ValueError("Golden scope or outcome changed")
    if run_root is not None:
        for name, record in provenance["runtime_receipts"].items():
            read_checked(run_root / name, record["sha256"], maximum=8 * 1024**2)
    rows = []
    for index, case in enumerate(result["cases"]):
        frames = case["frames"]
        if (
            not case["original_control_preserved"]
            or not case["candidate_admitted"]
            or case["baseline_core_sha256"] != case["archived_native_core_sha256"]
            or case["candidate_core_sha256"] != case["manifest_core_sha256"]
            or case["candidate_python_comparison"]["unequal_bits"] != 0
            or case["candidate_python_comparison"]["elements"] != 192 * frames
        ):
            raise ValueError("Golden core or control comparison changed")
        if run_root is not None:
            folder = run_root / f"output/{index:04}"
            baseline = (folder / "separate-product-core.f32").read_bytes()
            candidate = (folder / "fused-product-core.f32").read_bytes()
            if len(candidate) != 192 * frames * 4 or len(baseline) != len(candidate):
                raise ValueError("Golden core geometry changed")
            if hashlib.sha256(candidate).hexdigest() != case["manifest_core_sha256"]:
                raise ValueError("Golden core differs from archived Mac reference")
            if hashlib.sha256(baseline).hexdigest() != case["archived_native_core_sha256"]:
                raise ValueError("Golden original control differs")
            coordinates = [
                {"bin": i // frames, "frame": i % frames}
                for i, (a, b) in enumerate(
                    zip(struct.iter_unpack("<I", baseline), struct.iter_unpack("<I", candidate))
                )
                if a != b
            ]
            observed = case["baseline_python_comparison"]
            if (
                len(coordinates) != observed["unequal_bits"]
                or coordinates[:32] != observed["first_different_coordinates"]
            ):
                raise ValueError("Golden original difference coordinates changed")
        rows.append(
            {
                "id": case["id"],
                "decoded_seconds": case["pcm_samples"] / 22050,
                "frames": frames,
                "original_differences": case["baseline_python_comparison"]["unequal_bits"],
                "candidate_differences": case["candidate_python_comparison"]["unequal_bits"],
            }
        )
    for receipt in provenance["resources"].values():
        if (
            not receipt["complete"]
            or receipt["worker_exit_code"]
            or receipt["remaining_owned_pids"]
        ):
            raise ValueError("Incomplete supervised golden execution")
    return {
        "cases": rows,
        "core_elements": sum(row["frames"] * 192 for row in rows),
        "original_differences": sum(row["original_differences"] for row in rows),
        "candidate_differences": 0,
        "retained_cores_verified": run_root is not None,
        "new_accuracy_or_portability_claim": False,
        "resources": provenance["resources"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument(
        "--goldens", action="store_true", help="Verify all nine retained golden cases"
    )
    args = parser.parse_args()
    function = summarize_goldens if args.goldens else summarize
    print(json.dumps(function(run_root=args.run_root), indent=2))
