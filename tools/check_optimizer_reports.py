"""Check recorded SAM/ensemble arithmetic and provenance without fitting or inference."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "experiments/studies"
CONVENTIONS = {
    "repo_symmetric_fifths": "mirex",
    "existing_asymmetric_ascending_fifth": "mirex_asymmetric",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)


def validate_pair(old: dict[str, Any], new: dict[str, Any], movements: dict[str, Any]) -> None:
    for domain, count in (("beatport", 126), ("fmak", 250)):
        require(old[domain]["ids"] == new[domain]["ids"], "Paired development IDs differ")
        require(
            len(set(old[domain]["ids"])) == count == old[domain]["tracks"] == new[domain]["tracks"],
            "Wrong development denominator",
        )
        for convention, metric in CONVENTIONS.items():
            row = movements[domain][convention]
            require(
                row["categories"] == ["exact", "fifth", "relative", "parallel", "other"],
                "Wrong movement category order",
            )
            matrix = np.asarray(row["row_control_column_candidate"])
            require(
                matrix.shape == (5, 5) and bool(np.all(matrix >= 0)) and int(matrix.sum()) == count,
                "Movement matrix denominator differs",
            )
            require(
                close(float(matrix[0].sum()) / count, old[domain]["exact"]),
                "Old exact arithmetic differs",
            )
            require(
                close(float(matrix[:, 0].sum()) / count, new[domain]["exact"]),
                "New exact arithmetic differs",
            )
            weights = np.array([1, 0.5, 0.3, 0.2, 0])
            require(
                close(float(matrix.sum(axis=1) @ weights) / count, old[domain][metric]),
                "Old weighted arithmetic differs",
            )
            require(
                close(float(matrix.sum(axis=0) @ weights) / count, new[domain][metric]),
                "New weighted arithmetic differs",
            )
            require(
                int(matrix[1:, 0].sum()) == row["exact_promotions"]
                and int(matrix[0, 1:].sum()) == row["exact_demotions"],
                "Paired exact movement counts differ",
            )


def check(root: Path = STUDIES) -> dict[str, Any]:
    sam = root / "k10-sam"
    ensemble = root / "k10-ensemble"
    control = json.loads((sam / "data/control-result.json").read_text())
    candidate = json.loads((sam / "data/candidate-result.json").read_text())
    result = json.loads((ensemble / "data/result.json").read_text())
    require(
        control["status"] == candidate["status"] == result["status"] == "complete",
        "Incomplete report",
    )
    require(
        control["control_reproduced"]
        and candidate["control_reproduced"]
        and candidate["control_proof"]["exact_history_predictions_state"],
        "Fresh exact control proof missing",
    )
    require(
        candidate["control_proof"]["result_sha256"] == sha(sam / "data/control-result.json"),
        "Fresh control artifact hash differs",
    )
    require(candidate["control_metrics"] == control["metrics"], "Candidate paired control differs")
    require(
        candidate["identity"]["source_sha256"] == control["identity"]["source_sha256"],
        "SAM arms used different executed source",
    )
    require(
        candidate["identity"]["sam_spec"]["rho"] == 0.05
        and candidate["identity"]["sam_spec"]["enabled"]
        and not control["identity"]["sam_spec"]["enabled"],
        "Frozen SAM specification differs",
    )
    for arm, raw in (("control", control), ("candidate", candidate)):
        run_path = sam / f"data/{arm}-run.json.gz"
        run_bytes = gzip.decompress(run_path.read_bytes())
        run_sha = hashlib.sha256(run_bytes).hexdigest()
        run = json.loads(run_bytes)
        require(run_sha == raw["run_sha256"], "Actual SAM run hash differs")
        require(
            len(run["history"]) == raw["history_epochs"] == raw["stop_epoch"],
            "History/stop horizon differs",
        )
        histories = json.loads((sam / "data/histories.json").read_text())[arm]
        require(
            histories["history"] == run["history"]
            and histories["best_epoch"] == run["best_epoch"]
            and histories["source_run_sha256"] == run_sha,
            "Figure history does not match actual run",
        )
        best = max(run["history"], key=lambda entry: entry["mirex"])["epoch"]
        require(
            best == raw["selected_epoch"] == run["best_epoch"],
            "Earliest strict-best selection differs",
        )
        require(
            run["parameters"] == 109128
            and run["feature_manifest_sha256"] == raw["identity"]["feature_manifest_sha256"],
            "Parameter/cache identity differs",
        )
        require(
            sum(track_id.startswith("beatport:") for track_id in run["fit_ids"]) == 1230
            and sum(track_id.startswith("fmak:") for track_id in run["fit_ids"]) == 976,
            "Fitting domain counts differ",
        )
        require(
            not run["calibration_used"]
            and not run["final_holdout_used"]
            and len(run["fit_ids"]) == 2206
            and len(run["validation_ids"]) == 376,
            "Dataset roles differ",
        )
    validate_pair(control["metrics"], candidate["metrics"], candidate["confusion_movements"])
    sam_gate = candidate["metrics"]["fmak"]["mirex"] - control["metrics"]["fmak"][
        "mirex"
    ] >= 0.01 and all(
        candidate["metrics"][domain][metric] >= control["metrics"][domain][metric]
        for domain, metric in (("beatport", "mirex"), ("beatport", "exact"), ("fmak", "exact"))
    )
    require(
        sam_gate == candidate["method_gate"]["qualifies_for_confirmation"] is False,
        "SAM negative gate differs",
    )
    require(
        result["fixed_member_seeds"] == [13, 37, 73]
        and result["canonical_seed"] == 13
        and result["ensemble_candidates"] == result["ensemble_replicates"] == 1,
        "Fixed single-cohort estimand differs",
    )
    require(
        result["member_predictions_reproduced"]
        and result["identity"]["temperature"] == 1
        and result["identity"]["weights"] == [1 / 3] * 3
        and result["identity"]["accumulation_dtype"] == "float64",
        "Fixed probability-combination identity differs",
    )
    require(
        result["member_metrics"]["13"] == control["metrics"],
        "Canonical ensemble single differs from exact SAM control",
    )
    for seed in ("13", "37", "73"):
        validate_pair(
            result["member_metrics"][seed], result["metrics"], result["confusion_movements"][seed]
        )
        for domain in ("beatport", "fmak"):
            for metric in ("mirex", "mirex_asymmetric", "exact"):
                delta = (
                    result["metrics"][domain][metric]
                    - result["member_metrics"][seed][domain][metric]
                )
                require(
                    close(delta, result["comparisons"][seed][domain][metric]),
                    "Ensemble member delta differs",
                )
    uniform = all(
        result["metrics"][domain][metric] >= result["member_metrics"][seed][domain][metric]
        for seed in ("13", "37", "73")
        for domain in ("beatport", "fmak")
        for metric in ("mirex", "exact")
    )
    require(
        uniform == result["uniformly_nonregressing_vs_all_members"] is False,
        "All-member nonregression differs",
    )
    for domain in ("beatport", "fmak"):
        require(
            close(
                result["metrics"][domain]["mirex"] - result["bp_control_mean"][domain]["mirex"],
                result["bp_reference_hurdle_delta"][domain],
            ),
            "BP-mean hurdle arithmetic differs",
        )
    canonical = result["member_metrics"]["13"]
    teacher = result["metrics"]["fmak"]["mirex"] - canonical["fmak"]["mirex"] >= 0.01 and all(
        result["metrics"][domain][metric] >= canonical[domain][metric]
        for domain, metric in (("beatport", "mirex"), ("beatport", "exact"), ("fmak", "exact"))
    )
    bp_gate = (
        result["bp_reference_hurdle_delta"]["fmak"] >= 0.02
        and result["bp_reference_hurdle_delta"]["beatport"] >= 0
    )
    require(
        teacher == result["teacher_gate"]["qualifies_as_teacher"] is False
        and bp_gate == result["bp_reference_hurdle_pass"] is False,
        "Ensemble negative gates differ",
    )
    require(
        not result["promoted"]
        and not result["distillation_authorized"]
        and not candidate["promoted"],
        "Promotion/distillation must remain false",
    )
    inputs = {}
    for directory, report in ((sam, candidate), (ensemble, result)):
        require(
            sha(directory / "PROTOCOL.md") == report["identity"]["protocol_sha256"],
            "Frozen protocol bytes differ",
        )
        require(
            sha(directory / "data/reviewed-sources.json")
            == report["identity"]["reviewed_sources_sha256"],
            "Reviewed source-map bytes differ",
        )
        proof = json.loads((directory / "data/execution-provenance.json").read_text())
        require(
            all(receipt == "0" for receipt in proof["exit_receipts"].values()),
            "Authoritative execution failed",
        )
        raw = directory / ("data/candidate-result.json" if directory == sam else "data/result.json")
        server_key = (
            "candidate-seed13/sam-result.json" if directory == sam else "result/result.json"
        )
        require(sha(raw) == proof["artifacts"][server_key], "Local/server raw result differs")
        inputs[str(raw.relative_to(ROOT))] = sha(raw)
    return dict(
        passed=True,
        operations="recorded-artifact arithmetic/provenance only; no fit/inference/scoring",
        sam_gate=False,
        ensemble_teacher_gate=False,
        ensemble_bp_hurdle=False,
        paired_matrices_verified=16,
        input_sha256=inputs,
    )


if __name__ == "__main__":
    print(json.dumps(check(), sort_keys=True, indent=2))
