"""Read-only prior-audit ledger verification; never retries an attempted query."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.bp_gs_scope import classify, outcome
from tools.run_cqt_expansion import sha256

ROLES = ("config", "execution-plan", "partial", "failure", "permit")
ENGINE = "a0d315f3f6f6f62b33b7453119350807702eec35a17723b9079fca30c4328f71"
MATCHER = "04c8b51b6f37223d0db7544f45c87e8d7a11c8c70f2993741ecc75b5c052aa90"


def prior_ledger(
    plan: dict[str, Any], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], set[str]]:
    bound = config["continuation"]
    root = Path(bound["root"])
    if (
        not root.is_absolute()
        or root.is_symlink()
        or root.resolve() == Path(config["output_root"]).resolve()
    ):
        raise ValueError("Prior audit must be a separate absolute read-only root")
    documents = {}
    for role in ROLES:
        path = root / (role + ".json")
        if path.is_symlink() or sha256(path) != bound[role + "_sha256"]:
            raise ValueError("Prior ledger receipt differs: " + role)
        documents[role] = json.loads(path.read_bytes())
    old = documents["config"]
    previous = documents["execution-plan"]
    if old["mode"] != "audit" or old["output_root"] != str(root) or previous != plan:
        raise ValueError("Prior config/cohort/execution plan mismatch")
    for key in (
        "query_manifest_sha256",
        "runtime_files",
        "cache_root",
        "vendor_path",
        "qualification_receipt_sha256",
    ):
        if old[key] != config[key]:
            raise ValueError("Prior scientific inputs/runtime differ: " + key)
    for name, expected in (
        ("experiments/fma_fingerprints.py", ENGINE),
        ("experiments/fma_alignment_matcher.py", MATCHER),
    ):
        if old["source_sha256"][name] != expected or config["source_sha256"][name] != expected:
            raise ValueError("Continuation cannot change the numerical engine")
    for name in ("experiments/bp_gs_scope.py", "tools/audit_bp_gs.py"):
        if old["source_sha256"][name] != config["source_sha256"][name]:
            raise ValueError("Continuation cannot change CP/scope/decision policy")
    qualification_config = Path(bound["qualification_config"])
    if sha256(qualification_config) != bound["qualification_config_sha256"]:
        raise ValueError("Inherited qualification config differs")
    qualification = json.loads(qualification_config.read_bytes())
    receipt_path = Path(config["qualification_receipt"])
    if sha256(receipt_path) != config["qualification_receipt_sha256"]:
        raise ValueError("Inherited qualification guardian receipt differs")
    guardian = json.loads(receipt_path.read_bytes())
    evidence = receipt_path.parent / "result.json"
    if (
        guardian.get("status") != "complete"
        or guardian["config_sha256"] != bound["qualification_config_sha256"]
        or sha256(evidence) != guardian["result_sha256"]
        or qualification["mode"] != "qualification"
    ):
        raise ValueError("Inherited qualification result/evidence/config binding differs")
    qualified = json.loads(evidence.read_bytes())
    if len(qualified["results"]) != 250 or not qualified.get("qualification_passed"):
        raise ValueError("Inherited250 numerical qualification missing")
    qualified_plan = json.loads((receipt_path.parent / "execution-plan.json").read_bytes())
    if qualified_plan != plan:
        raise ValueError("Inherited qualification index/dictionary/CP-effective plan differs")
    for number, row in enumerate(qualified["results"]):
        if row["evidence"] != f"{number:06}.json.gz":
            raise ValueError("Inherited qualification evidence numbering differs")
        path = receipt_path.parent / "evidence" / row["evidence"]
        compressed_evidence = path.read_bytes()
        if (
            hashlib.sha256(compressed_evidence).hexdigest() != row["evidence_sha256"]
            or hashlib.sha256(gzip.decompress(compressed_evidence)).hexdigest()
            != row["decoded_sha256"]
        ):
            raise ValueError("Inherited qualification raw evidence differs")
    for name in (
        "experiments/fma_fingerprints.py",
        "experiments/fma_alignment_matcher.py",
        "experiments/bp_gs_scope.py",
        "tools/audit_bp_gs.py",
    ):
        if qualification["source_sha256"][name] != old["source_sha256"][name]:
            raise ValueError("Qualification numerical closure differs from original audit")
    if qualification["runtime_files"] != old["runtime_files"]:
        raise ValueError("Qualification runtime closure differs")
    if (
        documents["failure"].get("status") != "incomplete"
        or documents["permit"].get("config_sha256") != bound["config_sha256"]
        or documents["permit"].get("mode") != "audit"
        or documents["permit"].get("authorized") is not True
    ):
        raise ValueError("Prior audit authorization/failure identity differs")
    rows = documents["partial"]["results"]
    attempted = [row["id"] for row in rows]
    cohort = {row["id"]: row for row in plan["cohort"]}
    if (
        not rows
        or len(rows) >= len(cohort)
        or len(set(attempted)) != len(attempted)
        or not set(attempted) <= set(cohort)
        or attempted != bound["attempted_ids"]
    ):
        raise ValueError("Prior attempted population differs or contains repeated IDs")
    archive_order = bound["archive_order_ids"]
    if (
        len(archive_order) != len(cohort)
        or len(set(archive_order)) != len(cohort)
        or set(archive_order) != set(cohort)
        or archive_order[: len(attempted)] != attempted
    ):
        raise ValueError("Prior attempts must be the frozen archive-order prefix")
    unvisited = archive_order[len(attempted) :]
    if unvisited != bound["unvisited_ids"]:
        raise ValueError("Frozen continuation must contain exactly the unvisited IDs")
    imported = []
    for number, row in enumerate(rows):
        expected = f"{number:06}.json.gz"
        if row["evidence"] != expected:
            raise ValueError("Prior evidence numbering/path differs")
        evidence = root / "evidence" / expected
        query = root / "queries" / f"{number:06}.npz"
        for path, digest in ((evidence, row["evidence_sha256"]), (query, row["query_sha256"])):
            if path.is_symlink() or sha256(path) != digest:
                raise ValueError("Prior query/evidence bytes differ")
        decoded_bytes = gzip.decompress(evidence.read_bytes())
        if hashlib.sha256(decoded_bytes).hexdigest() != row["decoded_sha256"]:
            raise ValueError("Prior decoded evidence hash differs")
        decoded = json.loads(decoded_bytes)
        matched = decoded["match"]
        decisions = []
        for match in matched["matches"]:
            decision = classify(match, match["chromaprint_distance"])
            if decision != match["decision"]:
                raise ValueError("Prior alignment decision differs from frozen policy")
            decisions.append(decision)
        expected_outcome = outcome(
            exact=False,
            complete=matched["complete"],
            informative=row["informative"],
            decisions=decisions,
        )
        if (
            decoded["id"] != row["id"]
            or row["panel"] != "audit"
            or row["complete"] != matched["complete"]
            or row["reason"] != matched.get("reason")
            or row["outcome"] != expected_outcome
            or row["alignments"] != len(matched["matches"])
            or row["identity_flagged"] != ("identity_evidence" in decisions)
            or row["review_flagged"] != ("review" in decisions)
        ):
            raise ValueError("Prior query summary disagrees with retained evidence")
        if (
            "extraction" in row
            and row["extraction"]["container_sha256"] != cohort[row["id"]]["container_sha256"]
        ):
            raise ValueError("Prior extraction/cohort source differs")
        imported.append(
            dict(
                row,
                primary769=row["id"] in set(plan["primary_ids"]),
                provenance="prior_audit",
                prior_root=str(root),
                prior_query=str(query),
                prior_evidence=str(evidence),
            )
        )
    return imported, set(unvisited)


def cohort_ledger(plan: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Repeated audit ledger ID")
    return [
        dict(
            id=row["id"],
            primary769=row["id"] in set(plan["primary_ids"]),
            attempted=row["id"] in by_id,
            complete=by_id.get(row["id"], {}).get("complete", False),
            outcome=by_id.get(row["id"], {}).get("outcome", "insufficient_information"),
            status=(
                "unvisited"
                if row["id"] not in by_id
                else "complete"
                if by_id[row["id"]]["complete"]
                else "query_incomplete"
            ),
        )
        for row in plan["cohort"]
    ]


def population_status(plan: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Repeated final audit ID")
    output = {}
    for name, ids in (
        ("primary769", plan["primary_ids"]),
        ("sensitivity837", [row["id"] for row in plan["cohort"]]),
    ):
        attempted = [by_id[identity] for identity in ids if identity in by_id]
        complete = sum(row["complete"] for row in attempted)
        output[name] = dict(
            total=len(ids),
            attempted=len(attempted),
            complete=complete,
            incomplete=len(attempted) - complete,
            unvisited=len(ids) - len(attempted),
            unresolved=len(ids) - complete,
            outcomes={
                label: sum(row["outcome"] == label for row in attempted)
                for label in (
                    "definite_detected_overlap",
                    "review",
                    "insufficient_information",
                    "no_detected_overlap",
                )
            },
        )
    return output
