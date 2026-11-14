"""Synthetic immutable-prefix and continuation checks; no real fingerprint queries."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments import gtzan_overlap_continuation as continuation


def save(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return continuation.sha256(path)


def fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    prior = tmp_path / "prior"
    qualified = tmp_path / "qualified"
    prior.mkdir()
    qualified.mkdir()
    closure = {
        "experiments/fma_fingerprints.py": continuation.ENGINE,
        "experiments/fma_alignment_matcher.py": continuation.MATCHER,
        "experiments/bp_gs_scope.py": "policy",
        "tools/audit_bp_gs.py": "cp",
    }
    plan = dict(
        cohort=[
            dict(id=f"id{index}", member=f"genres/{index}.wav", container_sha256=f"source{index}")
            for index in range(3)
        ],
        primary_ids=["id0", "id2"],
        index_sha256="index",
    )
    config: dict[str, Any] = dict(
        mode="audit",
        output_root=str(tmp_path / "new"),
        source_sha256=closure,
        query_manifest_sha256="manifest",
        runtime_files=[dict(path="runtime", sha256="pin")],
        cache_root="cache",
        vendor_path="vendor",
    )
    qualification = dict(config, mode="qualification", output_root=str(qualified))
    qual_sha = save(qualified / "config.json", qualification)
    qual_rows = []
    for number in range(250):
        raw = json.dumps(dict(control=number)).encode()
        compressed = gzip.compress(raw, mtime=0)
        path = qualified / "evidence" / f"{number:06}.json.gz"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(compressed)
        qual_rows.append(
            dict(
                evidence=path.name,
                evidence_sha256=continuation.sha256(path),
                decoded_sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    result_sha = save(qualified / "result.json", dict(results=qual_rows, qualification_passed=True))
    receipt_sha = save(
        qualified / "guardian-result.json",
        dict(status="complete", config_sha256=qual_sha, result_sha256=result_sha),
    )
    save(qualified / "execution-plan.json", plan)
    config.update(
        qualification_receipt=str(qualified / "guardian-result.json"),
        qualification_receipt_sha256=receipt_sha,
    )
    old = dict(config, output_root=str(prior))
    bound = dict(
        root=str(prior),
        attempted_ids=["id0", "id1"],
        unvisited_ids=["id2"],
        archive_order_ids=["id0", "id1", "id2"],
        qualification_config=str(qualified / "config.json"),
        qualification_config_sha256=qual_sha,
    )
    documents = {"config": old, "execution-plan": plan, "failure": dict(status="incomplete")}
    bound.update(
        {
            role + "_sha256": save(prior / (role + ".json"), value)
            for role, value in documents.items()
        }
    )
    bound["permit_sha256"] = save(
        prior / "permit.json",
        dict(mode="audit", authorized=True, config_sha256=bound["config_sha256"]),
    )
    rows = []
    for number in range(2):
        complete = number == 0
        match: dict[str, Any] = dict(complete=complete, matches=[], posting_visits=1)
        if not complete:
            match["reason"] = "vote_memory_budget"
        raw = json.dumps(dict(id=f"id{number}", match=match)).encode()
        path = prior / "evidence" / f"{number:06}.json.gz"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(gzip.compress(raw, mtime=0))
        query = prior / "queries" / f"{number:06}.npz"
        query.parent.mkdir(exist_ok=True)
        np.savez(query, landmarks=np.zeros((0, 2)), chromaprint=np.zeros(0), chromaprint_step=0.1)
        rows.append(
            dict(
                id=f"id{number}",
                evidence=path.name,
                evidence_sha256=continuation.sha256(path),
                decoded_sha256=hashlib.sha256(raw).hexdigest(),
                query_sha256=continuation.sha256(query),
                complete=complete,
                reason=match.get("reason"),
                informative=True,
                alignments=0,
                identity_flagged=False,
                review_flagged=False,
                panel="audit",
                outcome="no_detected_overlap" if complete else "insufficient_information",
            )
        )
    bound["partial_sha256"] = save(
        prior / "partial.json", dict(results=rows, remaining_are_not_negatives=True)
    )
    config["continuation"] = bound
    return plan, config, rows


def test_import_preserves_complete_and_incomplete_rows_without_retry(tmp_path: Path) -> None:
    plan, config, rows = fixture(tmp_path)
    imported, unvisited = continuation.prior_ledger(plan, config)
    assert unvisited == {"id2"}
    for original, copied in zip(rows, imported, strict=True):
        assert all(copied[key] == value for key, value in original.items())
    assert imported[1]["outcome"] == "insufficient_information"
    status = continuation.population_status(plan, imported)["sensitivity837"]
    assert status == dict(
        total=3,
        attempted=2,
        complete=1,
        incomplete=1,
        unvisited=1,
        unresolved=2,
        outcomes={
            "definite_detected_overlap": 0,
            "review": 0,
            "insufficient_information": 1,
            "no_detected_overlap": 1,
        },
    )
    assert [entry["status"] for entry in continuation.cohort_ledger(plan, imported)] == [
        "complete",
        "query_incomplete",
        "unvisited",
    ]


@pytest.mark.parametrize(
    "failure", ["missing_npz", "bad_evidence", "partial_tamper", "cohort", "duplicate", "runtime"]
)
def test_prior_import_rejects_tamper_and_incompatible_ledger(tmp_path: Path, failure: str) -> None:
    plan, config, rows = fixture(tmp_path)
    prior = Path(config["continuation"]["root"])
    if failure == "missing_npz":
        (prior / "queries/000000.npz").unlink()
    elif failure == "bad_evidence":
        (prior / "evidence/000000.json.gz").write_bytes(b"tampered")
    elif failure == "partial_tamper":
        (prior / "partial.json").write_text("tampered")
    elif failure == "cohort":
        plan["primary_ids"] = ["id1"]
    elif failure == "duplicate":
        rows[1]["id"] = rows[0]["id"]
        config["continuation"]["partial_sha256"] = save(prior / "partial.json", dict(results=rows))
        config["continuation"]["attempted_ids"] = ["id0", "id0"]
    else:
        config["runtime_files"] = []
    with pytest.raises((ValueError, OSError)):
        continuation.prior_ledger(plan, config)


def test_global_identity_failure_still_aborts_and_retains_unvisited_ledger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from unittest.mock import MagicMock

    from tools import run_gtzan_overlap as guardian

    output = tmp_path / "new"
    value = dict(
        mode="audit", output_root=str(output), deadline_epoch=10**12, caps=dict(guardian.MAXIMUMS)
    )
    plan = dict(cohort=[dict(id=f"id{index}") for index in range(3)], primary_ids=["id0"])
    monkeypatch.setattr(guardian, "load", lambda path: {} if path.name == "permit" else value)
    monkeypatch.setattr(guardian, "sha256", lambda path: "bound")
    monkeypatch.setattr(guardian, "check_contract", lambda *args: None)
    monkeypatch.setattr(guardian, "environment", lambda *args: None)
    monkeypatch.setattr(guardian, "verify_bindings", lambda *args: None)
    monkeypatch.setattr(guardian, "execution_plan", lambda *args: plan)
    monkeypatch.setattr(guardian, "start_identity", lambda pid: 11)
    monkeypatch.setattr(guardian, "process_tree", lambda pid: (100, 0))
    calls = 0

    def reserve(*args: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Protected identity changed")

    monkeypatch.setattr(guardian, "reserve_check", reserve)
    child = MagicMock(pid=12345)
    child.poll.return_value = None
    monkeypatch.setattr(guardian.subprocess, "Popen", lambda *args, **kwargs: child)
    stopped = MagicMock()
    monkeypatch.setattr(guardian, "terminate_own", stopped)
    with pytest.raises(RuntimeError, match="Protected identity"):
        guardian.supervise(tmp_path / "config", tmp_path / "permit", output)
    stopped.assert_called_once_with(child, 11)
    failure = json.loads((output / "failure.json").read_bytes())
    assert [row["status"] for row in failure["cohort_ledger"]] == ["unvisited"] * 3
    assert not (output / "guardian-result.json").exists()
