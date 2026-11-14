"""Synthetic metadata/index tests; no real recording query or label scoring."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.bp_gs_scope import classify, outcome, scope
from experiments.fma_alignment_matcher import AllOffsetIndex
from experiments.fma_fingerprints import PostingIndex
from tools.collect_bp_lineage import bind_sources, label_record, parser_from_source


def test_collector_binds_loader_and_every_run_source(tmp_path: Path) -> None:
    for name in ("experiments/k10_multidomain.py", "train.py", "drone/datasets.py"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("# synthetic source\n")
    sources = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("experiments/k10_multidomain.py", "train.py", "drone/datasets.py")
    }
    runs = [dict(source_sha256=sources), dict(source_sha256=dict(sources))]
    bind_sources(tmp_path, tmp_path / "drone/datasets.py", runs)
    runs[1]["source_sha256"]["train.py"] = "changed"
    with pytest.raises(ValueError, match="source map"):
        bind_sources(tmp_path, tmp_path / "drone/datasets.py", runs)
    with pytest.raises(ValueError, match="canonical"):
        bind_sources(tmp_path, tmp_path / "different.py", runs)


def test_label_parse_reasons_and_unrelated_failure_not_swallowed(tmp_path: Path) -> None:
    parser = parser_from_source(Path(__file__).parents[1] / "drone/datasets.py")
    path = tmp_path / "123 Human readable label.txt"
    for text, reason in (
        (b"C minor", "parseable"),
        (b"unknown annotation", "unparseable_text"),
        (b"\xff", "invalid_utf8"),
    ):
        path.write_bytes(text)
        record = label_record(path, parser)
        assert record["parse_status_reason"] == reason
        assert record["filename"] == path.name
        assert "key" not in record

    def broken(text: str) -> tuple[str, str] | None:
        raise RuntimeError("Unexpected parser failure")

    path.write_bytes(b"C major")
    with pytest.raises(RuntimeError, match="Unexpected"):
        label_record(path, broken)


def test_target_scope_preserves_all_positive_and_weak_offsets(tmp_path: Path) -> None:
    path = tmp_path / "index.sqlite"
    writer = PostingIndex(path)
    short = np.arange(25, dtype=np.int64) * 4
    long = 1000 + np.arange(22, dtype=np.int64) * 20
    weak = 3000 + np.arange(5, dtype=np.int64) * 30
    query = np.column_stack((np.r_[short, long, weak], np.arange(52, dtype=np.int64)))
    for reference_id in (0, 1, 2):
        reference = query.copy()
        reference[:25, 0] += 100
        reference[25:47, 0] += 1000
        reference[47:, 0] += 5000
        writer.add(reference_id, f"synthetic:{reference_id}", reference)
    writer.finish()
    writer.db.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    index = AllOffsetIndex(path)
    expected = [match for match in index.match(query)["matches"] if match["ref_id"] in (1, 2)]
    scope(index, path, [1, 2])
    actual = index.match(query)
    assert actual["complete"]
    assert actual["matches"] == expected
    assert {match["ref_id"] for match in actual["matches"]} == {1, 2}
    assert any(
        match["count"] == 5 and classify(match, None) == "below_screening_support"
        for match in actual["matches"]
    )
    assert any(
        match["count"] == 22 and classify(match, 0.0) == "identity_evidence"
        for match in actual["matches"]
    )
    assert not index.match(query, deadline=time.monotonic() - 1)["complete"]
    assert not index.match(query, maximum_vote_cells=1)["complete"]
    with pytest.raises(ValueError, match="Unexpected postings"):
        index.db.execute("SELECT ref,frame FROM postings")
    index.db.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert (
        outcome(exact=False, complete=False, informative=True, decisions=[])
        == "insufficient_information"
    )


@pytest.mark.parametrize(
    "text,branch",
    [
        ("unknown", "wrong_token_count"),
        ("C dorian", "unsupported_mode"),
        ("H minor", "unsupported_tonic"),
        ("c minor", "unsupported_tonic"),
    ],
)
def test_unparseable_reasons_follow_canonical_branches(
    tmp_path: Path, text: str, branch: str
) -> None:
    parser = parser_from_source(Path(__file__).parents[1] / "drone/datasets.py")
    path = tmp_path / "123 readable.txt"
    path.write_text(text)
    record = label_record(path, parser)
    assert record["parse_status_reason"] == "unparseable_text"
    assert record["unparseable_branch"] == branch
    assert not record["parseable"]


def test_index_artifact_and_dictionary_guards_are_preserved(tmp_path: Path) -> None:
    from experiments.fma_fingerprints import CONTRACT
    from tools.audit_bp_gs import verify_dictionary

    path = tmp_path / "index.sqlite"
    query = np.column_stack((np.arange(22, dtype=np.int64) * 20, np.arange(22, dtype=np.int64)))
    writer = PostingIndex(path)
    writer.add(0, "synthetic:0", query)
    writer.finish()
    writer.db.close()
    artifact = tmp_path / "000000.npz"
    artifact.write_bytes(b"immutable synthetic witness")
    descriptor = dict(
        contract=CONTRACT,
        complete_coverage=True,
        references=1,
        postings=22,
        source_sha256={
            "experiments/fma_fingerprints.py": "a0d315f3f6f6f62b33b7453119350807702eec35a17723b9079fca30c4328f71"
        },
    )
    options: dict[str, Any] = dict(
        descriptor=descriptor,
        query_extraction_contract=CONTRACT,
        expected_index_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        fingerprint_root=tmp_path,
        artifact_sha256={artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()},
        expected_references=1,
    )
    index = AllOffsetIndex.from_verified_artifacts(path, **options)
    verify_dictionary(index, [dict(ref_id=0, id="synthetic:0", hashes=22)])
    with pytest.raises(ValueError, match="dictionary"):
        verify_dictionary(index, [dict(ref_id=0, id="wrong", hashes=22)])
    index.db.close()
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint SHA"):
        AllOffsetIndex.from_verified_artifacts(path, **options)
    artifact.write_bytes(b"immutable synthetic witness")
    with path.open("ab") as handle:
        handle.write(b"changed index")
    with pytest.raises(ValueError, match="index SHA"):
        AllOffsetIndex.from_verified_artifacts(path, **options)


def test_frozen_matcher_source_bytes_retained() -> None:
    root = Path(__file__).parents[1]
    assert (
        hashlib.sha256((root / "experiments/fma_fingerprints.py").read_bytes()).hexdigest()
        == "a0d315f3f6f6f62b33b7453119350807702eec35a17723b9079fca30c4328f71"
    )
    assert (
        hashlib.sha256((root / "experiments/fma_alignment_matcher.py").read_bytes()).hexdigest()
        == "04c8b51b6f37223d0db7544f45c87e8d7a11c8c70f2993741ecc75b5c052aa90"
    )


def test_pcm_metadata_join_detects_cross_roles_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import csv
    import gzip
    import io
    import json

    from tools import audit_bp_gs_pcm as pcm

    records = [
        dict(domain="fmak", id=f"fmak:{index}", pcm_sha256=f"pcm{index}") for index in range(5489)
    ]
    records[11]["pcm_sha256"] = records[10]["pcm_sha256"]
    records += [
        dict(domain="beatport", id="beatport:1", pcm_sha256="pcm10"),
        dict(domain="beatport", id="beatport:2", pcm_sha256="pcm12"),
    ]
    raw = json.dumps(dict(records=records)).encode()
    references = tmp_path / "references.gz"
    references.write_bytes(gzip.compress(raw, mtime=0))
    monkeypatch.setattr(pcm, "FULL_SHA", hashlib.sha256(raw).hexdigest())
    roles = (
        "PROPOSED_expanded_fit",
        "PROPOSED_inner_selection",
        "PROPOSED_calibration",
        "PROPOSED_development_confirmation",
    )
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["id", "proposed_role", "component"])
    writer.writeheader()
    for index in range(5489):
        writer.writerow(
            dict(
                id=f"fmak:{index:06d}",
                proposed_role=roles[(index - 10) % 4],
                component=f"synthetic:{index}",
            )
        )
    ledger = tmp_path / "ledger.gz"
    ledger.write_bytes(gzip.compress(stream.getvalue().encode(), mtime=0))
    before = ledger.read_bytes()
    prepared = tmp_path / "prepared.json"
    prepared.write_text(
        json.dumps(dict(rows=[dict(id="beatport:1", domain="beatport", role="train")]))
    )
    monkeypatch.setattr(pcm, "PREPARED_SHA", pcm.digest(prepared))
    release = tmp_path / "release.json"
    release.write_text(json.dumps(dict(metadata=dict(train_ids=["2"]))))
    proposal = tmp_path / "proposal.json"
    proposal.write_text(json.dumps(dict(status="PROPOSED_NOT_ACTIVATED")))
    result = pcm.audit(references, ledger, proposal, prepared, release)
    assert result["fma_cross_role_pcm_groups"] == 1
    assert result["bp_fitting_to_fma_evaluation"]["current_fitting"]["matching_bp_ids"] == 1
    assert result["bp_fitting_to_fma_evaluation"]["release_fitting"]["matching_bp_ids"] == 1
    assert ledger.read_bytes() == before
    assert not result["roles_changed"]


def test_packaged_metadata_plan_and_collector_provenance() -> None:
    import gzip
    import json

    from tools.audit_bp_gs import sources

    data = Path(__file__).parents[1] / "experiments/studies/bp-gs-overlap/data"
    plan = json.loads(gzip.decompress((data / "plan.json.gz").read_bytes()))
    assert plan["source_sha256"] == sources()
    assert plan["counts"]["total_planned_queries"] == 2334
    assert len(plan["queries"]) == len({row["record"]["id"] for row in plan["queries"]}) == 1486
    assert len(plan["gs_records"]) == 604
    assert len(plan["controls"]) == 244
    assert plan["counts"]["release_train_validation_container_intersection"] == 2
    assert plan["source_label_parse_reasons"] == dict(
        parseable=1363, wrong_token_count=104, unsupported_mode=17, unsupported_tonic=2
    )
    for version in (1, 2):
        witness = json.loads((data / f"collector-provenance-v{version}.json").read_bytes())
        assert witness["exit_code"] == 0
        assert (
            hashlib.sha256(witness["source_utf8"].encode()).hexdigest() == witness["source_sha256"]
        )
        assert (
            hashlib.sha256((data / f"collector-v{version}.json").read_bytes()).hexdigest()
            == witness["stdout_sha256"]
        )
    index = json.loads((data / "artifact-index.json").read_bytes())
    for name, witness in index.items():
        raw = (data / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == witness["sha256"]
        if name.endswith(".gz"):
            assert hashlib.sha256(gzip.decompress(raw)).hexdigest() == witness["decoded_sha256"]
    pcm = json.loads((data / "pcm-role-check.json").read_bytes())
    assert (
        pcm["source_sha256"]
        == hashlib.sha256((data.parents[3] / "tools/audit_bp_gs_pcm.py").read_bytes()).hexdigest()
    )
    assert not plan["real_queries_executed"]


def test_reviewed_plan_sha_and_source_corruption_fail_before_queries(tmp_path: Path) -> None:
    import json
    from argparse import Namespace

    from tools.audit_bp_gs import digest, run

    path = tmp_path / "plan.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Reviewed plan SHA"):
        run(Namespace(plan=path, plan_sha256="wrong", output=tmp_path / "untouched"))
    path.write_text(json.dumps(dict(source_sha256={"wrong": "source"})))
    with pytest.raises(ValueError, match="Frozen source"):
        run(Namespace(plan=path, plan_sha256=digest(path), output=tmp_path / "untouched"))
    assert not (tmp_path / "untouched").exists()
