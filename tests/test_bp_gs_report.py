"""Pin observed validation receipts and bind report examples to original evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/studies/bp-gs-overlap/data"


def test_original_root_validation_is_preserved_without_reattribution() -> None:
    execution = DATA / "execution"
    receipt = json.loads((execution / "original-root-check-receipt.json").read_bytes())
    assert receipt["executor"] == "root"
    assert receipt["receipt_kind"] == "copy_of_observed_tool_result_not_a_new_execution"
    assert receipt["exit_code"] == 0
    assert (
        hashlib.sha256((execution / "original-root-checker.py.txt").read_bytes()).hexdigest()
        == receipt["source_sha256"]
    )
    assert hashlib.sha256(receipt["stdout"].encode()).hexdigest() == receipt["stdout_sha256"]
    observed = json.loads(receipt["stdout"])
    assert observed["passed"]
    assert observed["queries"] == 2334
    assert observed["verified_alignment_records"] == 4315263
    assert (
        hashlib.sha256(gzip.decompress((execution / "result.json.gz").read_bytes())).hexdigest()
        == observed["result_sha256"]
    )


def test_report_counts_exposures_and_examples_bind_to_original_records() -> None:
    report = json.loads(gzip.decompress((DATA / "diagnostic.json.gz").read_bytes()))
    result = json.loads(gzip.decompress((DATA / "execution/result.json.gz").read_bytes()))
    plan = json.loads(gzip.decompress((DATA / "plan.json.gz").read_bytes()))
    assert (
        report["source_sha256"]
        == hashlib.sha256(
            (DATA / "execution/original-report-helper.py.txt").read_bytes()
        ).hexdigest()
    )
    assert report["outcomes"] == dict(
        definite_detected_overlap=0, review=539, insufficient_information=0, no_detected_overlap=947
    )
    assert sum(report["outcomes"].values()) == 1486
    assert report["scientific_alignment_records"] == sum(
        row["alignments"] for row in result["results"] if row["kind"] == "bp-audit"
    )
    assert (
        sum(report["review_alignment_categories"].values())
        == report["review_alignment_records"]
        == 162928
    )
    assert report["model_exposure"]["release"]["review:fitting:direct_id"]["count"] == 459
    for name in ("control13", "control37", "control73", "joint13", "joint37", "joint73"):
        row = report["model_exposure"][name]
        assert row["review:fitting:direct_id"]["count"] == 458
        assert row["review:validation:direct_id"]["count"] == 43
        assert (
            row["review:fitting:container_alias_only"]["ids"]
            == row["review:fitting:pcm_alias_only"]["ids"]
            == ["beatport:299977"]
        )
    ids = {row["id"]: row["ref_id"] for row in plan["gs_records"]}
    for group in report["examples"].values():
        for example in group:
            raw = (DATA / "execution/examples" / example["evidence"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == example["evidence_sha256"]
            original = json.loads(gzip.decompress(raw))
            assert original["id"] == example["query_id"]
            match = next(
                row
                for row in original["match"]["matches"]
                if row["ref_id"] == ids[example["target_id"]]
                and row["offset_bin"] == example["offset_bin"]
                and row["phase"] == example["phase"]
            )
            assert match["decision"] == "review"
            assert match["chromaprint_distance"] == example["chromaprint_distance"]


def test_report_loader_reads_identical_plain_and_gzip_bytes(tmp_path: Path) -> None:
    from tools.report_bp_gs import payload_bytes, read_payload

    raw = b'{"recording":"synthetic", "decisions":["review"]}\n'
    plain, packed = tmp_path / "result.json", tmp_path / "result.json.gz"
    plain.write_bytes(raw)
    packed.write_bytes(gzip.compress(raw, mtime=0))
    assert read_payload(plain) == read_payload(packed) == json.loads(raw)
    assert payload_bytes(plain) == payload_bytes(packed) == raw
    index = json.loads((DATA / "execution/artifact-index.json").read_bytes())
    original = gzip.decompress((DATA / "execution/result.json.gz").read_bytes())
    assert (
        hashlib.sha256(original).hexdigest()
        == index["result.json.gz"]["decoded_sha256"]
        == "a3eb3809f89617c0435a2ee0e0adbfc4178bb5c837bdc55f48e06012826eece7"
    )
