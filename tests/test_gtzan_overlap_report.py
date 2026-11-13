"""Published overlap evidence cannot silently become admission or lose failed cases."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools.report_gtzan_overlap import DATA, MANIFEST, check_rows, summarize, verify


@pytest.fixture(scope="module")
def evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    payload, _ = verify()
    manifest = json.loads(gzip.decompress(MANIFEST.read_bytes()))
    return payload, manifest


def test_frozen_cohorts_keep_review_and_failed_query_separate(
    evidence: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    payload, manifest = evidence
    result = summarize(payload, manifest)
    assert result["populations"]["primary769"] == {
        "recordings": 769,
        "complete": 769,
        "outcomes": {"no_detected_overlap": 602, "review": 167},
    }
    assert result["populations"]["sensitivity837"] == {
        "recordings": 837,
        "complete": 836,
        "outcomes": {"insufficient_information": 1, "no_detected_overlap": 656, "review": 180},
    }
    assert result["genres"]["primary769"]["hiphop"]["recordings"] == 73
    assert result["genres"]["primary769"]["classical"]["recordings"] == 0
    assert result["analyzed_duration_seconds"]["observed"] == 836
    assert result["analyzed_duration_seconds"]["maximum"] < 31
    assert result["admitted_for_scoring"] is False


@pytest.mark.parametrize(
    "defect",
    [
        "drop",
        "duplicate",
        "retry",
        "membership",
        "failed_negative",
        "decision",
        "accounting",
        "source",
        "admit",
    ],
)
def test_inconsistent_projection_is_rejected(
    evidence: tuple[dict[str, Any], dict[str, Any]], defect: str
) -> None:
    original, manifest = evidence
    payload = copy.deepcopy(original)
    rows = payload["rows"]
    if defect == "drop":
        rows.pop()
    elif defect == "duplicate":
        rows[-1] = rows[0]
    elif defect == "retry":
        rows[376]["execution"] = "continuation"
    elif defect == "membership":
        rows[0]["primary769"] = not rows[0]["primary769"]
    elif defect == "failed_negative":
        rows[376]["outcome"] = "no_detected_overlap"
    elif defect == "decision":
        row = next(row for row in rows if row["supported_alignments"])
        row["supported_alignments"][0]["decision"] = "identity_evidence"
    elif defect == "accounting":
        rows[0]["below_screening_support"] += 1
    elif defect == "source":
        rows[0]["extraction"]["container_sha256"] = "0" * 64
    else:
        payload["admitted_for_scoring"] = True
    with pytest.raises(ValueError):
        check_rows(payload, manifest)


def test_published_metadata_has_no_private_operational_identifiers() -> None:
    paths = [*DATA.iterdir(), DATA.parent / "gtzan-overlap-fresh-controls.json", MANIFEST]
    for path in paths:
        raw = path.read_bytes()
        if path.name.endswith(".gz"):
            raw = gzip.decompress(raw)
        for forbidden in (b"/home/", b"/Users/", b"GPU-", b'"pid":', b'"start_ticks":'):
            assert forbidden not in raw, path


def test_portable_controls_reconstruct_the_original_executed_manifest() -> None:
    from tools.prepare_gtzan_overlap import REFERENCES
    from tools.report_gtzan_overlap import encode
    from tools.run_gtzan_overlap import resolve_fresh_controls

    public = json.loads((DATA.parent / "gtzan-overlap-fresh-controls.json").read_bytes())
    references = json.loads(gzip.decompress(REFERENCES.read_bytes()))["records"]
    original = copy.deepcopy(public)
    witness = original.pop("publication")["original_executed_manifest_sha256"]
    original["controls"] = resolve_fresh_controls(public, references)
    for row in original["controls"]:
        del row["expected_ref"]
    execution = json.loads((DATA / "execution-source.json").read_bytes())
    source_name = "experiments/studies/v2-data-contract/data/gtzan-overlap-fresh-controls.json"
    assert hashlib.sha256(encode(original)).hexdigest() == witness
    assert witness == execution["source_files"][source_name]
    assert hashlib.sha256(encode(public)).hexdigest() != witness


def test_corrupt_publication_fails_before_reporting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import report_gtzan_overlap as module

    (tmp_path / "audit.json.gz").write_bytes(b"not the original publication")
    monkeypatch.setattr(module, "DATA", tmp_path)
    with pytest.raises(ValueError, match="projection changed"):
        verify()
