"""The incomplete corpus result remains a failure; read preserved metadata only."""

import gzip
import json
from pathlib import Path

import pytest

from tools.musicfm_crop_failure_report import verify


def evidence() -> dict[str, str]:
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/studies/musicfm-crop-io/data/corpus-failure-raw.json.gz"
    )
    return json.loads(gzip.decompress(path.read_bytes()))


def test_frozen_corpus_failure_is_not_a_completed_cache() -> None:
    result = verify(evidence())
    assert result["status"] == "FAILED_NOT_QUALIFIED_NO_RESUME"
    assert result["passed_prefix"] == 2288
    assert result["unprocessed"] == 3349
    assert result["validation_processed"] == 0


def test_passing_prefix_cannot_be_relabelled_failed() -> None:
    files = evidence()
    name = "corpus/prepared-v1/records/0001.json"
    row = json.loads(files[name])
    row["crop"]["passed"] = False
    files[name] = json.dumps(row)
    with pytest.raises(ValueError, match="gate"):
        verify(files)


def test_missing_decoder_warning_evidence_is_rejected() -> None:
    files = evidence()
    row = json.loads(files["corpus/prepared-v1/records/2289.json"])
    del files["corpus/prepared-v1/" + row["full_execution"]["stderr"]]
    with pytest.raises(KeyError):
        verify(files)


def test_changed_array_diagnostic_source_is_rejected() -> None:
    files = evidence()
    files["diagnostic/diagnose.py"] += "\n"
    with pytest.raises(ValueError, match="diagnostic"):
        verify(files)
