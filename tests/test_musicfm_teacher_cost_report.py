"""Evidence-integrity and independent vector checks; no model/audio/runtime dependency."""

from __future__ import annotations

import copy
import gzip
import json
from io import BytesIO
from pathlib import Path

import pytest

from tools.musicfm_teacher_cost_report import LABELS, STUDY, load_raw, paired_vectors, summarize


def test_original_failure_controlled_stop_and_corrected_pass_are_all_retained() -> None:
    report = summarize()
    old, fixed = (report["executions"][label] for label in LABELS)
    assert old["composed_gate"]["failed_tracks"] == 5
    assert old["composed_gate"]["max_absolute_error"] == 0.0012251086533069666
    assert fixed["composed_gate"]["passed"]
    assert fixed["composed_gate"]["max_absolute_error"] == 3.2306259806214044e-7
    assert old["composed_gate"]["argmax_disagreements"] == 0
    assert old["initial_state_sha256"] == fixed["initial_state_sha256"]
    assert not report["controlled_stop"]["complete"]
    assert not report["real_task_accuracy_measured"]
    assert not report["training_authorized_by_this_report"]
    assert not fixed["projection_measured"]
    assert 10 < fixed["model_steady_plus_validation_projection_hours"] < 12


def test_failure_is_not_attributed_to_tiny_tails() -> None:
    old = summarize()["executions"][LABELS[0]]["composed_gate"]["per_song"]
    assert [row["id"] for row in old if row["failed"]] == [
        "beatport:1059355",
        "beatport:1077935",
        "beatport:11077",
        "beatport:1147737",
        "beatport:1274431",
    ]
    assert all(not row["failed"] for row in old if 0 < row["tail_samples"] <= 24000)


@pytest.mark.parametrize("damage", ["nan", "changed-value", "context", "domain-schedule"])
def test_vector_proof_rejects_corruption_or_different_executed_schedule(damage: str) -> None:
    raw, _ = load_raw(STUDY)
    result = json.loads(raw[LABELS[1] + "/result.json"])
    result = copy.deepcopy(result)
    if damage in ("nan", "changed-value"):
        result["composed_batch_proof"]["candidate_logits"][0][0] = (
            float("nan") if damage == "nan" else 100.0
        )
    elif damage == "context":
        result["batched_validation"]["per_song"][0]["samples"] -= 1
    else:
        result["batched_validation"]["domain_schedule"][0]["ids"].reverse()
    with pytest.raises(ValueError):
        paired_vectors(result)


def test_raw_bundle_rejects_compressed_byte_damage(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "cost-index.json").write_bytes((STUDY / "data/cost-index.json").read_bytes())
    packed = bytearray((STUDY / "data/cost-raw.json.gz").read_bytes())
    packed[-10] ^= 1
    (data / "cost-raw.json.gz").write_bytes(packed)
    with pytest.raises(ValueError, match="compressed evidence"):
        load_raw(tmp_path)


def test_raw_archive_is_losslessly_repacked_deterministically() -> None:
    packed = (STUDY / "data/cost-raw.json.gz").read_bytes()
    output = BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as stream:
        stream.write(gzip.decompress(packed))
    assert packed[3] & 8 == 0  # No originating filename.
    assert int.from_bytes(packed[4:8], "little") == 0
    assert gzip.decompress(output.getvalue()) == gzip.decompress(packed)
    # Preserve recorded compressed bytes; zlib implementations may encode identically
    # decoded bytes differently across platforms. load_raw verifies BOTH hashes.
