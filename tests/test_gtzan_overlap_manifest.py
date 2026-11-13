"""Label-free query projections and release-role aliases, without audio or matching."""

from __future__ import annotations

import copy
import gzip
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from tools.prepare_gtzan_overlap import (
    CHECKPOINT_SHA,
    OUTPUT,
    compressed,
    digest,
    encode,
    manifest,
    pinned,
    queries,
    release_anchor,
)


def fixture() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ledger = []
    inventory = []
    for i in range(3):
        ident = f"hiphop.{i:05d}"
        member = f"genres/hiphop/{ident}.wav"
        ledger.append(
            dict(
                id=ident,
                audio_member=member,
                metadata_eligible=i != 1,
                raw_single_key_reference=True,
                expected_member_md5=str(i) * 32,
                drone_ref=17,
                annotation_integer=13,
            )
        )
        inventory.append(
            dict(
                member=member,
                expected_md5=str(i) * 32,
                md5=str(i) * 32,
                sha256=str(i) * 64,
                member_identity_verified=True,
                overlap_cleared=False,
                admitted_for_scoring=False,
                bytes=2044,
                wave=dict(
                    channels=1,
                    sample_rate=22050,
                    sample_width=2,
                    frames=1000,
                    duration_seconds=1000 / 22050,
                ),
            )
        )
    return ledger, inventory


def release_fixture() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    refs = [
        dict(
            ref_id=i,
            id=f"beatport:{i}",
            domain="beatport",
            status="complete",
            container_sha256=str(i) * 64,
            pcm_sha256=str(i) * 64,
        )
        for i in range(4)
    ]
    refs[2]["container_sha256"] = refs[0]["container_sha256"]
    refs[3]["pcm_sha256"] = refs[1]["pcm_sha256"]
    release = dict(
        checkpoint_sha256=CHECKPOINT_SHA,
        metadata=dict(
            train_ids=["0"],
            validation_ids=["beatport:1"],
            training_audio_sha256=["0" * 64, "1" * 64],
        ),
    )
    return release, refs


def test_eligibility_and_label_nonleakage() -> None:
    ledger, inventory = fixture()
    result = queries(ledger, inventory)
    assert [q.id for q in result] == ["hiphop.00000", "hiphop.00002"]
    assert len(queries(ledger, inventory, eligibility_field="raw_single_key_reference")) == 3
    raw = encode([asdict(q) for q in result])
    assert b"drone_ref" not in raw and b"annotation_integer" not in raw
    ledger[0]["drone_ref"] = 23
    ledger[0]["annotation_integer"] = 99
    assert queries(ledger, list(reversed(inventory))) == result


@pytest.mark.parametrize("which", ["id", "member", "inventory"])
def test_duplicate_inputs_rejected(which: str) -> None:
    ledger, inventory = fixture()
    if which == "inventory":
        inventory.append(copy.deepcopy(inventory[0]))
    else:
        field = "id" if which == "id" else "audio_member"
        ledger[1][field] = ledger[0][field]
    with pytest.raises(ValueError, match="Duplicate"):
        queries(ledger, inventory)


@pytest.mark.parametrize("fault", ["md5", "admission", "missing", "wave", "sha256", "eligibility"])
def test_query_tampering_rejected(fault: str) -> None:
    ledger, inventory = fixture()
    if fault == "md5":
        inventory[0]["md5"] = "f" * 32
    elif fault == "admission":
        inventory[0]["overlap_cleared"] = True
    elif fault == "missing":
        inventory.pop()
    elif fault == "wave":
        inventory[0]["wave"]["duration_seconds"] = 1
    elif fault == "sha256":
        inventory[0]["sha256"] = "bad"
    else:
        ledger[0]["metadata_eligible"] = 1
    with pytest.raises(ValueError):
        queries(ledger, inventory)


def test_release_declared_roles_and_both_alias_types() -> None:
    release, refs = release_fixture()
    anchor = release_anchor(release, refs)
    assert anchor["declared_counts"] == {"fitting": 1, "selection": 1}
    rows = anchor["known_exact_exposure_aliases"]
    assert rows[2]["release_roles"] == {"fitting": ["container_sha256"]}
    assert rows[3]["release_roles"] == {"selection": ["pcm_sha256"]}
    # Current candidate roles and old review flags cannot change release exposure.
    refs[0]["role"] = "validation"
    refs[0]["review_required"] = False
    assert release_anchor(release, refs) == anchor


@pytest.mark.parametrize(
    "fault", ["checkpoint", "duplicate", "cross_role", "uncovered", "witness", "ref_duplicate"]
)
def test_release_anchor_rejects_tampering(fault: str) -> None:
    release, refs = release_fixture()
    meta = release["metadata"]
    if fault == "checkpoint":
        release["checkpoint_sha256"] = "f" * 64
    elif fault == "duplicate":
        meta["train_ids"] = ["0", "beatport:0"]
    elif fault == "cross_role":
        meta["validation_ids"] = ["0"]
    elif fault == "uncovered":
        meta["train_ids"] = ["100"]
    elif fault == "witness":
        meta["training_audio_sha256"] = ["f" * 64]
    else:
        refs.append(copy.deepcopy(refs[0]))
    with pytest.raises(ValueError):
        release_anchor(release, refs)


def test_pin_checks_exact_metadata_bytes(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_bytes(b"original")
    expected = digest(path.read_bytes())
    assert pinned(path, expected) == b"original"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Pinned metadata"):
        pinned(path, expected)


def test_committed_manifest_reproduces_from_terminal_metadata() -> None:
    result = manifest()
    assert compressed(result) == OUTPUT.read_bytes()
    assert gzip.decompress(OUTPUT.read_bytes()) == encode(result)
    assert compressed(result) == compressed(result)
    assert OUTPUT.read_bytes()[4:8] == b"\0" * 4
    assert result["query_count"] == 769 and result["query_counts_by_genre"]["hiphop"] == 73
    assert result["sensitivity_query_count"] == 837
    assert not result["admitted_for_scoring"] and not result["fingerprint_admission"]
    assert result["release_exposure"]["declared_counts"] == {"fitting": 1237, "selection": 126}
    for row in result["queries"] + result["sensitivity_queries"]:
        assert set(row) == {
            "id",
            "member",
            "wav_md5",
            "wav_sha256",
            "bytes",
            "frames",
            "sample_rate",
            "duration_seconds",
        }
