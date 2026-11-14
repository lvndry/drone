from __future__ import annotations

import copy
from typing import Any

import pytest

from experiments.adapt import development_split
from experiments.small import signature


def fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    groups = [f"artist:{index // 2}" for index in range(202)]
    meta = {
        "tracks": [{"sha256": f"audio:{index // 2}"} for index in range(202)],
        "refs": [0] * 202,
    }
    partition = {
        "metadata_sha256": signature(meta),
        "development_indices": list(range(200)),
        "final_holdout_indices": [200, 201],
    }
    audit = {"groups": groups, "eligible_development_indices": list(range(200))}
    return meta, partition, audit


def test_adaptation_keeps_artists_audio_and_final_holdout_separate() -> None:
    meta, partition, audit = fixture()
    result = development_split(meta, partition, audit)
    assert result == development_split(meta, partition, audit)
    assert all(result.values())
    assert len({meta["tracks"][index]["sha256"] for index in result["train"]}) == len(
        result["train"]
    )
    for first, second in (
        ("train", "validation"),
        ("train", "calibration"),
        ("validation", "calibration"),
    ):
        assert not {audit["groups"][index] for index in result[first]} & {
            audit["groups"][index] for index in result[second]
        }
    assert not set(sum(result.values(), [])) & {200, 201}


def test_adaptation_rejects_eligible_artist_shared_with_final_holdout() -> None:
    meta, partition, audit = fixture()
    audit["groups"][0] = audit["groups"][200]
    with pytest.raises(AssertionError):
        development_split(meta, partition, audit)


def test_adaptation_rejects_conflicting_audio_labels() -> None:
    meta, partition, audit = fixture()
    meta = copy.deepcopy(meta)
    meta["refs"][1] = 1
    partition["metadata_sha256"] = signature(meta)
    with pytest.raises(ValueError, match="Conflicting"):
        development_split(meta, partition, audit)
