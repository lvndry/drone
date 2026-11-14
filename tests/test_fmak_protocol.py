from __future__ import annotations

import copy
from typing import Any

from tools.fmak_protocol import partition


def test_final_holdout_is_label_independent_and_keeps_recordings_together() -> None:
    meta: dict[str, Any] = {
        "ids": [str(index) for index in range(100)],
        "tracks": [{"sha256": str(index // 2), "key": ["C", "major"]} for index in range(100)],
        "refs": [0] * 100,
    }
    original = partition(meta, holdout_size=20)
    changed = copy.deepcopy(meta)
    changed["refs"] = [index % 24 for index in range(100)]
    for track in changed["tracks"]:
        track["key"] = ["A", "minor"]
    relabeled = partition(changed, holdout_size=20)
    assert original["final_holdout_indices"] == relabeled["final_holdout_indices"]
    development = {meta["tracks"][index]["sha256"] for index in original["development_indices"]}
    test = {meta["tracks"][index]["sha256"] for index in original["final_holdout_indices"]}
    assert not development & test
    assert len(original["final_holdout_indices"]) == 20
    excluded = partition(meta, excluded_audio=["0"], holdout_size=20)
    assert excluded["excluded_indices"] == [0, 1]
    assert not set(excluded["excluded_indices"]) & set(excluded["final_holdout_indices"])
