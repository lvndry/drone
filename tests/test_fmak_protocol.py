from __future__ import annotations

import copy
from typing import Any

from tools.fmak_protocol import partition


def test_final_holdout_is_label_independent_and_keeps_recordings_together() -> None:
    meta: dict[str, Any] = {
        "ids": [str(i) for i in range(100)],
        "tracks": [{"sha256": str(i // 2), "key": ["C", "major"]} for i in range(100)],
        "refs": [0] * 100,
    }
    a = partition(meta, holdout_size=20)
    changed = copy.deepcopy(meta)
    changed["refs"] = [i % 24 for i in range(100)]
    for track in changed["tracks"]:
        track["key"] = ["A", "minor"]
    b = partition(changed, holdout_size=20)
    assert a["final_holdout_indices"] == b["final_holdout_indices"]
    dev = {meta["tracks"][i]["sha256"] for i in a["development_indices"]}
    test = {meta["tracks"][i]["sha256"] for i in a["final_holdout_indices"]}
    assert not dev & test
    assert len(a["final_holdout_indices"]) == 20
    excluded = partition(meta, excluded_audio=["0"], holdout_size=20)
    assert excluded["excluded_indices"] == [0, 1]
    assert not set(excluded["excluded_indices"]) & set(excluded["final_holdout_indices"])
