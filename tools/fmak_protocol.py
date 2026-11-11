"""Create a label-independent, exact-audio-grouped FMAK final holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Collection
from pathlib import Path
from typing import Any

from experiments.reference.train import load_prepared


def partition(
    meta: dict[str, Any], excluded_audio: Collection[str] = (), holdout_size: int = 1000
) -> dict[str, Any]:
    excluded_audio = set(excluded_audio)
    grouped = defaultdict(list)
    excluded = []
    for index, track in enumerate(meta["tracks"]):
        if track["sha256"] in excluded_audio:
            excluded.append(index)
        else:
            grouped[track["sha256"]].append(index)
    ordered = sorted(
        grouped,
        key=lambda digest: hashlib.sha256(("drone-fmak-holdout-v1:" + digest).encode()).hexdigest(),
    )
    test, dev = [], []
    for digest in ordered:
        rows = grouped[digest]
        if len(test) + len(rows) <= holdout_size:
            test.extend(rows)
        else:
            dev.extend(rows)
    if len(test) != holdout_size or not dev:
        raise ValueError("Cannot form the requested grouped holdout and nonempty development set")
    test.sort()
    dev.sort()
    return {
        "policy": "drone-fmak-holdout-v1",
        "development_indices": dev,
        "final_holdout_indices": test,
        "excluded_indices": excluded,
        "development_ids": [meta["ids"][i] for i in dev],
        "final_holdout_ids": [meta["ids"][i] for i in test],
        "excluded_ids": [meta["ids"][i] for i in excluded],
        "unique_audio_groups": len(grouped),
        "metadata_sha256": hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--beatport-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, meta = load_prepared(args.data, "fmak")
    _, beatport = load_prepared(args.beatport_data, "train")
    result = partition(meta, [t["sha256"] for t in beatport["tracks"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if json.loads(args.output.read_text()) != result:
            raise ValueError("Existing partition differs; refusing to overwrite a frozen protocol")
    else:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        {
            "development": len(result["development_indices"]),
            "final_holdout": len(result["final_holdout_indices"]),
            "excluded": len(result["excluded_indices"]),
        }
    )


if __name__ == "__main__":
    main()
