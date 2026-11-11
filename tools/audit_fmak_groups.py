"""Audit official artist/album groups against the frozen FMAK holdout; no scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.reference.train import load_prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, meta = load_prepared(args.data, "fmak")
    split_bytes, metadata_bytes = args.partition.read_bytes(), args.metadata.read_bytes()
    split = json.loads(split_bytes)
    official = json.loads(metadata_bytes)
    assert (
        split["metadata_sha256"]
        == hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    )
    parents = {}

    def root(node: str) -> str:
        parents.setdefault(node, node)
        if parents[node] != node:
            parents[node] = root(parents[node])
        return parents[node]

    nodes = []
    missing = []
    for i, identity in enumerate(meta["ids"]):
        row = official["tracks"].get(identity, {})
        known = [
            f"{kind}:{row.get(kind + '_id', '')}"
            for kind in ("artist", "album")
            if row.get(kind + "_id", "").isdigit() and int(row[kind + "_id"]) > 0
        ]
        known.append("audio:" + meta["tracks"][i]["sha256"])
        if not row.get("artist_id", "").isdigit() or int(row["artist_id"]) <= 0:
            missing.append(i)
        if not known:
            known = [f"unknown:{identity}"]
        for node in known[1:]:
            parents[root(node)] = root(known[0])
        nodes.append(known[0])
    groups = [root(node) for node in nodes]
    held_out_groups = {groups[i] for i in split["final_holdout_indices"]}
    quarantine = [
        i for i in split["development_indices"] if groups[i] in held_out_groups or i in missing
    ]
    quarantine_set = set(quarantine)
    eligible = [i for i in split["development_indices"] if i not in quarantine_set]
    assert not {groups[i] for i in eligible} & held_out_groups
    result = {
        "policy": "artist-album-audio-connected-components-primary-credits-v1",
        "partition_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "official_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "tracks_csv_sha256": official["tracks_csv_sha256"],
        "tracks": len(groups),
        "components": len(set(groups)),
        "largest_component": max(Counter(groups).values()),
        "missing_metadata_indices": missing,
        "groups": groups,
        "quarantined_development_indices": quarantine,
        "eligible_development_indices": eligible,
        "final_holdout_unchanged": True,
        "limitations": "Exact audio and primary artist/album IDs; no acoustic near-duplicate or remix audit.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        {
            "eligible_development": len(eligible),
            "quarantined": len(quarantine),
            "missing_metadata": len(missing),
            "components": len(set(groups)),
        }
    )


if __name__ == "__main__":
    main()
