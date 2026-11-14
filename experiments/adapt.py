"""One fixed small-model data-adaptation pilot, with artist-purged FMAK splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.reference.checkpoints import load_checkpoint
from experiments.reference.train import load_prepared
from experiments.small import (
    Recipe,
    atomic_json,
    development_metrics,
    load_development,
    run_recipe,
    signature,
    snapshot_sources,
)

POLICY = "fmak-artist-album-audio-purged-70-20-10-hash-v1"


def development_split(
    meta: dict[str, Any], partition: dict[str, Any], audit: dict[str, Any]
) -> dict[str, list[int]]:
    """Fix label-independent group roles; no final-related artist enters fitting."""
    assert partition["metadata_sha256"] == signature(meta)
    eligible = audit["eligible_development_indices"]
    assert set(eligible) <= set(partition["development_indices"])
    assert not set(eligible) & set(partition["final_holdout_indices"])
    groups = audit["groups"]
    assert not {groups[index] for index in eligible} & {
        groups[index] for index in partition["final_holdout_indices"]
    }
    parts = {"train": [], "validation": [], "calibration": []}
    labels = {}
    seen_training = set()
    for index in eligible:
        digest = meta["tracks"][index]["sha256"]
        reference = meta["refs"][index]
        if digest in labels and labels[digest] != reference:
            raise ValueError("Conflicting labels on identical eligible audio")
        labels[digest] = reference
        value = (
            int(hashlib.sha256((POLICY + ":" + groups[index]).encode()).hexdigest(), 16) / 2**256
        )
        role = "train" if value < 0.7 else "validation" if value < 0.9 else "calibration"
        if role == "train":
            if digest in seen_training:
                continue
            seen_training.add(digest)
        parts[role].append(index)
    if not all(parts.values()):
        raise ValueError("Need nonempty fit, validation and calibration groups")
    for first, second in (
        ("train", "validation"),
        ("train", "calibration"),
        ("validation", "calibration"),
    ):
        assert not {groups[index] for index in parts[first]} & {
            groups[index] for index in parts[second]
        }
        assert not {meta["tracks"][index]["sha256"] for index in parts[first]} & {
            meta["tracks"][index]["sha256"] for index in parts[second]
        }
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beatport", type=Path, required=True)
    parser.add_argument("--fmak", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(13, 37, 73), default=13)
    args = parser.parse_args()
    torch.set_num_threads(16)
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_sources(args.output)
    beatport_features, beatport_metadata, beatport_train, beatport_validation = load_development(
        args.beatport
    )
    fmak_features, fmak_metadata = load_prepared(args.fmak, "fmak")
    partition_bytes = args.partition.read_bytes()
    group_bytes = args.groups.read_bytes()
    partition = json.loads(partition_bytes)
    audit = json.loads(group_bytes)
    assert audit["partition_sha256"] == hashlib.sha256(partition_bytes).hexdigest()
    assert not {track["sha256"] for track in beatport_metadata["tracks"]} & {
        track["sha256"] for track in fmak_metadata["tracks"]
    }
    parts = development_split(fmak_metadata, partition, audit)
    split_record = {
        "policy": POLICY,
        "partition_sha256": hashlib.sha256(partition_bytes).hexdigest(),
        "group_audit_sha256": hashlib.sha256(group_bytes).hexdigest(),
        "fmak": parts,
        "beatport_train": beatport_train,
        "beatport_validation": beatport_validation,
        "selection": "Combined validation per-track symmetric MIREX; exact epoch ties keep earlier epoch",
        "gate": "FMAK validation gain >=0.02 and Beatport validation loss <=0.02 versus frozen seed13 candidate",
        "final_holdout_used": False,
        "calibration_used": False,
    }
    path = args.output / "split.json"
    if path.exists() and json.loads(path.read_bytes()) != split_record:
        raise ValueError("Frozen adaptation split differs")
    atomic_json(path, split_record)
    print(
        {
            "beatport_train": len(beatport_train),
            "beatport_validation": len(beatport_validation),
            "fmak": {role: len(indices) for role, indices in parts.items()},
        },
        flush=True,
    )
    # Only fitting/validation rows enter the merged training bundle. Calibration
    # and final-holdout arrays are never passed to training or metrics.
    entries = [
        ("beatport", beatport_features, beatport_metadata, index) for index in beatport_train
    ]
    entries += [("fmak", fmak_features, fmak_metadata, index) for index in parts["train"]]
    training_count = len(entries)
    entries += [
        ("beatport", beatport_features, beatport_metadata, index) for index in beatport_validation
    ]
    beatport_validation_count = len(beatport_validation)
    entries += [("fmak", fmak_features, fmak_metadata, index) for index in parts["validation"]]
    features = np.stack([array[index] for _, array, _, index in entries])
    metadata = {
        "feature_spec": beatport_metadata["feature_spec"],
        "label_schema": beatport_metadata["label_schema"],
        "frames": beatport_metadata["frames"],
        "ids": [tag + ":" + meta["ids"][index] for tag, _, meta, index in entries],
        "refs": [meta["refs"][index] for _, _, meta, index in entries],
        "tracks": [
            dict(meta["tracks"][index], id=tag + ":" + meta["ids"][index])
            for tag, _, meta, index in entries
        ],
        "split_record": split_record,
    }
    training_indices = list(range(training_count))
    validation_indices = list(range(training_count, len(entries)))
    assert (
        beatport_metadata["feature_spec"] == fmak_metadata["feature_spec"]
        and beatport_metadata["frames"] == fmak_metadata["frames"]
    )
    recipe = Recipe("frequency-lr-3e4-joint-data", pooling="frequency", lr=3e-4)
    domains = {
        "beatport": validation_indices[:beatport_validation_count],
        "fmak": validation_indices[beatport_validation_count:],
    }
    baseline, _ = load_checkpoint(args.baseline)
    baseline_metrics = {
        name: development_metrics(
            baseline, features[indices], [metadata["refs"][index] for index in indices]
        )
        for name, indices in domains.items()
    }
    atomic_json(args.output / "baseline.json", baseline_metrics)
    run_recipe(
        features,
        metadata,
        training_indices,
        validation_indices,
        recipe,
        seed=args.seed,
        output=args.output / f"seed{args.seed}",
        max_seconds=600,
        split_policy=POLICY,
    )
    model, _ = load_checkpoint(args.output / f"seed{args.seed}/best.pt")
    metrics = {
        name: development_metrics(
            model, features[indices], [metadata["refs"][index] for index in indices]
        )
        for name, indices in domains.items()
    }
    delta = {name: metrics[name]["mirex"] - baseline_metrics[name]["mirex"] for name in domains}
    result = {
        "seed": args.seed,
        "recipe": recipe.name,
        "metrics": metrics,
        "delta_mirex": delta,
        "qualifies_for_confirmation": delta["fmak"] >= 0.02 and delta["beatport"] >= -0.02,
        "calibration_used": False,
        "final_holdout_used": False,
    }
    atomic_json(args.output / "result.json", result)
    print(
        {"delta_mirex": delta, "qualifies_for_confirmation": result["qualifies_for_confirmation"]},
        flush=True,
    )


if __name__ == "__main__":
    main()
