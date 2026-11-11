"""One fixed small-model data-adaptation pilot, with artist-purged FMAK splits."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from drone.checkpoints import load_checkpoint
from experiments.small import (
    Recipe,
    atomic_json,
    development_metrics,
    load_development,
    run_recipe,
    signature,
    snapshot_sources,
)
from train import load_prepared

POLICY = "fmak-artist-album-audio-purged-70-20-10-hash-v1"


def development_split(meta, partition, audit):
    """Fix label-independent group roles; no final-related artist enters fitting."""
    assert partition["metadata_sha256"] == signature(meta)
    eligible = audit["eligible_development_indices"]
    assert set(eligible) <= set(partition["development_indices"])
    assert not set(eligible) & set(partition["final_holdout_indices"])
    groups = audit["groups"]
    assert not {groups[i] for i in eligible} & {
        groups[i] for i in partition["final_holdout_indices"]
    }
    parts = {"train": [], "validation": [], "calibration": []}
    labels = {}
    seen_training = set()
    for i in eligible:
        digest, ref = meta["tracks"][i]["sha256"], meta["refs"][i]
        if digest in labels and labels[digest] != ref:
            raise ValueError("Conflicting labels on identical eligible audio")
        labels[digest] = ref
        value = int(hashlib.sha256((POLICY + ":" + groups[i]).encode()).hexdigest(), 16) / 2**256
        role = "train" if value < 0.7 else "validation" if value < 0.9 else "calibration"
        if role == "train":
            if digest in seen_training:
                continue
            seen_training.add(digest)
        parts[role].append(i)
    if not all(parts.values()):
        raise ValueError("Need nonempty fit, validation and calibration groups")
    for first, second in (
        ("train", "validation"),
        ("train", "calibration"),
        ("validation", "calibration"),
    ):
        assert not {groups[i] for i in parts[first]} & {groups[i] for i in parts[second]}
        assert not {meta["tracks"][i]["sha256"] for i in parts[first]} & {
            meta["tracks"][i]["sha256"] for i in parts[second]
        }
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beatport", type=Path, required=True)
    parser.add_argument("--fmak", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(16)
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_sources(args.output)
    bp, bm, btr, bval = load_development(args.beatport)
    fm, mm = load_prepared(args.fmak, "fmak")
    partition_bytes, group_bytes = args.partition.read_bytes(), args.groups.read_bytes()
    partition, audit = json.loads(partition_bytes), json.loads(group_bytes)
    assert audit["partition_sha256"] == hashlib.sha256(partition_bytes).hexdigest()
    assert not {t["sha256"] for t in bm["tracks"]} & {t["sha256"] for t in mm["tracks"]}
    parts = development_split(mm, partition, audit)
    split_record = {
        "policy": POLICY,
        "partition_sha256": hashlib.sha256(partition_bytes).hexdigest(),
        "group_audit_sha256": hashlib.sha256(group_bytes).hexdigest(),
        "fmak": parts,
        "beatport_train": btr,
        "beatport_validation": bval,
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
            "beatport_train": len(btr),
            "beatport_validation": len(bval),
            "fmak": {k: len(v) for k, v in parts.items()},
        },
        flush=True,
    )
    # Only fitting/validation rows enter the merged training bundle. Calibration
    # and final-holdout arrays are never passed to training or metrics.
    entries = [("beatport", bp, bm, i) for i in btr]
    entries += [("fmak", fm, mm, i) for i in parts["train"]]
    ntrain = len(entries)
    entries += [("beatport", bp, bm, i) for i in bval]
    nbval = len(bval)
    entries += [("fmak", fm, mm, i) for i in parts["validation"]]
    features = np.stack([array[i] for _, array, _, i in entries])
    metadata = {
        "feature_spec": bm["feature_spec"],
        "label_schema": bm["label_schema"],
        "frames": bm["frames"],
        "ids": [tag + ":" + meta["ids"][i] for tag, _, meta, i in entries],
        "refs": [meta["refs"][i] for _, _, meta, i in entries],
        "tracks": [
            dict(meta["tracks"][i], id=tag + ":" + meta["ids"][i]) for tag, _, meta, i in entries
        ],
        "split_record": split_record,
    }
    tr, val = list(range(ntrain)), list(range(ntrain, len(entries)))
    assert bm["feature_spec"] == mm["feature_spec"] and bm["frames"] == mm["frames"]
    recipe = Recipe("frequency-lr-3e4-joint-data", pooling="frequency", lr=3e-4)
    domains = {"beatport": val[:nbval], "fmak": val[nbval:]}
    baseline, _ = load_checkpoint(args.baseline)
    baseline_metrics = {
        name: development_metrics(
            baseline, features[indices], [metadata["refs"][i] for i in indices]
        )
        for name, indices in domains.items()
    }
    atomic_json(args.output / "baseline.json", baseline_metrics)
    run_recipe(
        features,
        metadata,
        tr,
        val,
        recipe,
        seed=13,
        output=args.output / "seed13",
        max_seconds=600,
        split_policy=POLICY,
    )
    model, _ = load_checkpoint(args.output / "seed13/best.pt")
    metrics = {
        name: development_metrics(model, features[indices], [metadata["refs"][i] for i in indices])
        for name, indices in domains.items()
    }
    delta = {name: metrics[name]["mirex"] - baseline_metrics[name]["mirex"] for name in domains}
    result = {
        "seed": 13,
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
