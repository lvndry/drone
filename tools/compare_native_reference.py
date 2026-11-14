"""Score Euphoria's native predictions on precisely the frozen FMAK development set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from drone.keys import WEIGHTS, KeyIndices, key_relation
from experiments.reference.train import load_prepared


def weights(predictions: KeyIndices, references: KeyIndices, symmetric: bool = True) -> np.ndarray:
    return np.array(
        [
            WEIGHTS[key_relation(int(prediction), int(reference), symmetric=symmetric)]
            if prediction >= 0
            else 0
            for prediction, reference in zip(predictions, references, strict=True)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--native-provenance", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, meta = load_prepared(args.data, "fmak")
    partition_bytes = args.partition.read_bytes()
    split = json.loads(partition_bytes)
    partition_hash = hashlib.sha256(partition_bytes).hexdigest()
    assert (
        split["metadata_sha256"]
        == hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    )
    provenance = json.loads(args.native_provenance.read_bytes())
    candidate = json.loads(args.candidate.read_bytes())
    groups = json.loads(args.groups.read_bytes())
    for record in (provenance, candidate, groups):
        assert record["partition_sha256"] == partition_hash
    assert not provenance["final_holdout_scored"] and not candidate["final_holdout_scored"]
    ids = [meta["ids"][index] for index in split["development_indices"]]
    references = np.array([meta["refs"][index] for index in split["development_indices"]])
    with args.native.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(ids) and len({row["id"] for row in rows}) == len(ids)
    native = {row["id"]: row for row in rows}
    assert set(native) == set(ids)
    native_predictions = np.array([int(native[identity]["prediction"]) for identity in ids])
    assert ((native_predictions >= -1) & (native_predictions < 24)).all()
    cnn = np.array(candidate["metrics"]["predictions"])
    assert len(cnn) == len(references) and ((cnn >= 0) & (cnn < 24)).all()
    native_scores = weights(native_predictions, references)
    cnn_scores = weights(cnn, references)
    delta = cnn_scores - native_scores
    labels = [groups["groups"][index] for index in split["development_indices"]]
    unique = sorted(set(labels))
    position = {name: index for index, name in enumerate(unique)}
    group_index = np.array([position[name] for name in labels])
    sizes = np.bincount(group_index)
    totals = np.bincount(group_index, weights=delta)
    generator = np.random.default_rng(13)
    draws = generator.integers(len(unique), size=(5000, len(unique)))
    bootstrap = totals[draws].sum(1) / sizes[draws].sum(1)
    valid = native_predictions >= 0
    result = {
        "corpus": "FMAKv2 development only",
        "partition_sha256": partition_hash,
        "final_holdout_scored": False,
        "tracks": len(ids),
        "native_provenance": provenance,
        "native_csv_sha256": hashlib.sha256(args.native.read_bytes()).hexdigest(),
        "candidate_checkpoint_sha256": candidate["checkpoint_sha256"],
        "native_metrics": {
            "exact": float(np.mean(native_predictions == references)),
            "mirex": float(native_scores.mean()),
            "mirex_asymmetric": float(
                weights(native_predictions, references, symmetric=False).mean()
            ),
            "mode_accuracy": float(np.mean(valid & (native_predictions % 2 == references % 2))),
            "missing_predictions": int((~valid).sum()),
            "predicted_counts": np.bincount(native_predictions[valid], minlength=24).tolist(),
        },
        "candidate_metrics": {
            metric: candidate["metrics"][metric] for metric in ("exact", "mirex", "mode_accuracy")
        },
        "candidate_minus_native_mirex": float(delta.mean()),
        "paired_artist_album_cluster_bootstrap_95ci": np.quantile(
            bootstrap, [0.025, 0.975]
        ).tolist(),
        "bootstrap": {"repetitions": 5000, "seed": 13, "clusters": len(unique)},
        "candidate_minus_native_common_decoded_mirex": float(delta[valid].mean())
        if valid.any()
        else None,
        "native_cpu_seconds_p50_p95": np.quantile(
            [float(native[identity]["cpu_seconds"]) for identity in ids], [0.5, 0.95]
        ).tolist(),
        "limitations": "Development result; primary artist/album grouping only. Native timings use the concurrent worker count in provenance and include decode; no matched runtime speed claim.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
