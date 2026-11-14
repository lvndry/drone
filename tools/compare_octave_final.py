"""Compare frozen final predictions with fixed native Euphoria; no model selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from drone.keys import WEIGHTS, KeyIndices, key_relation


def scores(predictions: KeyIndices, references: KeyIndices, symmetric: bool = True) -> np.ndarray:
    return np.array(
        [
            WEIGHTS[key_relation(int(prediction), int(reference), symmetric=symmetric)]
            if prediction >= 0
            else 0
            for prediction, reference in zip(predictions, references, strict=True)
        ]
    )


def compare(neural: dict[str, Any], native_predictions: KeyIndices) -> dict[str, Any]:
    references = np.array(neural["references"])
    predictions = np.array(neural["predictions"])
    native = np.array(native_predictions)
    assert len(native) == len(references) and ((native >= -1) & (native < 24)).all()
    cnn_weights = scores(predictions, references)
    native_weights = scores(native, references)
    valid = native >= 0
    groups, inverse = np.unique(neural["groups"], return_inverse=True)
    sizes = np.bincount(inverse)
    generator = np.random.default_rng(13)
    draws = generator.integers(len(groups), size=(5000, len(groups)))
    intervals = {}
    for name, difference in (
        ("mirex", cnn_weights - native_weights),
        ("exact", (predictions == references).astype(float) - (native == references)),
    ):
        totals = np.bincount(inverse, weights=difference)
        values = totals[draws].sum(1) / sizes[draws].sum(1)
        intervals[name] = {
            "delta": float(difference.mean()),
            "paired_cluster_bootstrap_95ci": np.quantile(values, [0.025, 0.975]).tolist(),
        }
    return {
        "tracks": len(references),
        "native": {
            "exact": float(np.mean(native == references)),
            "mirex": float(native_weights.mean()),
            "mirex_asymmetric": float(scores(native, references, symmetric=False).mean()),
            "mode_accuracy": float(np.mean(valid & (native % 2 == references % 2))),
            "missing_predictions": int((~valid).sum()),
        },
        "neural": {
            key: neural[key]
            for key in ("exact", "mirex", "mirex_asymmetric", "mode_accuracy", "uncertainty")
        },
        "paired": intervals,
        "common_decoded_tracks": int(valid.sum()),
        "mirex_delta_common_decoded": float((cnn_weights - native_weights)[valid].mean())
        if valid.any()
        else None,
        "bootstrap": {"seed": 13, "repetitions": 5000, "clusters": len(groups)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "fmak",
        "native-csv",
        "native-provenance",
        "giantsteps",
        "native-giantsteps",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    fmak = json.loads(args.fmak.read_bytes())
    giantsteps = json.loads(args.giantsteps.read_bytes())
    native_giantsteps = json.loads(args.native_giantsteps.read_bytes())
    provenance = json.loads(args.native_provenance.read_bytes())
    assert fmak["freeze_sha256"] == giantsteps["freeze_sha256"] == provenance["freeze_sha256"]
    assert giantsteps["test_metadata_sha256"] == native_giantsteps["test_metadata_sha256"]
    assert fmak["mode"] == "fmak-final" and giantsteps["mode"] == "giantsteps"
    assert fmak["tracks"] == 1000 and giantsteps["tracks"] == 604
    with args.native_csv.open() as stream:
        rows = list(csv.DictReader(stream))
    lookup = {row["id"]: int(row["prediction"]) for row in rows}
    assert len(rows) == len(lookup) == len(fmak["ids"]) and set(lookup) == set(fmak["ids"])
    result: dict[str, Any] = {
        "model_choice": fmak["model_choice"],
        "freeze_sha256": fmak["freeze_sha256"],
        "native_provenance": provenance,
        "fmak_final": compare(fmak, [lookup[identity] for identity in fmak["ids"]]),
        "giantsteps": compare(giantsteps, native_giantsteps["predictions"]),
        "comparison_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": "Our artist-purged FMAK partition is not directly comparable to published all-FMAK/different-holdout results. GiantSteps was exposed earlier in this research program; its intervals use track hashes, not artist groups. Confidence intervals do not remove annotation ambiguity or unrecognized remix overlap.",
    }
    result["meets_observed_quality_gate"] = all(
        float(result[name]["paired"][metric]["delta"]) >= 0
        for name in ("fmak_final", "giantsteps")
        for metric in ("mirex", "exact")
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
