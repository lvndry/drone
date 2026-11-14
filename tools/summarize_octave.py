"""Audit fixed octave confirmation and select only preregistered inference candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from drone.keys import WEIGHTS, KeyIndices, key_relation
from experiments.octave import device_setup, metrics, probabilities
from experiments.octave_model import FEATURE_SPEC, OctaveNet
from experiments.small import signature


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "pilot", "data", "native-beatport", "native-fmak"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    seeds = (13, 37, 73)
    runs = [read(args.root / f"seed{seed}/run.json") for seed in seeds]
    assert all(run["status"] == "complete" for run in runs)
    assert [run["identity"]["seed"] for run in runs] == list(seeds)
    identities = [dict(run["identity"]) for run in runs]
    for identity in identities:
        identity.pop("seed")
    assert identities[0] == identities[1] == identities[2]
    pilot = read(args.pilot / "run.json")
    repeated_predictions = pilot["domain_metrics"] == runs[0]["domain_metrics"]
    repeated_history = pilot["history"] == runs[0]["history"]
    assert repeated_predictions, "Repeated seed13 validation differs; investigate before freezing"
    manifest = read(args.data / "manifest.json")
    assert signature(manifest) == identities[0]["feature_manifest_sha256"]
    assert manifest["feature_spec"] == FEATURE_SPEC
    rows = manifest["rows"]
    validation_indices = [index for index, row in enumerate(rows) if row["role"] == "validation"]
    references = np.array([row["ref"] for row in rows])[validation_indices]
    positions = {
        name: [
            position
            for position, row_index in enumerate(validation_indices)
            if rows[row_index]["domain"] == name
        ]
        for name in ("beatport", "fmak")
    }
    native = {"beatport": read(args.native_beatport), "fmak": read(args.native_fmak)}
    for name in positions:
        expected = [rows[validation_indices[position]]["id"] for position in positions[name]]
        actual = native[name]["ids"]
        if name == "beatport":
            actual = ["beatport:" + identity for identity in actual]
        assert actual == expected
    device_setup(args.device)
    array = np.load(args.data / "features.npy", mmap_mode="r", allow_pickle=False)
    seed_probabilities = []
    checkpoints = []
    for seed, run in zip(seeds, runs, strict=True):
        path = args.root / f"seed{seed}/best.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["format_version"] == "octave-supervised-pilot-v1"
        assert payload["feature_spec"] == FEATURE_SPEC and payload["identity"] == run["identity"]
        model = OctaveNet().to(args.device)
        model.load_state_dict(payload["state_dict"])
        probability = probabilities(model, array, validation_indices, args.device)
        for name, index in positions.items():
            assert metrics(probability[index], references[index]) == run["domain_metrics"][name]
        seed_probabilities.append(probability)
        checkpoints.append(
            {
                "seed": seed,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        del model
    candidates = {
        "canonical-seed13": seed_probabilities[0],
        "ensemble-all-three": np.mean(seed_probabilities, axis=0),
    }
    evaluated = {
        name: {
            "combined": metrics(candidate_probability, references),
            **{
                domain: metrics(candidate_probability[index], references[index])
                for domain, index in positions.items()
            },
        }
        for name, candidate_probability in candidates.items()
    }
    canonical = evaluated["canonical-seed13"]
    ensemble = evaluated["ensemble-all-three"]
    use_ensemble = ensemble["combined"]["mirex"] >= canonical["combined"]["mirex"] + 0.01 and all(
        ensemble[name]["mirex"] >= canonical[name]["mirex"] - 0.005 for name in positions
    )
    choice = "ensemble-all-three" if use_ensemble else "canonical-seed13"
    means = {}
    for domain in positions:
        values = [run["domain_metrics"][domain]["mirex"] for run in runs]
        means[domain] = {
            "mirex_mean": statistics.mean(values),
            "mirex_sample_sd": statistics.stdev(values),
            "exact_mean": statistics.mean(run["domain_metrics"][domain]["exact"] for run in runs),
            "seed_mirex": dict(zip(map(str, seeds), values, strict=True)),
            "native_mirex": native[domain]["mirex"],
            "native_exact": native[domain]["exact"],
        }
    eligible = all(
        means[name]["mirex_mean"] >= native[name]["mirex"]
        and evaluated[choice][name]["mirex"] >= native[name]["mirex"]
        for name in positions
    )
    paired = {}
    for name, index in positions.items():
        truth = references[index]
        cnn = evaluated[choice][name]["predictions"]
        dsp = native[name]["predictions"]

        def scores(predictions: KeyIndices) -> np.ndarray:
            return np.array(
                [
                    WEIGHTS[key_relation(int(prediction), int(reference))] if prediction >= 0 else 0
                    for prediction, reference in zip(predictions, truth, strict=True)
                ]
            )

        paired[name] = {
            "delta_mirex": float((scores(cnn) - scores(dsp)).mean()),
            "oracle_max_mirex_diagnostic": float(np.maximum(scores(cnn), scores(dsp)).mean()),
        }
    result = {
        "choice": choice,
        "eligible_for_final_evaluation": eligible,
        "feature_spec": FEATURE_SPEC,
        "feature_manifest_sha256": signature(manifest),
        "identity_without_seed": identities[0],
        "selected_checkpoints": checkpoints if use_ensemble else checkpoints[:1],
        "all_checkpoints": checkpoints,
        "confirmation": means,
        "candidates": evaluated,
        "paired_native_diagnostics": paired,
        "seed13_predictions_reproduce_pilot": repeated_predictions,
        "seed13_histories_reproduce_pilot": repeated_history,
        "seed13_max_loss_difference": max(
            abs(pilot_epoch["loss"] - repeated_epoch["loss"])
            for pilot_epoch, repeated_epoch in zip(
                pilot["history"], runs[0]["history"], strict=True
            )
        ),
        "seed13_validation_epoch_score_differences": sum(
            pilot_epoch["validation_mirex"] != repeated_epoch["validation_mirex"]
            for pilot_epoch, repeated_epoch in zip(
                pilot["history"], runs[0]["history"], strict=True
            )
        ),
        "elapsed_seconds": sum(run["elapsed_seconds"] for run in runs),
        "calibration_used": False,
        "final_holdout_used": False,
        "selection_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": "Selected validation results; no final quality claim. Oracle is an unattainable diagnostic, not a trained fusion model.",
    }
    target = args.root / "selection.json"
    if target.exists() and read(target) != result:
        raise ValueError("Existing selection differs; refusing to overwrite")
    target.write_text(json.dumps(result, indent=2) + "\n")
    seed13_probabilities, seed37_probabilities, seed73_probabilities = seed_probabilities
    np.savez(
        args.root / "validation-probabilities.npz",
        seed13=seed13_probabilities,
        seed37=seed37_probabilities,
        seed73=seed73_probabilities,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "choice",
                    "eligible_for_final_evaluation",
                    "confirmation",
                    "seed13_histories_reproduce_pilot",
                    "paired_native_diagnostics",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
