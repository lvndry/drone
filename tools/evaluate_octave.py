"""Calibrate a frozen eligible octave candidate, then evaluate sealed datasets once."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drone.keys import score_breakdown, score_mirex
from experiments.adapt import development_split
from experiments.octave import decode, device_setup, probabilities
from experiments.octave_model import FEATURE_SPEC, OctaveNet, compress
from experiments.reference.datasets import load_giantsteps
from experiments.reference.features import audio_digest
from experiments.reference.train import load_prepared
from experiments.small import atomic_json, signature


def calibrated(probability: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.maximum(probability, 1e-12)) / temperature
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def uncertainty(probability: np.ndarray, refs: np.ndarray) -> dict[str, float]:
    confidence, prediction = probability.max(1), probability.argmax(1)
    correct = prediction == refs
    ece = 0.0
    for left, right in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:], strict=True):
        chosen = (confidence > left) & (confidence <= right)
        if chosen.any():
            ece += chosen.mean() * abs(correct[chosen].mean() - confidence[chosen].mean())
    one_hot = np.eye(24)[refs]
    return {
        "nll": float(-np.log(np.maximum(probability[np.arange(len(refs)), refs], 1e-12)).mean()),
        "brier": float(np.square(probability - one_hot).sum(1).mean()),
        "ece_15_equal_width": float(ece),
    }


def evaluation_decode(row: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Recover broken MP3 seeking/duration metadata without changing valid inputs."""
    try:
        return decode(row), None
    except (subprocess.CalledProcessError, ValueError) as error:
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                row["path"],
                "-threads",
                "1",
                "-ac",
                "1",
                "-ar",
                "22050",
                "-f",
                "f32le",
                "pipe:1",
            ],
            capture_output=True,
            check=True,
            timeout=120,
        )
        waveform = np.frombuffer(result.stdout, dtype="<f4")
        if not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError(f"Full decode also failed: {row['id']}") from error
        duration = len(waveform) / 22050
        segments = []
        for fraction in (0.25, 0.75):
            start = max(0, min(duration - 32, duration * fraction - 16))
            first = round(start * 22050)
            segment = waveform[first : first + 705600]
            segments.append(np.pad(segment, (0, 705600 - len(segment))))
        return np.stack(segments), {
            "id": row["id"],
            "route": "full-decode-actual-duration",
            "fast_error": type(error).__name__,
            "actual_duration_seconds": duration,
        }


def rows_for(
    args: argparse.Namespace, selection: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, meta = load_prepared(args.fmak, "fmak")
    partition_bytes, group_bytes = args.partition.read_bytes(), args.groups.read_bytes()
    partition, audit = json.loads(partition_bytes), json.loads(group_bytes)
    assert partition["metadata_sha256"] == signature(meta)
    assert audit["partition_sha256"] == hashlib.sha256(partition_bytes).hexdigest()
    if args.mode == "calibrate":
        indices = development_split(meta, partition, audit)["calibration"]
    elif args.mode == "fmak-final":
        indices = partition["final_holdout_indices"]
        assert len(indices) == 1000
    else:
        _, gs = load_prepared(args.beatport, "test")
        tracks = load_giantsteps(args.beatport / "giantsteps")
        lookup = {t.id: t for t in tracks}
        rows = [
            {
                "id": "giantsteps:" + identity,
                "path": str(lookup[identity].mp3),
                "ref": gs["refs"][i],
                "sha256": gs["tracks"][i]["sha256"],
                "group": gs["tracks"][i]["sha256"],
            }
            for i, identity in enumerate(gs["ids"])
        ]
        assert len(rows) == 604
        return rows, {
            "test_metadata_sha256": signature(gs),
            "giantsteps_program_exposure_disclosed": True,
        }
    rows = [
        {
            "id": "fmak:" + meta["ids"][i],
            "path": str(args.fmak / "audio" / f"{meta['ids'][i]}.mp3"),
            "ref": meta["refs"][i],
            "sha256": meta["tracks"][i]["sha256"],
            "group": audit["groups"][i],
        }
        for i in indices
    ]
    return rows, {
        "partition_sha256": hashlib.sha256(partition_bytes).hexdigest(),
        "group_audit_sha256": hashlib.sha256(group_bytes).hexdigest(),
    }


def features_for(
    rows: list[dict[str, Any]], cache: Path, device: str
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from nnAudio.features import VQT

    cache = cache / signature(FEATURE_SPEC)[:16]
    cache.mkdir(parents=True, exist_ok=True)
    fallback_path = cache / "decode-fallbacks.json"
    fallbacks = json.loads(fallback_path.read_bytes()) if fallback_path.exists() else {}
    transform = VQT(
        sr=22050,
        hop_length=512,
        fmin=27.5,
        n_bins=99,
        bins_per_octave=12,
        gamma=0,
        trainable=False,
        verbose=False,
    ).to(device)
    start = time.monotonic()
    missing = [row for row in rows if not (cache / (row["sha256"] + ".npy")).exists()]
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for offset in range(0, len(missing), 8):
            if time.monotonic() - start > 600:
                raise TimeoutError("Ten-minute frozen-evaluation feature cap reached")
            batch = missing[offset : offset + 8]
            waves = list(pool.map(evaluation_decode, batch))
            for row, (wave, fallback) in zip(batch, waves, strict=True):
                if fallback is not None:
                    fallbacks[row["id"]] = fallback
                result = compress(transform(torch.from_numpy(wave).to(device))).cpu().numpy()
                assert result.shape == (2, 99, 1379) and np.isfinite(result).all()
                path = cache / (row["sha256"] + ".npy")
                pending = path.with_suffix(".pending")
                with pending.open("wb") as stream:
                    np.save(stream, result)
                pending.replace(path)
            atomic_json(fallback_path, fallbacks)
            if (offset + len(batch)) % 80 == 0 or offset + len(batch) == len(missing):
                print(
                    f"Frozen {len(rows)}-row evaluation: prepared {offset + len(batch)}/{len(missing)} new features",
                    flush=True,
                )
    array = np.stack(
        [np.load(cache / (row["sha256"] + ".npy"), allow_pickle=False) for row in rows]
    )
    return array, [fallbacks[row["id"]] for row in rows if row["id"] in fallbacks]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "fmak-final", "giantsteps"), required=True)
    for name in ("selection", "fmak", "beatport", "partition", "groups", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    selection_bytes = args.selection.read_bytes()
    selection = json.loads(selection_bytes)
    if not selection["eligible_for_final_evaluation"]:
        raise ValueError("Candidate has not passed the frozen development gate")
    assert selection["feature_spec"] == FEATURE_SPEC
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / (args.mode + ".json")
    if target.exists():
        old = json.loads(target.read_bytes())
        assert old["selection_sha256"] == hashlib.sha256(selection_bytes).hexdigest()
        print(f"Already evaluated: {target}; refusing repeated scoring", flush=True)
        return
    freeze_path = args.output / "freeze.json"
    freeze = None
    if args.mode != "calibrate":
        freeze = json.loads(freeze_path.read_bytes())
        assert freeze["selection_sha256"] == hashlib.sha256(selection_bytes).hexdigest()
        assert freeze["feature_spec"] == FEATURE_SPEC
    device_setup(args.device)
    models, used_audio, used_ids, validation_ids = [], set(), set(), set()
    for record in selection["selected_checkpoints"]:
        path = Path(record["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["format_version"] == "octave-supervised-pilot-v1"
        assert payload["feature_spec"] == FEATURE_SPEC
        used_audio.update(payload["training_audio_sha256"])
        used_ids.update(payload["train_ids"])
        validation_ids.update(payload["validation_ids"])
        model = OctaveNet().to(args.device)
        model.load_state_dict(payload["state_dict"])
        models.append(model)
    rows, provenance = rows_for(args, selection)
    for row in rows:
        if audio_digest(row["path"]) != row["sha256"]:
            raise ValueError("Evaluation audio differs from the pinned manifest")
    assert not used_audio & {row["sha256"] for row in rows}
    assert not (used_ids | validation_ids) & {row["id"] for row in rows}
    evaluation_ids = args.output / (args.mode + "-identities.json")
    identities = {
        "feature_spec": FEATURE_SPEC,
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "rows": rows,
        **provenance,
    }
    if evaluation_ids.exists():
        assert json.loads(evaluation_ids.read_bytes()) == identities
    else:
        atomic_json(evaluation_ids, identities)
    array, fallbacks = features_for(rows, args.output / "features", args.device)
    predictions = [
        probabilities(model, array, list(range(len(rows))), args.device) for model in models
    ]
    raw = np.mean(predictions, axis=0)
    refs = np.array([row["ref"] for row in rows])
    result = {
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "identities_sha256": signature(identities),
        "tracks": len(rows),
        "mode": args.mode,
        "feature_spec": FEATURE_SPEC,
        "model_choice": selection["choice"],
        "decode_fallbacks": fallbacks,
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **provenance,
    }
    if args.mode == "calibrate":
        temperatures = np.exp(np.linspace(np.log(0.25), np.log(4), 401))
        losses = [
            uncertainty(calibrated(raw, temperature), refs)["nll"] for temperature in temperatures
        ]
        temperature = float(temperatures[int(np.argmin(losses))])
        calibrated_prob = calibrated(raw, temperature)
        assert np.array_equal(raw.argmax(1), calibrated_prob.argmax(1))
        result.update(
            temperature=temperature,
            raw_uncertainty=uncertainty(raw, refs),
            fitted_uncertainty=uncertainty(calibrated_prob, refs),
            accuracy_used_for_selection=False,
        )
        freeze = {
            "selection_sha256": result["selection_sha256"],
            "feature_spec": FEATURE_SPEC,
            "selected_checkpoints": selection["selected_checkpoints"],
            "temperature": temperature,
            "calibration_identity_sha256": result["identities_sha256"],
            "calibration_rows": len(rows),
            "temperature_search": "401 log-spaced values in [0.25,4], calibration NLL only",
            "final_scores_observed_at_freeze": False,
            "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if freeze_path.exists():
            previous_freeze = json.loads(freeze_path.read_bytes())
            freeze["frozen_at_utc"] = previous_freeze["frozen_at_utc"]
            assert previous_freeze == freeze
        else:
            atomic_json(freeze_path, freeze)
    else:
        assert freeze is not None
        probability = calibrated(raw, freeze["temperature"])
        pred = probability.argmax(1)
        assert np.array_equal(pred, raw.argmax(1))
        exact, mirex = score_mirex(pred, refs)
        matrix = np.zeros((24, 24), dtype=int)
        np.add.at(matrix, (refs, pred), 1)
        result.update(
            freeze_sha256=hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
            exact=exact,
            mirex=mirex,
            mirex_asymmetric=score_mirex(pred, refs, symmetric=False)[1],
            mode_accuracy=float(np.mean(pred % 2 == refs % 2)),
            uncertainty=uncertainty(probability, refs),
            counts=dict(score_breakdown(pred, refs)),
            confusion_matrix=matrix.tolist(),
            predictions=pred.tolist(),
            ids=[row["id"] for row in rows],
            groups=[row["group"] for row in rows],
            references=refs.tolist(),
        )
    np.savez(args.output / (args.mode + "-probabilities.npz"), raw=raw, refs=refs)
    atomic_json(target, result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key
                in (
                    "mode",
                    "tracks",
                    "exact",
                    "mirex",
                    "mode_accuracy",
                    "temperature",
                    "raw_uncertainty",
                    "fitted_uncertainty",
                    "uncertainty",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
