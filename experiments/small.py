"""Bounded small-model diagnostics and screening; never reads held-out features.

Run with python -m experiments.small. JSON records, immutable source copies and
resumable optimizer state live in the chosen output directory, outside git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from drone.keys import KeyIndices, score_breakdown, score_mirex
from experiments.reference.checkpoints import load_checkpoint, save_checkpoint
from experiments.reference.features import FEATURE_SPEC
from experiments.reference.model import KeyCNN, params_count
from experiments.reference.train import (
    SEED,
    augment_batch,
    evaluate_set,
    load_prepared,
    validation_split,
)


@dataclass(frozen=True)
class Recipe:
    name: str
    pooling: str = "global"
    augmentation: bool = True
    smoothing: float = 0.1
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    mode_balance: bool = False


# Fixed before looking at the screen. No medium/large candidates.
RECIPES = (
    Recipe("control"),
    Recipe("global-no-augmentation", augmentation=False),
    Recipe("frequency", pooling="frequency"),
    Recipe("frequency-no-augmentation", pooling="frequency", augmentation=False),
    Recipe("frequency-no-smoothing", pooling="frequency", smoothing=0),
    Recipe("frequency-dropout-01", pooling="frequency", dropout=0.1),
    Recipe("frequency-lr-3e4", pooling="frequency", lr=3e-4),
    Recipe("global-mode-balanced", mode_balance=True),
)


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(pending, path)


def source_manifest(root: str | Path | None = None) -> dict[str, str]:
    root = Path(root or Path(__file__).resolve().parents[1])
    paths = [root / "train.py", root / "pyproject.toml", root / "uv.lock"]
    for directory in ("drone", "experiments"):
        paths += sorted((root / directory).rglob("*.py"))
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file() and not path.name.startswith(".")
    }


def signature(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def snapshot_sources(output: Path) -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    hashes = source_manifest(root)
    for name, expected in hashes.items():
        target = output / "source" / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(
                "Output directory belongs to different source; use a new experiment ID"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    return hashes


def load_development(data: str | Path) -> tuple[np.ndarray, dict[str, Any], list[int], list[int]]:
    features, meta = load_prepared(data, "train")
    train, validation = validation_split(
        meta["refs"], SEED, [track["sha256"] for track in meta["tracks"]]
    )
    if set(meta["ids"][index] for index in train) & set(meta["ids"][index] for index in validation):
        raise ValueError("Duplicate IDs cross the training/validation boundary")
    if set(meta["tracks"][index]["sha256"] for index in train) & set(
        meta["tracks"][index]["sha256"] for index in validation
    ):
        raise ValueError("Identical audio crosses the training/validation boundary")
    # No test features or labels are opened here.
    return features, meta, train, validation


def probabilities(model: nn.Module, features: np.ndarray, batch: int = 128) -> np.ndarray:
    model.eval()
    results = []
    with torch.inference_mode():
        for start in range(0, len(features), batch):
            results.append(
                torch.softmax(
                    model(torch.from_numpy(features[start : start + batch]).transpose(1, 2)), dim=1
                ).numpy()
            )
    return np.concatenate(results)


def development_metrics(
    model: nn.Module, features: np.ndarray, references: KeyIndices
) -> dict[str, Any]:
    probability = probabilities(model, features)
    prediction = probability.argmax(1)
    reference_array = np.asarray(references)
    predicted_classes = [int(predicted) for predicted in prediction]
    reference_classes = [int(reference) for reference in reference_array]
    exact, mirex = score_mirex(predicted_classes, reference_classes)
    matrix = np.zeros((24, 24), dtype=int)
    np.add.at(matrix, (reference_array, prediction), 1)
    support = matrix.sum(1)
    recalls = np.divide(matrix.diagonal(), support, out=np.zeros(24), where=support > 0)
    return {
        "exact": exact,
        "mirex": mirex,
        "mirex_asymmetric": score_mirex(predicted_classes, reference_classes, symmetric=False)[1],
        "mode_accuracy": float(np.mean(prediction % 2 == reference_array % 2)),
        "macro_recall_present_classes": float(recalls[support > 0].mean()),
        "nll": float(
            -np.log(
                np.maximum(probability[np.arange(len(reference_array)), reference_array], 1e-12)
            ).mean()
        ),
        "counts": dict(score_breakdown(predicted_classes, reference_classes)),
        "predicted_counts": np.bincount(prediction, minlength=24).tolist(),
        "reference_counts": np.bincount(reference_array, minlength=24).tolist(),
        "confusion_matrix": matrix.tolist(),
        "predictions": prediction.tolist(),
    }


def mode_weights(references: KeyIndices) -> torch.Tensor:
    modes = np.bincount(np.asarray(references) % 2, minlength=2)
    if np.any(modes == 0):
        raise ValueError("Mode balancing needs both major and minor examples")
    return torch.tensor(
        [len(references) / (2 * modes[key % 2]) for key in range(24)], dtype=torch.float32
    )


def initial_state(
    recipe: Recipe, seed: int, epochs: int, references: KeyIndices
) -> tuple[
    KeyCNN, torch.optim.AdamW, torch.optim.lr_scheduler.CosineAnnealingLR, nn.CrossEntropyLoss
]:
    torch.manual_seed(seed)
    model = KeyCNN(pooling=recipe.pooling, dropout=recipe.dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loss = nn.CrossEntropyLoss(
        weight=mode_weights(references) if recipe.mode_balance else None,
        label_smoothing=recipe.smoothing,
    )
    return model, optimizer, scheduler, loss


def run_recipe(
    features: np.ndarray,
    meta: dict[str, Any],
    training: Sequence[int],
    validation: Sequence[int],
    recipe: Recipe,
    *,
    seed: int,
    output: Path,
    epochs: int = 120,
    batch: int = 64,
    threads: int = 16,
    max_seconds: float = 300,
    stop_after: int | None = None,
    split_policy: str = "stratified-seed13-validation-priority-audio-dedup-v1",
) -> dict[str, Any]:
    """Train/resume one recipe. A hard per-run wall budget is checked every batch."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "recipe": asdict(recipe),
        "seed": seed,
        "epochs": epochs,
        "batch": batch,
        "threads": threads,
        "feature_spec": FEATURE_SPEC,
        "manifest_sha256": signature(meta),
        "split_sha256": signature({"train": training, "validation": validation}),
        "split_policy": split_policy,
        "source_sha256": source_manifest(),
    }
    result_path = output / "run.json"
    state_path = output / "last.pt"
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old["identity"] != identity:
            raise ValueError("Existing run identity differs; refusing to reuse it")
        if old["status"] == "complete":
            return old
    torch.set_num_threads(threads)
    references = np.asarray(meta["refs"])
    model, optimizer, scheduler, criterion = initial_state(
        recipe, seed, epochs, references[training]
    )
    rng = np.random.default_rng(seed)
    record: dict[str, Any] = {
        "identity": identity,
        "status": "running",
        "history": [],
        "best_epoch": 0,
        "parameters": params_count(model),
        "elapsed_seconds": 0.0,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": "cpu",
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "hardware": platform.platform(),
    }
    first_epoch = 1
    best = -1.0
    if state_path.exists():
        state = torch.load(state_path, weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Resume checkpoint identity differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        rng.bit_generator.state = state["numpy_rng"]
        record = state["record"]
        first_epoch = state["epoch"] + 1
        best = state["best"]
    prior_elapsed = record["elapsed_seconds"]
    start = time.monotonic()
    atomic_json(result_path, record)
    try:
        for epoch in range(first_epoch, epochs + 1):
            model.train()
            order = rng.permutation(training)
            loss_sum = 0.0
            for offset in range(0, len(order), batch):
                if prior_elapsed + time.monotonic() - start > max_seconds:
                    raise TimeoutError(f"Run exceeded its {max_seconds}s training wall budget")
                indices = order[offset : offset + batch]
                if recipe.augmentation:
                    inputs, targets = augment_batch(
                        features[indices], references[indices], rng.integers(-6, 6, len(indices))
                    )
                else:
                    inputs = features[indices]
                    targets = references[indices]
                optimizer.zero_grad()
                loss = criterion(
                    model(torch.from_numpy(inputs).transpose(1, 2)), torch.from_numpy(targets)
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(indices)
            scheduler.step()
            _, exact, mirex = evaluate_set(
                model, features[validation], references[validation], "cpu"
            )
            record["history"].append(
                {
                    "epoch": epoch,
                    "loss": loss_sum / len(training),
                    "validation_exact": exact,
                    "validation_mirex": mirex,
                }
            )
            if mirex > best:
                best = mirex
                record["best_epoch"] = epoch
                save_checkpoint(
                    output / "best.pt",
                    model,
                    frames=meta["frames"],
                    recipe=asdict(recipe),
                    seed=seed,
                    validation_mirex=mirex,
                    train_ids=[meta["ids"][index] for index in training],
                    validation_ids=[meta["ids"][index] for index in validation],
                    source_sha256=identity["source_sha256"],
                    training_audio_sha256=[meta["tracks"][index]["sha256"] for index in training],
                )
            record["elapsed_seconds"] = float(prior_elapsed + time.monotonic() - start)
            atomic_json(result_path, record)
            state = {
                "identity": identity,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": rng.bit_generator.state,
                "record": record,
                "epoch": epoch,
                "best": best,
            }
            pending = state_path.with_suffix(".pending")
            torch.save(state, pending)
            os.replace(pending, state_path)
            if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
                print(
                    f"{recipe.name} seed={seed} epoch={epoch}/{epochs} val={mirex:.4f} "
                    f"best={best:.4f} elapsed={record['elapsed_seconds']:.1f}s",
                    flush=True,
                )
            if stop_after is not None and epoch >= stop_after and epoch < epochs:
                record["status"] = "paused"
                atomic_json(result_path, record)
                return record
        model, _ = load_checkpoint(output / "best.pt")
        record["train"] = development_metrics(model, features[training], references[training])
        record["validation"] = development_metrics(
            model, features[validation], references[validation]
        )
        record["best_checkpoint_sha256"] = hashlib.sha256(
            (output / "best.pt").read_bytes()
        ).hexdigest()
        record["status"] = "complete"
        record["elapsed_seconds"] = float(prior_elapsed + time.monotonic() - start)
        atomic_json(result_path, record)
        return record
    except Exception as error:
        record.update(
            status="failed",
            error=str(error),
            elapsed_seconds=float(prior_elapsed + time.monotonic() - start),
        )
        atomic_json(result_path, record)
        raise


def diagnostics(
    features: np.ndarray,
    meta: dict[str, Any],
    training: Sequence[int],
    validation: Sequence[int],
    checkpoint: Path,
    output: Path,
    threads: int = 16,
) -> dict[str, Any]:
    torch.set_num_threads(threads)
    references = np.asarray(meta["refs"])
    model, payload = load_checkpoint(checkpoint)
    expected_spec = payload["feature_spec"]
    if expected_spec != meta["feature_spec"]:
        raise ValueError("Baseline and current feature metadata differ")
    frozen_sources = payload["metadata"].get("source_sha256", {})
    current_sources = source_manifest()
    # Show differing files explicitly; architecture additions do not rewrite baseline provenance.
    differences = [
        name
        for name, digest in frozen_sources.items()
        if not Path(name).name.startswith(".") and current_sources.get(name) != digest
    ]
    original_training, original_validation = validation_split(meta["refs"], SEED)
    result = {
        "split_audit": {
            "original_training_rows": len(original_training),
            "clean_training_rows": len(training),
            "original_validation_rows": len(original_validation),
            "clean_validation_rows": len(validation),
            "removed_training_ids": [
                meta["ids"][index] for index in original_training if index not in set(training)
            ],
            "validation_ids_preserved": original_validation == validation,
        },
        "baseline_warning": "Checkpoint trained on original split with duplicate-audio leakage; diagnostic only",
        "baseline_checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "feature_shape": list(features.shape),
        "finite": bool(np.isfinite(features).all()),
        "zero_variance_tracks": np.flatnonzero(features.std(axis=(1, 2)) < 1e-8).tolist(),
        "train_tracks": len(training),
        "validation_tracks": len(validation),
        "class_counts": dict(Counter(map(int, references))),
        "source_differences_from_baseline": differences,
        "train": development_metrics(model, features[training], references[training]),
        "validation": development_metrics(model, features[validation], references[validation]),
        "overfit": {},
    }
    # One/two examples per class, from training only; no validation/test fitting.
    subset = [
        index
        for key in range(24)
        for index in [candidate for candidate in training if references[candidate] == key][:2]
    ]
    inputs = torch.from_numpy(features[subset]).transpose(1, 2)
    targets = torch.from_numpy(references[subset])
    for pooling in ("global", "frequency"):
        torch.manual_seed(13)
        candidate = KeyCNN(pooling=pooling, dropout=0)
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=0.003, weight_decay=0)
        start = time.monotonic()
        for step in range(200):
            candidate.train()
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(candidate(inputs), targets)
            loss.backward()
            optimizer.step()
            if (step + 1) % 20 == 0:
                candidate.eval()
                with torch.inference_mode():
                    accuracy = float((candidate(inputs).argmax(1) == targets).float().mean())
                if accuracy >= 0.99:
                    break
        result["overfit"][pooling] = {
            "examples": len(subset),
            "steps": step + 1,
            "exact": accuracy,
            "loss": float(loss.detach()),
            "seconds": time.monotonic() - start,
        }
    atomic_json(output / "diagnostics.json", result)
    print(
        json.dumps(
            {
                "validation": {
                    name: value
                    for name, value in result["validation"].items()
                    if name in ("exact", "mirex", "mode_accuracy")
                },
                "overfit": result["overfit"],
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def confirmation_recipes(
    summary: dict[str, Any], manifest_hash: str, split_hash: str, sources: dict[str, str]
) -> list[Recipe]:
    """Promote at most two challengers with >=2 validation MIREX points over control."""
    if (
        summary["status"] != "complete"
        or summary["manifest_sha256"] != manifest_hash
        or summary["split_sha256"] != split_hash
        or summary["source_sha256"] != sources
    ):
        raise ValueError(
            "Confirmation requires a completed screen with matching data, split and source"
        )
    results = summary["results"]
    if len(results) != len(RECIPES) or any(result["status"] != "complete" for result in results):
        raise ValueError("All eight screening recipes must finish before promotion")
    by_name = {result["identity"]["recipe"]["name"]: result for result in results}
    if set(by_name) != {recipe.name for recipe in RECIPES}:
        raise ValueError("Screen contains unexpected recipe names")
    for recipe in RECIPES:
        if by_name[recipe.name]["identity"]["recipe"] != asdict(recipe):
            raise ValueError("Screen recipe parameters differ from the registered experiment")
    threshold = by_name["control"]["validation"]["mirex"] + 0.02
    ranked = sorted(
        results,
        key=lambda result: (result["validation"]["mirex"], result["validation"]["exact"]),
        reverse=True,
    )
    names = ["control"] + [
        result["identity"]["recipe"]["name"]
        for result in ranked
        if result["identity"]["recipe"]["name"] != "control"
        and result["validation"]["mirex"] >= threshold
    ][:2]
    return [recipe for recipe in RECIPES if recipe.name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("diagnose", "screen", "confirm"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--screen", type=Path, help="Completed screening directory for confirmation"
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-run-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.threads < 1 or args.epochs < 1 or args.max_run_seconds < 1:
        parser.error("threads, epochs and budget must be positive")
    if args.command == "diagnose" and not args.baseline:
        parser.error("diagnose requires --baseline")
    if args.command == "confirm" and not args.screen:
        parser.error("confirm requires --screen")
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = snapshot_sources(args.output)
    features, meta, training, validation = load_development(args.data)
    if args.command == "diagnose":
        diagnostics(features, meta, training, validation, args.baseline, args.output, args.threads)
        return
    if args.command == "screen":
        jobs = [(recipe, 13) for recipe in RECIPES]
    else:
        screen = json.loads((args.screen / "summary.json").read_text())
        selected = confirmation_recipes(
            screen,
            signature(meta),
            signature({"train": training, "validation": validation}),
            hashes,
        )
        jobs = [(recipe, seed) for recipe in selected for seed in (13, 37, 73)]
    summary = {
        "promotion_rule": "control plus up to two challengers >=0.02 validation MIREX above control",
        "command": args.command,
        "source_sha256": hashes,
        "manifest_sha256": signature(meta),
        "split_sha256": signature({"train": training, "validation": validation}),
        "split_policy": "stratified-seed13-validation-priority-audio-dedup-v1",
        "jobs": [{"recipe": asdict(recipe), "seed": seed} for recipe, seed in jobs],
        "status": "running",
        "results": [],
    }
    atomic_json(args.output / "summary.json", summary)
    tranche_start = time.monotonic()
    for recipe, seed in jobs:
        if time.monotonic() - tranche_start > 3600:
            summary["status"] = "budget_exhausted"
            atomic_json(args.output / "summary.json", summary)
            raise TimeoutError("Tranche exceeded its one-hour wall budget")
        output = args.output / f"{recipe.name}-seed{seed}"
        print(f"Starting {output.name}", flush=True)
        result = run_recipe(
            features,
            meta,
            training,
            validation,
            recipe,
            seed=seed,
            output=output,
            epochs=args.epochs,
            threads=args.threads,
            max_seconds=args.max_run_seconds,
        )
        summary["results"].append(result)
        atomic_json(args.output / "summary.json", summary)
    summary["status"] = "complete"
    atomic_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
