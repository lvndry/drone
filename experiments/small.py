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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from drone.keys import score_breakdown, score_mirex
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


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(pending, path)


def source_manifest(root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    paths = [root / "train.py", root / "pyproject.toml", root / "uv.lock"]
    for directory in ("drone", "experiments"):
        paths += sorted((root / directory).rglob("*.py"))
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
        if p.is_file() and not p.name.startswith(".")
    }


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def snapshot_sources(output):
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


def load_development(data):
    features, meta = load_prepared(data, "train")
    train, val = validation_split(meta["refs"], SEED, [t["sha256"] for t in meta["tracks"]])
    if set(meta["ids"][i] for i in train) & set(meta["ids"][i] for i in val):
        raise ValueError("Duplicate IDs cross the training/validation boundary")
    if set(meta["tracks"][i]["sha256"] for i in train) & set(
        meta["tracks"][i]["sha256"] for i in val
    ):
        raise ValueError("Identical audio crosses the training/validation boundary")
    # No test features or labels are opened here.
    return features, meta, train, val


def probabilities(model, features, batch=128):
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


def development_metrics(model, features, refs):
    prob = probabilities(model, features)
    pred = prob.argmax(1)
    refs = np.asarray(refs)
    exact, mirex = score_mirex([int(c) for c in pred], [int(c) for c in refs])
    matrix = np.zeros((24, 24), dtype=int)
    np.add.at(matrix, (refs, pred), 1)
    support = matrix.sum(1)
    recalls = np.divide(matrix.diagonal(), support, out=np.zeros(24), where=support > 0)
    return {
        "exact": exact,
        "mirex": mirex,
        "mirex_asymmetric": score_mirex(
            [int(c) for c in pred], [int(c) for c in refs], symmetric=False
        )[1],
        "mode_accuracy": float(np.mean(pred % 2 == refs % 2)),
        "macro_recall_present_classes": float(recalls[support > 0].mean()),
        "nll": float(-np.log(np.maximum(prob[np.arange(len(refs)), refs], 1e-12)).mean()),
        "counts": dict(score_breakdown([int(c) for c in pred], [int(c) for c in refs])),
        "predicted_counts": np.bincount(pred, minlength=24).tolist(),
        "reference_counts": np.bincount(refs, minlength=24).tolist(),
        "confusion_matrix": matrix.tolist(),
        "predictions": pred.tolist(),
    }


def mode_weights(refs):
    modes = np.bincount(np.asarray(refs) % 2, minlength=2)
    if np.any(modes == 0):
        raise ValueError("Mode balancing needs both major and minor examples")
    return torch.tensor([len(refs) / (2 * modes[k % 2]) for k in range(24)], dtype=torch.float32)


def initial_state(recipe, seed, epochs, refs):
    torch.manual_seed(seed)
    model = KeyCNN(pooling=recipe.pooling, dropout=recipe.dropout)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loss = nn.CrossEntropyLoss(
        weight=mode_weights(refs) if recipe.mode_balance else None, label_smoothing=recipe.smoothing
    )
    return model, optimizer, scheduler, loss


def run_recipe(
    features,
    meta,
    tr,
    val,
    recipe,
    *,
    seed,
    output,
    epochs=120,
    batch=64,
    threads=16,
    max_seconds=300,
    stop_after=None,
    split_policy="stratified-seed13-validation-priority-audio-dedup-v1",
):
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
        "split_sha256": signature({"train": tr, "validation": val}),
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
    refs = np.asarray(meta["refs"])
    model, optimizer, scheduler, criterion = initial_state(recipe, seed, epochs, refs[tr])
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
    first_epoch, best = 1, -1.0
    if state_path.exists():
        state = torch.load(state_path, weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Resume checkpoint identity differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        rng.bit_generator.state = state["numpy_rng"]
        record, first_epoch, best = state["record"], state["epoch"] + 1, state["best"]
    prior_elapsed = record["elapsed_seconds"]
    start = time.monotonic()
    atomic_json(result_path, record)
    try:
        for epoch in range(first_epoch, epochs + 1):
            model.train()
            order = rng.permutation(tr)
            loss_sum = 0.0
            for offset in range(0, len(order), batch):
                if prior_elapsed + time.monotonic() - start > max_seconds:
                    raise TimeoutError(f"Run exceeded its {max_seconds}s training wall budget")
                indices = order[offset : offset + batch]
                if recipe.augmentation:
                    x, y = augment_batch(
                        features[indices], refs[indices], rng.integers(-6, 6, len(indices))
                    )
                else:
                    x, y = features[indices], refs[indices]
                optimizer.zero_grad()
                loss = criterion(model(torch.from_numpy(x).transpose(1, 2)), torch.from_numpy(y))
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(indices)
            scheduler.step()
            _, exact, mirex = evaluate_set(model, features[val], refs[val], "cpu")
            record["history"].append(
                {
                    "epoch": epoch,
                    "loss": loss_sum / len(tr),
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
                    train_ids=[meta["ids"][i] for i in tr],
                    validation_ids=[meta["ids"][i] for i in val],
                    source_sha256=identity["source_sha256"],
                    training_audio_sha256=[meta["tracks"][i]["sha256"] for i in tr],
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
        record["train"] = development_metrics(model, features[tr], refs[tr])
        record["validation"] = development_metrics(model, features[val], refs[val])
        record["best_checkpoint_sha256"] = hashlib.sha256(
            (output / "best.pt").read_bytes()
        ).hexdigest()
        record["status"] = "complete"
        record["elapsed_seconds"] = float(prior_elapsed + time.monotonic() - start)
        atomic_json(result_path, record)
        return record
    except Exception as exc:
        record.update(
            status="failed",
            error=str(exc),
            elapsed_seconds=float(prior_elapsed + time.monotonic() - start),
        )
        atomic_json(result_path, record)
        raise


def diagnostics(features, meta, tr, val, checkpoint, output, threads=16) -> dict[str, Any]:
    torch.set_num_threads(threads)
    refs = np.asarray(meta["refs"])
    model, payload = load_checkpoint(checkpoint)
    expected_spec = payload["feature_spec"]
    if expected_spec != meta["feature_spec"]:
        raise ValueError("Baseline and current feature metadata differ")
    frozen_sources = payload["metadata"].get("source_sha256", {})
    current_sources = source_manifest()
    # Show differing files explicitly; architecture additions do not rewrite baseline provenance.
    differences = [
        p
        for p, h in frozen_sources.items()
        if not Path(p).name.startswith(".") and current_sources.get(p) != h
    ]
    original_tr, original_val = validation_split(meta["refs"], SEED)
    result = {
        "split_audit": {
            "original_training_rows": len(original_tr),
            "clean_training_rows": len(tr),
            "original_validation_rows": len(original_val),
            "clean_validation_rows": len(val),
            "removed_training_ids": [meta["ids"][i] for i in original_tr if i not in set(tr)],
            "validation_ids_preserved": original_val == val,
        },
        "baseline_warning": "Checkpoint trained on original split with duplicate-audio leakage; diagnostic only",
        "baseline_checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "feature_shape": list(features.shape),
        "finite": bool(np.isfinite(features).all()),
        "zero_variance_tracks": np.flatnonzero(features.std(axis=(1, 2)) < 1e-8).tolist(),
        "train_tracks": len(tr),
        "validation_tracks": len(val),
        "class_counts": dict(Counter(map(int, refs))),
        "source_differences_from_baseline": differences,
        "train": development_metrics(model, features[tr], refs[tr]),
        "validation": development_metrics(model, features[val], refs[val]),
        "overfit": {},
    }
    # One/two examples per class, from training only; no validation/test fitting.
    subset = [i for key in range(24) for i in [j for j in tr if refs[j] == key][:2]]
    x = torch.from_numpy(features[subset]).transpose(1, 2)
    y = torch.from_numpy(refs[subset])
    for pooling in ("global", "frequency"):
        torch.manual_seed(13)
        candidate = KeyCNN(pooling=pooling, dropout=0)
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=0.003, weight_decay=0)
        start = time.monotonic()
        for step in range(200):
            candidate.train()
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(candidate(x), y)
            loss.backward()
            optimizer.step()
            if (step + 1) % 20 == 0:
                candidate.eval()
                with torch.inference_mode():
                    accuracy = float((candidate(x).argmax(1) == y).float().mean())
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
                    k: v
                    for k, v in result["validation"].items()
                    if k in ("exact", "mirex", "mode_accuracy")
                },
                "overfit": result["overfit"],
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def confirmation_recipes(summary, manifest_hash, split_hash, sources):
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
    if len(results) != len(RECIPES) or any(r["status"] != "complete" for r in results):
        raise ValueError("All eight screening recipes must finish before promotion")
    by_name = {r["identity"]["recipe"]["name"]: r for r in results}
    if set(by_name) != {r.name for r in RECIPES}:
        raise ValueError("Screen contains unexpected recipe names")
    for recipe in RECIPES:
        if by_name[recipe.name]["identity"]["recipe"] != asdict(recipe):
            raise ValueError("Screen recipe parameters differ from the registered experiment")
    threshold = by_name["control"]["validation"]["mirex"] + 0.02
    ranked = sorted(
        results, key=lambda r: (r["validation"]["mirex"], r["validation"]["exact"]), reverse=True
    )
    names = ["control"] + [
        r["identity"]["recipe"]["name"]
        for r in ranked
        if r["identity"]["recipe"]["name"] != "control" and r["validation"]["mirex"] >= threshold
    ][:2]
    return [recipe for recipe in RECIPES if recipe.name in names]


def main():
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
    features, meta, tr, val = load_development(args.data)
    if args.command == "diagnose":
        diagnostics(features, meta, tr, val, args.baseline, args.output, args.threads)
        return
    if args.command == "screen":
        jobs = [(recipe, 13) for recipe in RECIPES]
    else:
        screen = json.loads((args.screen / "summary.json").read_text())
        selected = confirmation_recipes(
            screen,
            signature(meta),
            signature({"train": tr, "validation": val}),
            hashes,
        )
        jobs = [(recipe, seed) for recipe in selected for seed in (13, 37, 73)]
    summary = {
        "promotion_rule": "control plus up to two challengers >=0.02 validation MIREX above control",
        "command": args.command,
        "source_sha256": hashes,
        "manifest_sha256": signature(meta),
        "split_sha256": signature({"train": tr, "validation": val}),
        "split_policy": "stratified-seed13-validation-priority-audio-dedup-v1",
        "jobs": [{"recipe": asdict(r), "seed": seed} for r, seed in jobs],
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
            tr,
            val,
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
