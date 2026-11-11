"""Bounded K10 data-only comparison on frozen development identities."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drone.checkpoints import load_checkpoint, save_checkpoint
from drone.features import (
    CROP_FRAMES,
    FEATURE_SPEC,
    N_TOTAL,
    audio_digest,
    cqt_path,
    extract_cqt,
    window,
)
from drone.keys import LABEL_SCHEMA, score_mirex, transpose_key
from drone.model import KeyCNN, params_count
from experiments.octave import device_setup
from experiments.small import atomic_json, snapshot_sources
from train import evaluate_features

FROZEN_SHA256 = "5f995f516430332a37f591ec299a36f9a46c16b2cd01cd887700837fd600bac9"


def validate_rows(rows: list[dict[str, Any]]) -> None:
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate development identities")
    for r in rows:
        if r["role"] not in ("train", "validation") or r["domain"] not in ("beatport", "fmak"):
            raise ValueError("Only fitting and validation rows from the two domains are allowed")
        if not r["id"].startswith(r["domain"] + ":"):
            raise ValueError("Domain and identity disagree")
        if not 0 <= r["ref"] < 24:
            raise ValueError("Invalid key label")
    fit = {r["sha256"] for r in rows if r["role"] == "train"}
    val = {r["sha256"] for r in rows if r["role"] == "validation"}
    if fit & val:
        raise ValueError("Identical audio crosses fitting and validation")
    if len(fit) != sum(r["role"] == "train" for r in rows):
        raise ValueError("Fitting audio must be unique")
    labels = {}
    for r in rows:
        if r["sha256"] in labels and labels[r["sha256"]] != r["ref"]:
            raise ValueError("Conflicting labels on identical audio")
        labels[r["sha256"]] = r["ref"]


def prepare_one(task: tuple[dict[str, Any], Path, Path]) -> dict[str, Any]:
    row, output, existing = task
    if audio_digest(row["path"]) != row["sha256"]:
        raise ValueError(f"Audio content changed: {row['id']}")
    destination = Path(cqt_path(output, row["sha256"]))
    original = Path(cqt_path(existing, row["sha256"]))
    reused = False
    if not destination.exists() and original.is_file():
        cached = np.load(original, allow_pickle=False)
        if (
            cached.ndim == 2
            and cached.shape[0] == N_TOTAL
            and cached.shape[1] >= 8
            and np.isfinite(cached).all()
            and cached.dtype == np.float16
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(original, destination)
            except OSError:
                shutil.copyfile(original, destination)
            reused = True
    array = extract_cqt(row["path"], output)
    if array.shape[1] < 8 or not np.isfinite(array).all():
        raise ValueError(f"Invalid CQT: {row['id']}")
    return dict(
        row,
        cache=str(destination.relative_to(output)),
        frames=array.shape[1],
        cache_sha256=audio_digest(destination),
        reused_cache=reused,
    )


def prepare(args: argparse.Namespace) -> None:
    raw = args.manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_SHA256:
        raise ValueError("The frozen development manifest differs")
    rows = json.loads(raw)["rows"]
    validate_rows(rows)
    counts = {
        f"{domain}_{role}": sum(r["domain"] == domain and r["role"] == role for r in rows)
        for domain in ("beatport", "fmak")
        for role in ("train", "validation")
    }
    if counts != dict(
        beatport_train=1230, beatport_validation=126, fmak_train=976, fmak_validation=250
    ):
        raise ValueError("Development counts differ from the declared experiment")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = snapshot_sources(args.output)
    start = time.monotonic()
    completed = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        tasks = [(r, args.output, args.existing_cache) for r in rows]
        for row in pool.map(prepare_one, tasks):
            completed.append(row)
            if len(completed) % 50 == 0:
                print(
                    f"K10 CQT {len(completed)}/{len(rows)} elapsed={time.monotonic() - start:.1f}s",
                    flush=True,
                )
            if time.monotonic() - start > 1200:
                raise TimeoutError("Preparation exceeded its 20-minute budget")
    atomic_json(
        args.output / "manifest.json",
        {
            "feature_spec": FEATURE_SPEC,
            "label_schema": LABEL_SCHEMA,
            "frozen_manifest_sha256": FROZEN_SHA256,
            "rows": completed,
            "counts": counts,
            "versions": {n: importlib.metadata.version(n) for n in ("numpy", "librosa", "torch")},
            "source_sha256": sources,
            "elapsed_seconds": time.monotonic() - start,
            "complete": True,
            "calibration_used": False,
            "final_holdout_used": False,
        },
    )


def load_data(data: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[np.ndarray]]:
    manifest = json.loads((data / "manifest.json").read_bytes())
    if not manifest["complete"] or manifest["feature_spec"] != FEATURE_SPEC:
        raise ValueError("Incomplete or incompatible feature preparation")
    if manifest["frozen_manifest_sha256"] != FROZEN_SHA256:
        raise ValueError("Different development identities")
    rows = manifest["rows"]
    validate_rows(rows)
    arrays = []
    for r in rows:
        path = data / r["cache"]
        if audio_digest(path) != r["cache_sha256"]:
            raise ValueError(f"Changed CQT cache: {r['id']}")
        array = np.load(path, allow_pickle=False)
        if array.shape != (N_TOTAL, r["frames"]) or not np.isfinite(array).all():
            raise ValueError(f"Invalid cached feature: {r['id']}")
        arrays.append(array)
    return manifest, rows, arrays


def domain_metrics(
    pred: np.ndarray, refs: np.ndarray, rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result = {}
    for domain in ("beatport", "fmak"):
        indices = [i for i, r in enumerate(rows) if r["domain"] == domain]
        p, y = pred[indices], refs[indices]
        exact, mirex = score_mirex(p, y)
        result[domain] = {
            "tracks": len(indices),
            "exact": exact,
            "mirex": mirex,
            "mirex_asymmetric": score_mirex(p, y, symmetric=False)[1],
            "mode_accuracy": float(np.mean(p % 2 == y % 2)),
            "predictions": p.tolist(),
            "ids": [rows[i]["id"] for i in indices],
        }
    return result


def train(args: argparse.Namespace) -> None:
    if (args.output / "run.json").exists():
        raise ValueError("Use a fresh run directory; this pilot does not silently restart")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = snapshot_sources(args.output)
    device_setup(args.device)
    capacity_candidate = getattr(args, "capacity_probe", False)
    capacity_active = capacity_candidate or getattr(args, "capacity_control", False)
    capacity_memory = {}
    if capacity_candidate:
        from experiments.capacity import memory_guard

        capacity_memory = memory_guard(args.device)
    elif capacity_active:
        capacity_memory = dict(allocator_fraction=0.04, required_free_bytes=8 * 1024**3)
    budget_seconds = 2700 if capacity_candidate else 1200
    manifest, rows, arrays = load_data(args.data)
    fitting = [
        i
        for i, r in enumerate(rows)
        if r["role"] == "train" and (args.arm == "joint" or r["domain"] == "beatport")
    ]
    validation = [i for i, r in enumerate(rows) if r["role"] == "validation"]
    val_rows = [rows[i] for i in validation]
    val_features = [window(arrays[i]).astype(np.float32) for i in validation]
    val_refs = np.array([rows[i]["ref"] for i in validation])
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = KeyCNN(width=2.0).to(args.device) if capacity_candidate else KeyCNN().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 120)
    record = {
        "status": "running",
        "arm": args.arm,
        "seed": args.seed,
        "parameters": params_count(model),
        "source_sha256": sources,
        "feature_manifest_sha256": audio_digest(args.data / "manifest.json"),
        "config": dict(
            epochs=120,
            batch=32,
            lr=0.001,
            weight_decay=0.0001,
            smoothing=0.0,
            patience=25,
            device=args.device,
        ),
        "fit_ids": [rows[i]["id"] for i in fitting],
        "validation_ids": [r["id"] for r in val_rows],
        "history": [],
        "calibration_used": False,
        "final_holdout_used": False,
    }
    probe = None
    if getattr(args, "ema_probe", False):
        from experiments.ema import EmaProbe

        probe = EmaProbe(args, record, model, rows, arrays, fitting)
    capacity = None
    if capacity_active:
        from experiments.capacity import CapacityProbe

        capacity = CapacityProbe(args, record, capacity_memory)
    best, best_epoch, start = -1, 0, time.monotonic()
    if capacity_active and args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, 121):
        model.train()
        order = rng.permutation(fitting)
        loss_sum = 0.0
        for offset in range(0, len(order), 32):
            idx = order[offset : offset + 32]
            xs, ys = [], []
            for i in idx:
                shift = int(rng.integers(-4, 5))
                feature = window(arrays[i], shift).astype(np.float32)
                if feature.shape[1] < CROP_FRAMES:
                    feature = np.pad(feature, ((0, 0), (0, CROP_FRAMES - feature.shape[1])))
                else:
                    first = int(rng.integers(0, feature.shape[1] - CROP_FRAMES + 1))
                    feature = feature[:, first : first + CROP_FRAMES]
                xs.append(feature)
                ys.append(transpose_key(rows[i]["ref"], -shift))
            optimizer.zero_grad()
            logits = model(torch.from_numpy(np.stack(xs)).to(args.device))
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(ys, device=args.device))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            if probe is not None:
                probe.step(model)
            loss_sum += float(loss.detach()) * len(idx)
            if time.monotonic() - start > budget_seconds:
                record.update(status="budget_exhausted", elapsed_seconds=time.monotonic() - start)
                atomic_json(args.output / "run.json", record)
                if probe is not None:
                    probe.budget_exhausted(start)
                if capacity is not None:
                    capacity.failure(record, start, "Fixed fit budget exhausted", budget=True)
                return
        scheduler.step()
        pred = evaluate_features(model, val_features, args.device, batch=4)
        _, score = score_mirex(pred, val_refs)
        record["history"].append(dict(epoch=epoch, loss=loss_sum / len(order), mirex=score))
        if score > best:
            best, best_epoch = score, epoch
            save_checkpoint(
                args.output / "best.pt",
                model,
                epoch=epoch,
                experiment="k10-multidomain-development-v1",
                arm=args.arm,
                seed=args.seed,
                source_sha256=sources,
                **({"width": 2.0} if capacity_candidate else {}),
            )
            atomic_json(
                args.output / "best-validation.json", domain_metrics(pred, val_refs, val_rows)
            )
            if probe is not None:
                probe.save_best(model)
        record.update(best_epoch=best_epoch, elapsed_seconds=time.monotonic() - start)
        atomic_json(args.output / "run.json", record)
        print(
            f"K10 {args.arm} seed={args.seed} epoch={epoch} val={score:.4f} best={best:.4f} elapsed={record['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if epoch - best_epoch >= 25:
            break
    record.update(
        status="complete",
        early_stopped=epoch < 120,
        domain_metrics=json.loads((args.output / "best-validation.json").read_bytes()),
        checkpoint_sha256=audio_digest(args.output / "best.pt"),
    )
    atomic_json(args.output / "run.json", record)
    if probe is not None:
        probe.finish(record, val_features, val_refs, val_rows, start)
    if capacity is not None:
        result = capacity.finish(record, val_features, val_refs, val_rows, start)
        if result["status"] != "complete":
            raise RuntimeError(f"Capacity probe failed: {result.get('error', result['status'])}")


def diagnose(args: argparse.Namespace) -> None:
    device_setup(args.device)
    _, rows, arrays = load_data(args.data)
    indices = [i for i, r in enumerate(rows) if r["role"] == "validation"]
    val_rows = [rows[i] for i in indices]
    features = [window(arrays[i]).astype(np.float32) for i in indices]
    refs = np.array([r["ref"] for r in val_rows])
    model, _ = load_checkpoint(args.checkpoint, args.device)
    corrected = evaluate_features(model, features, args.device, batch=4)
    padded = []
    for offset in range(0, len(features), 4):
        chunk = features[offset : offset + 4]
        x = np.zeros((len(chunk), 192, max(f.shape[1] for f in chunk)), dtype=np.float32)
        for j, f in enumerate(chunk):
            x[j, :, : f.shape[1]] = f
        with torch.inference_mode():
            padded.extend(model(torch.from_numpy(x).to(args.device)).argmax(1).cpu().tolist())
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output / "diagnostic.json",
        {
            "checkpoint_sha256": audio_digest(args.checkpoint),
            "status": "diagnostic_shipped_weights_not_an_independent_control",
            "corrected": domain_metrics(corrected, refs, val_rows),
            "padded_batch4": domain_metrics(np.array(padded), refs, val_rows),
            "changed_predictions": int(np.sum(corrected != padded)),
            "calibration_used": False,
            "final_holdout_used": False,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "train", "diagnose"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--existing-cache", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--arm", choices=("control", "joint"))
    p.add_argument("--seed", type=int, choices=(13, 37, 73), default=13)
    probes = p.add_mutually_exclusive_group()
    probes.add_argument("--ema-probe", action="store_true")
    probes.add_argument("--capacity-probe", action="store_true")
    probes.add_argument("--capacity-control", action="store_true")
    p.add_argument("--paired-control", type=Path)
    p.add_argument("--reference-run", type=Path)
    p.add_argument("--reference-bp", type=Path)
    p.add_argument("--reference-checkpoint", type=Path)
    p.add_argument("--pilot-run", type=Path)
    args = p.parse_args()
    if args.ema_probe and (
        args.command != "train"
        or args.arm != "joint"
        or args.reference_run is None
        or args.reference_bp is None
    ):
        p.error("--ema-probe requires train --arm joint --reference-run and --reference-bp")
    if (args.capacity_probe or args.capacity_control) and (
        args.command != "train"
        or args.arm != "joint"
        or args.reference_run is None
        or args.reference_bp is None
        or (args.capacity_probe and args.paired_control is None)
    ):
        p.error(
            "Capacity needs joint training/archived references; candidate also needs --paired-control"
        )
    if args.paired_control is not None and not args.capacity_probe:
        p.error("--paired-control requires --capacity-probe")
    try:
        {"prepare": prepare, "train": train, "diagnose": diagnose}[args.command](args)
    except torch.OutOfMemoryError as exc:
        if args.capacity_probe or args.capacity_control:
            from experiments.capacity import record_oom

            record_oom(args, exc)
        raise


if __name__ == "__main__":
    main()
