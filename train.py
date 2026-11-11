"""Prepare features, train on Beatport, and evaluate once on held-out GiantSteps.

Training follows the K10 (MIREX 2026, MIT) protocol: 192 core CQT bins,
320-frame crops, uniform +/-4-semitone shift as an exact re-slice of the
padded spectrogram, plain cross-entropy, AdamW 1e-3, cosine schedule.

`prepare-corrections` and `finetune` adapt a trained checkpoint to the keys a
person corrected in Euphoria, mixed with the base training data, and report
GiantSteps and held-out corrections before and after.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from drone.checkpoints import load_checkpoint, save_checkpoint
from drone.corrections import match_corrections, read_corrections_export, split_corrections
from drone.datasets import Track, load_beatport, load_giantsteps, prepare, sample_features
from drone.features import CROP_FRAMES, FEATURE_ID, FEATURE_SPEC, N_CORE, audio_digest
from drone.keys import (
    LABEL_SCHEMA,
    WEIGHTS,
    encode_key,
    key_relation,
    score_breakdown,
    score_mirex,
)
from drone.model import KeyCNN, params_count
from drone.splits import validation_split

DATA = Path(os.environ.get("DRONE_DATA", Path(__file__).resolve().parent / "data"))
SEED = 13
EPOCHS = 120
BATCH = 32
LR = 1e-3
FINETUNE_EPOCHS = 10
FINETUNE_LR = 1e-4
CORRECTIONS_SHARE = 0.25
CORRECTIONS_FIT = "corrections-fit"
CORRECTIONS_HELD_OUT = "corrections-held-out"


def evaluate_features(
    model: nn.Module, features: Sequence[np.ndarray], device: str, batch: int = 16
) -> np.ndarray:
    """Classify variable-length arrays at true length with bounded batching."""
    if not len(features) or batch < 1:
        raise ValueError("Evaluation needs nonempty features and a positive batch")
    model.eval()
    pred = np.empty(len(features), dtype=np.int64)
    for i in range(0, len(features), batch):
        groups = {}
        for j, feature in enumerate(features[i : i + batch], start=i):
            groups.setdefault(feature.shape[1], []).append((j, feature))
        for entries in groups.values():
            indices, arrays = zip(*entries, strict=True)
            x = torch.from_numpy(np.stack(arrays)).to(device)
            with torch.inference_mode():
                pred[list(indices)] = model(x).argmax(1).cpu().numpy()
    return pred


def evaluate_tracks(
    model: nn.Module, tracks: list[Track], track_dir: str | Path, device: str, batch: int = 16
) -> np.ndarray:
    """Evaluate full tracks without making predictions depend on padded peers."""
    if not len(tracks) or batch < 1:
        raise ValueError("Evaluation needs nonempty tracks and a positive batch")
    pred = []
    for i in range(0, len(tracks), batch):
        features = [prepare(t, track_dir)[0] for t in tracks[i : i + batch]]
        pred.append(evaluate_features(model, features, device, batch))
    return np.concatenate(pred)


def prepared_path(data: str | Path, tag: str) -> Path:
    return Path(data) / "features" / FEATURE_ID / f"{tag}.json"


def prepare_dir(
    tag: str, tracks: list[Track], track_dir: str | Path, *, data: str | Path = DATA
) -> dict[str, Any]:
    """Warm the content-addressed CQT cache and write the track manifest."""
    if not tracks:
        raise ValueError(f"No tracks for {tag}; download the datasets with bash fetch.sh")
    metadata = {
        "feature_spec": FEATURE_SPEC,
        "label_schema": LABEL_SCHEMA,
        "crop": CROP_FRAMES,
        "tracks": [
            {
                "id": track.id,
                "path": str(track.audio),
                "sha256": audio_digest(track.audio),
                "key": list(track.key),
            }
            for track in tracks
        ],
        "refs": [encode_key(*track.key) for track in tracks],
        "ids": [track.id for track in tracks],
    }
    out = prepared_path(data, tag)
    if out.is_file():
        try:
            cached = json.loads(out.read_text())
        except (ValueError, OSError):
            pass
        else:
            if cached == metadata:
                return metadata
    start = time.monotonic()
    for i, track in enumerate(tracks):
        prepare(track, track_dir)
        if (i + 1) % 50 == 0:
            print(f"{tag}: {i + 1}/{len(tracks)} ({time.monotonic() - start:.0f}s)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        pending = Path(scratch) / "manifest.json"
        pending.write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(pending, out)
    print(f"{tag}: done {len(tracks)} tracks in {time.monotonic() - start:.0f}s", flush=True)
    return metadata


def load_manifest(data: str | Path, tag: str) -> dict[str, Any]:
    path = prepared_path(data, tag)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {tag} manifest. Run uv run train.py prepare.")
    metadata = json.loads(path.read_text())
    if (
        metadata.get("feature_spec") != FEATURE_SPEC
        or metadata.get("label_schema") != LABEL_SCHEMA
        or not metadata.get("refs")
        or len(metadata["ids"]) != len(metadata["refs"])
    ):
        raise ValueError(f"Invalid {tag} manifest; run uv run train.py prepare")
    return metadata


def assert_disjoint(train_ids: Sequence[str], test_ids: Sequence[str]) -> None:
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"Training/test overlap: {sorted(overlap)[:10]}")


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train_model(
    data: str | Path,
    device: str,
    *,
    epochs: int = EPOCHS,
    batch: int = BATCH,
    seed: int = SEED,
    lr: float = LR,
    width: float = 1.0,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.0,
) -> Path:
    data = Path(data)
    meta = load_manifest(data, "train")
    test_meta = load_manifest(data, "test")
    assert_disjoint(meta["ids"], test_meta["ids"])
    assert_disjoint(
        [t["sha256"] for t in meta["tracks"]], [t["sha256"] for t in test_meta["tracks"]]
    )

    ordered = [Track(t["id"], t["path"], tuple(t["key"])) for t in meta["tracks"]]
    refs = meta["refs"]
    tr_idx, val_idx = validation_split(refs, SEED, [t["sha256"] for t in meta["tracks"]])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if epochs < 1 or batch < 1 or lr <= 0 or weight_decay < 0 or not 0 <= label_smoothing < 1:
        raise ValueError("Invalid training hyperparameters")
    model = KeyCNN(width=width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    run_root = data / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    run = Path(
        tempfile.mkdtemp(
            prefix=f"w{width}-seed{seed}-{time.strftime('%Y%m%d-%H%M%S')}-", dir=run_root
        )
    )
    checkpoint = run / "drone.pt"
    config = {
        "epochs": epochs,
        "batch": batch,
        "seed": seed,
        "lr": lr,
        "device": str(device),
        "split_seed": SEED,
        "width": width,
        "crop": CROP_FRAMES,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
    }
    source_files = [Path(__file__), *sorted((Path(__file__).parent / "drone").glob("*.py"))]
    source_hashes = {
        str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_files
    }
    record: dict[str, Any] = {
        "source_sha256": source_hashes,
        "config": config,
        "feature_spec": FEATURE_SPEC,
        "label_schema": LABEL_SCHEMA,
        "git_revision": git_revision(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "training_manifest": meta,
        "train_ids": [meta["ids"][i] for i in tr_idx],
        "validation_ids": [meta["ids"][i] for i in val_idx],
        "held_out_ids": test_meta["ids"],
        "history": [],
    }
    lock = Path(__file__).parent / "uv.lock"
    record["lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None

    val_tracks = [ordered[i] for i in val_idx]
    best_val, start = -1.0, time.monotonic()
    print(f"device {device}, params {params_count(model):,}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(tr_idx)
        loss_sum = 0.0
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            feats, y = [], []
            for j in idx:
                feat, cls = sample_features(ordered[j], data, rng, T=CROP_FRAMES)
                feats.append(feat)
                y.append(cls)
            T = max(f.shape[1] for f in feats)
            x = torch.zeros(len(feats), N_CORE, T)
            for k, feat in enumerate(feats):
                x[k, :, : feat.shape[1]] = torch.from_numpy(feat)
            optimizer.zero_grad()
            loss = criterion(
                model(x.to(device)),
                torch.from_numpy(np.asarray(y, dtype=np.int64)).to(device),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(idx)
        scheduler.step()
        val_pred = evaluate_tracks(model, val_tracks, data, device)
        exact, mirex = score_mirex([int(c) for c in val_pred], [refs[i] for i in val_idx])
        record["history"].append(
            {
                "epoch": epoch,
                "loss": loss_sum / len(order),
                "validation_exact": exact,
                "validation_mirex": mirex,
            }
        )
        if mirex > best_val:
            best_val = mirex
            record["best_epoch"] = epoch
            save_checkpoint(
                checkpoint,
                model,
                width=width,
                config=config,
                epoch=epoch,
                validation_mirex=mirex,
                git_revision=record["git_revision"],
                train_ids=record["train_ids"],
                validation_ids=record["validation_ids"],
                source_sha256=source_hashes,
                training_audio_sha256=[t["sha256"] for t in meta["tracks"]],
            )
        record["elapsed_seconds"] = time.monotonic() - start
        (run / "run.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"epoch {epoch}/{epochs} loss {loss_sum / len(order):.4f} "
            f"val MIREX {100 * mirex:.1f}% best {100 * best_val:.1f}%",
            flush=True,
        )
    record["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (run / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Best checkpoint: {checkpoint}\nRun `uv run train.py evaluate {checkpoint}` when ready.")
    return checkpoint


def evaluate_checkpoint(data: str | Path, checkpoint: str | Path, device: str) -> dict[str, Any]:
    data = Path(data)
    meta = load_manifest(data, "test")
    model, payload = load_checkpoint(checkpoint, device)
    trained_ids = payload.get("metadata", {}).get("train_ids", [])
    validation_ids = payload.get("metadata", {}).get("validation_ids", [])
    assert_disjoint(trained_ids + validation_ids, meta["ids"])
    assert_disjoint(
        payload.get("metadata", {}).get("training_audio_sha256", []),
        [t["sha256"] for t in meta["tracks"]],
    )

    ordered = [Track(t["id"], t["path"], tuple(t["key"])) for t in meta["tracks"]]
    pred = evaluate_tracks(model, ordered, data, device)
    classes = [int(c) for c in pred]
    exact, mirex = score_mirex(classes, meta["refs"])
    cats = score_breakdown(classes, meta["refs"])
    result: dict[str, Any] = {
        "tracks": len(pred),
        "exact": exact,
        "mirex": mirex,
        "mirex_asymmetric": score_mirex(classes, meta["refs"], symmetric=False)[1],
        "counts": dict(cats),
        "feature_spec": FEATURE_SPEC,
        "test_manifest": meta,
        "predictions": classes,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
    }
    print(
        f"Held-out GiantSteps ({len(pred)} tracks): exact {100 * exact:.1f}% MIREX {100 * mirex:.1f}%"
    )
    print(
        " ".join(
            f"{key}: {cats[key]}" for key in ("exact", "fifth", "relative", "parallel", "other")
        )
    )
    return result


def prepare_corrections(
    data: str | Path, export: str | Path, audio_directory: str | Path
) -> dict[str, Any]:
    """Match an Euphoria corrections export to audio and cache its features.

    The matched tracks are split by content hash into a set to fit and a
    held-out set, each written as a manifest like the base datasets.
    """
    data = Path(data)
    parsed = read_corrections_export(export)
    matched = match_corrections(parsed.key_corrections, audio_directory)
    test_meta = load_manifest(data, "test")
    test_hashes = {t["sha256"] for t in test_meta["tracks"]}
    held_out_test = [t.id for t in matched.tracks if audio_digest(t.audio) in test_hashes]
    if held_out_test:
        raise ValueError(
            f"{len(held_out_test)} corrected tracks are GiantSteps test audio; "
            "remove them from the export so the benchmark stays held out"
        )
    fit, held_out = split_corrections(matched.tracks)
    prepare_dir(CORRECTIONS_FIT, fit, data, data=data)
    prepare_dir(CORRECTIONS_HELD_OUT, held_out, data, data=data)
    summary = {
        "key_corrections": len(parsed.key_corrections),
        "tempo_only_skipped": parsed.tempo_only_count,
        "matched": len(matched.tracks),
        "unmatched": len(matched.unmatched_hashes),
        "fit": len(fit),
        "held_out": len(held_out),
    }
    print(
        f"{summary['key_corrections']} key corrections "
        f"({summary['tempo_only_skipped']} tempo-only skipped), "
        f"{summary['matched']} matched to audio, {summary['unmatched']} not found; "
        f"{summary['fit']} to fit, {summary['held_out']} held out",
        flush=True,
    )
    return summary


def score_tracks(
    model: nn.Module, meta: dict[str, Any], data: str | Path, device: str
) -> dict[str, Any]:
    ordered = [Track(t["id"], t["path"], tuple(t["key"])) for t in meta["tracks"]]
    classes = [int(c) for c in evaluate_tracks(model, ordered, data, device)]
    exact, mirex = score_mirex(classes, meta["refs"])
    return {
        "tracks": len(classes),
        "exact": exact,
        "mirex": mirex,
        "counts": dict(score_breakdown(classes, meta["refs"])),
        "predictions": classes,
    }


def paired_change(
    before: dict[str, Any], after: dict[str, Any], refs: Sequence[int]
) -> dict[str, int]:
    """Tracks whose MIREX credit rose, fell or stayed, between two checkpoints."""
    change = {"improved": 0, "worsened": 0, "unchanged": 0}
    for old, new, ref in zip(before["predictions"], after["predictions"], refs, strict=True):
        old_credit = WEIGHTS[key_relation(old, ref)]
        new_credit = WEIGHTS[key_relation(new, ref)]
        if new_credit > old_credit:
            change["improved"] += 1
        elif new_credit < old_credit:
            change["worsened"] += 1
        else:
            change["unchanged"] += 1
    return change


def correction_batches(
    base_count: int,
    correction_count: int,
    batch: int,
    share: float,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One epoch of (base indices, correction indices) per batch.

    An epoch is one pass over the base data. Every batch holds the same number
    of corrections, round(batch * share), drawn by cycling through shuffled
    passes over the corrections, so their share of each gradient step is fixed
    whatever the size of the personal set.
    """
    per_batch = max(1, round(batch * share))
    base_per_batch = batch - per_batch
    if base_per_batch < 1:
        raise ValueError("corrections share must leave room for base data in each batch")
    base_order = rng.permutation(base_count)
    steps = -(-base_count // base_per_batch)
    needed = steps * per_batch
    passes = -(-needed // correction_count)
    correction_order = np.concatenate([rng.permutation(correction_count) for _ in range(passes)])
    return [
        (
            base_order[step * base_per_batch : (step + 1) * base_per_batch],
            correction_order[step * per_batch : (step + 1) * per_batch],
        )
        for step in range(steps)
    ]


def finetune_on_corrections(
    data: str | Path,
    checkpoint: str | Path,
    device: str,
    *,
    epochs: int = FINETUNE_EPOCHS,
    batch: int = BATCH,
    lr: float = FINETUNE_LR,
    share: float = CORRECTIONS_SHARE,
    seed: int = SEED,
    weight_decay: float = 1e-4,
) -> Path:
    """Fine-tune `checkpoint` on corrections mixed with the base training data.

    Each batch is `share` corrections and the rest base tracks, both cropped and
    pitch shifted as in training. The last epoch is kept: the held-out sets are
    only scored, never used to choose a checkpoint. The report holds GiantSteps
    and held-out corrections scores for the starting and fine-tuned weights.
    """
    if epochs < 1 or batch < 2 or lr <= 0 or weight_decay < 0 or not 0 < share < 1:
        raise ValueError("Invalid fine-tuning hyperparameters")
    data = Path(data)
    base_meta = load_manifest(data, "train")
    test_meta = load_manifest(data, "test")
    fit_meta = load_manifest(data, CORRECTIONS_FIT)
    held_meta = load_manifest(data, CORRECTIONS_HELD_OUT)
    test_hashes = [t["sha256"] for t in test_meta["tracks"]]
    fit_hashes = [t["sha256"] for t in fit_meta["tracks"]]
    held_hashes = [t["sha256"] for t in held_meta["tracks"]]
    assert_disjoint(fit_hashes + held_hashes, test_hashes)
    assert_disjoint(fit_hashes, held_hashes)
    corrected_audio = set(fit_hashes) | set(held_hashes)
    base_tracks = [
        Track(t["id"], t["path"], tuple(t["key"]))
        for t in base_meta["tracks"]
        if t["sha256"] not in corrected_audio
    ]
    fit_tracks = [Track(t["id"], t["path"], tuple(t["key"])) for t in fit_meta["tracks"]]

    model, payload = load_checkpoint(checkpoint, device)
    width = payload.get("model_config", {}).get("width", 1.0)
    before = {
        "giantsteps": score_tracks(model, test_meta, data, device),
        "corrections_held_out": score_tracks(model, held_meta, data, device),
    }

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    run_root = data / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    run = Path(
        tempfile.mkdtemp(
            prefix=f"finetune-share{share}-seed{seed}-{time.strftime('%Y%m%d-%H%M%S')}-",
            dir=run_root,
        )
    )
    config = {
        "base_checkpoint": str(checkpoint),
        "base_checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "epochs": epochs,
        "batch": batch,
        "lr": lr,
        "weight_decay": weight_decay,
        "corrections_share": share,
        "corrections_per_batch": max(1, round(batch * share)),
        "seed": seed,
        "device": str(device),
        "base_tracks": len(base_tracks),
        "corrections_fit": len(fit_tracks),
        "corrections_held_out": len(held_meta["tracks"]),
    }
    history = []
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        samples = 0
        for base_indices, correction_indices in correction_batches(
            len(base_tracks), len(fit_tracks), batch, share, rng
        ):
            chosen = [base_tracks[i] for i in base_indices]
            chosen += [fit_tracks[i] for i in correction_indices]
            feats, y = [], []
            for track in chosen:
                feat, cls = sample_features(track, data, rng, T=CROP_FRAMES)
                feats.append(feat)
                y.append(cls)
            x = torch.zeros(len(feats), N_CORE, CROP_FRAMES)
            for k, feat in enumerate(feats):
                x[k, :, : feat.shape[1]] = torch.from_numpy(feat)
            optimizer.zero_grad()
            loss = criterion(
                model(x.to(device)),
                torch.from_numpy(np.asarray(y, dtype=np.int64)).to(device),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(chosen)
            samples += len(chosen)
        scheduler.step()
        history.append({"epoch": epoch, "loss": loss_sum / samples})
        print(f"epoch {epoch}/{epochs} loss {loss_sum / samples:.4f}", flush=True)

    tuned = run / "drone.pt"
    base_metadata = payload.get("metadata", {})
    save_checkpoint(
        tuned,
        model,
        width=width,
        config=config,
        epoch=epochs,
        git_revision=git_revision(),
        train_ids=list(base_metadata.get("train_ids", []))
        + [t.id for t in base_tracks]
        + [t.id for t in fit_tracks],
        validation_ids=list(base_metadata.get("validation_ids", [])),
        training_audio_sha256=list(base_metadata.get("training_audio_sha256", []))
        + [t["sha256"] for t in base_meta["tracks"]]
        + fit_hashes,
    )
    after = {
        "giantsteps": score_tracks(model, test_meta, data, device),
        "corrections_held_out": score_tracks(model, held_meta, data, device),
    }
    report: dict[str, Any] = {
        "config": config,
        "history": history,
        "elapsed_seconds": time.monotonic() - start,
        "checkpoint_sha256": hashlib.sha256(tuned.read_bytes()).hexdigest(),
    }
    references = {"giantsteps": test_meta["refs"], "corrections_held_out": held_meta["refs"]}
    for name, refs in references.items():
        report[name] = {
            "before": before[name],
            "after": after[name],
            "paired_change": paired_change(before[name], after[name], refs),
        }
        print(
            f"{name} ({before[name]['tracks']} tracks): MIREX "
            f"{100 * before[name]['mirex']:.2f}% -> {100 * after[name]['mirex']:.2f}%, "
            f"exact {100 * before[name]['exact']:.2f}% -> {100 * after[name]['exact']:.2f}%; "
            f"{report[name]['paired_change']}",
            flush=True,
        )
    (run / "finetune.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Fine-tuned checkpoint: {tuned}\nReport: {run / 'finetune.json'}")
    return tuned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "train", "evaluate", "prepare-corrections", "finetune")
    )
    parser.add_argument("checkpoint", nargs="?")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--epochs", type=int, help=f"default {EPOCHS}, or {FINETUNE_EPOCHS} for finetune"
    )
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--lr", type=float, help=f"default {LR}, or {FINETUNE_LR} for finetune")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--threads", type=int, default=1, help="CPU intra-op threads (process-wide)"
    )
    parser.add_argument("--output", type=Path, help="Write evaluation JSON to this path")
    parser.add_argument(
        "--corrections", type=Path, help="An export from `euphoria corrections export`"
    )
    parser.add_argument(
        "--audio", type=Path, help="Folder holding the corrected tracks, searched recursively"
    )
    parser.add_argument(
        "--corrections-share",
        type=float,
        default=CORRECTIONS_SHARE,
        help="Fraction of each fine-tuning batch drawn from corrections",
    )
    args = parser.parse_args()
    finetuning = args.command == "finetune"
    if args.epochs is None:
        args.epochs = FINETUNE_EPOCHS if finetuning else EPOCHS
    if args.lr is None:
        args.lr = FINETUNE_LR if finetuning else LR
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs/batch-size must be positive")
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        if args.command == "prepare":
            beatport = load_beatport(args.data, args.data / "labels")
            giantsteps = load_giantsteps(args.data / "giantsteps")
            assert_disjoint([t.id for t in beatport], [t.id for t in giantsteps])
            prepare_dir("train", beatport, args.data, data=args.data)
            prepare_dir("test", giantsteps, args.data / "giantsteps", data=args.data)
        elif args.command == "prepare-corrections":
            if not args.corrections or not args.audio:
                parser.error("prepare-corrections requires --corrections and --audio")
            prepare_corrections(args.data, args.corrections, args.audio)
        elif args.command == "finetune":
            if not args.checkpoint:
                parser.error("finetune requires a checkpoint path, such as weights/drone-v1.pt")
            finetune_on_corrections(
                args.data,
                args.checkpoint,
                args.device,
                epochs=args.epochs,
                batch=args.batch_size,
                lr=args.lr,
                share=args.corrections_share,
                seed=args.seed,
                weight_decay=args.weight_decay,
            )
        elif args.command == "train":
            train_model(
                args.data,
                args.device,
                epochs=args.epochs,
                batch=args.batch_size,
                seed=args.seed,
                lr=args.lr,
                width=args.width,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
            )
        else:
            if not args.checkpoint:
                parser.error("evaluate requires a checkpoint path")
            result = evaluate_checkpoint(args.data, args.checkpoint, args.device)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
