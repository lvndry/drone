"""Prepare features, train on Beatport, and evaluate once on held-out GiantSteps."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from drone.keys import LABEL_SCHEMA, encode_key, score_breakdown, score_mirex, transpose_key
from experiments.reference.checkpoints import load_checkpoint, save_checkpoint
from experiments.reference.datasets import load_beatport, load_giantsteps, prepare
from experiments.reference.features import (
    FEATURE_ID,
    FEATURE_SPEC,
    N_SEM,
    audio_digest,
    shift_features,
)
from experiments.reference.model import MODEL_SIZES, KeyCNN, params_count

DATA = Path(os.environ.get("DRONE_DATA", Path(__file__).resolve().parents[2] / "data"))
SEED = 13
T = 300
EPOCHS = 120
BATCH = 64
LR = 1e-3


def evaluate_set(model, feats, refs, device, batch=128):
    if not len(refs) or len(feats) != len(refs) or batch < 1:
        raise ValueError(
            "Evaluation needs matching nonempty features/references and a positive batch"
        )
    model.eval()
    pred = []
    with torch.inference_mode():
        for i in range(0, len(refs), batch):
            x = torch.from_numpy(feats[i : i + batch]).to(device)
            pred.append(model(x.transpose(1, 2)).argmax(1).cpu().numpy())
    pred = np.concatenate(pred)
    exact, mirex = score_mirex(pred, refs)
    return pred, exact, mirex


def prepared_path(data, tag):
    return Path(data) / "features" / FEATURE_ID / f"{tag}.npz"


def prepare_dir(tag, tracks, track_dir, *, data=DATA, frames=T):
    if not tracks:
        raise ValueError(f"No tracks for {tag}; download the datasets with bash fetch.sh")
    manifest = [
        {"id": track.id, "sha256": audio_digest(track.mp3), "key": list(track.key)}
        for track in tracks
    ]
    metadata = {
        "feature_spec": FEATURE_SPEC,
        "label_schema": LABEL_SCHEMA,
        "frames": frames,
        "tracks": manifest,
        "refs": [encode_key(*track.key) for track in tracks],
        "ids": [track.id for track in tracks],
    }
    out = prepared_path(data, tag)
    if out.is_file():
        features, cached = load_prepared(data, tag)
        if cached == metadata:
            return features, cached
    features = []
    start = time.monotonic()
    for i, track in enumerate(tracks):
        f, _ = prepare(track, track_dir, T=frames)
        features.append(f)
        if (i + 1) % 50 == 0:
            print(f"{tag}: {i + 1}/{len(tracks)} ({time.monotonic() - start:.0f}s)", flush=True)
    features = np.stack(features)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Replace only after the complete bundle has been written.
    import tempfile

    with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        pending = Path(scratch) / "features.npz"
        np.savez(pending, features=features, metadata=json.dumps(metadata))
        os.replace(pending, out)
    return features, metadata


def load_prepared(data, tag):
    path = prepared_path(data, tag)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing current {tag} features. Run uv run python -m experiments.reference.train prepare."
        )
    with np.load(path, allow_pickle=False) as bundle:
        feats, metadata = bundle["features"], json.loads(str(bundle["metadata"]))
    if metadata.get("feature_spec") != FEATURE_SPEC or metadata.get("label_schema") != LABEL_SCHEMA:
        raise ValueError(
            "Stale feature cache; run uv run python -m experiments.reference.train prepare"
        )
    refs = metadata["refs"]
    if (
        not refs
        or feats.shape != (len(refs), metadata["frames"], N_SEM)
        or feats.dtype != np.float32
        or not np.isfinite(feats).all()
        or len(metadata["ids"]) != len(refs)
    ):
        raise ValueError(
            f"Invalid {tag} feature bundle; run uv run python -m experiments.reference.train prepare"
        )
    for ref in refs:
        transpose_key(ref, 0)
    return feats, metadata


def assert_disjoint(train_ids, test_ids):
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"Training/test overlap: {sorted(overlap)[:10]}")


def validation_split(refs, seed=SEED, audio_hashes=None):
    """Fixed seeded, stratified split; training augmentation never touches validation."""
    rng = np.random.default_rng(seed)
    val = []
    for key in sorted(set(refs)):
        indices = np.flatnonzero(np.asarray(refs) == key)
        rng.shuffle(indices)
        if len(indices) > 1:
            val.extend(indices[: max(1, len(indices) // 10)].tolist())
    val = sorted(val)
    chosen = set(val)
    train = [i for i in range(len(refs)) if i not in chosen]
    if audio_hashes is not None:
        if len(audio_hashes) != len(refs):
            raise ValueError("Audio identities must align with key labels")
        labels_by_audio = {}
        for digest, ref in zip(audio_hashes, refs, strict=True):
            if digest in labels_by_audio and labels_by_audio[digest] != ref:
                raise ValueError("Identical audio has conflicting labels; resolve before splitting")
            labels_by_audio[digest] = ref
        # Preserve the original validation identities. Quarantine training copies
        # of validation audio, and count each remaining recording only once.
        seen = set()
        unique_val = []
        for i in val:
            if audio_hashes[i] not in seen:
                unique_val.append(i)
                seen.add(audio_hashes[i])
        unique_train = []
        for i in train:
            if audio_hashes[i] not in seen:
                unique_train.append(i)
                seen.add(audio_hashes[i])
        train, val = unique_train, unique_val
    if not train or not val:
        raise ValueError(
            "Need at least two distinct examples in a class for a training/validation split"
        )
    return train, val


def augment_batch(feats, refs, shifts):
    if not len(refs) or len(feats) != len(refs) or len(shifts) != len(refs):
        raise ValueError("Augmentation requires matching nonempty features, labels, and shifts")
    return (
        np.stack([shift_features(f, s) for f, s in zip(feats, shifts, strict=True)]),
        np.asarray(
            [transpose_key(r, s) for r, s in zip(refs, shifts, strict=True)], dtype=np.int64
        ),
    )


def git_revision():
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
    data,
    device,
    *,
    epochs=EPOCHS,
    batch=BATCH,
    seed=SEED,
    lr=LR,
    model_size="small",
    dropout=0.3,
    weight_decay=1e-4,
    label_smoothing=0.0,
):
    feats, meta = load_prepared(data, "train")
    # Refuse leaked data before starting training; do not score the test set here.
    _, test_meta = load_prepared(data, "test")
    assert_disjoint(meta["ids"], test_meta["ids"])
    assert_disjoint(
        [t["sha256"] for t in meta["tracks"]], [t["sha256"] for t in test_meta["tracks"]]
    )
    tr_idx, val_idx = validation_split(meta["refs"], SEED, [t["sha256"] for t in meta["tracks"]])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if epochs < 1 or batch < 1 or lr <= 0 or weight_decay < 0 or not 0 <= label_smoothing < 1:
        raise ValueError("Invalid training hyperparameters")
    model = KeyCNN(size=model_size, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    run = Path(data) / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    checkpoint = run / "drone.pt"
    config = {
        "epochs": epochs,
        "batch": batch,
        "seed": seed,
        "lr": lr,
        "device": str(device),
        "split_seed": SEED,
        "model_size": model_size,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
    }
    root = Path(__file__).resolve().parents[2]
    source_files = [
        *sorted(Path(__file__).parent.glob("*.py")),
        root / "drone" / "keys.py",
        root / "drone" / "download.py",
    ]
    source_hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files
    }
    record = {
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
    lock = root / "uv.lock"
    record["lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None
    best_val, start = -1.0, time.monotonic()
    print(f"device {device}, params {params_count(model):,}", flush=True)
    refs = meta["refs"]
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(tr_idx)
        loss_sum = 0.0
        for i in range(0, len(order), batch):
            idx = order[i : i + batch]
            # Twelve shifts, uniform modulo 12. Mode proportions are unchanged.
            shifts = rng.integers(-6, 6, size=len(idx))
            x, y = augment_batch(feats[idx], [refs[j] for j in idx], shifts)
            optimizer.zero_grad()
            loss = criterion(
                model(torch.from_numpy(x).to(device).transpose(1, 2)),
                torch.from_numpy(y).to(device),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(idx)
        scheduler.step()
        _, exact, mirex = evaluate_set(model, feats[val_idx], [refs[i] for i in val_idx], device)
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
                frames=meta["frames"],
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
    print(
        f"Best checkpoint: {checkpoint}\nRun `uv run python -m experiments.reference.train evaluate {checkpoint}` when ready."
    )
    return checkpoint


def evaluate_checkpoint(data, checkpoint, device):
    feats, meta = load_prepared(data, "test")
    model, payload = load_checkpoint(checkpoint, device)
    if payload["frames"] != meta["frames"]:
        raise ValueError("Checkpoint and test features have different frame counts; rerun prepare")
    trained_ids = payload.get("metadata", {}).get("train_ids", [])
    validation_ids = payload.get("metadata", {}).get("validation_ids", [])
    assert_disjoint(trained_ids + validation_ids, meta["ids"])
    assert_disjoint(
        payload.get("metadata", {}).get("training_audio_sha256", []),
        [t["sha256"] for t in meta["tracks"]],
    )
    pred, exact, mirex = evaluate_set(model, feats, meta["refs"], device)
    cats = score_breakdown(pred, meta["refs"])
    result = {
        "tracks": len(pred),
        "exact": exact,
        "mirex": mirex,
        "mirex_asymmetric": score_mirex(pred, meta["refs"], symmetric=False)[1],
        "counts": dict(cats),
        "feature_spec": FEATURE_SPEC,
        "test_manifest": meta,
        "predictions": pred.tolist(),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("checkpoint", nargs="?")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--frames", type=int, default=T)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model-size", choices=MODEL_SIZES, default="small")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--threads", type=int, default=1, help="CPU intra-op threads (process-wide)"
    )
    parser.add_argument("--output", type=Path, help="Write evaluation JSON to this path")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.frames < 148:
        parser.error("epochs/batch-size must be positive and frames must be at least 148")
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        if args.command == "prepare":
            beatport = load_beatport(args.data, args.data / "labels")
            giantsteps = load_giantsteps(args.data / "giantsteps")
            assert_disjoint([t.id for t in beatport], [t.id for t in giantsteps])
            prepare_dir("train", beatport, args.data / "audio", data=args.data, frames=args.frames)
            prepare_dir(
                "test", giantsteps, args.data / "giantsteps", data=args.data, frames=args.frames
            )
        elif args.command == "train":
            train_model(
                args.data,
                args.device,
                epochs=args.epochs,
                batch=args.batch_size,
                seed=args.seed,
                lr=args.lr,
                model_size=args.model_size,
                dropout=args.dropout,
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
