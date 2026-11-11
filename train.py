"""Prepare features, train on Beatport, and evaluate once on held-out GiantSteps.

Training follows the K10 (MIREX 2026, MIT) protocol: 192 core CQT bins,
320-frame crops, uniform +/-4-semitone shift as an exact re-slice of the
padded spectrogram, plain cross-entropy, AdamW 1e-3, cosine schedule.
"""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from drone.checkpoints import load_checkpoint, save_checkpoint
from drone.datasets import load_beatport, load_giantsteps, prepare, sample_features
from drone.features import CROP_FRAMES, FEATURE_ID, FEATURE_SPEC, N_CORE, audio_digest
from drone.keys import LABEL_SCHEMA, encode_key, score_breakdown, score_mirex
from drone.model import KeyCNN, params_count

DATA = Path(os.environ.get("DRONE_DATA", Path(__file__).resolve().parent / "data"))
SEED = 13
EPOCHS = 120
BATCH = 32
LR = 1e-3


def evaluate_tracks(model, tracks, track_dir, device, batch=16):
    """Per-track evaluation at full length (K10 does not pad at inference)."""
    model.eval()
    pred = []
    for i in range(0, len(tracks), batch):
        xs, ys = [], []
        for track in tracks[i : i + batch]:
            features, _ = prepare(track, track_dir)
            xs.append(features)
        T = max(x.shape[1] for x in xs)
        x = torch.zeros(len(xs), N_CORE, T)
        for j, feat in enumerate(xs):
            x[j, :, : feat.shape[1]] = torch.from_numpy(feat)
        with torch.inference_mode():
            pred.append(model(x.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(pred)


def prepared_path(data, tag):
    return Path(data) / "features" / FEATURE_ID / f"{tag}.json"


def prepare_dir(tag, tracks, track_dir, *, data=DATA):
    """Warm the content-addressed CQT cache and write the track manifest."""
    if not tracks:
        raise ValueError(f"No tracks for {tag}; download the datasets with bash fetch.sh")
    metadata = {
        "feature_spec": FEATURE_SPEC,
        "label_schema": LABEL_SCHEMA,
        "crop": CROP_FRAMES,
        "tracks": [
            {"id": track.id, "path": str(track.mp3), "sha256": audio_digest(track.mp3), "key": list(track.key)}
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


def load_manifest(data, tag):
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


def assert_disjoint(train_ids, test_ids):
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"Training/test overlap: {sorted(overlap)[:10]}")


def validation_split(refs, seed=SEED):
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
    if not train or not val:
        raise ValueError("Need at least two examples in a class for a training/validation split")
    return train, val


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
    width=1.0,
    weight_decay=1e-4,
    label_smoothing=0.0,
):
    data = Path(data)
    meta = load_manifest(data, "train")
    test_meta = load_manifest(data, "test")
    assert_disjoint(meta["ids"], test_meta["ids"])
    assert_disjoint(
        [t["sha256"] for t in meta["tracks"]], [t["sha256"] for t in test_meta["tracks"]]
    )
    from drone.datasets import Track

    ordered = [Track(t["id"], t["path"], tuple(t["key"])) for t in meta["tracks"]]
    refs = meta["refs"]
    tr_idx, val_idx = validation_split(refs, SEED)
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
    run = Path(tempfile.mkdtemp(prefix=f"w{width}-seed{seed}-{time.strftime('%Y%m%d-%H%M%S')}-", dir=run_root))
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
        exact, mirex = score_mirex(val_pred, [refs[i] for i in val_idx])
        record["history"].append(
            {"epoch": epoch, "loss": loss_sum / len(order), "validation_exact": exact, "validation_mirex": mirex}
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


def evaluate_checkpoint(data, checkpoint, device):
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
    from drone.datasets import Track

    ordered = [Track(t["id"], t["path"], tuple(t["key"])) for t in meta["tracks"]]
    pred = evaluate_tracks(model, ordered, data, device)
    exact, mirex = score_mirex(pred, meta["refs"])
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
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=1, help="CPU intra-op threads (process-wide)")
    parser.add_argument("--output", type=Path, help="Write evaluation JSON to this path")
    args = parser.parse_args()
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
