"""Train and evaluate the key detection CNN.

Usage:
  python train.py prepare            # decode + CQT features for train and test sets
  python train.py train              # train, keep best model by validation MIREX
  python train.py evaluate <ckpt>    # report the held-out GiantSteps 604 results

MIREX weights match crates/euphoria-bench: exact 1.0, fifth 0.5, relative 0.3,
parallel 0.2.
"""

import argparse
import sys
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drone.datasets import load_beatport, load_giantsteps, prepare
from drone.features import TONICS
from drone.model import KeyCNN, params_count

DATA = os.environ.get("DRONE_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
FEAT = os.path.join(DATA, "features")
RUN = os.path.join(DATA, "runs")

SEED = 13
T = 300
EPOCHS = 120
BATCH = 64
LR = 1e-3
AUG_SEMITONES = 12  # full circle of shifts, so every key is equally represented


# --- MIREX scoring in the 24-class space (index = tonic * 2 + mode) ---

def score_mirex(pred, ref):
    """Returns (exact, fifth, relative, parallel, other, mirex) as fractions."""
    n = len(ref)
    counts = [0, 0, 0, 0, 0]
    for p, r in zip(pred, ref):
        pt, pm = p % 12, p // 12
        rt, rm = r % 12, r // 12
        if p == r:
            counts[0] += 1
        elif pt == (rt + 7) % 12 and pm == rm:
            counts[1] += 1
        elif pm != rm and pt == (rt + 9) % 12:
            counts[2] += 1
        elif pt == rt:
            counts[3] += 1
        else:
            counts[4] += 1
    mirex = (counts[0] + 0.5 * counts[1] + 0.3 * counts[2] + 0.2 * counts[3]) / n
    return counts[0] / n, mirex


def evaluate_set(model, feats, refs, device, batch=128):
    model.eval()
    pred = []
    with torch.no_grad():
        for i in range(0, len(refs), batch):
            x = torch.from_numpy(feats[i : i + batch]).to(device)
            logits = model(x.transpose(1, 2))
            pred.append(logits.argmax(1).cpu().numpy())
    pred = np.concatenate(pred)
    exact, mirex = score_mirex(pred.tolist(), refs)
    return pred, exact, mirex


def class_id_of(track):
    return TONICS.index(track.key[0]) * 2 + (1 if track.key[1] == "minor" else 0)


def prepare_dir(tag, tracks, track_dir):
    out = os.path.join(FEAT, f"{tag}_feats.npy")
    refs_path = os.path.join(FEAT, f"{tag}_refs.json")
    os.makedirs(FEAT, exist_ok=True)
    if os.path.exists(out) and os.path.exists(refs_path):
        return np.load(out), json.load(open(refs_path))
    feats, refs, ids = [], [], []
    t0 = time.time()
    for i, track in enumerate(tracks):
        f, _ = prepare(track, track_dir, T=T, augment=0)
        feats.append(f)
        refs.append(class_id_of(track))
        ids.append(track.id)
        if (i + 1) % 50 == 0:
            print(f"  {tag}: {i + 1}/{len(tracks)} ({time.time() - t0:.0f}s)", flush=True)
    np.save(out, np.stack(feats))
    json.dump({"refs": refs, "ids": ids}, open(refs_path, "w"))
    print(f"  {tag}: done {len(tracks)} tracks in {time.time() - t0:.0f}s", flush=True)
    return np.stack(feats), refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command")
    ap.add_argument("ckpt", nargs="?", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if args.command == "prepare":
        beatport = load_beatport(DATA, os.path.join(DATA, "labels"))
        print(f"beatport tracks: {len(beatport)}")
        giantsteps = load_giantsteps(os.path.join(DATA, "giantsteps"))
        print(f"giantsteps tracks: {len(giantsteps)}")
        prepare_dir("train", beatport, os.path.join(DATA, "audio"))
        prepare_dir("test", giantsteps, os.path.join(DATA, "giantsteps"))

    elif args.command in ("train", "evaluate"):
        feats, refs = np.load(os.path.join(FEAT, "train_feats.npy")), None
        meta = json.load(open(os.path.join(FEAT, "train_refs.json")))
        refs = meta["refs"]
        # validation split: every 10th track, never touched for training
        val_idx = list(range(9, len(refs), 10))
        tr_idx = [i for i in range(len(refs)) if i not in set(val_idx)]
        val_refs = [refs[i] for i in val_idx]

        model = KeyCNN().to(device)
        print(f"device {device}, params {params_count(model) / 1e6:.2f}M")

        if args.command == "evaluate":
            ckpt = torch.load(args.ckpt, map_location=device)
            model.load_state_dict(ckpt)
        else:
            os.makedirs(RUN, exist_ok=True)
            ckpt_path = os.path.join(RUN, f"key-cnn-run{time.strftime('%Y%m%d-%H%M%S')}.pt")
            criterion = nn.CrossEntropyLoss()
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
            best_val, best_epoch = -1.0, 0
            for epoch in range(1, EPOCHS + 1):
                model.train()
                order = np.random.permutation(tr_idx)
                loss_sum, t0 = 0.0, time.time()
                for i in range(0, len(order), BATCH):
                    idx = order[i : i + BATCH]
                    x = np.stack([feats[j] for j in idx])
                    shift = np.random.randint(-AUG_SEMITONES // 2, AUG_SEMITONES // 2 + 1, len(idx))
                    y = np.array([((refs[j] % 12) + s) % 12 * 2 + refs[j] // 12 for j, s in zip(idx, shift)])
                    for j, s in enumerate(idx):
                        if s:
                            x[j] = np.roll(x[j], s, axis=1)
                    xb = torch.from_numpy(x).to(device)
                    yb = torch.from_numpy(y.astype(np.int64)).to(device)
                    opt.zero_grad()
                    loss = criterion(model(xb.transpose(1, 2)), yb)
                    loss.backward()
                    opt.step()
                    loss_sum += loss.item()
                sched.step()
                val_pred, _, val_mirex = evaluate_set(model, feats[val_idx], val_refs, device)
                if val_mirex > best_val:
                    best_val, best_epoch = val_mirex, epoch
                    torch.save(model.state_dict(), ckpt_path)
                if epoch % 5 == 0 or epoch == 1:
                    print(
                        f"epoch {epoch}/{EPOCHS} loss {loss_sum / len(order) * BATCH:.4f} "
                        f"val MIREX {100 * val_mirex:.1f}% (best {100 * best_val:.1f}% @ {best_epoch}) "
                        f"{time.time() - t0:.0f}s",
                        flush=True,
                    )
            print(f"training done, best val MIREX {100 * best_val:.1f}% @ epoch {best_epoch}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

        test_feats, test_meta = (
            np.load(os.path.join(FEAT, "test_feats.npy")),
            json.load(open(os.path.join(FEAT, "test_refs.json"))),
        )
        pred, exact, mirex = evaluate_set(model, test_feats, test_meta["refs"], device)
        # per-class breakdown for the table
        from collections import Counter

        cats = Counter()
        for p, r in zip(pred, test_meta["refs"]):
            pt, pm = p % 12, p // 12
            rt, rm = r % 12, r // 12
            if p == r:
                cats["exact"] += 1
            elif pt == (rt + 7) % 12 and pm == rm:
                cats["fifth"] += 1
            elif pm != rm and pt == (rt + 9) % 12:
                cats["relative"] += 1
            elif pt == rt:
                cats["parallel"] += 1
            else:
                cats["other"] += 1
        n = len(test_meta["refs"])
        print(
            f"held-out GiantSteps 604: exact {100 * exact:.1f}%  "
            + "  ".join(f"{k} {100 * cats[k] / n:.1f}%" for k in ("fifth", "relative", "parallel", "other"))
            + f"  MIREX {100 * mirex:.1f}%",
            flush=True,
        )
        if args.command == "train":
            print(ckpt_path, flush=True)


if __name__ == "__main__":
    main()
