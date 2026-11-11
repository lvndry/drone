"""Capped, supervised CQT/octave pilot. Does not open calibration/final audio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from drone.keys import LABEL_SCHEMA, score_mirex
from experiments.adapt import development_split
from experiments.octave_model import FEATURE_SPEC, OctaveNet, compress
from experiments.reference.features import audio_digest
from experiments.reference.train import load_prepared
from experiments.small import atomic_json, load_development, signature, snapshot_sources


def device_setup(device: str) -> None:
    torch.set_num_threads(4)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
        free, _ = torch.cuda.mem_get_info()
        if free < 8 * 1024**3:
            raise RuntimeError("Need 8 GiB currently free; do not interrupt vLLM")
        torch.cuda.set_per_process_memory_fraction(0.04)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False


def selected_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    _, bm, btr, bval = load_development(args.beatport)
    _, fm = load_prepared(args.fmak, "fmak")
    partition, audit = json.loads(args.partition.read_bytes()), json.loads(args.groups.read_bytes())
    assert audit["partition_sha256"] == hashlib.sha256(args.partition.read_bytes()).hexdigest()
    parts = development_split(fm, partition, audit)
    assert not {t["sha256"] for t in bm["tracks"]} & {t["sha256"] for t in fm["tracks"]}
    bp_audio = {p.stem.split(" ")[0]: p for p in (args.beatport / "audio/audio").glob("*.mp3")}
    rows = []
    for domain, meta, indices, role in (
        ("beatport", bm, btr, "train"),
        ("fmak", fm, parts["train"], "train"),
        ("beatport", bm, bval, "validation"),
        ("fmak", fm, parts["validation"], "validation"),
    ):
        for i in indices:
            identity = meta["ids"][i]
            path = (
                bp_audio[identity]
                if domain == "beatport"
                else args.fmak / "audio" / f"{identity}.mp3"
            )
            rows.append(
                {
                    "id": domain + ":" + identity,
                    "path": str(path),
                    "sha256": meta["tracks"][i]["sha256"],
                    "ref": meta["refs"][i],
                    "domain": domain,
                    "role": role,
                }
            )
    assert not {r["sha256"] for r in rows if r["role"] == "train"} & {
        r["sha256"] for r in rows if r["role"] == "validation"
    }
    return rows


def decode(row: dict[str, Any]) -> np.ndarray:
    assert audio_digest(row["path"]) == row["sha256"]
    duration = float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                row["path"],
            ],
            text=True,
        ).strip()
    )
    segments = []
    for fraction in (0.25, 0.75):
        start = max(0, min(duration - 32, duration * fraction - 16))
        output = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-ss",
                str(start),
                "-i",
                row["path"],
                "-t",
                "32",
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
            timeout=60,
        )
        x = np.frombuffer(output.stdout, dtype="<f4")
        if not len(x) or not np.isfinite(x).all():
            raise ValueError(f"Invalid decoded segment: {row['id']}")
        segments.append(np.pad(x[: 704000 + 1600], (0, max(0, 705600 - len(x)))))
    return np.stack(segments)


def prepare(args: argparse.Namespace) -> None:
    from importlib.metadata import distribution, version

    from nnAudio.features import VQT

    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_sources(args.output)
    device_setup(args.device)
    rows = selected_rows(args)
    identity = {
        "feature_spec": FEATURE_SPEC,
        "rows": rows,
        "partition_sha256": hashlib.sha256(args.partition.read_bytes()).hexdigest(),
        "group_sha256": hashlib.sha256(args.groups.read_bytes()).hexdigest(),
        "source_sha256": signature(snapshot_sources(args.output)),
        "nnaudio": version("nnAudio"),
        "scipy": version("scipy"),
        "torch": str(torch.__version__),
        "dependency_records_sha256": {
            name: hashlib.sha256(
                (distribution(name).read_text("RECORD") or "").encode()
            ).hexdigest()
            for name in ("nnAudio", "scipy", "torch", "numpy")
        },
    }
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_bytes()) != identity:
        raise ValueError("Existing feature run differs")
    atomic_json(manifest, identity)
    shape = (len(rows), 2, 99, 1379)
    path, status = args.output / "features.npy", args.output / "preparation.json"
    completed = json.loads(status.read_bytes())["completed"] if status.exists() else 0
    if completed and not path.exists():
        raise ValueError("Missing partial feature array")
    array = np.lib.format.open_memmap(
        path, mode="r+" if path.exists() else "w+", dtype=np.float32, shape=shape
    )
    transform = VQT(
        sr=22050,
        hop_length=512,
        fmin=27.5,
        n_bins=99,
        bins_per_octave=12,
        gamma=0,
        trainable=False,
        verbose=False,
    ).to(args.device)
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
        for offset in range(completed, len(rows), 8):
            if time.monotonic() - start > 1200:
                raise TimeoutError(
                    "Twenty-minute feature preparation cap reached; resume retains completed rows"
                )
            waves = list(pool.map(decode, rows[offset : offset + 8]))
            for local, wave in enumerate(waves):
                magnitude = transform(torch.from_numpy(wave).to(args.device))
                features = compress(magnitude).cpu().numpy()
                if features.shape != shape[1:] or not np.isfinite(features).all():
                    raise ValueError(f"Unexpected CQT shape/content: {features.shape}")
                array[offset + local] = features
            array.flush()
            completed = offset + len(waves)
            atomic_json(
                status,
                {
                    "completed": completed,
                    "tracks": len(rows),
                    "elapsed_seconds_this_process": time.monotonic() - start,
                    "complete": completed == len(rows),
                    "bytes": int(array.nbytes),
                },
            )
            if completed % 80 == 0 or completed == len(rows):
                print(
                    f"CQT prepared {completed}/{len(rows)} in {time.monotonic() - start:.1f}s",
                    flush=True,
                )


def probabilities(
    model: nn.Module, array: np.ndarray, indices: Sequence[int], device: str, batch: int = 16
) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch):
            selected = indices[start : start + batch]
            total = None
            for segment in (0, 1):
                x = np.stack([array[i, segment, 12:96] for i in selected])
                p = torch.softmax(model(torch.from_numpy(x).to(device)), 1).cpu().numpy()
                total = p if total is None else total + p
            output.append(total / 2)
    return np.concatenate(output)


def metrics(prob: np.ndarray, refs: np.ndarray) -> dict[str, Any]:
    pred = prob.argmax(1)
    exact, mirex = score_mirex(pred, refs)
    return {
        "exact": exact,
        "mirex": mirex,
        "mode_accuracy": float(np.mean(pred % 2 == refs % 2)),
        "predictions": pred.tolist(),
    }


def train(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    sources = snapshot_sources(args.output)
    device_setup(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    manifest = json.loads((args.data / "manifest.json").read_bytes())
    assert manifest["feature_spec"] == FEATURE_SPEC
    assert json.loads((args.data / "preparation.json").read_bytes())["complete"]
    rows = manifest["rows"]
    array = np.load(args.data / "features.npy", mmap_mode="r", allow_pickle=False)
    tr = [i for i, row in enumerate(rows) if row["role"] == "train"]
    val = [i for i, row in enumerate(rows) if row["role"] == "validation"]
    refs = np.array([row["ref"] for row in rows])
    reference = json.loads(args.reference.read_bytes())
    identity = {
        "feature_manifest_sha256": signature(manifest),
        "source_sha256": sources,
        "seed": args.seed,
        "epochs": 120,
        "batch": 16,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "smoothing": 0.1,
        "device": args.device,
        "selection": "combined validation per-track symmetric MIREX",
        "early_stopping_patience": 25,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
    }
    model = OctaveNet().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 120)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    record: dict[str, Any] = {
        "identity": identity,
        "status": "running",
        "history": [],
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch": str(torch.__version__),
    }
    best, first_epoch, prior = -1.0, 1, 0.0
    last = args.output / "last.pt"
    if last.exists():
        state = torch.load(last, weights_only=True, map_location="cpu")
        assert state["identity"] == identity
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        rng.bit_generator.state = state["numpy_rng"]
        record, best, first_epoch, prior = (
            state["record"],
            state["best"],
            state["epoch"] + 1,
            state["record"]["elapsed_seconds"],
        )
    start = time.monotonic()
    for epoch in range(first_epoch, 121):
        model.train()
        order = rng.permutation(tr)
        loss_sum = 0.0
        for offset in range(0, len(order), 16):
            if prior + time.monotonic() - start > 900:
                record.update(
                    status="budget-stopped", elapsed_seconds=prior + time.monotonic() - start
                )
                atomic_json(args.output / "run.json", record)
                raise TimeoutError(
                    "Fifteen-minute fitting cap reached; last epoch remains resumable"
                )
            selected = order[offset : offset + 16]
            segments, crops = rng.integers(0, 2, len(selected)), rng.integers(0, 16, len(selected))
            x = np.stack(
                [
                    array[i, segment, crop : crop + 84]
                    for i, segment, crop in zip(selected, segments, crops, strict=True)
                ]
            )
            labels = ((refs[selected] // 2 - crops) % 12) * 2 + refs[selected] % 2
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model(torch.from_numpy(x).to(args.device)), torch.from_numpy(labels).to(args.device)
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(selected)
        scheduler.step()
        prob = probabilities(model, array, val, args.device)
        result = metrics(prob, refs[val])
        record["history"].append(
            {
                "epoch": epoch,
                "loss": loss_sum / len(tr),
                "validation_mirex": result["mirex"],
                "validation_exact": result["exact"],
            }
        )
        if result["mirex"] > best:
            best = result["mirex"]
            record["best_epoch"] = epoch
            torch.save(
                {
                    "format_version": "octave-supervised-pilot-v1",
                    "feature_spec": FEATURE_SPEC,
                    "label_schema": LABEL_SCHEMA,
                    "state_dict": model.state_dict(),
                    "identity": identity,
                    "train_ids": [rows[i]["id"] for i in tr],
                    "validation_ids": [rows[i]["id"] for i in val],
                    "training_audio_sha256": [rows[i]["sha256"] for i in tr],
                },
                args.output / "best.pt",
            )
            domain_metrics = {}
            for domain in ("beatport", "fmak"):
                positions = [j for j, i in enumerate(val) if rows[i]["domain"] == domain]
                domain_metrics[domain] = metrics(prob[positions], refs[np.array(val)[positions]])
            atomic_json(args.output / "best-validation.json", domain_metrics)
        record["elapsed_seconds"] = prior + time.monotonic() - start
        atomic_json(args.output / "run.json", record)
        pending = last.with_suffix(".pending")
        torch.save(
            {
                "identity": identity,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if args.device.startswith("cuda")
                else [],
                "numpy_rng": rng.bit_generator.state,
                "record": record,
                "best": best,
                "epoch": epoch,
            },
            pending,
        )
        os.replace(pending, last)
        print(
            f"Octave epoch={epoch}/120 val={result['mirex']:.4f} best={best:.4f} elapsed={record['elapsed_seconds']:.1f}s",
            flush=True,
        )
        if epoch - record["best_epoch"] >= 25:
            record["early_stopped"] = True
            break
    domain_metrics = json.loads((args.output / "best-validation.json").read_bytes())
    delta = {
        domain: domain_metrics[domain]["mirex"] - reference["metrics"][domain]["mirex"]
        for domain in domain_metrics
    }
    record.update(
        status="complete",
        domain_metrics=domain_metrics,
        delta_vs_adapted_seed13=delta,
        qualifies_for_confirmation=delta["fmak"] >= 0.02 and delta["beatport"] >= -0.02,
        calibration_used=False,
        final_holdout_used=False,
    )
    atomic_json(args.output / "run.json", record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, choices=(13, 37, 73), default=13)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("beatport", "fmak", "partition", "groups", "data", "reference"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else train)(args)


if __name__ == "__main__":
    main()
