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
    _, beatport_metadata, beatport_train, beatport_validation = load_development(args.beatport)
    _, fmak_metadata = load_prepared(args.fmak, "fmak")
    partition = json.loads(args.partition.read_bytes())
    audit = json.loads(args.groups.read_bytes())
    assert audit["partition_sha256"] == hashlib.sha256(args.partition.read_bytes()).hexdigest()
    parts = development_split(fmak_metadata, partition, audit)
    assert not {track["sha256"] for track in beatport_metadata["tracks"]} & {
        track["sha256"] for track in fmak_metadata["tracks"]
    }
    beatport_audio = {
        audio_path.stem.split(" ")[0]: audio_path
        for audio_path in (args.beatport / "audio/audio").glob("*.mp3")
    }
    rows = []
    for domain, meta, indices, role in (
        ("beatport", beatport_metadata, beatport_train, "train"),
        ("fmak", fmak_metadata, parts["train"], "train"),
        ("beatport", beatport_metadata, beatport_validation, "validation"),
        ("fmak", fmak_metadata, parts["validation"], "validation"),
    ):
        for index in indices:
            identity = meta["ids"][index]
            path = (
                beatport_audio[identity]
                if domain == "beatport"
                else args.fmak / "audio" / f"{identity}.mp3"
            )
            rows.append(
                {
                    "id": domain + ":" + identity,
                    "path": str(path),
                    "sha256": meta["tracks"][index]["sha256"],
                    "ref": meta["refs"][index],
                    "domain": domain,
                    "role": role,
                }
            )
    assert not {row["sha256"] for row in rows if row["role"] == "train"} & {
        row["sha256"] for row in rows if row["role"] == "validation"
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
        samples = np.frombuffer(output.stdout, dtype="<f4")
        if not len(samples) or not np.isfinite(samples).all():
            raise ValueError(f"Invalid decoded segment: {row['id']}")
        segments.append(np.pad(samples[: 704000 + 1600], (0, max(0, 705600 - len(samples)))))
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
    path = args.output / "features.npy"
    status = args.output / "preparation.json"
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
                inputs = np.stack([array[index, segment, 12:96] for index in selected])
                probability = (
                    torch.softmax(model(torch.from_numpy(inputs).to(device)), 1).cpu().numpy()
                )
                total = probability if total is None else total + probability
            output.append(total / 2)
    return np.concatenate(output)


def metrics(probability: np.ndarray, references: np.ndarray) -> dict[str, Any]:
    prediction = probability.argmax(1)
    classes = [int(predicted) for predicted in prediction]
    exact, mirex = score_mirex(classes, [int(reference) for reference in references])
    return {
        "exact": exact,
        "mirex": mirex,
        "mode_accuracy": float(np.mean(prediction % 2 == references % 2)),
        "predictions": classes,
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
    training = [index for index, row in enumerate(rows) if row["role"] == "train"]
    validation = [index for index, row in enumerate(rows) if row["role"] == "validation"]
    references = np.array([row["ref"] for row in rows])
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
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "torch": str(torch.__version__),
    }
    best = -1.0
    first_epoch = 1
    prior = 0.0
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
        record = state["record"]
        best = state["best"]
        first_epoch = state["epoch"] + 1
        prior = state["record"]["elapsed_seconds"]
    start = time.monotonic()
    for epoch in range(first_epoch, 121):
        model.train()
        order = rng.permutation(training)
        loss_sum = 0.0
        for offset in range(0, len(order), 16):
            if prior + time.monotonic() - start > 900:
                record.update(
                    status="budget-stopped", elapsed_seconds=float(prior + time.monotonic() - start)
                )
                atomic_json(args.output / "run.json", record)
                raise TimeoutError(
                    "Fifteen-minute fitting cap reached; last epoch remains resumable"
                )
            selected = order[offset : offset + 16]
            segments = rng.integers(0, 2, len(selected))
            crops = rng.integers(0, 16, len(selected))
            inputs = np.stack(
                [
                    array[index, segment, crop : crop + 84]
                    for index, segment, crop in zip(selected, segments, crops, strict=True)
                ]
            )
            labels = ((references[selected] // 2 - crops) % 12) * 2 + references[selected] % 2
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model(torch.from_numpy(inputs).to(args.device)),
                torch.from_numpy(labels).to(args.device),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(selected)
        scheduler.step()
        probability = probabilities(model, array, validation, args.device)
        result = metrics(probability, references[validation])
        record["history"].append(
            {
                "epoch": epoch,
                "loss": loss_sum / len(training),
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
                    "train_ids": [rows[index]["id"] for index in training],
                    "validation_ids": [rows[index]["id"] for index in validation],
                    "training_audio_sha256": [rows[index]["sha256"] for index in training],
                },
                args.output / "best.pt",
            )
            domain_metrics: dict[str, dict[str, Any]] = {}
            for domain in ("beatport", "fmak"):
                positions = [
                    position
                    for position, index in enumerate(validation)
                    if rows[index]["domain"] == domain
                ]
                domain_metrics[domain] = metrics(
                    probability[positions], references[np.array(validation)[positions]]
                )
            atomic_json(args.output / "best-validation.json", domain_metrics)
        record["elapsed_seconds"] = float(prior + time.monotonic() - start)
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
    delta: dict[str, float] = {
        domain: float(domain_metrics[domain]["mirex"])
        - float(reference["metrics"][domain]["mirex"])
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
