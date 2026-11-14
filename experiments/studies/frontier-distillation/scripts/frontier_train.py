"""Frontier experiments for the K10 key classifier.

pretrain: label-free transposition-equivariance training on cached CQTs.
train:    the v2a supervised recipe, optionally initialised from a pretrained checkpoint.

Both use the same in-memory CQT store. Supervised runs select the checkpoint on the
internal validation split only; the valid, holdout and GiantSteps panels are scored
once, after training, on the selected checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from environment import required_path

SOURCE_DIRECTORY = required_path("DRONE_SOURCE")
PREPARED_DIRECTORY = required_path("DRONE_PREP")
GIANTSTEPS_DIRECTORY = required_path("DRONE_GIANTSTEPS")
WORK_DIRECTORY = required_path("DRONE_WORK")

sys.path.insert(0, str(SOURCE_DIRECTORY))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rescore_gsplus import load_gsplus, load_original  # noqa: E402
from torch import nn  # noqa: E402

from drone.checkpoints import load_checkpoint, save_checkpoint  # noqa: E402
from drone.features import BINS_PER_SEMI, CROP_FRAMES, N_CORE, PAD_SEMI, extract_cqt  # noqa: E402
from drone.keys import score_mirex, transpose_key  # noqa: E402
from drone.model import KeyCNN  # noqa: E402
from drone.splits import validation_split  # noqa: E402

DROPPED_TRACK_IDS = {"99017"}
FEATURE_VIEW = os.environ.get("DRONE_FEATURE_VIEW", "plain")
OMEGA = 2 * math.pi / 12


def load_arrays(paths: list[Path]) -> list[np.ndarray]:
    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(np.load, paths))


def core_slice(array: np.ndarray, shift: int) -> np.ndarray:
    low = (PAD_SEMI + shift) * BINS_PER_SEMI
    return array[low : low + N_CORE]


def mask_edges(
    features: np.ndarray, maximum_bins: int, generator: np.random.Generator
) -> np.ndarray:
    low = int(generator.integers(0, maximum_bins + 1))
    high = int(generator.integers(0, maximum_bins + 1))
    features[:low] = 0.0
    if high:
        features[-high:] = 0.0
    return features


def wide_slice(array: np.ndarray, shift: int) -> np.ndarray:
    """Core slice for shifts beyond the cached pad; bins outside the CQT range are silent."""
    if abs(shift) <= PAD_SEMI:
        return core_slice(array, shift)
    extra = (abs(shift) - PAD_SEMI) * BINS_PER_SEMI
    padded = np.pad(array, ((extra, extra), (0, 0)))
    low = (PAD_SEMI + shift) * BINS_PER_SEMI + extra
    return padded[low : low + N_CORE]


def crop(array: np.ndarray, shift: int, frames: int, generator: np.random.Generator) -> np.ndarray:
    """Choose the time window before shifting, so long tracks are never copied whole."""
    if array.shape[1] <= frames:
        features = wide_slice(array, shift)
        padded = np.zeros((N_CORE, frames), dtype=np.float32)
        padded[:, : features.shape[1]] = features
        return padded
    start = int(generator.integers(0, array.shape[1] - frames + 1))
    return wide_slice(array[:, start : start + frames], shift).astype(np.float32)


def predict_tracks(model: nn.Module, arrays: list[np.ndarray], device: str) -> list[int]:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for array in arrays:
            features = torch.from_numpy(core_slice(array, 0).astype(np.float32)).unsqueeze(0)
            predictions.append(int(model(features.to(device)).argmax()))
    return predictions


def scored(predictions: list[int], references: list[int]) -> dict:
    exact, mirex = score_mirex(predictions, references)
    return {"n": len(references), "exact": round(100 * exact, 2), "mirex": round(100 * mirex, 2)}


def load_manifest() -> dict[str, list[dict]]:
    rows = json.loads((PREPARED_DIRECTORY / "manifest.json").read_text())["tracks"]
    panels: dict[str, list[dict]] = {"train": [], "valid": [], "holdout": []}
    for row in rows:
        if row["track_id"] in DROPPED_TRACK_IDS and row["panel"] == "train":
            continue
        panels[row["panel"]].append(row)
    return panels


def feature_path(row: dict) -> Path:
    """Harmonic-view features where they exist; the six clips too short for HPSS keep the plain view."""
    if FEATURE_VIEW == "harmonic":
        harmonic = WORK_DIRECTORY / "harmonic-cqt" / f"{row['audio_sha256']}.npy"
        if harmonic.exists():
            return harmonic
    return PREPARED_DIRECTORY / row["cqt_path"]


def manifest_arrays(rows: list[dict]) -> list[np.ndarray]:
    return load_arrays([feature_path(row) for row in rows])


def giantsteps_arrays(audio_files: list[Path]) -> list[np.ndarray]:
    if FEATURE_VIEW == "harmonic":
        return load_arrays(
            [
                WORK_DIRECTORY / "harmonic-gs" / f"{path.name.split('.')[0]}.npy"
                for path in audio_files
            ]
        )
    return [extract_cqt(str(path), WORK_DIRECTORY / "gs-cqt-cache") for path in audio_files]


def final_panels(model: nn.Module, device: str, panels: dict[str, list[dict]]) -> dict:
    report = {}
    for panel in ("valid", "holdout"):
        rows = panels[panel]
        predictions = predict_tracks(model, manifest_arrays(rows), device)
        report[panel] = scored(predictions, [int(row["key_label"]) for row in rows])
    audio_files = sorted(
        path
        for path in (GIANTSTEPS_DIRECTORY / "audio").glob("*.mp3")
        if not path.name.startswith("._")
    )
    track_ids = [path.name.split(".")[0] for path in audio_files]
    arrays = giantsteps_arrays(audio_files)
    predictions = dict(zip(track_ids, predict_tracks(model, arrays, device)))
    original = load_original(GIANTSTEPS_DIRECTORY / "annotations" / "key")
    gsplus, _ = load_gsplus(WORK_DIRECTORY / "gsplus-download" / "annotations" / "new_annotations")
    original_ids = [track for track in track_ids if track in original]
    gsplus_ids = [track for track in track_ids if track in gsplus]
    clean_ids = [track for track in gsplus_ids if gsplus[track]["clean"]]
    report["gs_original"] = scored(
        [predictions[track] for track in original_ids], [original[track] for track in original_ids]
    )
    report["gsplus_primary"] = scored(
        [predictions[track] for track in gsplus_ids],
        [gsplus[track]["primary"] for track in gsplus_ids],
    )
    report["gsplus_clean"] = scored(
        [predictions[track] for track in clean_ids],
        [gsplus[track]["primary"] for track in clean_ids],
    )
    return report


def tonic_phasor(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tonic = probabilities.view(-1, 12, 2).sum(-1)
    angles = OMEGA * torch.arange(12, device=probabilities.device)
    return (tonic * torch.cos(angles)).sum(-1), -(tonic * torch.sin(angles)).sum(-1)


def roll_classes(probabilities: torch.Tensor, tonic_steps: torch.Tensor) -> torch.Tensor:
    grid = probabilities.view(-1, 12, 2)
    positions = (torch.arange(12, device=grid.device).unsqueeze(0) - tonic_steps.unsqueeze(1)) % 12
    return torch.gather(grid, 1, positions.unsqueeze(-1).expand(-1, -1, 2)).reshape(-1, 24)


def equivariance_loss(
    logits_first: torch.Tensor,
    logits_second: torch.Tensor,
    shift_first: torch.Tensor,
    shift_second: torch.Tensor,
    spread_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """A view raised by `shift` semitones has its tonic lowered by `shift` classes.

    The spectral term is minimal only when both tonic distributions are one-hot and
    offset by the known shift difference, so it cannot be satisfied by a constant output.
    Undoing each view's shift gives the tonic of the unshifted track; requiring those to
    spread across a batch rules out predicting the shift itself from the spectral envelope.
    """
    probabilities_first = logits_first.softmax(-1)
    probabilities_second = logits_second.softmax(-1)
    real_first, imaginary_first = tonic_phasor(probabilities_first)
    real_second, imaginary_second = tonic_phasor(probabilities_second)
    cross_real = real_first * real_second + imaginary_first * imaginary_second
    cross_imaginary = imaginary_first * real_second - real_first * imaginary_second
    difference = (shift_first - shift_second).float()
    spectral = (
        0.5
        * (
            (cross_real - torch.cos(OMEGA * difference)) ** 2
            + (cross_imaginary - torch.sin(OMEGA * difference)) ** 2
        ).mean()
    )

    steps = shift_first - shift_second
    first_in_second_frame = roll_classes(probabilities_first, steps).detach()
    second_in_first_frame = roll_classes(probabilities_second, -steps).detach()
    consistency = 0.5 * (
        -(first_in_second_frame * logits_second.log_softmax(-1)).sum(-1).mean()
        - (second_in_first_frame * logits_first.log_softmax(-1)).sum(-1).mean()
    )

    mode_mean = (
        torch.cat([probabilities_first, probabilities_second]).view(-1, 12, 2).sum(1).mean(0)
    )
    mode_entropy = -(mode_mean * torch.log(mode_mean + 1e-8)).sum()

    canonical = torch.cat(
        [
            roll_classes(probabilities_first, shift_first),
            roll_classes(probabilities_second, shift_second),
        ]
    )
    canonical_tonic_mean = canonical.view(-1, 12, 2).sum(-1).mean(0)
    tonic_spread = -(canonical_tonic_mean * torch.log(canonical_tonic_mean + 1e-8)).sum()

    tonic_first = probabilities_first.view(-1, 12, 2).sum(-1).argmax(-1)
    tonic_second = probabilities_second.view(-1, 12, 2).sum(-1).argmax(-1)
    agreement = ((tonic_first + steps) % 12 == tonic_second).float().mean()
    total = spectral + 0.5 * consistency - mode_entropy - spread_weight * tonic_spread
    return total, {
        "spectral": float(spectral.detach()),
        "consistency": float(consistency.detach()),
        "mode_entropy": float(mode_entropy.detach()),
        "tonic_spread": float(tonic_spread.detach()),
        "shift_agreement": float(agreement),
    }


def best_relabeling(predictions: list[int], references: list[int]) -> tuple[int, int, float]:
    """Label-free training fixes classes only up to a tonic rotation and a mode swap."""
    best = (0, 0, -1.0)
    for rotation in range(12):
        for swap in (0, 1):
            mapped = [
                ((value // 2 + rotation) % 12) * 2 + ((value % 2) ^ swap) for value in predictions
            ]
            exact, _ = score_mirex(mapped, references)
            if exact > best[2]:
                best = (rotation, swap, exact)
    return best


def relabel_head(model: KeyCNN, rotation: int, swap: int) -> None:
    source = [0] * 24
    for old_class in range(24):
        new_class = ((old_class // 2 + rotation) % 12) * 2 + ((old_class % 2) ^ swap)
        source[new_class] = old_class
    index = torch.tensor(source, device=model.head.weight.device)
    with torch.no_grad():
        model.head.weight.copy_(model.head.weight[index].clone())
        model.head.bias.copy_(model.head.bias[index].clone())


def view(
    array: np.ndarray, shift: int, arguments: argparse.Namespace, generator: np.random.Generator
) -> np.ndarray:
    features = np.array(crop(array, shift, arguments.frames, generator), dtype=np.float32)
    if arguments.edge_mask:
        features = mask_edges(features, arguments.edge_mask, generator)
    return features


def pretrain(arguments: argparse.Namespace) -> None:
    device = arguments.device
    output = Path(arguments.out)
    output.mkdir(parents=True, exist_ok=True)
    panels = load_manifest()
    train_rows = panels["train"]
    arrays = manifest_arrays(train_rows)
    labelled_count = len(arrays)
    if arguments.unlabeled:
        extra_paths = sorted(Path(arguments.unlabeled).glob("*.npy"))
        arrays += load_arrays(extra_paths)
    print(
        f"pretrain tracks {len(arrays)} ({labelled_count} from the fitting pool, labels unused)",
        flush=True,
    )

    torch.manual_seed(arguments.seed)
    generator = np.random.default_rng(arguments.seed)
    model = KeyCNN(width=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps_per_epoch = len(arrays) // arguments.batch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, total_steps=arguments.epochs * steps_per_epoch, pct_start=0.05
    )
    history = []
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        started = time.time()
        order = generator.permutation(len(arrays))
        sums: dict[str, float] = {}
        for step in range(steps_per_epoch):
            batch = order[step * arguments.batch : (step + 1) * arguments.batch]
            shifts_first = generator.integers(-PAD_SEMI, PAD_SEMI + 1, len(batch))
            shifts_second = generator.integers(-PAD_SEMI, PAD_SEMI + 1, len(batch))
            first = np.stack(
                [
                    view(arrays[track_index], int(shift), arguments, generator)
                    for track_index, shift in zip(batch, shifts_first)
                ]
            )
            second = np.stack(
                [
                    view(arrays[track_index], int(shift), arguments, generator)
                    for track_index, shift in zip(batch, shifts_second)
                ]
            )
            logits = model(torch.from_numpy(np.concatenate([first, second])).to(device))
            loss, parts = equivariance_loss(
                logits[: len(batch)],
                logits[len(batch) :],
                torch.from_numpy(shifts_first).to(device),
                torch.from_numpy(shifts_second).to(device),
                arguments.spread_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            for name, value in parts.items():
                sums[name] = sums.get(name, 0.0) + value
        entry = {"epoch": epoch} | {
            name: round(value / steps_per_epoch, 4) for name, value in sums.items()
        }
        if epoch % arguments.probe_every == 0 or epoch == arguments.epochs:
            sample = list(range(0, labelled_count, 4))
            predictions = predict_tracks(
                model, [arrays[track_index] for track_index in sample], device
            )
            references = [int(train_rows[track_index]["key_label"]) for track_index in sample]
            rotation, swap, exact = best_relabeling(predictions, references)
            entry |= {
                "probe_rotation": rotation,
                "probe_swap": swap,
                "probe_exact": round(100 * exact, 2),
            }
        entry["seconds"] = round(time.time() - started, 1)
        history.append(entry)
        print(json.dumps(entry), flush=True)
        (output / "pretrain.json").write_text(
            json.dumps({"config": vars(arguments), "history": history}, indent=1)
        )

    predictions = predict_tracks(model, arrays[:labelled_count], device)
    references = [int(row["key_label"]) for row in train_rows]
    rotation, swap, exact = best_relabeling(predictions, references)
    relabel_head(model, rotation, swap)
    aligned = predict_tracks(model, arrays[:labelled_count], device)
    summary = {
        "rotation": rotation,
        "swap": swap,
        "fitting_pool_after_relabeling": scored(aligned, references),
        "panels_after_relabeling": final_panels(model, device, panels),
    }
    save_checkpoint(
        output / "pretrained-aligned.pt", model, width=1.0, variant="equivariance-pretrain"
    )
    (output / "pretrain.json").write_text(
        json.dumps({"config": vars(arguments), "history": history, "summary": summary}, indent=1)
    )
    print("SUMMARY", json.dumps(summary), flush=True)


def shift_consistency(
    model: nn.Module,
    unlabeled_arrays: list[np.ndarray],
    arguments: argparse.Namespace,
    generator: np.random.Generator,
    device: str,
) -> tuple[torch.Tensor, float]:
    """Two differently shifted crops of one unlabeled clip must agree on the key.

    The first view's confident prediction, moved by the known shift difference,
    is the target for the second view.
    """
    chosen = generator.integers(0, len(unlabeled_arrays), arguments.unlabeled_batch)
    shifts_first = generator.integers(-PAD_SEMI, PAD_SEMI + 1, len(chosen))
    shifts_second = generator.integers(-PAD_SEMI, PAD_SEMI + 1, len(chosen))
    frames = arguments.unlabeled_frames
    first = np.stack(
        [
            crop(unlabeled_arrays[clip_index], int(shift), frames, generator)
            for clip_index, shift in zip(chosen, shifts_first)
        ]
    )
    second = np.stack(
        [
            crop(unlabeled_arrays[clip_index], int(shift), frames, generator)
            for clip_index, shift in zip(chosen, shifts_second)
        ]
    )
    logits = model(torch.from_numpy(np.concatenate([first, second])).to(device))
    steps = torch.from_numpy(shifts_first - shifts_second).to(device)
    target = roll_classes(logits[: len(chosen)].softmax(-1), steps).detach()
    confidence, pseudo_label = target.max(-1)
    keep = (confidence >= arguments.confidence).float()
    per_clip = nn.functional.cross_entropy(logits[len(chosen) :], pseudo_label, reduction="none")
    return (per_clip * keep).mean(), float(keep.mean())


def shifted_targets(probabilities: np.ndarray, shifts: np.ndarray, device: str) -> torch.Tensor:
    """Move unshifted-track key probabilities into the class frame of each shifted crop."""
    return roll_classes(
        torch.from_numpy(np.ascontiguousarray(probabilities)).to(device),
        -torch.from_numpy(shifts).to(device),
    )


def unlabeled_distillation(
    model: nn.Module,
    unlabeled_arrays: list[np.ndarray],
    soft_unlabeled: np.ndarray,
    dj_indices: np.ndarray,
    arguments: argparse.Namespace,
    generator: np.random.Generator,
    device: str,
) -> torch.Tensor:
    chosen = generator.integers(0, len(unlabeled_arrays), arguments.unlabeled_batch)
    if len(dj_indices) and arguments.dj_fraction > 0:
        from_dj = generator.random(len(chosen)) < arguments.dj_fraction
        chosen[from_dj] = dj_indices[generator.integers(0, len(dj_indices), int(from_dj.sum()))]
    shifts = generator.integers(-arguments.max_shift, arguments.max_shift + 1, len(chosen))
    frames = arguments.unlabeled_frames
    features = np.stack(
        [
            crop(unlabeled_arrays[clip_index], int(shift), frames, generator)
            for clip_index, shift in zip(chosen, shifts)
        ]
    )
    logits = model(torch.from_numpy(features).to(device))
    teacher = shifted_targets(soft_unlabeled[chosen], shifts, device)
    return -(teacher * logits.log_softmax(-1)).sum(-1).mean()


def train(arguments: argparse.Namespace) -> None:
    device = arguments.device
    output = Path(arguments.out)
    output.mkdir(parents=True, exist_ok=True)
    panels = load_manifest()
    train_rows = panels["train"]
    arrays = manifest_arrays(train_rows)
    references = [int(row["key_label"]) for row in train_rows]
    hashes = [row["audio_sha256"] for row in train_rows]
    fit_indices, validation_indices = validation_split(references, arguments.seed, hashes)
    held_out_indices: list[int] = []
    if arguments.folds:
        fold_of = [int(digest[:8], 16) % arguments.folds for digest in hashes]
        held_out_indices = [
            track_index
            for track_index in range(len(train_rows))
            if fold_of[track_index] == arguments.fold
        ]
        fit_indices = np.array(
            [track_index for track_index in fit_indices if fold_of[track_index] != arguments.fold]
        )
        validation_indices = [
            track_index
            for track_index in validation_indices
            if fold_of[track_index] != arguments.fold
        ]
        print(
            f"fold {arguments.fold}/{arguments.folds}: held out {len(held_out_indices)}", flush=True
        )
    suspect = np.zeros(len(train_rows), dtype=bool)
    if arguments.suspect_labels:
        flagged = np.load(arguments.suspect_labels)
        flagged_ids = {
            str(track)
            for track, is_suspect in zip(flagged["track_ids"], flagged["suspect"], strict=True)
            if is_suspect
        }
        suspect = np.array([row["track_id"] in flagged_ids for row in train_rows])
        if arguments.suspect_mode == "drop":
            fit_indices = np.array(
                [track_index for track_index in fit_indices if not suspect[track_index]]
            )
        print(f"suspect labels {int(suspect.sum())}, mode {arguments.suspect_mode}", flush=True)
    print(
        f"fit {len(fit_indices)} internal validation {len(validation_indices)} init {arguments.init}",
        flush=True,
    )

    torch.manual_seed(arguments.seed)
    generator = np.random.default_rng(arguments.seed)
    if arguments.init:
        model, _ = load_checkpoint(arguments.init, device)
    else:
        model = KeyCNN(width=arguments.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, arguments.epochs)
    unlabeled_arrays: list[np.ndarray] = []
    dj_indices = np.zeros(0, dtype=np.int64)
    if arguments.pool:
        pool = np.load(f"{arguments.pool}.npy", mmap_mode="r")
        pool_index = np.load(f"{arguments.pool}-index.npz")
        unlabeled_arrays = [
            pool[:, offset : offset + length]
            for offset, length in zip(pool_index["offsets"], pool_index["lengths"], strict=True)
        ]
        dj_indices = np.flatnonzero(pool_index["dj"])
        print(f"pool items {len(unlabeled_arrays)}, DJ-genre items {len(dj_indices)}", flush=True)
    elif arguments.unlabeled:
        unlabeled_arrays = load_arrays(sorted(Path(arguments.unlabeled).glob("*.npy")))
        print(f"unlabeled clips {len(unlabeled_arrays)}", flush=True)
    soft_train = None
    soft_unlabeled = None
    if arguments.soft_labels:
        soft = np.load(arguments.soft_labels)
        assert [str(value) for value in soft["train_ids"]] == [
            row["track_id"] for row in train_rows
        ]
        soft_train = soft["train"]
        soft_unlabeled = soft["unlabeled"]
        assert len(soft_unlabeled) == len(unlabeled_arrays)
    best_path = output / "best.pt"
    best_mirex = -1.0
    best_epoch = 0
    history = []
    validation_arrays = [arrays[track_index] for track_index in validation_indices]
    validation_references = [references[track_index] for track_index in validation_indices]
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        started = time.time()
        order = generator.permutation(fit_indices)
        loss_sum = 0.0
        confident_sum = 0.0
        confident_steps = 0
        for start in range(0, len(order), arguments.batch):
            batch = order[start : start + arguments.batch]
            shifts = generator.integers(-arguments.max_shift, arguments.max_shift + 1, len(batch))
            features = np.stack(
                [
                    crop(arrays[track_index], int(shift), arguments.crop, generator)
                    for track_index, shift in zip(batch, shifts)
                ]
            )
            targets = [
                transpose_key(references[track_index], -int(shift))
                for track_index, shift in zip(batch, shifts)
            ]
            logits = model(torch.from_numpy(features).to(device))
            hard_losses = nn.functional.cross_entropy(
                logits, torch.tensor(targets, dtype=torch.int64, device=device), reduction="none"
            )
            loss = hard_losses.mean()
            if soft_train is not None:
                teacher = shifted_targets(soft_train[batch], shifts, device)
                distillation_losses = -(teacher * logits.log_softmax(-1)).sum(-1)
                hard_weights = torch.full((len(batch),), arguments.hard_weight, device=device)
                hard_weights[torch.from_numpy(suspect[batch]).to(device)] = 0.0
                loss = (
                    hard_weights * hard_losses + (1 - hard_weights) * distillation_losses
                ).mean()
                assert soft_unlabeled is not None
                loss = loss + arguments.unlabeled_weight * unlabeled_distillation(
                    model,
                    unlabeled_arrays,
                    soft_unlabeled,
                    dj_indices,
                    arguments,
                    generator,
                    device,
                )
            if unlabeled_arrays and soft_train is None:
                ramp = min(1.0, epoch / arguments.ramp_epochs)
                consistency, confident = shift_consistency(
                    model, unlabeled_arrays, arguments, generator, device
                )
                loss = loss + arguments.consistency_weight * ramp * consistency
                confident_sum += confident
                confident_steps += 1
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch)
        scheduler.step()
        exact, mirex = score_mirex(
            predict_tracks(model, validation_arrays, device), validation_references
        )
        if mirex > best_mirex:
            best_mirex = mirex
            best_epoch = epoch
            save_checkpoint(
                best_path, model, width=arguments.width, seed=arguments.seed, epoch=epoch
            )
        entry = {
            "epoch": epoch,
            "loss": round(loss_sum / len(order), 4),
            "validation_exact": round(100 * exact, 2),
            "validation_mirex": round(100 * mirex, 2),
            "confident_fraction": round(confident_sum / max(confident_steps, 1), 3),
            "seconds": round(time.time() - started, 1),
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
    save_checkpoint(
        output / "final.pt",
        model,
        width=arguments.width,
        seed=arguments.seed,
        epoch=arguments.epochs,
    )
    selected, _ = load_checkpoint(str(best_path), device)
    if held_out_indices:
        from analyze import shifted_logits

        held_logits = shifted_logits(
            selected, [arrays[track_index] for track_index in held_out_indices], device
        )
        np.savez_compressed(
            output / "held-out.npz",
            logits=held_logits,
            track_ids=np.array(
                [train_rows[track_index]["track_id"] for track_index in held_out_indices]
            ),
            references=np.array([references[track_index] for track_index in held_out_indices]),
        )
    result = {
        "config": vars(arguments),
        "best_epoch": best_epoch,
        "best_validation_mirex": round(100 * best_mirex, 2),
        "selected": final_panels(selected, device, panels),
        "final_epoch": final_panels(model, device, panels),
        "history": history,
    }
    (output / "result.json").write_text(json.dumps(result, indent=1))
    print(
        "RESULT",
        json.dumps({key: result[key] for key in ("best_epoch", "selected", "final_epoch")}),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    pretrain_parser = commands.add_parser("pretrain")
    pretrain_parser.add_argument("--unlabeled", default=None)
    pretrain_parser.add_argument("--frames", type=int, default=160)
    pretrain_parser.add_argument("--probe-every", type=int, default=5)
    pretrain_parser.add_argument("--batch", type=int, default=64)
    pretrain_parser.add_argument("--spread-weight", type=float, default=0.0)
    pretrain_parser.add_argument("--edge-mask", type=int, default=0)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--init", default=None)
    train_parser.add_argument("--batch", type=int, default=32)
    train_parser.add_argument("--unlabeled", default=None)
    train_parser.add_argument("--unlabeled-batch", type=int, default=64)
    train_parser.add_argument("--unlabeled-frames", type=int, default=160)
    train_parser.add_argument("--consistency-weight", type=float, default=1.0)
    train_parser.add_argument("--confidence", type=float, default=0.9)
    train_parser.add_argument("--ramp-epochs", type=int, default=20)
    train_parser.add_argument("--soft-labels", default=None)
    train_parser.add_argument("--hard-weight", type=float, default=0.5)
    train_parser.add_argument("--unlabeled-weight", type=float, default=1.0)
    train_parser.add_argument("--width", type=float, default=1.0)
    train_parser.add_argument("--crop", type=int, default=CROP_FRAMES)
    train_parser.add_argument("--max-shift", type=int, default=PAD_SEMI)
    train_parser.add_argument("--folds", type=int, default=0)
    train_parser.add_argument("--pool", default=None)
    train_parser.add_argument("--dj-fraction", type=float, default=0.0)
    train_parser.add_argument("--fold", type=int, default=0)
    train_parser.add_argument(
        "--suspect-labels", default=None, help="npz with track_ids and a suspect flag per track"
    )
    train_parser.add_argument("--suspect-mode", choices=("teacher", "drop"), default="teacher")
    for subparser in (pretrain_parser, train_parser):
        subparser.add_argument("--out", required=True)
        subparser.add_argument("--seed", type=int, required=True)
        subparser.add_argument("--epochs", type=int, required=True)
        subparser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    if arguments.command == "pretrain":
        pretrain(arguments)
    else:
        train(arguments)


if __name__ == "__main__":
    main()
