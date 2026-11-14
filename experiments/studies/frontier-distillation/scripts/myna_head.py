"""Train a KeyMyna-style key head on frozen Myna window embeddings and score every panel.

Training draws windows from every cached playback-rate shift of each fitting
track, with the key moved up by the shift. A track's prediction averages the
window probabilities; scoring keeps every shift, rotated back to the unshifted
key, so shift averaging can be done afterwards. Writes an npz whose keys are
"<group>/<panel>", each (seeds, tracks, shifts, 24) log-probabilities, in the
track order of the frontier study's analysis panels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from environment import required_path
from frontier_train import GIANTSTEPS_DIRECTORY, WORK_DIRECTORY, load_manifest
from rescore_gsplus import load_gsplus, load_original
from torch import nn

SHIFTS = list(range(-6, 7))
EMBEDDING_SIZE = 384
KEY_COUNT = 24
CACHE_DIRECTORIES = [required_path("DRONE_MYNA_CACHE"), WORK_DIRECTORY / "myna-embeddings"]


def cache_name(source: str, track_id: str) -> str:
    if source == "fmakv2":
        return f"fmak-{int(track_id):06d}"
    if source == "giantsteps":
        return f"beatport-{track_id.split(':')[-1]}"
    return f"giantsteps-{track_id}"


def find_view(name: str, shift: int) -> Path | None:
    for directory in CACHE_DIRECTORIES:
        path = directory / f"{name}-shift{shift:+d}.npy"
        if path.exists():
            return path
    return None


def load_views(name: str) -> dict[int, np.ndarray]:
    views = {}
    for shift in SHIFTS:
        path = find_view(name, shift)
        if path is not None:
            array = np.load(path)
            if len(array):
                views[shift] = array
    return views


def transpose(key_class: int, semitones: int) -> int:
    return ((key_class // 2 + semitones) % 12) * 2 + key_class % 2


def rotate_back(probabilities: np.ndarray, shift: int) -> np.ndarray:
    """Move probabilities predicted on audio shifted up by `shift` back to the unshifted key."""
    return np.roll(probabilities.reshape(12, 2), -shift, axis=0).reshape(KEY_COUNT)


def is_internal_validation(track_id: str) -> bool:
    return hashlib.sha256(f"myna-head-validation:{track_id}".encode()).digest()[0] < 26


class KeyHead(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, KEY_COUNT),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.layers(embeddings)


def evaluation_panels() -> dict[str, tuple[list[str], list[int]]]:
    panels = load_manifest()
    loaded = {}
    for name in ("valid", "holdout"):
        rows = panels[name]
        loaded[name] = (
            [cache_name("fmakv2", row["track_id"]) for row in rows],
            [int(row["key_label"]) for row in rows],
        )
    audio_files = sorted(
        path
        for path in (GIANTSTEPS_DIRECTORY / "audio").glob("*.mp3")
        if not path.name.startswith("._")
    )
    track_ids = [path.name.split(".")[0] for path in audio_files]
    original = load_original(GIANTSTEPS_DIRECTORY / "annotations" / "key")
    gsplus, _ = load_gsplus(WORK_DIRECTORY / "gsplus-download" / "annotations" / "new_annotations")
    original_ids = [track for track in track_ids if track in original]
    gsplus_ids = [track for track in track_ids if track in gsplus]
    clean_ids = [track for track in gsplus_ids if gsplus[track]["clean"]]
    loaded["gs_original"] = (
        [f"giantsteps-{track}" for track in original_ids],
        [original[track] for track in original_ids],
    )
    loaded["gsplus_primary"] = (
        [f"giantsteps-{track}" for track in gsplus_ids],
        [gsplus[track]["primary"] for track in gsplus_ids],
    )
    loaded["gsplus_clean"] = (
        [f"giantsteps-{track}" for track in clean_ids],
        [gsplus[track]["primary"] for track in clean_ids],
    )
    return loaded


def track_log_probabilities(
    model: nn.Module, views: dict[int, np.ndarray], device: str
) -> np.ndarray:
    """(shifts, 24) log-probabilities in the unshifted key frame; uniform where a view is missing."""
    output = np.full((len(SHIFTS), KEY_COUNT), np.log(1 / KEY_COUNT), dtype=np.float32)
    with torch.inference_mode():
        for shift_index, shift in enumerate(SHIFTS):
            if shift not in views:
                continue
            probabilities = (
                model(torch.from_numpy(views[shift]).to(device)).softmax(-1).mean(0).cpu().numpy()
            )
            output[shift_index] = np.log(rotate_back(probabilities, shift) + 1e-9)
    return output


def accuracy(
    model: nn.Module, tracks: list[tuple[dict[int, np.ndarray], int]], device: str
) -> float:
    model.eval()
    correct = 0
    for views, label in tracks:
        log_probabilities = track_log_probabilities(
            model, {0: views[0]} if 0 in views else views, device
        )
        correct += int(log_probabilities[SHIFTS.index(0)].argmax() == label)
    return correct / max(len(tracks), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 15, 42])
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--group", default="myna")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    train_rows = load_manifest()["train"]
    with ThreadPoolExecutor(max_workers=32) as executor:
        train_views = list(
            executor.map(
                lambda row: load_views(cache_name(row["source"], row["track_id"])), train_rows
            )
        )
    fitting_embeddings = []
    fitting_labels = []
    validation_tracks = []
    for row, views in zip(train_rows, train_views, strict=True):
        if not views:
            continue
        label = int(row["key_label"])
        if is_internal_validation(row["track_id"]):
            validation_tracks.append((views, label))
            continue
        for shift, array in views.items():
            fitting_embeddings.append(array)
            fitting_labels.append(np.full(len(array), transpose(label, shift), dtype=np.int64))
    embeddings = torch.from_numpy(np.concatenate(fitting_embeddings))
    labels = torch.from_numpy(np.concatenate(fitting_labels))
    fitting_tracks = sum(
        1
        for row, views in zip(train_rows, train_views)
        if views and not is_internal_validation(row["track_id"])
    )
    print(
        f"fitting tracks {fitting_tracks}, windows {len(labels)}, internal validation tracks {len(validation_tracks)}",
        flush=True,
    )
    panels = evaluation_panels()
    panel_views = {}
    with ThreadPoolExecutor(max_workers=32) as executor:
        for panel, (names, _) in panels.items():
            panel_views[panel] = list(executor.map(load_views, names))
    for panel, views in panel_views.items():
        print(
            f"{panel}: {sum(1 for view in views if 0 in view)}/{len(views)} tracks with the unshifted view",
            flush=True,
        )
    stored: dict[str, list[np.ndarray]] = {panel: [] for panel in panels}
    report = {}
    for seed in arguments.seeds:
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        model = KeyHead(arguments.hidden, arguments.dropout).to(arguments.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=arguments.learning_rate, weight_decay=1e-2
        )
        criterion = nn.CrossEntropyLoss()
        best_accuracy = -1.0
        best_state = None
        for epoch in range(1, arguments.epochs + 1):
            model.train()
            order = torch.randperm(len(labels), generator=generator)
            for start in range(0, len(order), arguments.batch):
                batch = order[start : start + arguments.batch]
                logits = model(embeddings[batch].to(arguments.device))
                loss = criterion(logits, labels[batch].to(arguments.device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            validation_accuracy = accuracy(model, validation_tracks, arguments.device)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "loss": round(float(loss), 4),
                        "internal_validation_exact": round(100 * validation_accuracy, 2),
                    }
                ),
                flush=True,
            )
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_state = {
                    name: tensor.detach().clone() for name, tensor in model.state_dict().items()
                }
        assert best_state is not None, "the first epoch always sets the best state"
        model.load_state_dict(best_state)
        model.eval()
        report[seed] = {}
        for panel, (names, references) in panels.items():
            stacked = np.stack(
                [
                    track_log_probabilities(model, views, arguments.device)
                    for views in panel_views[panel]
                ]
            )
            stored[panel].append(stacked)
            single = 100 * float(
                (stacked[:, SHIFTS.index(0)].argmax(-1) == np.array(references)).mean()
            )
            averaged = 100 * float((stacked.mean(1).argmax(-1) == np.array(references)).mean())
            report[seed][panel] = {
                "single_exact": round(single, 2),
                "shift_average_exact": round(averaged, 2),
            }
        print(
            json.dumps(
                {
                    "seed": seed,
                    "best_internal_validation_exact": round(100 * best_accuracy, 2),
                    "panels": report[seed],
                }
            ),
            flush=True,
        )
    np.savez_compressed(
        arguments.out,
        **{f"{arguments.group}/{panel}": np.stack(arrays) for panel, arrays in stored.items()},  # ty: ignore[invalid-argument-type]
    )
    Path(arguments.out).with_suffix(".json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
