"""Score checkpoints singly, with pitch-shift averaging and as an ensemble.

For every checkpoint and every panel the logits of all nine shifted copies are
stored, each moved back to the unshifted class frame, so later analyses need no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from environment import required_path

sys.path.insert(0, str(required_path("DRONE_SOURCE")))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from frontier_train import (  # noqa: E402
    GIANTSTEPS_DIRECTORY,
    WORK_DIRECTORY,
    core_slice,
    giantsteps_arrays,
    load_manifest,
    manifest_arrays,
)
from rescore_gsplus import load_gsplus, load_original  # noqa: E402

from drone.checkpoints import load_checkpoint  # noqa: E402
from drone.features import PAD_SEMI  # noqa: E402
from drone.keys import score_breakdown, score_mirex  # noqa: E402

SHIFTS = list(range(-PAD_SEMI, PAD_SEMI + 1))


def shifted_logits(model: torch.nn.Module, arrays: list[np.ndarray], device: str) -> np.ndarray:
    """Return (tracks, shifts, 24) logits in the unshifted class frame."""
    model.eval()
    output = np.zeros((len(arrays), len(SHIFTS), 24), dtype=np.float32)
    with torch.inference_mode():
        for track_index, array in enumerate(arrays):
            batch = np.stack([core_slice(array, shift) for shift in SHIFTS]).astype(np.float32)
            logits = model(torch.from_numpy(batch).to(device)).float().cpu().numpy()
            for shift_index, shift in enumerate(SHIFTS):
                output[track_index, shift_index] = np.roll(
                    logits[shift_index].reshape(12, 2), shift, axis=0
                ).reshape(24)
    return output


def load_panels() -> dict[str, tuple[list[np.ndarray], list[int], list[str]]]:
    panels = load_manifest()
    loaded = {}
    for name in ("valid", "holdout"):
        rows = panels[name]
        loaded[name] = (
            manifest_arrays(rows),
            [int(row["key_label"]) for row in rows],
            [row["track_id"] for row in rows],
        )
    audio_files = sorted(
        path
        for path in (GIANTSTEPS_DIRECTORY / "audio").glob("*.mp3")
        if not path.name.startswith("._")
    )
    track_ids = [path.name.split(".")[0] for path in audio_files]
    arrays = giantsteps_arrays(audio_files)
    original = load_original(GIANTSTEPS_DIRECTORY / "annotations" / "key")
    gsplus, _ = load_gsplus(WORK_DIRECTORY / "gsplus-download" / "annotations" / "new_annotations")
    by_id = dict(zip(track_ids, arrays))
    original_ids = [track for track in track_ids if track in original]
    gsplus_ids = [track for track in track_ids if track in gsplus]
    clean_ids = [track for track in gsplus_ids if gsplus[track]["clean"]]
    loaded["gs_original"] = (
        [by_id[track] for track in original_ids],
        [original[track] for track in original_ids],
        original_ids,
    )
    loaded["gsplus_primary"] = (
        [by_id[track] for track in gsplus_ids],
        [gsplus[track]["primary"] for track in gsplus_ids],
        gsplus_ids,
    )
    loaded["gsplus_clean"] = (
        [by_id[track] for track in clean_ids],
        [gsplus[track]["primary"] for track in clean_ids],
        clean_ids,
    )
    return loaded


def summary(logits: np.ndarray, references: list[int]) -> dict:
    predictions = [int(value) for value in logits.argmax(-1)]
    exact, mirex = score_mirex(predictions, references)
    return {
        "exact": round(100 * exact, 2),
        "mirex": round(100 * mirex, 2),
        "breakdown": dict(score_breakdown(predictions, references)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group", action="append", required=True, help="name=checkpoint1,checkpoint2,..."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    panels = load_panels()
    zero_shift = SHIFTS.index(0)
    report: dict = {}
    stored: dict[str, np.ndarray] = {}
    for specification in arguments.group:
        name, checkpoint_list = specification.split("=", 1)
        checkpoints = checkpoint_list.split(",")
        report[name] = {}
        per_panel = {}
        for panel, (arrays, references, _) in panels.items():
            log_probabilities = []
            for checkpoint in checkpoints:
                model, _ = load_checkpoint(checkpoint, arguments.device)
                logits = shifted_logits(model, arrays, arguments.device)
                shifted = torch.from_numpy(logits).log_softmax(-1).numpy()
                log_probabilities.append(shifted)
            stack = np.stack(log_probabilities)
            stored[f"{name}/{panel}"] = stack
            singles = [summary(member[:, zero_shift], references) for member in stack]
            averaged = [summary(member.mean(1), references) for member in stack]
            per_panel[panel] = {
                "single_exact_mean": round(
                    float(np.mean([entry["exact"] for entry in singles])), 2
                ),
                "single_mirex_mean": round(
                    float(np.mean([entry["mirex"] for entry in singles])), 2
                ),
                "single_exact_each": [entry["exact"] for entry in singles],
                "shift_average_exact_mean": round(
                    float(np.mean([entry["exact"] for entry in averaged])), 2
                ),
                "shift_average_mirex_mean": round(
                    float(np.mean([entry["mirex"] for entry in averaged])), 2
                ),
                "ensemble": summary(stack[:, :, zero_shift].mean(0), references),
                "ensemble_shift_average": summary(stack.mean((0, 2)), references),
            }
        report[name] = per_panel
        print(name, json.dumps(per_panel), flush=True)
    Path(arguments.out).write_text(json.dumps(report, indent=1))
    np.savez_compressed(Path(arguments.out).with_suffix(".npz"), **stored)  # ty: ignore[invalid-argument-type]


if __name__ == "__main__":
    main()
