"""Conditional equal-shape encoder batching; original frontend remains batch one."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from experiments.musicfm_key import SongLogits
from experiments.musicfm_teacher import KeyTeacher


def batched_panel(
    teacher: KeyTeacher,
    ids: list[str],
    rows: dict[str, dict[str, Any]],
    read: Callable[[dict[str, Any], int, int], torch.Tensor],
    watch: Callable[[], None],
    *,
    batch_size: int,
    collect_predictions: bool,
    collect_logits: bool = False,
) -> dict[str, Any]:
    if batch_size != 16:
        raise ValueError("Only prequalified encoder batch16 may use this path")
    began = time.monotonic()
    device = teacher.head.weight.device
    teacher.eval()
    accumulators = {identity: SongLogits(rows[identity]["samples"]) for identity in ids}
    next_index = {identity: 0 for identity in ids}
    pending = {identity: {} for identity in ids}
    groups: dict[tuple[int, ...], list[tuple[str, int, torch.Tensor]]] = {}

    def flush(key: tuple[int, ...]) -> None:
        group = groups.pop(key)
        features = torch.cat([value for _, _, value in group], dim=0)
        _, hidden = getattr(teacher.backbone, "encoder")(features)
        logits = teacher.head(hidden[12].mean(dim=1))
        if (
            logits.shape != (len(group), 24)
            or logits.dtype != torch.float32
            or not bool(torch.isfinite(logits).all())
        ):
            raise ValueError("Nonfinite/wrong batched FP32 logits")
        values = logits.cpu().numpy()
        for (identity, index, _), value in zip(group, values, strict=True):
            pending[identity][index] = value.copy()
            accumulator = accumulators[identity]
            while next_index[identity] in pending[identity]:
                index = next_index[identity]
                accumulator.add(accumulator.windows[index], pending[identity].pop(index))
                next_index[identity] += 1
        watch()

    with torch.inference_mode(), torch.autocast(device.type, enabled=False):
        for identity in ids:
            for index, window in enumerate(accumulators[identity].windows):
                watch()
                wave = read(rows[identity], window.start, window.stop - window.start).to(
                    device=device
                )
                # Exact measured554aa50 policy: individual preprocessing+normalization, then cat.
                feature = getattr(teacher.backbone, "normalize")(
                    getattr(teacher.backbone, "preprocessing")(wave, ["melspec_2048"])
                )["melspec_2048"]
                if feature.shape[0] != 1 or feature.dtype != torch.float32:
                    raise ValueError("Original batch1 FP32 frontend required")
                key = tuple(feature.shape[1:])
                groups.setdefault(key, []).append((identity, index, feature))
                if len(groups[key]) == batch_size:
                    flush(key)
        for key in list(groups):
            flush(key)
    logits = [accumulators[identity].finish().tolist() for identity in ids]
    return {
        "tracks": len(ids),
        "windows": sum(len(a.windows) for a in accumulators.values()),
        "elapsed_seconds": time.monotonic() - began,
        "per_song": [
            {"id": i, "samples": rows[i]["samples"], "windows": len(accumulators[i].windows)}
            for i in ids
        ],
        **(
            {"predictions": [int(np.argmax(value)) for value in logits]}
            if collect_predictions
            else {}
        ),
        **({"logits": logits} if collect_logits else {}),
    }


def compare_panels(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if (
        reference["tracks"] != 376
        or candidate["tracks"] != 376
        or reference["windows"] != candidate["windows"]
    ):
        raise ValueError("Complete376-song composed proof required")
    if [(r["id"], r["samples"], r["windows"]) for r in reference["per_song"]] != [
        (r["id"], r["samples"], r["windows"]) for r in candidate["per_song"]
    ]:
        raise ValueError("Composed proof panel order/context differs")
    if reference.get("domain_schedule") != candidate.get("domain_schedule"):
        raise ValueError("Composed proof domain batching schedule differs")
    left, right = np.asarray(reference["logits"]), np.asarray(candidate["logits"])
    if (
        left.shape != (376, 24)
        or right.shape != left.shape
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("Invalid complete-song FP64 logits")
    error = float(np.max(np.abs(left - right)))
    disagreements = int(np.count_nonzero(left.argmax(1) != right.argmax(1)))
    return {
        "tracks": 376,
        "windows": reference["windows"],
        "max_absolute_logit_error": error,
        "argmax_disagreements": disagreements,
        "passed": error <= 1e-5 and disagreements == 0,
        "domain_schedule": reference.get("domain_schedule"),
        "per_song": reference["per_song"],
        "reference_logits": left.tolist(),
        "candidate_logits": right.tolist(),
        "label_blind": True,
        "task_scores_computed": False,
    }
