"""Metadata-only barrier for the one pinned teacher run, before any BN worker.

Callers must separately pin/read these original files and hash the selected
checkpoint. This verifies their relationships and original process retirement;
it is not an execution permit or a declaration that the teacher earned its gate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from experiments.k10_v2_pilot import selection_key

ORIGINAL_CONFIG_SHA = "069d1afc396c65fcaa1d7cb993db7d4c930aec93eae377a0a611842dfcbbac28"
ORIGINAL_PROCESSES = {1294: 174372685, 1307: 174372693}
FILES = frozenset({"config", "result", "controller", "success", "launch"})


@dataclass(frozen=True)
class TerminalTeacher:
    checkpoint_sha256: str
    result_sha256: str
    best_update: int
    original_gate_earned: bool
    historical_b1_seconds: float
    historical_b16_seconds: float

    @property
    def complete_evaluation_seconds(self) -> float:
        # Candidate B1/B16 proof plus one fresh untouched-baseline B1 replay.
        return 2 * self.historical_b1_seconds + self.historical_b16_seconds


def _seconds(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Missing finite positive comparable evaluation timing")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Missing finite positive comparable evaluation timing")
    return float(value)


def verify_terminal_teacher(
    files: Mapping[str, bytes],
    selected_checkpoint_sha256: str,
    live_start: Callable[[int], int | None],
    *,
    failure_receipts_absent: bool,
) -> TerminalTeacher:
    """Reject interim, mismatched, unaudited or still-live original teacher state.

    Timing admission consumes recorded timings, not original saved logit values.
    The future diagnostic must still verify its actual runtime and fresh baseline
    replay, source/crop hashes, all resource bounds and a separately sealed permit.
    """
    if set(files) != FILES:
        raise ValueError("Require all five original terminal metadata documents")
    if failure_receipts_absent is not True:
        raise ValueError("Original guardian/worker failure receipt prevents diagnostic admission")
    if hashlib.sha256(files["config"]).hexdigest() != ORIGINAL_CONFIG_SHA:
        raise ValueError("Different original teacher configuration")
    config, result, controller, success, launch = (
        json.loads(files[name]) for name in ("config", "result", "controller", "success", "launch")
    )
    result_sha = hashlib.sha256(files["result"]).hexdigest()
    if (
        config["mode"] != "fit"
        or config["eval_batch_size"] != 16
        or result["status"] != "COMPLETE_FIXED_PILOT_NOT_RELEASE_EVIDENCE"
        or result["updates"] != 8280
        or result["events"] != 120
        or len(result["history"]) != 120
        or [row["update"] for row in result["history"]] != list(range(69, 8281, 69))
        or result["provenance"]["config_sha256"] != ORIGINAL_CONFIG_SHA
        or result["provenance"]["source_sha256"] != config["source_sha256"]
        or result["confirmation_used"] is not False
        or result["calibration_used"] is not False
    ):
        raise ValueError("Original fixed teacher horizon/source was not completed")
    if (
        controller["state"] != "COMPLETE_NO_RAM_OWNERSHIP"
        or controller["worker_started"] is not True
        or controller["worker_exit"] != 0
        or success["mode"] != "fit"
        or success["result_sha256"] != result_sha
        or success["training_approved"] is not False
        or launch["config_sha256"] != ORIGINAL_CONFIG_SHA
        or launch["pid"] != 1307
        or launch["guardian_start_ticks"] != ORIGINAL_PROCESSES[1294]
        or any(live_start(pid) == start for pid, start in ORIGINAL_PROCESSES.items())
    ):
        raise ValueError("Original completion/ownership receipts or process retirement differ")
    caps = config["caps"]
    if (
        _seconds(controller["wall_seconds"]) > caps["wall_seconds"]
        or _seconds(controller["peaks"]["rss"]) > caps["rss_bytes"]
        or _seconds(controller["peaks"]["cpu"]) > caps["cpu_seconds"]
        or _seconds(controller["terminal_resources"]["rss_bytes"]) > caps["rss_bytes"]
        or _seconds(controller["terminal_resources"]["cpu_seconds"]) > caps["cpu_seconds"]
    ):
        raise ValueError("Original recorded resource evidence exceeded its fixed caps")
    best = max(result["history"], key=lambda row: selection_key(row["metrics"]))
    if (
        best["update"] != result["best_update"]
        or best["metrics"] != result["best_metrics"]
        or selected_checkpoint_sha256 != result["best_checkpoint_sha256"]
    ):
        raise ValueError("Original selected checkpoint/rule differs")
    gate = result["continuation_gate"]
    if (
        type(gate["continuation_earned"]) is not bool
        or gate["confirmation_scored"] is not False
        or gate["release_or_distillation_authorized"] is not False
        or set(gate["checks_against_each_control"]) != {"restricted", "expanded"}
    ):
        raise ValueError(
            "Original gate is missing, changed or already authorized a different action"
        )
    proof = result["selected_teacher_composed_batch_proof"]
    ids = [
        identity
        for domain in ("beatport", "fmak")
        for identity in result["best_metrics"][domain]["ids"]
    ]
    if (
        proof["passed"] is not True
        or proof["tracks"] != 376
        or proof["argmax_disagreements"] != 0
        or type(proof["max_absolute_logit_error"]) not in (float, int)
        or not 0 <= proof["max_absolute_logit_error"] <= 1e-5
        or len(ids) != 376
        or len(set(ids)) != 376
        or [row["id"] for row in proof["per_song"]] != ids
    ):
        raise ValueError("Original selected-checkpoint B1/B16 audit did not pass")
    b1 = sum(_seconds(row["seconds"]) for row in proof["per_song"])
    b16 = max(
        sum(
            _seconds(row["validation"][domain]["elapsed_seconds"])
            for domain in ("beatport", "fmak")
        )
        for row in result["history"]
    )
    return TerminalTeacher(
        selected_checkpoint_sha256,
        result_sha,
        result["best_update"],
        gate["continuation_earned"],
        b1,
        b16,
    )
