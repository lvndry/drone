"""Private, guarded BN diagnostic worker. Never trains or promotes a model.

An execution needs a terminal original teacher, sealed source/config/UTC permit,
actual-source qualification and timing admission. There is no automatic launch.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from experiments.musicfm_bn import Clip
from experiments.musicfm_bn_calibration import calibrate_statistics
from experiments.musicfm_bn_panel import metadata_bytes, panel_rows
from experiments.musicfm_bn_qualification import qualify, timing_admission
from experiments.musicfm_bn_state import Watch, snapshot_state, verify_state_change
from experiments.musicfm_bn_terminal import (
    ORIGINAL_CONFIG_SHA,
    TerminalTeacher,
    verify_terminal_teacher,
)
from experiments.musicfm_eval import compare_panels
from experiments.musicfm_inputs import CachedProvider, apply_identity_overlay, merge_disk_cache
from experiments.musicfm_pilot import (
    checkpoint,
    evaluate_selection_schedule,
    hashed,
    load_teacher,
    metrics,
    validate_inputs,
    verify_cache,
)
from experiments.musicfm_teacher import KeyTeacher
from tools.musicfm_crop_io import file_identity, save, sha
from tools.run_musicfm_bn import (
    ORIGINAL,
    live_start,
    original_failures_absent,
    read_original,
    validate_launch,
    verify_import_paths,
)
from tools.run_musicfm_teacher import worker_watch


def useful_signal(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return candidate["fmak"]["exact_correct"] >= baseline["fmak"]["exact_correct"] + 5 and all(
        candidate["beatport"][key] >= baseline["beatport"][key]
        for key in ("exact_correct", "symmetric_units10", "asymmetric_units10")
    )


def bound_json(path: Path, digest: str, watch: Watch) -> dict[str, Any]:
    """Parse the exact verified metadata bytes; never reopen after hashing."""
    watch()
    with path.open("rb") as stream:
        raw = stream.read(64 * 1024**2 + 1)
    watch()
    if len(raw) > 64 * 1024**2 or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Bound metadata bytes changed: " + str(path))
    decoded = gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw
    if len(decoded) > 64 * 1024**2:
        raise ValueError("Decoded metadata exceeds its fixed bound")
    result = json.loads(decoded)
    if not isinstance(result, dict):
        raise ValueError("Expected a metadata object")
    return result


def score_admission(
    remaining: float, historical: float, overhead: float
) -> dict[str, bool | float]:
    """No remaining calibration: reserve the full fresh comparison plus safety."""
    if (
        any(
            isinstance(x, bool) or not math.isfinite(x) or x < 0
            for x in (remaining, historical, overhead)
        )
        or historical == 0
    ):
        raise ValueError("Finite comparable timing and nonnegative remaining budget required")
    required = 2 * historical + overhead + 120
    return dict(
        fits_budget=required <= remaining, required_seconds=required, remaining_seconds=remaining
    )


def selected_state(
    teacher: KeyTeacher, path: Path, terminal: TerminalTeacher, watch: Watch
) -> None:
    before = file_identity(path)
    if path.is_symlink() or hashed(path, watch) != terminal.checkpoint_sha256:
        raise ValueError("Original selected checkpoint bytes changed")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (
        payload["provenance"]["selected_update"] != terminal.best_update
        or payload["provenance"]["config_sha256"] != ORIGINAL_CONFIG_SHA
    ):
        raise ValueError("Selected state has a different update identity")
    teacher.load_state_dict(payload["state_dict"], strict=True)
    teacher.eval()
    del payload
    if file_identity(path) != before:
        raise ValueError("Original selected checkpoint changed during loading")
    watch()


def worker(config_path: Path, approval_path: Path, output: Path, guardian_pid: int) -> None:
    config_path, approval_path, output = (
        config_path.resolve(),
        approval_path.resolve(),
        output.absolute(),
    )
    raw = config_path.read_bytes()
    config, approval = json.loads(raw), json.loads(approval_path.read_bytes())
    config_sha = hashlib.sha256(raw).hexdigest()
    validate_launch(config, approval, config_sha)
    verify_import_paths(config["source_sha256"])
    if os.getppid() != guardian_pid or output != Path(config["output_root"]) / "worker":
        raise ValueError("Private worker requires its sealed output and actual live parent")
    original_files = read_original(config)
    terminal = verify_terminal_teacher(
        original_files,
        config["selected_checkpoint_sha256"],
        live_start,
        failure_receipts_absent=original_failures_absent(),
    )
    original = json.loads(original_files["config"])
    original_result = json.loads(original_files["result"])
    runtime = dict(original)
    for key in (
        "deadline_epoch",
        "caps",
        "protected_starts",
        "allocator_bytes",
        "torch_threads",
        "cpus",
    ):
        runtime[key] = config[key]
    runtime["_config_sha256"] = config_sha
    watch, watch_statistics = worker_watch(runtime, output.parent, guardian_pid)
    watch()
    output.mkdir(exist_ok=False)
    began = time.monotonic()
    phase = "INPUTS"
    provider: CachedProvider | None = None
    complete = False
    try:
        if original["input_provider"] != "disk":
            raise ValueError("Only the original immutable disk cache is admitted")
        for name in (
            "plan",
            "labels",
            "cache_manifest",
            "identity_overlay",
            "identity_receipt",
            "ram_config",
            "disk_cache_manifest",
        ):
            if hashed(Path(original[name]), watch) != original[name + "_sha256"]:
                raise ValueError("Original input receipt changed: " + name)
        root = Path(__file__).resolve().parents[1]
        for name, digest in original["evidence_sha256"].items():
            if hashed(root / name, watch) != digest:
                raise ValueError("Original upstream/configuration/statistics evidence changed")
        plan = bound_json(Path(original["plan"]), original["plan_sha256"], watch)
        disk = bound_json(
            Path(original["disk_cache_manifest"]), original["disk_cache_manifest_sha256"], watch
        )
        if (
            disk.get("complete") is not True
            or disk.get("provider") != "readonly-disk-full-f32"
            or disk["canonical_manifest_sha256"] != original["cache_manifest_sha256"]
            or disk["physical_overlay_sha256"] != original["identity_overlay_sha256"]
        ):
            raise ValueError("Different canonical disk-input binding")
        rows = {row["id"]: row for row in disk["rows"]}
        if len(rows) != len(disk["rows"]):
            raise ValueError("Duplicate disk-input IDs")
        projection = panel_rows(plan, rows)  # Does not inspect key/target fields.
        clips = tuple(
            Clip(
                row["domain"],
                row["id"],
                row["samples"],
                row["start"],
                row["length"],
                Fraction(row["weight_numerator"], row["weight_denominator"]),
            )
            for row in projection
        )
        verify_cache({clip.identity: rows[clip.identity] for clip in clips}, watch)
        provider = CachedProvider(rows, watch)
        crop_hashes = {}
        for clip, row in zip(clips, projection, strict=True):
            values = provider.read(clip.identity, clip.start, clip.length)
            digest = hashlib.sha256(values.tobytes()).hexdigest()
            crop_hashes[clip.identity] = digest
            row["crop_sha256"] = digest
        panel_manifest = dict(
            config_sha256=config_sha,
            original_plan_sha256=original["plan_sha256"],
            disk_cache_manifest_sha256=original["disk_cache_manifest_sha256"],
            identity_overlay_sha256=original["identity_overlay_sha256"],
            projection_sha256=hashlib.sha256(metadata_bytes(plan, rows)).hexdigest(),
            protocol_sha256=sha(root / "experiments/studies/musicfm-key-pilot/BN_PROTOCOL.md"),
            rows=projection,
        )
        save(output / "sealed-fitting-panel.json", panel_manifest)

        def read(clip: Clip) -> Tensor:
            assert provider is not None
            values = provider.read(clip.identity, clip.start, clip.length)
            if hashlib.sha256(values.tobytes()).hexdigest() != crop_hashes[clip.identity]:
                raise ValueError("Read differs from the sealed canonical fitting crop")
            return torch.from_numpy(values).unsqueeze(0)

        phase = "MODEL_LOAD"
        selected = ORIGINAL / "run-v1/worker/best.pt"
        if selected.is_symlink() or hashed(selected, watch) != terminal.checkpoint_sha256:
            raise ValueError("Original selected checkpoint hash differs before CUDA work")
        if torch.cuda.device_count() != 1 or uuid.UUID(
            str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
        ) != uuid.UUID(config["gpu_uuid"].removeprefix("GPU-")):
            raise ValueError("Unexpected physical CUDA device")
        teacher = load_teacher(runtime, output, watch)
        selected_state(teacher, selected, terminal, watch)
        verify_import_paths(config["source_sha256"])
        checkpoint(
            output / "qualification-original.pt",
            teacher,
            {
                "purpose": "serialization-cost-only",
                "source_checkpoint_sha256": terminal.checkpoint_sha256,
            },
            watch,
        )
        overhead = time.monotonic() - began
        phase = "QUALIFICATION"
        qualification = qualify(teacher, clips, read, watch)
        save(output / "qualification.json", asdict(qualification))
        admission = timing_admission(
            qualification,
            remaining_wall_seconds=config["deadline_epoch"] - time.time(),
            historical_complete_seconds=terminal.complete_evaluation_seconds,
            measured_overhead_seconds=overhead,
        )
        save(output / "admission.json", asdict(admission))
        if not admission.fits_budget:
            save(
                output / "result.json",
                dict(
                    status="BN_DIAGNOSTIC_BUDGET_INCONCLUSIVE_NO_SCORING",
                    promotion_authorized=False,
                    training_performed=False,
                    selection_scored=False,
                    qualification=asdict(qualification),
                    admission=asdict(admission),
                ),
            )
            complete = True
            return
        phase = "CALIBRATION"
        selected_state(teacher, selected, terminal, watch)
        calibrated = calibrate_statistics(teacher, clips, read, watch)
        if calibrated.prefix_forwards != 9216 or dict(calibrated.crop_sha256) != crop_hashes:
            raise ValueError("Incomplete calibration coverage or different sealed crops")
        save(
            output / "moments.json",
            dict(
                changed_buffers=calibrated.changed_buffers,
                prefix_forwards=calibrated.prefix_forwards,
                moments=[
                    dict(mean=row.mean.tolist(), variance=row.variance.tolist())
                    for row in calibrated.moments
                ],
                fitting_panel_sha256=sha(output / "sealed-fitting-panel.json"),
            ),
        )
        candidate_path = output / "candidate.pt"
        sealed_state = snapshot_state(teacher, watch)
        candidate_sha = checkpoint(
            candidate_path,
            teacher,
            dict(
                original_checkpoint_sha256=terminal.checkpoint_sha256,
                config_sha256=config_sha,
                moments_sha256=sha(output / "moments.json"),
                promotion_authorized=False,
            ),
            watch,
        )
        phase = "SEALED_BEFORE_SELECTION"
        save(
            output / "sealed-candidate.json",
            dict(checkpoint_sha256=candidate_sha, epoch=time.time(), selection_scored=False),
        )

        if hashed(candidate_path, watch) != candidate_sha:
            raise ValueError("Sealed calibrated checkpoint changed before reload")
        teacher.load_state_dict(
            torch.load(candidate_path, map_location="cpu", weights_only=True, mmap=True)[
                "state_dict"
            ],
            strict=True,
        )
        verify_state_change(sealed_state, snapshot_state(teacher, watch))
        scoring_budget = score_admission(
            config["deadline_epoch"] - time.time(), terminal.complete_evaluation_seconds, overhead
        )
        save(output / "score-admission.json", scoring_budget)
        if not scoring_budget["fits_budget"]:
            save(
                output / "result.json",
                dict(
                    status="BN_DIAGNOSTIC_BUDGET_INCONCLUSIVE_NO_SCORING",
                    promotion_authorized=False,
                    training_performed=False,
                    selection_scored=False,
                    candidate_checkpoint_sha256=candidate_sha,
                    qualification=asdict(qualification),
                    admission=asdict(admission),
                    score_admission=scoring_budget,
                ),
            )
            complete = True
            return

        # Only now open reference keys and the original validation waveform rows.
        labels = bound_json(Path(original["labels"]), original["labels_sha256"], watch)
        canonical = bound_json(
            Path(original["cache_manifest"]), original["cache_manifest_sha256"], watch
        )
        validation_rows = apply_identity_overlay(
            validate_inputs(plan, labels, canonical),
            bound_json(
                Path(original["identity_overlay"]), original["identity_overlay_sha256"], watch
            ),
            original["cache_manifest_sha256"],
        )
        validation_rows = merge_disk_cache(
            validation_rows,
            disk,
            original["cache_manifest_sha256"],
            original["identity_overlay_sha256"],
        )
        panels = {"beatport": plan["bp_selection_guard"], "fmak": plan["fma_inner_selection"]}
        verify_cache(
            {identity: validation_rows[identity] for ids in panels.values() for identity in ids},
            watch,
        )
        phase = "CANDIDATE_AUDIT"
        candidate = evaluate_selection_schedule(
            teacher, panels, validation_rows, watch, collect_logits=True
        )
        fast = evaluate_selection_schedule(
            teacher, panels, validation_rows, watch, collect_logits=True, batch_size=16
        )
        candidate_proof = compare_panels(candidate, fast)
        if not candidate_proof["passed"]:
            raise ValueError("Calibrated selected-state B1/B16 numerical audit failed")
        phase = "BASELINE_REPLAY"
        selected_state(teacher, selected, terminal, watch)
        baseline = evaluate_selection_schedule(
            teacher, panels, validation_rows, watch, collect_logits=True
        )

        def scores(panel: dict[str, Any]) -> dict[str, Any]:
            offset, measured = 0, {}
            for domain, ids in panels.items():
                logits = torch.tensor(
                    panel["logits"][offset : offset + len(ids)], dtype=torch.float64
                )
                measured[domain] = metrics(ids, logits.argmax(1).tolist(), labels["labels"])
                offset += len(ids)
            return measured

        candidate_metrics, baseline_metrics = scores(candidate), scores(baseline)
        if baseline_metrics != original_result["best_metrics"]:
            raise ValueError(
                "Fresh original baseline does not reproduce the terminal selected predictions"
            )
        if (
            sha(candidate_path) != candidate_sha
            or hashed(selected, watch) != terminal.checkpoint_sha256
        ):
            raise ValueError("Sealed candidate/original artifact changed during comparison")
        if any(parameter.grad is not None for parameter in teacher.parameters()):
            raise ValueError("Diagnostic unexpectedly created gradients")
        phase = "COMPLETE"
        save(
            output / "result.json",
            dict(
                status="COMPLETE_SINGLE_BN_DIAGNOSTIC_NOT_PROMOTED",
                promotion_authorized=False,
                training_performed=False,
                original_gate_unchanged=original_result["continuation_gate"],
                useful_signal=useful_signal(candidate_metrics, baseline_metrics),
                candidate_metrics=candidate_metrics,
                baseline_metrics=baseline_metrics,
                candidate=candidate,
                baseline=baseline,
                candidate_composed_batch_proof=candidate_proof,
                candidate_checkpoint_sha256=candidate_sha,
                candidate_strict_state_reload=True,
                original_checkpoint_sha256=terminal.checkpoint_sha256,
                original_result_sha256=terminal.result_sha256,
                config_sha256=config_sha,
                qualification=asdict(qualification),
                admission=asdict(admission),
                score_admission=scoring_budget,
                elapsed_seconds=time.monotonic() - began,
                watcher=watch_statistics,
                confirmation_used=False,
                reserved_calibration_used=False,
            ),
        )
        complete = True
    except BaseException as failure:
        save(
            output / "failure.json",
            dict(
                phase=phase,
                error=repr(failure),
                selection_may_have_started=phase
                in {"CANDIDATE_AUDIT", "BASELINE_REPLAY", "COMPLETE"},
            ),
        )
        raise
    finally:
        if provider is not None:
            provider.close(failed=not complete)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "approval", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--guardian-pid", type=int, required=True)
    args = parser.parse_args()
    worker(args.config, args.approval, args.output, args.guardian_pid)


if __name__ == "__main__":
    main()
