"""Source-bound disposable cost probe and fresh full MusicFM fit; no automatic launch."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import resource
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from drone.keys import score_mirex
from experiments.k10_v2_pilot import paired_batches, selection_key
from experiments.musicfm_eval import batched_panel, compare_panels
from experiments.musicfm_inputs import (
    BatchProvider,
    CachedProvider,
    StreamProvider,
    apply_identity_overlay,
    merge_disk_cache,
    ordered_batches,
)
from experiments.musicfm_key import song_windows
from experiments.musicfm_sampling import batches
from experiments.musicfm_teacher import (
    KeyTeacher,
    accumulated_update,
    evaluate_song,
    make_optimizer,
    seed_training,
)
from tools.musicfm_crop_io import file_identity, load, save, sha
from tools.preflight_musicfm import EVIDENCE, WEIGHT_BYTES, WEIGHT_SHA, prepare_upstream
from tools.prepare_musicfm_ram_cache import read_crop, save_or_replace

PLAN_SHA = "c3e59ffb764ca5f51444447b0a18a69543228e60a9e3723d925ddd82b0fc8373"
LABEL_SHA = "b8942b5913b8ebacdf2fc9e0ebca72a934aaa7c56d4e1a97d0f60fac84354323"
FIRST16_SHA = "5fbbc3e28ef085356ad4c743a4a64d0fde6b6b67bf5e7b16c83e23728113ad82"
UPDATES, EVERY = 8280, 69


def hashed(path: Path, watch: Callable[[], None]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            watch()
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(
    plan: dict[str, Any], labels: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    pools = {
        "fit": plan["bp_fit"] + plan["expanded_fma_fit"],
        "validation": plan["bp_selection_guard"] + plan["fma_inner_selection"],
    }
    if (
        len(plan["bp_fit"]),
        len(plan["expanded_fma_fit"]),
        len(plan["bp_selection_guard"]),
        len(plan["fma_inner_selection"]),
    ) != (1229, 4033, 126, 250):
        raise ValueError("Different approved pools")
    expected = {identity: role for role, ids in pools.items() for identity in ids}
    rows = {row["id"]: row for row in manifest["rows"]}
    if (
        manifest.get("complete") is not True
        or len(rows) != 5638
        or len(manifest["rows"]) != 5638
        or set(rows) != set(expected)
        or set(labels["labels"]) != set(expected)
    ):
        raise ValueError("Incomplete/extra/reserved cache membership")
    for identity, row in rows.items():
        if (
            row["role"] != expected[identity]
            or row["target"] != labels["labels"][identity]
            or type(row["target"]) is not int
            or not 0 <= row["target"] < 24
            or row["samples"] <= 1024
            or row["bytes"] != row["samples"] * 4
        ):
            raise ValueError("Cache role/target/sample contract differs")
        if row["role"] == "fit" and row["samples"] < 1200:
            raise ValueError("Unsupported microbatch-one BatchNorm input")
    return rows


def verify_cache(rows: dict[str, dict[str, Any]], watch: Callable[[], None]) -> None:
    for row in rows.values():
        watch()
        path = Path(row["cache"])
        if (
            path.is_symlink()
            or list(file_identity(path)) != list(row["cache_identity"])
            or path.stat().st_size != row["bytes"]
            or path.stat().st_mode & 0o222
            or hashed(path, watch) != row["waveform_sha256"]
        ):
            raise ValueError("Immutable full waveform changed")


def metrics(ids: list[str], predictions: list[int], labels: dict[str, int]) -> dict[str, Any]:
    refs = [labels[i] for i in ids]
    exact, sym = score_mirex(predictions, refs)
    asym = score_mirex(predictions, refs, symmetric=False)[1]
    return {
        "tracks": len(ids),
        "ids": ids,
        "predictions": predictions,
        "exact_correct": sum(p == r for p, r in zip(predictions, refs, strict=True)),
        "symmetric_units10": round(sym * len(ids) * 10),
        "asymmetric_units10": round(asym * len(ids) * 10),
        "exact": exact,
        "mirex": sym,
        "mirex_asymmetric": asym,
    }


def wave(row: dict[str, Any], start: int, length: int, watch: Callable[[], None]) -> torch.Tensor:
    path = Path(row["cache"])
    source = Path(row["path"])
    before = file_identity(path)
    raw = file_identity(source)
    if list(raw) != list(row["source_stable_identity"]):
        raise ValueError("Validation source physical identity changed")
    result = torch.from_numpy(read_crop(row, start, length, watch=watch)).unsqueeze(0)
    if file_identity(path) != before or file_identity(source) != raw:
        raise ValueError("Validation cache/source changed during read")
    return result


def evaluate_panel(
    teacher: KeyTeacher,
    ids: list[str],
    rows: dict[str, dict[str, Any]],
    watch: Callable[[], None],
    *,
    collect_predictions: bool,
    collect_logits: bool = False,
    batch_size: int = 1,
) -> dict[str, Any]:
    if batch_size == 16:
        return batched_panel(
            teacher,
            ids,
            rows,
            lambda row, start, length: wave(row, start, length, watch),
            watch,
            batch_size=16,
            collect_predictions=collect_predictions,
            collect_logits=collect_logits,
        )
    if batch_size != 1:
        raise ValueError("Unqualified evaluation batch")
    collected = []
    began = time.monotonic()
    predictions = []
    windows = 0
    timings = []
    for identity in ids:
        watch()
        row = rows[identity]
        started = time.monotonic()

        def inputs() -> Iterator[tuple[Any, torch.Tensor]]:
            for window in song_windows(row["samples"]):
                yield window, wave(row, window.start, window.stop - window.start, watch)

        logits = evaluate_song(teacher, row["samples"], inputs())
        if collect_predictions:
            predictions.append(int(np.argmax(logits)))
        if collect_logits:
            collected.append(logits.tolist())
        windows += len(song_windows(row["samples"]))
        timings.append(
            {
                "id": identity,
                "samples": row["samples"],
                "windows": len(song_windows(row["samples"])),
                "seconds": time.monotonic() - started,
            }
        )
        watch()
    return {
        "tracks": len(ids),
        "windows": windows,
        "elapsed_seconds": time.monotonic() - began,
        "per_song": timings,
        **({"predictions": predictions} if collect_predictions else {}),
        **({"logits": collected} if collect_logits else {}),
    }


def evaluate_selection_schedule(
    teacher: KeyTeacher,
    panels: dict[str, list[str]],
    rows: dict[str, dict[str, Any]],
    watch: Callable[[], None],
    *,
    collect_logits: bool,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Replay fit's separate BP/FMA calls, including each domain's partial batches."""
    if set(panels) != {"beatport", "fmak"}:
        raise ValueError("Explicit BP and FMA selection panels required")
    measured = [
        evaluate_panel(
            teacher,
            panels[domain],
            rows,
            watch,
            collect_predictions=False,
            collect_logits=collect_logits,
            batch_size=batch_size,
        )
        for domain in ("beatport", "fmak")
    ]
    return {
        "tracks": sum(panel["tracks"] for panel in measured),
        "windows": sum(panel["windows"] for panel in measured),
        "elapsed_seconds": sum(panel["elapsed_seconds"] for panel in measured),
        "per_song": [row for panel in measured for row in panel["per_song"]],
        "domain_schedule": [
            {"domain": domain, "ids": panels[domain], "tracks": panel["tracks"]}
            for domain, panel in zip(("beatport", "fmak"), measured, strict=True)
        ],
        **(
            {"logits": [row for panel in measured for row in panel["logits"]]}
            if collect_logits
            else {}
        ),
    }


def load_teacher(config: dict[str, Any], output: Path, watch: Callable[[], None]) -> KeyTeacher:
    watch()
    weights = Path(config["weights"])
    if weights.stat().st_size != WEIGHT_BYTES or hashed(weights, watch) != WEIGHT_SHA:
        raise ValueError("Only original pinned MSD checkpoint may initialize this run")
    for name, row in config["module_receipts"].items():
        if hashed(Path(row["path"]), watch) != row["sha256"]:
            raise ValueError("Isolated provisioned environment changed: " + name)
    torch.set_num_threads(config["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available() or torch.__version__ != "2.15.0+cu132":
        raise ValueError("Reviewed CUDA Torch environment required")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(config["allocator_bytes"] / total, 0)
    prepare_upstream(output)
    sys.path.insert(0, str(output / "upstream"))
    upstream = importlib.import_module("musicfm.model.musicfm_25hz")
    cls = getattr(importlib.import_module("transformers"), "Wav2Vec2ConformerConfig")
    options = json.loads((EVIDENCE / "conformer-config.json").read_bytes())
    options["_attn_implementation"] = "eager"
    with patch.object(cls, "from_pretrained", return_value=cls.from_dict(options)):
        backbone = upstream.MusicFM25Hz(
            is_flash=False, stat_path=str(EVIDENCE / "msd-stats.json"), model_path=""
        )
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    if not all(key.startswith("model.") for key in state):
        raise ValueError("Unexpected checkpoint prefix")
    backbone.load_state_dict({key[6:]: value for key, value in state.items()}, strict=True)
    del state
    for name, receipt in config["module_receipts"].items():
        actual_name = {
            "musicfm": "musicfm.model.musicfm_25hz",
            "conformer": "transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer",
        }.get(name, name)
        module = importlib.import_module(actual_name)
        if module.__file__ is None or sha(Path(module.__file__)) != receipt["sha256"]:
            raise ValueError("Imported environment shadowed: " + name)
    teacher = KeyTeacher(backbone, seed=13).float().to("cuda")
    seed_training(13)
    if any(p.dtype != torch.float32 for p in teacher.parameters()):
        raise ValueError("FP32 master parameters required")
    watch()
    return teacher


def finite_state(teacher: KeyTeacher) -> None:
    if any(not bool(torch.isfinite(value).all()) for value in teacher.state_dict().values()):
        raise ValueError("Nonfinite model parameters/buffers; abort whole run")


def checkpoint(
    path: Path, teacher: KeyTeacher, provenance: dict[str, Any], watch: Callable[[], None]
) -> str:
    watch()
    finite_state(teacher)
    temporary = path.with_suffix(".pending")
    torch.save({"state_dict": teacher.state_dict(), "provenance": provenance}, temporary)
    watch()
    temporary.replace(path)
    return hashed(path, watch)


def cpu_seconds() -> float:
    values = [resource.getrusage(who) for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    return sum(row.ru_utime + row.ru_stime for row in values)


def probe(
    teacher: KeyTeacher,
    rows: dict[str, dict[str, Any]],
    ids: list[str],
    selection: dict[str, list[str]],
    watch: Callable[[], None],
    provider: BatchProvider,
    batch_size: int = 1,
) -> dict[str, Any]:
    if len(ids) != 16 or len(set(ids)) != 16 or any(rows[i]["role"] != "fit" for i in ids):
        raise ValueError("Probe requires the original fitting16 without replacement")
    optimizer = make_optimizer(teacher)
    times = []
    from experiments.k10_v2_pilot import Draw

    draws = [
        [Draw(ids[(32 * u + i) % 16], "probe", 32 * u + i) for i in range(32)] for u in range(3)
    ]
    phase_began = time.monotonic()
    cpu_began = cpu_seconds()
    for update, (batch, arrays) in enumerate(ordered_batches(draws, rows, provider)):
        examples = [
            (torch.from_numpy(a).unsqueeze(0), (32 * update + i) % 24) for i, a in enumerate(arrays)
        ]
        watch()
        torch.cuda.synchronize()
        began = time.monotonic()
        accumulated_update(teacher, optimizer, examples)
        finite_state(teacher)
        torch.cuda.synchronize()
        times.append(time.monotonic() - began)
        examples.clear()
        arrays.clear()
        watch()
    training_elapsed = time.monotonic() - phase_began
    training_cpu = cpu_seconds() - cpu_began
    validation_cpu = cpu_seconds()
    panel = evaluate_selection_schedule(
        teacher, selection, rows, watch, collect_logits=batch_size == 16
    )
    proof = None
    if batch_size == 16:
        fast = evaluate_selection_schedule(
            teacher, selection, rows, watch, collect_logits=True, batch_size=16
        )
        proof = compare_panels(panel, fast)
        panel.pop("logits")
        fast.pop("logits")
    else:
        fast = None
    return {
        "status": "DISPOSABLE_COST_ONLY",
        "synthetic_target_updates": 3,
        "accumulation": 32,
        "update_seconds": times,
        "training_pipeline_seconds": training_elapsed,
        "training_pipeline_cpu_seconds": training_cpu,
        "input_wait_seconds_per_update": list(getattr(provider, "resolve_wait_seconds", [])),
        "validation_cpu_seconds": cpu_seconds() - validation_cpu,
        "validation": panel,
        "batched_validation": fast,
        "composed_batch_proof": proof,
        "task_scores_computed": False,
        "weights_saved_or_reused": False,
        "optimizer_state_parameters": len(optimizer.state),
        "optimizer_state_dtypes": sorted(
            {
                str(t.dtype)
                for state in optimizer.state.values()
                for t in state.values()
                if torch.is_tensor(t)
            }
        ),
    }


def fit(
    teacher: KeyTeacher,
    rows: dict[str, dict[str, Any]],
    plan: dict[str, Any],
    labels: dict[str, int],
    output: Path,
    provenance: dict[str, Any],
    watch: Callable[[], None],
    provider: BatchProvider,
    batch_size: int = 1,
) -> dict[str, Any]:
    optimizer = make_optimizer(teacher)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)
    panels = {"beatport": plan["bp_selection_guard"], "fmak": plan["fma_inner_selection"]}
    history = []
    best = None
    selected = None
    best_metrics = None
    trace = hashlib.sha256()
    coverage = Counter()
    losses = 0.0
    stream = ordered_batches(batches(plan["bp_fit"], plan["expanded_fma_fit"]), rows, provider)
    for update, (draws, arrays) in enumerate(stream, 1):
        examples = [
            (torch.from_numpy(a).unsqueeze(0), labels[d.identity])
            for d, a in zip(draws, arrays, strict=True)
        ]
        for draw in draws:
            coverage[draw.identity] += 1
            if draw.domain == "beatport":
                trace.update(f"{draw.identity}:{draw.augmentation_seed}\n".encode())
        watch()
        losses += accumulated_update(teacher, optimizer, examples)
        finite_state(teacher)
        examples.clear()
        arrays.clear()
        watch()
        if update % EVERY:
            continue
        measured = {}
        timing = {}
        for domain, ids in panels.items():
            panel = evaluate_panel(
                teacher, ids, rows, watch, collect_predictions=True, batch_size=batch_size
            )
            measured[domain] = metrics(ids, panel.pop("predictions"), labels)
            timing[domain] = panel
        key = selection_key(measured)
        if best is None or key > best:
            best = key
            selected = update
            best_metrics = measured
            best_sha = checkpoint(
                output / "best.pt", teacher, {**provenance, "selected_update": update}, watch
            )
        history.append(
            {
                "update": update,
                "train_ce": losses / EVERY,
                "selection_exact": float(key[0]),
                "selection_mirex": float(key[1]),
                "metrics": measured,
                "validation": timing,
            }
        )
        losses = 0.0
        scheduler.step()
        save_or_replace(
            output / "progress.json",
            {
                "complete": False,
                "updates": update,
                "events": len(history),
                "best_update": selected,
                "history": history,
            },
        )
    if update != UPDATES or len(history) != 120:
        raise ValueError("Incomplete frozen horizon")
    expected_trace = provenance.get("canonical_bp_trace_sha256")
    if expected_trace is not None and trace.hexdigest() != expected_trace:
        raise ValueError(
            "Actual BP ID/augmentation-seed subsequence differs from fixed control replay"
        )
    final_sha = checkpoint(
        output / "final.pt", teacher, {**provenance, "final_update": update}, watch
    )
    selected_proof = None
    if batch_size == 16:
        if hashed(output / "best.pt", watch) != best_sha:
            raise ValueError("Selected checkpoint changed")
        teacher.load_state_dict(
            torch.load(
                output / "best.pt", map_location=teacher.head.weight.device, weights_only=True
            )["state_dict"],
            strict=True,
        )
        reference = evaluate_selection_schedule(teacher, panels, rows, watch, collect_logits=True)
        fast = evaluate_selection_schedule(
            teacher, panels, rows, watch, collect_logits=True, batch_size=16
        )
        selected_proof = compare_panels(reference, fast)
    return {
        "selected_teacher_composed_batch_proof": selected_proof,
        "status": "COMPLETE_FIXED_PILOT_NOT_RELEASE_EVIDENCE",
        "updates": update,
        "events": 120,
        "best_update": selected,
        "best_metrics": best_metrics,
        "best_checkpoint_sha256": best_sha,
        "final_checkpoint_sha256": final_sha,
        "bp_id_draw_trace_sha256": trace.hexdigest(),
        "coverage": dict(coverage),
        "history": history,
        "confirmation_used": False,
        "calibration_used": False,
    }


def canonical_bp_trace(
    bp: list[str], fma: list[str], frames: dict[str, int] | None = None, updates: int = 8280
) -> tuple[str, str | None]:
    canonical = hashlib.sha256()
    augmented = hashlib.sha256()
    for batch in paired_batches(bp, fma, updates=updates):
        for draw in batch:
            if draw.domain != "beatport":
                continue
            canonical.update(f"{draw.identity}:{draw.augmentation_seed}\n".encode())
            if frames is not None:
                rng = np.random.default_rng(draw.augmentation_seed)
                shift = int(rng.integers(-4, 5))
                first = (
                    int(rng.integers(0, frames[draw.identity] - 320 + 1))
                    if frames[draw.identity] >= 320
                    else 0
                )
                augmented.update(
                    f"{draw.identity}:{draw.augmentation_seed}:{shift}:{first}\n".encode()
                )
    return canonical.hexdigest(), augmented.hexdigest() if frames is not None else None


def continuation_gate(measured: dict[str, Any], controls: dict[str, Any]) -> dict[str, Any]:
    if set(controls) != {"restricted", "expanded"}:
        raise ValueError("Both frozen K10 controls required")
    checks = {}
    for name, control in controls.items():
        old = control["best_metrics"]
        for domain in ("beatport", "fmak"):
            if measured[domain]["ids"] != old[domain]["ids"]:
                raise ValueError("Teacher/control panel order differs")
        checks[name] = {
            "fma_exact_plus5": measured["fmak"]["exact_correct"]
            >= old["fmak"]["exact_correct"] + 5,
            "bp_nondecline": all(
                measured["beatport"][key] >= old["beatport"][key]
                for key in ("exact_correct", "symmetric_units10", "asymmetric_units10")
            ),
        }
    return {
        "continuation_earned": all(all(row.values()) for row in checks.values()),
        "checks_against_each_control": checks,
        "confirmation_scored": False,
        "release_or_distillation_authorized": False,
    }


def bind_cost_provenance(config: dict[str, Any], cost: dict[str, Any]) -> None:
    """A measured disposable probe must use the prospective fit's invariant recipe."""
    expected = {
        "source_sha256": config["source_sha256"],
        "input_provider": config["input_provider"],
        "pretrained_checkpoint_sha256": config["weight_sha256"],
        "plan_sha256": config["plan_sha256"],
        "labels_sha256": config["labels_sha256"],
        "cache_manifest_sha256": config["cache_manifest_sha256"],
        "identity_overlay_sha256": config["identity_overlay_sha256"],
        "disk_cache_manifest_sha256": config["disk_cache_manifest_sha256"],
        "decoder_reference_sha256": config["ram_config_sha256"],
        "module_receipts": config["module_receipts"],
        "canonical_bp_trace_sha256": config["canonical_bp_trace_sha256"],
        "eval_batch": config["eval_batch_size"],
        "seed_after_strict_load": 13,
    }
    observed = cost.get("provenance", {})
    if any(key not in observed or observed[key] != value for key, value in expected.items()):
        raise ValueError("Disposable cost source/data/evaluation provenance differs from fit")


def make_plan(args: argparse.Namespace) -> dict[str, Any]:
    if (
        sha(args.plan) != PLAN_SHA
        or sha(args.labels) != LABEL_SHA
        or sha(args.first16) != FIRST16_SHA
    ):
        raise ValueError("Pinned cohort/annotation/original16 identity differs")
    ram, manifest = load(args.ram_config), load(args.manifest)
    rows = validate_inputs(load(args.plan), load(args.labels), manifest)
    apply_identity_overlay(rows, load(args.identity_overlay), sha(args.manifest))
    identity_receipt = load(args.identity_receipt)
    if (
        identity_receipt.get("status") != "COMPLETE_READ_ONLY_IDENTITIES_REFRESHED_NOT_ACTIVATED"
        or identity_receipt.get("overlay_sha256") != sha(args.identity_overlay)
        or identity_receipt.get("manifest_sha256") != sha(args.manifest)
    ):
        raise ValueError("Unbound or failed source identity refresh receipt")
    if args.provider == "disk":
        if args.disk_cache is None:
            raise ValueError("Complete disk-cache manifest required")
        merge_disk_cache(
            apply_identity_overlay(rows, load(args.identity_overlay), sha(args.manifest)),
            load(args.disk_cache),
            sha(args.manifest),
            sha(args.identity_overlay),
        )
    if manifest["config_sha256"] != sha(args.ram_config):
        raise ValueError("Wrong completed RAM preparation")
    env = load(args.environment)
    caps = load(args.caps)
    if any(
        type(caps.get(key)) not in (int, float) or not math.isfinite(caps[key]) or caps[key] <= 0
        for key in ("wall_seconds", "cpu_seconds", "rss_bytes", "output_bytes")
    ):
        raise ValueError("Finite positive externally reviewed caps required")
    if args.mode == "fit":
        if args.cost_probe is None or sha(args.cost_probe) != caps.get("basis_probe_sha256"):
            raise ValueError("Full-fit budget pending exact measured disposable probe")
        cost = load(args.cost_probe)
        if (
            cost.get("status") != "DISPOSABLE_COST_ONLY"
            or cost.get("validation", {}).get("tracks") != 376
            or cost.get("synthetic_target_updates") != 3
            or cost.get("task_scores_computed") is not False
        ):
            raise ValueError("Incomplete or wrong cost evidence")
        if cost.get("provenance", {}).get("input_provider") != args.provider:
            raise ValueError("Cost evidence measured a different input provider")
    if args.eval_batch == 16:
        if (
            args.batch_proof is None
            or sha(args.batch_proof)
            != "11c9137eeb934ca1a4b66a879cf97f5b76c3e6cd36a974aa614560b9978cae71"
        ):
            raise ValueError("Exact qualified B1-frontend/B16-encoder proof required")
        if args.mode == "fit" and cost.get("composed_batch_proof", {}).get("passed") is not True:
            raise ValueError("Full376 composed cost proof must pass before16fit")
    first = load(args.first16)["first_batch"]
    if (
        len(first) != 16
        or len({r["id"] for r in first}) != 16
        or any(
            r["id"] not in load(args.plan)["bp_fit"] + load(args.plan)["expanded_fma_fit"]
            for r in first
        )
    ):
        raise ValueError("Original fitting16 changed")
    names = [
        "experiments/musicfm_pilot.py",
        "experiments/musicfm_teacher.py",
        "experiments/musicfm_sampling.py",
        "experiments/musicfm_key.py",
        "experiments/musicfm_inputs.py",
        "experiments/musicfm_eval.py",
        "tools/musicfm_stream_crop.py",
        "tools/musicfm_source_contract.py",
        "tools/run_musicfm_teacher.py",
        "tools/prepare_musicfm_ram_cache.py",
        "tools/run_musicfm_ram_cache.py",
        "tools/preflight_musicfm.py",
        "drone/keys.py",
    ]
    names += [
        "experiments/k10_v2_pilot.py",
        "tools/musicfm_crop_io.py",
        "tools/run_cqt_expansion.py",
        "tools/run_k10_v2_pair.py",
    ]
    repository = Path(__file__).resolve().parents[1]
    controls = {
        arm: repository / "experiments/studies/k10-v2-pilot/data/results" / (arm + ".json.gz")
        for arm in ("restricted", "expanded")
    }
    frames_path = (
        repository
        / "experiments/studies/k10-v2-pilot/data/canonical-eligibility-v2/feature-manifest-v2.json.gz"
    )
    frame_rows = {r["id"]: r["frames"] for r in load(frames_path)["rows"]}
    original = load(args.plan)
    canonical, augmented = canonical_bp_trace(
        original["bp_fit"], original["expanded_fma_fit"], frame_rows
    )
    if any(load(path)["bp_trace_sha256"] != augmented for path in controls.values()):
        raise ValueError("Old executed K10 augmented trace does not match frozen metadata replay")
    result = {
        "status": "PROPOSED_NO_EXECUTION_AUTHORIZATION",
        "canonical_bp_trace_sha256": canonical,
        "old_k10_augmented_trace_replay_sha256": augmented,
        "controls": {
            name: {"path": str(path), "sha256": sha(path)} for name, path in controls.items()
        },
        "evidence_sha256": {
            str(p.relative_to(repository)): sha(p) for p in EVIDENCE.rglob("*") if p.is_file()
        },
        "mode": args.mode,
        "input_provider": args.provider,
        "disk_cache_manifest": str(args.disk_cache) if args.disk_cache else None,
        "disk_cache_manifest_sha256": sha(args.disk_cache) if args.disk_cache else None,
        "plan": str(args.plan),
        "labels": str(args.labels),
        "cache_manifest": str(args.manifest),
        "identity_overlay": str(args.identity_overlay),
        "identity_overlay_sha256": sha(args.identity_overlay),
        "identity_receipt": str(args.identity_receipt),
        "identity_receipt_sha256": sha(args.identity_receipt),
        "ram_config": str(args.ram_config),
        "plan_sha256": PLAN_SHA,
        "labels_sha256": LABEL_SHA,
        "cache_manifest_sha256": sha(args.manifest),
        "ram_config_sha256": sha(args.ram_config),
        "weights": str(args.weights),
        "weight_sha256": WEIGHT_SHA,
        "module_receipts": env["module_receipts"],
        "probe_ids": [r["id"] for r in first],
        "source_sha256": {n: sha(Path(__file__).resolve().parents[1] / n) for n in names},
        "allocator_bytes": 12 * 1024**3,
        "gpu_uuid": env["gpu_uuid"],
        "cpus": [8, 9],
        "decoder_cpus": [10, 11, 12, 13],
        "torch_threads": 1,
        "protected_starts": ram["protected_starts"],
        "caps": caps,
        "eval_batch_size": args.eval_batch,
        "batch_qualification_sha256": sha(args.batch_proof) if args.batch_proof else None,
        "not_before_epoch": None,
        "deadline_epoch": None,
        "no_resume_or_retry": True,
        "basis_initial_state_sha256": cost["provenance"]["initial_state_sha256"]
        if args.mode == "fit"
        else None,
    }
    if args.mode == "fit":
        bind_cost_provenance(result, cost)
    return result


def worker(config_path: Path, approval_path: Path, output: Path, guardian_pid: int) -> None:
    from tools.run_musicfm_teacher import worker_watch

    config, approval = load(config_path), load(approval_path)
    if config["mode"] not in {"probe", "fit"}:
        raise ValueError("Unknown execution mode")
    if (
        os.getppid() != guardian_pid
        or approval.get("config_sha256") != sha(config_path)
        or approval.get("authorized") is not True
        or config["eval_batch_size"] not in (1, 16)
        or (
            config["mode"] == "probe"
            and (
                approval.get("allow_probe") is not True
                or approval.get("allow_training") is not False
            )
        )
        or (config["mode"] == "fit" and approval.get("allow_training") is not True)
    ):
        raise ValueError("Private worker requires its exact live guardian and approval")
    config["_config_sha256"] = sha(config_path)
    watch, watch_statistics = worker_watch(config, output.parent, guardian_pid)
    watch()
    root = Path(__file__).resolve().parents[1]
    if any(sha(root / name) != expected for name, expected in config["source_sha256"].items()):
        raise ValueError("Frozen numerical/guard source changed")
    for name, key in [
        ("plan", "plan_sha256"),
        ("labels", "labels_sha256"),
        ("cache_manifest", "cache_manifest_sha256"),
        ("ram_config", "ram_config_sha256"),
        ("identity_overlay", "identity_overlay_sha256"),
        ("identity_receipt", "identity_receipt_sha256"),
    ]:
        if sha(Path(config[name])) != config[key]:
            raise ValueError("Bound input changed: " + name)
    if any(sha(root / name) != value for name, value in config["evidence_sha256"].items()):
        raise ValueError("Pinned upstream/source/config/stat evidence changed")
    controls = {}
    for name, record in config["controls"].items():
        path = Path(record["path"])
        if sha(path) != record["sha256"]:
            raise ValueError("Frozen K10 comparator changed")
        controls[name] = load(path)
    plan, labels, manifest = (
        load(Path(config["plan"])),
        load(Path(config["labels"])),
        load(Path(config["cache_manifest"])),
    )
    rows = apply_identity_overlay(
        validate_inputs(plan, labels, manifest),
        load(Path(config["identity_overlay"])),
        config["cache_manifest_sha256"],
    )
    if config["input_provider"] == "disk":
        path = Path(config["disk_cache_manifest"])
        if sha(path) != config["disk_cache_manifest_sha256"]:
            raise ValueError("Disk cache manifest changed")
        rows = merge_disk_cache(
            rows, load(path), config["cache_manifest_sha256"], config["identity_overlay_sha256"]
        )
    from tools.musicfm_crop_io import runtime_files

    ram = load(Path(config["ram_config"]))
    if runtime_files(Path(ram["ffmpeg"])) != ram["runtime_file_sha256"]:
        raise ValueError("Pinned full-stream decoder runtime changed")
    output.mkdir(exist_ok=False)
    began = time.monotonic()
    verify_cache(
        rows
        if config["input_provider"] == "disk"
        else {i: r for i, r in rows.items() if r["role"] == "validation"},
        watch,
    )
    verified = time.monotonic()
    teacher = load_teacher(config, output, watch)
    loaded = time.monotonic()
    initial = hashlib.sha256()
    for name, value in teacher.state_dict().items():
        watch()
        initial.update(name.encode())
        initial.update(value.detach().cpu().numpy().tobytes())
    provenance = {
        "initial_state_sha256": initial.hexdigest(),
        "input_provider": config["input_provider"],
        "canonical_bp_trace_sha256": config["canonical_bp_trace_sha256"],
        "config_sha256": sha(config_path),
        "source_sha256": config["source_sha256"],
        "pretrained_checkpoint_sha256": WEIGHT_SHA,
        "plan_sha256": PLAN_SHA,
        "labels_sha256": LABEL_SHA,
        "cache_manifest_sha256": config["cache_manifest_sha256"],
        "identity_overlay_sha256": config["identity_overlay_sha256"],
        "disk_cache_manifest_sha256": config["disk_cache_manifest_sha256"],
        "decoder_reference_sha256": config["ram_config_sha256"],
        "module_receipts": config["module_receipts"],
        "seed_after_strict_load": 13,
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
        "precision": "FP32 masters/frontend/eval;BF16 training encoder/head",
        "eval_batch": config["eval_batch_size"],
    }
    if (
        config["mode"] == "fit"
        and provenance["initial_state_sha256"] != config["basis_initial_state_sha256"]
    ):
        raise ValueError(
            "Fresh fit initialization differs from disposable probe original initialization"
        )
    teacher.register_forward_pre_hook(lambda module, inputs: watch())
    provider = (
        CachedProvider(rows, watch)
        if config["input_provider"] == "disk"
        else StreamProvider(
            rows,
            output / "streams",
            Path(load(Path(config["ram_config"]))["ffmpeg"]),
            config["decoder_cpus"],
            watch,
        )
    )
    try:
        if config["mode"] == "probe":
            result = probe(
                teacher,
                rows,
                config["probe_ids"],
                {"beatport": plan["bp_selection_guard"], "fmak": plan["fma_inner_selection"]},
                watch,
                provider,
                config["eval_batch_size"],
            )
        elif config["mode"] == "fit" and approval.get("allow_training") is True:
            result = fit(
                teacher,
                rows,
                plan,
                labels["labels"],
                output,
                provenance,
                watch,
                provider,
                config["eval_batch_size"],
            )
            result["continuation_gate"] = continuation_gate(result["best_metrics"], controls)
            if (
                config["eval_batch_size"] == 16
                and not result["selected_teacher_composed_batch_proof"]["passed"]
            ):
                result["continuation_gate"]["continuation_earned"] = False
        else:
            raise ValueError("Full fit not explicitly authorized")
        save(
            output / "result.json",
            {
                **result,
                "provenance": provenance,
                "cache_verify_seconds": verified - began,
                "model_load_seconds": loaded - verified,
                "elapsed_seconds": time.monotonic() - began,
                "max_cuda_allocated": torch.cuda.max_memory_allocated(),
                "max_cuda_reserved": torch.cuda.max_memory_reserved(),
                "guardian_watch": watch_statistics,
            },
        )
    except BaseException as error:
        provider.close(failed=True)
        save(
            output / "failure.json",
            {"complete": False, "error": repr(error), "provenance": provenance},
        )
        raise
    else:
        provider.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    for name in (
        "plan",
        "labels",
        "first16",
        "ram-config",
        "manifest",
        "identity-overlay",
        "identity-receipt",
        "environment",
        "weights",
        "caps",
        "output",
    ):
        plan.add_argument("--" + name, type=Path, required=True)
    plan.add_argument("--mode", choices=["probe", "fit"], required=True)
    plan.add_argument("--provider", choices=["disk", "stream"], default="disk")
    plan.add_argument("--disk-cache", type=Path)
    plan.add_argument("--cost-probe", type=Path)
    plan.add_argument("--eval-batch", type=int, choices=[1, 16], default=1)
    plan.add_argument("--batch-proof", type=Path)
    sealing = commands.add_parser("seal")
    for name in ("config", "output"):
        sealing.add_argument("--" + name, type=Path, required=True)
    for name in ("begin", "end"):
        sealing.add_argument("--" + name, required=True)
    run = commands.add_parser("worker")
    for name in ("config", "approval", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--guardian-pid", type=int, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        save(args.output, make_plan(args))
    elif args.command == "seal":
        from tools.prepare_musicfm_waveforms import seal

        save(args.output, seal(args.config, args.begin, args.end))
    else:
        worker(args.config, args.approval, args.output, args.guardian_pid)


if __name__ == "__main__":
    main()
