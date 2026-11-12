"""Fixed13/37/73 joint-K10 probability ensemble; inference-only research artifact."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from drone.features import CROP_FRAMES, FEATURE_SPEC, N_CORE, N_TOTAL, audio_digest
from drone.keys import LABEL_SCHEMA
from experiments.ema import read_reference
from experiments.k10_multidomain import FROZEN_SHA256, domain_metrics, validate_rows
from experiments.octave import device_setup
from experiments.paired import compare, movements
from experiments.small import atomic_json

SEEDS = (13, 37, 73)
VERSION = "k10-joint-three-seed-probability-ensemble-v1"
ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "experiments/studies/k10-ensemble"
PROTOCOL = STUDY / "PROTOCOL.md"
REVIEWED = ROOT / "experiments/studies/k10-capacity/data/reviewed-sources.json"
SOURCE_FILES = (
    "experiments/ensemble.py",
    "experiments/paired.py",
    "experiments/ema.py",
    "experiments/k10_multidomain.py",
    "experiments/octave.py",
    "experiments/small.py",
    "drone/features.py",
    "drone/keys.py",
    "drone/checkpoints.py",
    "pyproject.toml",
    "uv.lock",
)


def probabilities(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[1] != 24 or not np.isfinite(logits).all():
        raise ValueError("Need finite track-by24 logits")
    shifted = logits.astype(np.float64) - logits.astype(np.float64).max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def combine(logits: list[np.ndarray], seeds: tuple[int, ...] = SEEDS) -> np.ndarray:
    if seeds != SEEDS or len(logits) != 3 or any(z.shape != logits[0].shape for z in logits):
        raise ValueError("Need fixed ordered canonical13/37/73 members with identical shape")
    return np.stack([probabilities(z) for z in logits]).mean(axis=0, dtype=np.float64)


def teacher_report(
    members: list[dict[str, Any]], ensemble: dict[str, Any], bp: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(members) != 3 or len(bp) != 3:
        raise ValueError("Need three members and BP references")
    bp_mean = {
        d: {
            m: float(np.mean([r[d][m] for r in bp])) for m in ("mirex", "mirex_asymmetric", "exact")
        }
        for d in ("beatport", "fmak")
    }
    member_summary = {
        d: {
            m: dict(
                mean=float(np.mean([r[d][m] for r in members])),
                minimum=min(r[d][m] for r in members),
                maximum=max(r[d][m] for r in members),
            )
            for m in ("mirex", "mirex_asymmetric", "exact")
        }
        for d in ("beatport", "fmak")
    }
    delta = {d: ensemble[d]["mirex"] - bp_mean[d]["mirex"] for d in ("beatport", "fmak")}
    teacher = compare(members[0], ensemble)
    return dict(
        ensemble_candidates=1,
        ensemble_replicates=1,
        canonical_seed=13,
        fixed_member_seeds=list(SEEDS),
        member_summary=member_summary,
        comparisons={
            str(seed): {
                d: {
                    m: ensemble[d][m] - single[d][m] for m in ("mirex", "mirex_asymmetric", "exact")
                }
                for d in ("beatport", "fmak")
            }
            for seed, single in zip(SEEDS, members, strict=True)
        },
        teacher_gate=dict(
            delta=teacher["delta"], qualifies_as_teacher=teacher["qualifies_for_confirmation"]
        ),
        uniformly_nonregressing_vs_all_members=all(
            ensemble[d][m] >= single[d][m]
            for single in members
            for d in ("beatport", "fmak")
            for m in ("mirex", "exact")
        ),
        bp_control_mean=bp_mean,
        bp_reference_hurdle_delta=delta,
        bp_reference_hurdle_pass=delta["fmak"] >= 0.02 and delta["beatport"] >= 0,
        bp_reference_estimand="One ensemble vs mean of three BP-only controls; not three ensemble replications",
        uncertainty="One fixed cohort on reused epoch-selection development data; member spread is not ensemble standard error",
        promoted=False,
        distillation_authorized=False,
    )


def validate_payload(payload: dict[str, Any], run: dict[str, Any]) -> None:
    metadata = payload.get("metadata", {})
    if (
        payload.get("format_version") != 2
        or payload.get("feature_spec") != FEATURE_SPEC
        or payload.get("label_schema") != LABEL_SCHEMA
        or payload.get("crop") != CROP_FRAMES
        or payload.get("model_config") != {"width": 1.0}
        or any(metadata.get(k) != run[k] for k in ("arm", "seed", "source_sha256"))
        or metadata.get("epoch") != run["best_epoch"]
        or metadata.get("experiment") != "k10-multidomain-development-v1"
    ):
        raise ValueError("Checkpoint schema/model/source/selection metadata differs")


def verify_archive(source: Path, expected: dict[str, str], check: Callable[[], None]) -> None:
    for name, digest in expected.items():
        check()
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or audio_digest(path) != digest:
            raise ValueError(f"Archived source content differs: {name}")


def archived_model(source: Path) -> nn.Module:
    if N_CORE != 192:
        raise ValueError("Archived model requires192 core bins")
    spec = importlib.util.spec_from_file_location(
        "_ensemble_frozen_keycnn", source / "drone/model.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load verified archived model")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    constructor = cast(Callable[..., nn.Module], getattr(module, "KeyCNN"))
    model = constructor(width=1.0)
    if sum(p.numel() for p in model.parameters()) != 109128:
        raise ValueError("Archived K10 parameter count differs")
    return model


def validate_cohort(
    runs: list[dict[str, Any]],
    bp: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    manifest_sha: str,
    sources: dict[str, str],
) -> None:
    if (
        len(runs) != 3
        or len(bp) != 3
        or tuple(r["seed"] for r in runs) != SEEDS
        or tuple(r["seed"] for r in bp) != SEEDS
    ):
        raise ValueError("Member/reference order must be canonical13/37/73")
    for joint, control in zip(runs, bp, strict=True):
        if joint["source_sha256"] != sources or control["source_sha256"] != sources:
            raise ValueError("Run source differs from reviewed frozen673")
        for key in ("config", "feature_manifest_sha256", "fit_ids", "validation_ids"):
            if joint[key] != runs[0][key]:
                raise ValueError(f"Joint member identities differ: {key}")
        if joint["feature_manifest_sha256"] != manifest_sha:
            raise ValueError("Cache manifest differs from joint run")
        for role, field in (("train", "fit_ids"), ("validation", "validation_ids")):
            if joint[field] != [r["id"] for r in rows if r["role"] == role]:
                raise ValueError(f"Manifest differs from {field}")
        for key in ("config", "feature_manifest_sha256", "validation_ids"):
            if joint[key] != control[key]:
                raise ValueError(f"BP reference differs: {key}")
        if control["fit_ids"] != [
            r["id"] for r in rows if r["role"] == "train" and r["domain"] == "beatport"
        ]:
            raise ValueError("BP fitting identities differ")


def terminal_sam(result_path: Path, exit_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_bytes())
    exit_code = int(exit_path.read_text().strip())
    if (
        result.get("artifact_version") != "k10-vanilla-sam-adamw-v1"
        or result.get("seed") != 13
        or result.get("status")
        not in (
            "complete",
            "budget_inconclusive",
            "resource_inconclusive",
            "reproduction_or_protocol_failed",
        )
        or (
            result["status"] == "complete" and (result.get("kind") != "candidate" or exit_code != 0)
        )
        or not 0 <= exit_code <= 255
    ):
        raise ValueError(
            "Need authoritative terminal SAM candidate result plus wrapper exit receipt"
        )
    return dict(
        result_sha256=audio_digest(result_path),
        exit_sha256=audio_digest(exit_path),
        status=result["status"],
        exit_code=exit_code,
    )


def extract(
    model: nn.Module, rows: list[dict[str, Any]], data: Path, device: str, check: Callable[[], None]
) -> np.ndarray:
    model.eval()
    output = np.empty((len(rows), 24), dtype=np.float32)
    for offset in range(0, len(rows), 4):
        groups: dict[int, list[tuple[int, np.ndarray]]] = {}
        for index, row in enumerate(rows[offset : offset + 4], start=offset):
            check()
            array = np.load(data / row["cache"], mmap_mode="r", allow_pickle=False)
            if (
                array.dtype != np.float16
                or array.shape != (N_TOTAL, row["frames"])
                or row["frames"] < 8
            ):
                raise ValueError("Cached CQT shape/dtype changed")
            core = np.asarray(array[8:200], dtype=np.float32)
            if not np.isfinite(core).all() or (core < 0).any():
                raise ValueError("Invalid core CQT")
            groups.setdefault(core.shape[1], []).append((index, core))
        for entries in groups.values():
            check()
            indices, arrays = zip(*entries, strict=True)
            with torch.inference_mode():
                output[list(indices)] = (
                    model(torch.from_numpy(np.stack(arrays)).to(device)).cpu().numpy()
                )
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite extracted logits")
    return output


def study(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError("Use fresh ensemble output")
    args.output.mkdir(parents=True)
    start = time.monotonic()

    def check() -> None:
        if time.monotonic() - start > 1200:
            raise TimeoutError("Fixed20-minute extraction/scoring budget exceeded")

    terminal = terminal_sam(args.sam_result, args.sam_exit)
    reviewed = json.loads(REVIEWED.read_bytes())["archived"]
    verify_archive(args.reference_source, reviewed["sources"], check)
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if (
        not manifest["complete"]
        or manifest["feature_spec"] != FEATURE_SPEC
        or manifest["label_schema"] != LABEL_SCHEMA
        or manifest["frozen_manifest_sha256"] != FROZEN_SHA256
    ):
        raise ValueError("Need fixed complete quarantined cache")
    rows = manifest["rows"]
    validate_rows(rows)
    counts = [
        sum(r["domain"] == d and r["role"] == role for r in rows)
        for d in ("beatport", "fmak")
        for role in ("train", "validation")
    ]
    if counts != [1230, 126, 976, 250]:
        raise ValueError("Wrong development counts")
    runs = [
        read_reference(path, "joint", seed)
        for path, seed in zip(args.joint_runs, SEEDS, strict=True)
    ]
    bp = [
        read_reference(path, "control", seed)
        for path, seed in zip(args.bp_runs, SEEDS, strict=True)
    ]
    validate_cohort(runs, bp, rows, audio_digest(manifest_path), reviewed["sources"])
    checkpoints = []
    for path, run in zip(args.joint_runs + args.bp_runs, runs + bp, strict=True):
        check()
        checkpoint = path.with_name("best.pt")
        if audio_digest(checkpoint) != run["checkpoint_sha256"]:
            raise ValueError("Checkpoint content changed")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        validate_payload(payload, run)
        checkpoints.append(
            dict(run_sha256=audio_digest(path), checkpoint_sha256=run["checkpoint_sha256"])
        )
    validation = [r for r in rows if r["role"] == "validation"]
    for row in validation:
        check()
        path = (args.data / row["cache"]).resolve()
        if (
            not path.is_relative_to(args.data.resolve())
            or audio_digest(path) != row["cache_sha256"]
        ):
            raise ValueError("Cache content changed or outside data root")
    device_setup(args.device)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    identity = dict(
        artifact_version=VERSION,
        member_seeds=list(SEEDS),
        temperature=1.0,
        weights=[1 / 3] * 3,
        accumulation_dtype="float64",
        tie="lowest canonical class index",
        feature_manifest_sha256=audio_digest(manifest_path),
        feature_spec=FEATURE_SPEC,
        label_schema=LABEL_SCHEMA,
        archived_source_commit=reviewed["commit"],
        archived_source_sha256=reviewed["sources"],
        source_sha256={name: audio_digest(ROOT / name) for name in SOURCE_FILES},
        protocol_sha256=audio_digest(PROTOCOL),
        reviewed_sources_sha256=audio_digest(REVIEWED),
        references=checkpoints,
        sam_terminal=terminal,
        budget_seconds=1200,
        memory_guard=dict(
            allocator_fraction=0.04,
            required_free_bytes=8 * 1024**3,
            free_bytes=free,
            total_bytes=total,
            allocator_cap_bytes=int(0.04 * total),
        ),
        torch=str(torch.__version__),
        numpy=np.__version__,
        training=False,
    )
    atomic_json(args.output / "identity.json", identity)
    (args.output / "PROTOCOL.md").write_bytes(PROTOCOL.read_bytes())
    y = np.array([r["ref"] for r in validation])
    logits = []
    metrics = []
    for seed, path, run in zip(SEEDS, args.joint_runs, runs, strict=True):
        check()
        model = archived_model(args.reference_source).to(args.device)
        payload = torch.load(path.with_name("best.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(payload["state_dict"], strict=True)
        values = extract(model, validation, args.data, args.device, check)
        scored = domain_metrics(values.argmax(1), y, validation)
        if scored != run["domain_metrics"]:
            raise ValueError(f"Member{seed} recorded predictions/metrics failed exact reproduction")
        atomic_json(
            args.output / f"member-{seed}.json",
            dict(
                seed=seed,
                ids=[r["id"] for r in validation],
                audio_sha256=[r["sha256"] for r in validation],
                refs=y.tolist(),
                logits=values.tolist(),
                metrics=scored,
            ),
        )
        logits.append(values)
        metrics.append(scored)
        del model, payload
    mean = combine(logits)
    scored = domain_metrics(mean.argmax(1), y, validation)
    artifact = args.output / "ensemble.json"
    atomic_json(
        artifact,
        dict(
            ids=[r["id"] for r in validation],
            audio_sha256=[r["sha256"] for r in validation],
            refs=y.tolist(),
            probabilities=mean.tolist(),
            predictions=mean.argmax(1).tolist(),
            metrics=scored,
        ),
    )
    report = teacher_report(metrics, scored, [run["domain_metrics"] for run in bp])
    report.update(
        artifact_version=VERSION,
        status="complete",
        identity=identity,
        member_predictions_reproduced=True,
        metrics=scored,
        member_metrics={str(seed): m for seed, m in zip(SEEDS, metrics, strict=True)},
        confusion_movements={
            str(seed): movements(m, scored, y, validation)
            for seed, m in zip(SEEDS, metrics, strict=True)
        },
        member_artifact_sha256={
            str(seed): audio_digest(args.output / f"member-{seed}.json") for seed in SEEDS
        },
        ensemble_artifact_sha256=audio_digest(artifact),
        elapsed_seconds=time.monotonic() - start,
        memory=dict(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        ),
    )
    check()
    atomic_json(args.output / "result.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--joint-runs", type=Path, nargs=3, required=True)
    parser.add_argument("--bp-runs", type=Path, nargs=3, required=True)
    parser.add_argument("--sam-result", type=Path, required=True)
    parser.add_argument("--sam-exit", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    args = parser.parse_args()
    try:
        study(args)
    except (ValueError, RuntimeError, TimeoutError, OSError) as error:
        if args.output.is_dir() and not (args.output / "result.json").exists():
            atomic_json(
                args.output / "failure.json",
                dict(
                    artifact_version=VERSION,
                    status="resource_inconclusive"
                    if isinstance(error, (TimeoutError, torch.OutOfMemoryError))
                    else "reproduction_or_protocol_failed",
                    error=str(error),
                    promoted=False,
                ),
            )
        raise


if __name__ == "__main__":
    main()
