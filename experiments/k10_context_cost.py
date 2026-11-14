"""Disposable CPU resource qualification; synthetic losses, no task scores or weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import os
import resource
import sys
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from experiments.k10_v2_pilot import Draw, paired_batches, validate_feature_manifest
from tools.musicfm_crop_io import file_identity, load, save, sha

PLAN = "c3e59ffb764ca5f51444447b0a18a69543228e60a9e3723d925ddd82b0fc8373"
LABELS = "b8942b5913b8ebacdf2fc9e0ebca72a934aaa7c56d4e1a97d0f60fac84354323"
FEATURES = "9f94f6a5c6a9e78b4c5d1013cf97b44d6d2fd69665348ebb497e2c817560d6cc"
INITIAL = "bb227d5d27d0f28c0065de14468a4ada92d79944ccbcb9da6a68c97e64daa219"
SOURCE_NAMES = (
    "experiments/k10_context_cost.py",
    "tools/run_k10_context_cost.py",
    "experiments/k10_v2_pilot.py",
    "drone/features.py",
    "drone/model.py",
    "drone/keys.py",
    "train.py",
    "tools/musicfm_crop_io.py",
    "tools/run_musicfm_teacher.py",
    "tools/preflight_musicfm.py",
    "tools/prepare_musicfm_ram_cache.py",
    "tools/prepare_musicfm_waveforms.py",
    "tools/run_musicfm_ram_cache.py",
    "tools/run_cqt_expansion.py",
    "tools/run_k10_v2_pair.py",
)


def make_plan(
    plan: Path, labels: Path, features: Path, protected: Path, runtime: Path
) -> dict[str, Any]:
    if (sha(plan), sha(labels), sha(features)) != (PLAN, LABELS, FEATURES):
        raise ValueError("Require exact admitted cohort, targets and CQT manifest")
    plan_document = load(plan)
    feature_manifest = load(features)
    validate_feature_manifest(plan_document, feature_manifest)
    if load(labels)["labels"] != plan_document["expected_refs"]:
        raise ValueError("Pinned numeric annotation export differs")
    rows = []
    for row in feature_manifest["rows"]:
        path = (features.parent / row["cache"]).resolve()
        if not path.is_relative_to(features.parent.resolve()) or not path.is_file():
            raise ValueError("Missing/escaped existing CQT")
        rows.append({**row, "cache_path": str(path), "cache_identity": file_identity(path)})
    starts = load(protected)["protected_starts"]
    if len(starts) != 6 or {start["pid"] for start in starts} != {
        1478814,
        3448680,
        1481756,
        3451013,
        1294,
        1307,
    }:
        raise ValueError("Require four protected vLLM and both active teacher identities")
    root = Path(__file__).resolve().parents[1]
    modules = load(runtime)["module_receipts"]
    library_root = Path(modules["torch"]["path"]).parent / "lib"
    return {
        "status": "PROPOSED_CPU_COST_ONLY_NO_EXECUTION",
        "plan": str(plan),
        "plan_sha256": PLAN,
        "labels": str(labels),
        "labels_sha256": LABELS,
        "features": str(features),
        "features_sha256": FEATURES,
        "protected_starts": starts,
        "protected_witness_sha256": sha(protected),
        "runtime_witness_sha256": sha(runtime),
        "interpreter": {"path": sys.executable, "sha256": sha(Path(sys.executable))},
        "native_cpu_sha256": {
            str(library_root / name): sha(library_root / name)
            for name in ("libtorch_cpu.so", "libtorch.so", "libc10.so")
        },
        "module_receipts": {
            name: modules[name]
            for name in ("torch", "torch._C", "numpy", "numpy._core._multiarray_umath")
        },
        "source_sha256": {name: sha(root / name) for name in SOURCE_NAMES},
        "rows": rows,
        "cpus": [14, 15],
        "torch_threads": 2,
        "interop_threads": 1,
        "nice": 19,
        "crop_frames": [320, 65],
        "updates_per_shape": 3,
        "batch": 32,
        "seed": 13,
        "gpu_uuid": "GPU-387d0c41-22e5-4c3b-efe8-1d8f1dd0494c",
        "caps": {
            "wall_seconds": 900,
            "cpu_seconds": 1200,
            "rss_bytes": 8 * 1024**3,
            "output_bytes": 1024**3,
        },
        "runtime": {"python": "3.12.3", "torch": "2.15.0+cu132", "numpy": "2.5.4"},
        "not_before_epoch": None,
        "deadline_epoch": None,
        "loss_target_policy": "global draw ordinal modulo24; original labels never enter loss",
        "full_task_fit_allowed": False,
    }


def contract(value: dict[str, Any]) -> None:
    if (
        value["cpus"] != [14, 15]
        or value["crop_frames"] != [320, 65]
        or value["updates_per_shape"] != 3
        or value["batch"] != 32
        or value["seed"] != 13
        or value["torch_threads"] != 2
        or value["interop_threads"] != 1
        or value["nice"] != 19
        or value["full_task_fit_allowed"] is not False
        or value["caps"]
        != {
            "wall_seconds": 900,
            "cpu_seconds": 1200,
            "rss_bytes": 8 * 1024**3,
            "output_bytes": 1024**3,
        }
    ):
        raise ValueError("Frozen disposable CPU cost contract differs")


def make_crop(feature: np.ndarray, draw: Draw, length: int) -> tuple[np.ndarray, dict[str, Any]]:
    from drone.features import window

    if length not in (65, 320):
        raise ValueError("Only the two prospective crop lengths are defined")
    rng = np.random.default_rng(draw.augmentation_seed)
    shift = int(rng.integers(-4, 5))
    view = window(feature, shift)
    first = int(rng.integers(0, view.shape[1] - length + 1)) if view.shape[1] >= length else 0
    result = view[:, first : first + length].astype(np.float32)
    actual = result.shape[1]
    result = np.pad(result, ((0, 0), (0, length - actual)))
    return result, {
        "id": draw.identity,
        "domain": draw.domain,
        "augmentation_seed": draw.augmentation_seed,
        "shift": shift,
        "first": first,
        "actual_frames": actual,
        "padding_frames": length - actual,
    }


def cpu_seconds() -> float:
    return sum(
        usage.ru_utime + usage.ru_stime
        for usage in (
            resource.getrusage(resource.RUSAGE_SELF),
            resource.getrusage(resource.RUSAGE_CHILDREN),
        )
    )


def qualify_shape(
    length: int,
    batches: list[list[Draw]],
    array: Callable[[str], np.ndarray],
    selection: list[str],
    rows: dict[str, dict[str, Any]],
    watch: Callable[[], None],
) -> dict[str, Any]:
    import torch

    from drone.features import window
    from drone.model import KeyCNN
    from train import evaluate_features

    if len(batches) != 3 or any(len(batch) != 32 for batch in batches):
        raise ValueError("Exactly three real fitting-draw batches of32 required")
    torch.manual_seed(13)
    model = KeyCNN().to("cpu")
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    if digest.hexdigest() != INITIAL:
        raise ValueError("Original seed13 K10 initialization differs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    updates = []
    witnesses = []
    start = time.monotonic()
    cpu_start = cpu_seconds()
    ordinal = 0
    for batch in batches:
        watch()
        prepared = []
        for draw in batch:
            crop, witness = make_crop(array(draw.identity), draw, length)
            prepared.append(crop)
            witness["synthetic_target"] = ordinal % 24
            witnesses.append(witness)
            ordinal += 1
        inputs = torch.from_numpy(np.stack(prepared))
        targets = torch.tensor([(ordinal - len(batch) + index) % 24 for index in range(len(batch))])
        began = time.monotonic()
        model.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite synthetic loss")
        loss.backward()
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise ValueError("Nonfinite gradient; no optimizer commit")
        optimizer.step()
        if any(not bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values()):
            raise ValueError("Nonfinite optimizer result")
        updates.append(time.monotonic() - began)
        watch()
    training = {"wall_seconds": time.monotonic() - start, "cpu_seconds": cpu_seconds() - cpu_start}
    training["actual_frames"] = sum(witness["actual_frames"] for witness in witnesses)
    training["padding_frames"] = sum(witness["padding_frames"] for witness in witnesses)
    if any(state["step"].item() != 3 for state in optimizer.state.values()):
        raise ValueError("Adam state did not complete exactly three commits")
    validation_started = time.monotonic()
    validation_cpu_start = cpu_seconds()
    features = [window(array(identity)).astype(np.float32) for identity in selection]
    counts = {"tracks": 0, "frames": 0, "forward_batches": 0}

    class FiniteEvaluation(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = model

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            watch()
            output = self.base(value)
            if output.shape != (value.shape[0], 24) or not bool(torch.isfinite(output).all()):
                raise ValueError("Invalid/nonfinite validation logits")
            counts["tracks"] += value.shape[0]
            counts["frames"] += value.shape[0] * value.shape[2]
            counts["forward_batches"] += 1
            return output

    # Original evaluator; its transient argmax result is discarded, never scored/retained.
    evaluate_features(FiniteEvaluation(), features, "cpu", batch=4)
    expected = sum(rows[identity]["frames"] for identity in selection)
    if counts["tracks"] != len(selection) or counts["frames"] != expected:
        raise ValueError("Incomplete true-length selection pass")
    watch()
    return {
        "crop_frames": length,
        "initial_state_sha256": digest.hexdigest(),
        "optimizer_commits": len(updates),
        "batch": 32,
        "synthetic_examples": ordinal,
        "update_compute_seconds": updates,
        "training": training,
        "draws": witnesses,
        "validation": {
            **counts,
            "wall_seconds": time.monotonic() - validation_started,
            "cpu_seconds": cpu_seconds() - validation_cpu_start,
            "finite": True,
            "context": [
                {"id": identity, "frames": rows[identity]["frames"]} for identity in selection
            ],
        },
        "task_scores_computed": False,
        "predictions_retained": False,
        "weights_retained": False,
        "optimizer_state_parameters": len(optimizer.state),
        "cumulative_process_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "rss_scope": "Linux lifetime high-water bound; second arm is not an independent peak",
    }


def worker(config: Path, approval: Path, output: Path, guardian: int) -> None:
    import torch

    from tools.run_k10_context_cost import worker_watch

    value = load(config)
    permit = load(approval)
    contract(value)
    if (
        os.getppid() != guardian
        or permit.get("authorized") is not True
        or permit.get("allow_cost") is not True
        or permit.get("allow_training") is not False
        or permit.get("config_sha256") != sha(config)
    ):
        raise ValueError("Exact private CPU cost-only guardian approval required")
    watch = worker_watch(value, output.parent, guardian, sha(config))
    root = Path(__file__).resolve().parents[1]
    if any(sha(root / name) != expected for name, expected in value["source_sha256"].items()):
        raise ValueError("Bound source changed")
    for name in ("plan", "labels", "features"):
        if sha(Path(value[name])) != value[name + "_sha256"]:
            raise ValueError("Pinned metadata changed")
    if (sys.version.split()[0], str(torch.__version__), np.__version__) != tuple(
        value["runtime"][key] for key in ("python", "torch", "numpy")
    ):
        raise ValueError("Qualified runtime versions differ")
    for name, receipt in value["module_receipts"].items():
        module = importlib.import_module(name)
        if module.__file__ is None or sha(Path(module.__file__)) != receipt["sha256"]:
            raise ValueError("Pinned runtime module shadowed: " + name)
    if sha(Path(sys.executable)) != value["interpreter"]["sha256"] or any(
        sha(Path(path)) != expected for path, expected in value["native_cpu_sha256"].items()
    ):
        raise ValueError("Pinned interpreter or native CPU kernel library changed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only cost must not initialize CUDA")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output.mkdir(exist_ok=False)
    began = time.monotonic()
    rows = {row["id"]: row for row in value["rows"]}
    for row in rows.values():
        watch()
        path = Path(row["cache_path"])
        before = file_identity(path)
        if list(before) != list(row["cache_identity"]) or sha(path) != row["cache_sha256"]:
            raise ValueError("CQT cache content/identity differs")
        if file_identity(path) != before:
            raise ValueError("CQT changed during verification")
    verified = time.monotonic()

    @lru_cache(maxsize=64)
    def array(identity: str) -> np.ndarray:
        watch()
        row = rows[identity]
        path = Path(row["cache_path"])
        if list(file_identity(path)) != list(row["cache_identity"]):
            raise ValueError("Immutable CQT changed")
        cqt = np.load(path, allow_pickle=False, mmap_mode="r")
        if (
            cqt.shape != (208, row["frames"])
            or cqt.dtype != np.float16
            or not np.isfinite(cqt).all()
        ):
            raise ValueError("Invalid CQT")
        if list(file_identity(path)) != list(row["cache_identity"]):
            raise ValueError("CQT changed during finite/shape verification")
        return cqt

    plan = load(Path(value["plan"]))
    batches = list(itertools.islice(paired_batches(plan["bp_fit"], plan["expanded_fma_fit"]), 3))
    selection = plan["bp_selection_guard"] + plan["fma_inner_selection"]
    arms = []
    for length in (320, 65):
        array.cache_clear()
        arms.append(qualify_shape(length, batches, array, selection, rows, watch))
        save(output / f"crop-{length}.json", arms[-1])

    def common(arm: dict[str, Any]) -> list[tuple[Any, ...]]:
        return [
            tuple(
                draw[key]
                for key in ("id", "domain", "augmentation_seed", "shift", "synthetic_target")
            )
            for draw in arm["draws"]
        ]

    if common(arms[0]) != common(arms[1]):
        raise ValueError("Crop arms have different ID/pitch/target streams")
    watch()
    save(
        output / "result.json",
        {
            "status": "COMPLETE_DISPOSABLE_CPU_COST_NO_TASK_FIT",
            "config_sha256": sha(config),
            "cache_verify_seconds": verified - began,
            "wall_seconds": time.monotonic() - began,
            "arms": arms,
            "source_sha256": value["source_sha256"],
            "runtime": value["runtime"],
            "allow_training": False,
            "task_scores_computed": False,
            "weights_retained": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    planning = subparsers.add_parser("plan")
    for name in ("plan", "labels", "features", "protected", "runtime", "output"):
        planning.add_argument("--" + name, type=Path, required=True)
    sealing = subparsers.add_parser("seal")
    for name in ("config", "output"):
        sealing.add_argument("--" + name, type=Path, required=True)
    for name in ("begin", "end"):
        sealing.add_argument("--" + name, required=True)
    run = subparsers.add_parser("worker")
    for name in ("config", "approval", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--guardian", type=int, required=True)
    args = parser.parse_args()
    if args.mode == "plan":
        save(
            args.output,
            make_plan(args.plan, args.labels, args.features, args.protected, args.runtime),
        )
    elif args.mode == "seal":
        from tools.prepare_musicfm_waveforms import seal

        contract(load(args.config))
        save(args.output, seal(args.config, args.begin, args.end))
    else:
        worker(args.config, args.approval, args.output, args.guardian)


if __name__ == "__main__":
    main()
