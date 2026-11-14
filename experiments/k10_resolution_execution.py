"""Hash-bound prospective GPU accuracy pair; separate fresh workers, no implicit permit."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from experiments.k10_context_cost import SOURCE_NAMES
from experiments.k10_gpu1_resource import (
    BACKEND_POLICY,
    GPU0,
    GPU1,
    POLICY,
)
from experiments.k10_gpu1_resource import (
    sources as probe_sources,
)
from experiments.k10_resolution_fit import MODELS, STATUS, arm, verify_data
from experiments.k10_v2_pilot import validate_feature_manifest
from tools.musicfm_crop_io import file_identity, load, save, sha

CAPS = {
    "wall_seconds": 1800,
    "cpu_seconds": 3600,
    "rss_bytes": 8 * 1024**3,
    "output_bytes": 512 * 1024**2,
}
QUALIFIED = {
    "sealed-config.json": "132978eae5395599d7e9ed0da0a12ef62ad7fd5f13646bb06d9147076577728c",
    "run-v1/controller.json": "c8c921ce1b48cc8462d4184aea93ebaded6bc033c723cbac4e3d2751335207f6",
    "run-v1/baseline/result.json": "57f57a9c6bb9c64cd961b71b4cff6a2e3c212446e0bf819e05dbafbc837191c8",
    "run-v1/candidate/result.json": "a59c1d1b666ef07bf9a568322b5cfee2fb0675eccd823657c9b755bb28f08324",
    "terminal-receipt.json": "8d81eec26afc20224df152087d51a97a452428d41da72e79d0593263c0fdfcbf",
}
ROOT = Path(__file__).resolve().parents[1]


def sources() -> dict[str, str]:
    names = {
        *SOURCE_NAMES,
        *probe_sources(),
        "experiments/k10_context_pair.py",
        "experiments/k10_resolution_fit.py",
        "experiments/k10_resolution_execution.py",
        "tools/run_k10_resolution_fit.py",
        "tools/run_k10_context_pair.py",
        "experiments/studies/k10-capacity/RESOLUTION_GPU_PAIR_PROTOCOL.md",
    }
    return {name: sha(ROOT / name) for name in sorted(names)}


def qualification(root: Path) -> dict[str, Any]:
    for name, expected in QUALIFIED.items():
        if sha(root / name) != expected:
            raise ValueError("Exact v3 qualification evidence changed: " + name)
    result = load(root / "run-v1/controller.json")
    if result.get("complete") is not True or result.get("cleanup_error") is not None:
        raise ValueError("V3 qualification incomplete")
    for name, initial in MODELS.items():
        arm_result = load(root / f"run-v1/{name}/result.json")
        if (
            not arm_result["complete"]
            or arm_result["initial_state_sha256"] != initial
            or arm_result["task_fit"]
            or arm_result["predictions_retained"]
            or arm_result["evaluation"]["selection376"]["tracks"] != 376
        ):
            raise ValueError("Qualification numerical/scope mismatch")
    return load(root / "sealed-config.json")


def prepare(
    qualified: Path, data_config: Path, not_before: float, deadline: float
) -> dict[str, Any]:
    sealed = qualification(qualified)
    data = load(data_config)
    value = {
        **data,
        "status": "PROPOSED_FRESH_GPU_RESOLUTION_PAIR",
        "caps": CAPS.copy(),
        "qualification_root": str(qualified),
        "qualification_sha256": QUALIFIED.copy(),
        "runtime": sealed["runtime"],
        "interpreter": sealed["interpreter"],
        "protected": sealed["protected"],
        "gpu_uuid": GPU1,
        "gpu0_uuid": GPU0,
        "policy": POLICY.copy(),
        "backend_policy": BACKEND_POLICY.copy(),
        "not_before_epoch": not_before,
        "deadline_epoch": deadline,
        "source_sha256": sources(),
        "model_order": ["baseline", "candidate"],
        "updates_per_model": 8280,
        "selection_events": 120,
        "crop_frames": 320,
        "launch_authorized": False,
    }
    contract(value)
    return value


def contract(value: dict[str, Any]) -> None:
    if (
        value.get("caps") != CAPS
        or value.get("backend_policy") != BACKEND_POLICY
        or value.get("policy") != POLICY
        or value.get("gpu_uuid") != GPU1
        or value.get("gpu0_uuid") != GPU0
        or value.get("model_order") != ["baseline", "candidate"]
        or value.get("updates_per_model") != 8280
        or value.get("selection_events") != 120
        or value.get("crop_frames") != 320
        or value.get("qualification_sha256") != QUALIFIED
        or not 0 < value["deadline_epoch"] - value["not_before_epoch"] <= CAPS["wall_seconds"]
    ):
        raise ValueError("Frozen GPU accuracy-pair contract differs")
    services = value["protected"]["protected_starts"]
    jobs = value["protected"]["active_job_starts"]
    if (
        {start["pid"] for start in services} != {1478814, 3448680, 1481756, 3451013}
        or len(services) != 4
        or {start["pid"] for start in jobs} != {1294, 1307, 3814436, 3814440}
        or len(jobs) != 4
    ):
        raise ValueError("Require exact eight protected starts")


def bindings(value: dict[str, Any]) -> None:
    contract(value)
    sealed = qualification(Path(value["qualification_root"]))
    if (
        value["policy"] != sealed["policy"]
        or value["protected"] != sealed["protected"]
        or any(sha(ROOT / name) != expected for name, expected in sealed["source_sha256"].items())
    ):
        raise ValueError("Qualified shared source/policy/protected identities changed")
    for name in ("historical_gpu_control", "groups"):
        if sha(Path(value[name])) != value[name + "_sha256"]:
            raise ValueError("Bound historical/group evidence changed")
    plan = load(Path(value["plan"]))
    if value["groups_sha256"] != plan["uncertainty_groups_sha256"]:
        raise ValueError("Frozen uncertainty groups differ")
    if value["runtime"] != sealed["runtime"] or value["interpreter"] != sealed["interpreter"]:
        raise ValueError("Qualified runtime changed")
    if (
        value["source_sha256"] != sources()
        or sha(Path(sys.executable)) != value["interpreter"]["sha256"]
    ):
        raise ValueError("Source/interpreter changed")
    if str(Path(sys.executable).resolve()) != str(Path(value["interpreter"]["path"]).resolve()):
        raise ValueError("Interpreter path differs")
    for pinned in [*value["runtime"]["modules"].values(), *value["runtime"]["files"]]:
        if sha(Path(pinned["path"])) != pinned["sha256"]:
            raise ValueError("Pinned runtime changed")
    sensor = value["runtime"]["nvidia_smi"]
    if sha(Path(sensor["path"])) != sensor["sha256"]:
        raise ValueError("Pinned sensor changed")


def worker(config: Path, approval: Path, output: Path, name: str, guardian: int) -> None:
    from tools.run_k10_gpu1_resource import environment, start_identity

    value = load(config)
    permit = load(approval)
    digest = sha(config)  # Cached once: never hash the large configuration in watch().
    bindings(value)
    if (
        name not in MODELS
        or permit.get("authorized") is not True
        or permit.get("allow_training") is not True
        or permit.get("allow_sequential_pair") is not True
        or permit.get("config_sha256") != digest
        or permit.get("deadline_epoch") != value["deadline_epoch"]
        or os.getppid() != guardian
    ):
        raise ValueError("Exact root-approved pair and owning guardian required")
    environment(value)
    for module_name, receipt in value["runtime"]["modules"].items():
        module = importlib.import_module(module_name)
        if (
            module.__file__ is None
            or module.__file__ != receipt["path"]
            or sha(Path(module.__file__)) != receipt["sha256"]
        ):
            raise ValueError("Shadowed runtime module")
    guardian_start = start_identity(guardian)
    last = 0.0

    def watch() -> None:
        nonlocal last
        now = time.time()
        if now >= value["deadline_epoch"]:
            raise TimeoutError("Fixed UTC deadline")
        if now - last < 0.05:
            return
        ready = load(output.parent / "ready.json")
        now = time.time()
        if (
            ready["guardian"] != guardian
            or ready["start_ticks"] != guardian_start
            or start_identity(guardian) != guardian_start
            or ready["config_sha256"] != digest
            or not 0 <= now - ready["epoch"] <= 0.6
        ):
            raise RuntimeError("Guardian identity/heartbeat stale")
        last = now

    rows = verify_data(value, watch)
    plan = load(Path(value["plan"]))
    validate_feature_manifest(plan, load(Path(value["features"])))

    @lru_cache(maxsize=64)
    def array(identity: str) -> np.ndarray:
        watch()
        row = rows[identity]
        path = Path(row["cache_path"])
        if list(file_identity(path)) != list(row["cache_identity"]):
            raise ValueError("CQT stat changed")
        cqt = np.load(path, allow_pickle=False, mmap_mode="r")
        if (
            cqt.shape != (208, row["frames"])
            or cqt.dtype != np.float16
            or not np.isfinite(cqt).all()
            or list(file_identity(path)) != list(row["cache_identity"])
        ):
            raise ValueError("CQT tensor invalid/changed")
        return cqt

    provenance = {
        "config_sha256": digest,
        "source_sha256": value["source_sha256"],
        "plan_sha256": value["plan_sha256"],
        "labels_sha256": value["labels_sha256"],
        "features_sha256": value["features_sha256"],
        "runtime": value["runtime"],
        "threads": 2,
        "seed": 13,
        "qualification_sha256": QUALIFIED,
    }
    result = arm(name, plan, rows, array, output, provenance, watch)
    if result["status"] != STATUS:
        raise ValueError("Incomplete selected-reload proof")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    seal = subparsers.add_parser("prepare")
    for name in ("qualification-root", "data-config", "output"):
        seal.add_argument("--" + name, type=Path, required=True)
    seal.add_argument("--not-before", type=float, required=True)
    seal.add_argument("--deadline", type=float, required=True)
    work = subparsers.add_parser("worker")
    for name in ("config", "approval", "output"):
        work.add_argument("--" + name, type=Path, required=True)
    work.add_argument("--model", choices=tuple(MODELS), required=True)
    work.add_argument("--guardian", type=int, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        save(
            args.output,
            prepare(args.qualification_root, args.data_config, args.not_before, args.deadline),
        )
    else:
        worker(args.config, args.approval, args.output, args.model, args.guardian)


if __name__ == "__main__":
    main()
