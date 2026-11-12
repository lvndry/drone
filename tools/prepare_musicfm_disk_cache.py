"""Prospective ordinary-disk full waveform cache; exact canonical identities, no model."""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tools.musicfm_crop_io import decode_command, file_identity, load, runtime_files, save, sha
from tools.musicfm_source_contract import apply_identity_overlay
from tools.prepare_musicfm_ram_cache import capacity_guard
from tools.prepare_musicfm_waveforms import Preparation, seal

CANONICAL = "7d56908dc5e84c165c7844d1e810ed678123ae7d0a93939c60a2e600b6402b4c"
OVERLAY = "6fc26efcb1dea503ef47177698e225eb6a0da673606772e1d6d1f29f717ee737"
LIMIT = 100_000_000_000


def config(canonical: Path, overlay: Path, reference: Path) -> dict[str, Any]:
    if sha(canonical) != CANONICAL or sha(overlay) != OVERLAY:
        raise ValueError("Wrong canonical input or audited physical overlay")
    original = load(canonical)
    prior = load(reference)
    if sha(reference) != original["config_sha256"]:
        raise ValueError("Different full decoder reference")
    rows = apply_identity_overlay({r["id"]: r for r in original["rows"]}, load(overlay), CANONICAL)
    if len(rows) != 5638 or sum(r["role"] == "fit" for r in rows.values()) != 5262:
        raise ValueError("Wrong approved fitting/selection membership")
    families: dict[str, tuple[str, int]] = {}
    for row in rows.values():
        key = row["waveform_sha256"]
        value = (row["role"], row["target"])
        if key in families and families[key] != value:
            raise ValueError("Conflicting/cross-role canonical family")
        families[key] = value
    root = Path(__file__).resolve().parents[1]
    names = [
        "tools/prepare_musicfm_disk_cache.py",
        "tools/musicfm_source_contract.py",
        "tools/musicfm_crop_io.py",
        "tools/prepare_musicfm_waveforms.py",
        "tools/run_cqt_expansion.py",
        "tools/prepare_musicfm_ram_cache.py",
        "tools/run_k10_v2_pair.py",
    ]
    return {
        "status": "PROPOSED_DISK_PREPARATION_NO_EXECUTION",
        "canonical": str(canonical),
        "canonical_sha256": CANONICAL,
        "overlay": str(overlay),
        "overlay_sha256": OVERLAY,
        "reference": str(reference),
        "reference_sha256": sha(reference),
        "ffmpeg": prior["ffmpeg"],
        "runtime_file_sha256": prior["runtime_file_sha256"],
        "source_sha256": {name: sha(root / name) for name in names},
        "protected_starts": prior["protected_starts"],
        "cpus": [10, 11, 12, 13],
        "workers": 4,
        "rows": list(rows.values()),
        "not_before_epoch": None,
        "deadline_epoch": None,
        "caps": {
            "cpu_seconds": 6000,
            "wall_seconds": 7200,
            "rss_bytes": 2 * 1024**3,
            "owned_bytes": LIMIT,
            "reference_bytes": 256 * 1024**2,
        },
        "no_retry_resume_or_training": True,
    }


class DiskPreparation(Preparation):
    def __init__(self, value: dict[str, Any], output: Path) -> None:
        super().__init__(value, output)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.slots = iter(value["cpus"])
        self.local = threading.local()
        self.validation_verified = 0

    def disk_usage(self, force: bool = False) -> int:
        # Atomic progress-file renames may race a sampled directory walk.
        if force or time.monotonic() - self.last_disk_time >= 1.0:
            total = 0
            for path in self.output.rglob("*"):
                try:
                    if path.is_file() and not path.is_symlink():
                        total += path.stat().st_size
                except FileNotFoundError:
                    continue
            self.last_disk_bytes = total
            self.last_disk_time = time.monotonic()
        return self.last_disk_bytes

    def progress(self, complete: bool) -> dict[str, Any]:
        return {
            **super().progress(complete),
            "validation_reused_verified": self.validation_verified,
            "cpu_scope": "Aggregate process+reaped children; concurrent per-file deltas are not exclusive",
        }

    def initialize(self) -> None:
        with self.lock:
            self.local.cpu = next(self.slots)

    def guard(self, force_disk: bool = False) -> None:
        if self.stop.is_set():
            raise RuntimeError("Disk preparation stopped")
        with self.lock:
            super().guard(force_disk)
            capacity_guard(self.config, self.output)

    def digest(
        self, path: Path, expected_sha: str, expected_bytes: int, finite: bool = False
    ) -> None:
        import numpy as np

        value = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024**2), b""):
                self.guard()
                value.update(block)
                size += len(block)
                if finite and (
                    len(block) % 4 or not np.isfinite(np.frombuffer(block, dtype="<f4")).all()
                ):
                    raise ValueError("Nonfinite/misaligned full waveform")
        if size != expected_bytes or value.hexdigest() != expected_sha:
            raise ValueError("Canonical byte/hash mismatch: " + str(path))

    def one(self, row: dict[str, Any], ordinal: int) -> dict[str, Any]:
        source = Path(row["path"])
        initial = file_identity(source)
        if list(initial) != list(row["source_stable_identity"]):
            raise ValueError("Physical source identity changed")
        self.digest(source, row["sha256"], initial[2])
        target = self.output / "fit" / f"wave-{ordinal:04d}.f32le"
        command = [
            "taskset",
            "-c",
            str(self.local.cpu),
            *decode_command(Path(self.config["ffmpeg"]), source, None, 288000),
        ]
        execution = self.decode(
            command,
            target,
            self.output / "logs" / f"{ordinal:04d}-full.stderr",
            min(row["bytes"], self.config["caps"]["reference_bytes"]),
        )
        execution["overlapping_all_child_cpu_delta_seconds"] = execution.pop("child_cpu_seconds")
        execution["cpu_accounting"] = "All concurrently reaped children; not exclusive per-file CPU"
        self.digest(target, row["waveform_sha256"], row["bytes"], True)
        if file_identity(source) != initial:
            raise ValueError("Source changed during full decode")
        target.chmod(0o444)
        result = {
            **row,
            "cache": str(target),
            "cache_identity": file_identity(target),
            "disk_preparation_execution": execution,
        }
        save(self.output / "records" / f"{ordinal:04d}.json", result)
        return result

    def validation(self, row: dict[str, Any]) -> None:
        path = Path(row["cache"])
        before = file_identity(path)
        if list(before) != list(row["cache_identity"]) or path.stat().st_mode & 0o222:
            raise ValueError("Validation cache changed/not readonly")
        self.digest(path, row["waveform_sha256"], row["bytes"])
        if file_identity(path) != before:
            raise ValueError("Validation changed while verifying")
        self.validation_verified += 1


def execute(config_path: Path, approval_path: Path, output: Path) -> None:
    value, approval = load(config_path), load(approval_path)
    if (
        approval.get("authorized") is not True
        or approval.get("allow_training") is not False
        or approval.get("config_sha256") != sha(config_path)
    ):
        raise ValueError("Exact preprocessing-only root approval required")
    root = Path(__file__).resolve().parents[1]
    if any(sha(root / name) != expected for name, expected in value["source_sha256"].items()):
        raise ValueError("Reviewed source changed")
    for path, key in [
        ("canonical", "canonical_sha256"),
        ("overlay", "overlay_sha256"),
        ("reference", "reference_sha256"),
    ]:
        if sha(Path(value[path])) != value[key]:
            raise ValueError("Pinned metadata changed")
    if runtime_files(Path(value["ffmpeg"])) != value["runtime_file_sha256"]:
        raise ValueError("Pinned full decoder runtime changed")
    affinity = getattr(os, "sched_getaffinity", None)
    if (
        affinity is None
        or affinity(0) != set(value["cpus"])
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise ValueError("Require exact CPU set and CUDA masked")
    if any(
        os.environ.get(k) != "1"
        for k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    ):
        raise ValueError("Require numeric threads1")
    if not value["not_before_epoch"] <= time.time() < value["deadline_epoch"]:
        raise ValueError("Outside approved finite window")
    output.mkdir(exist_ok=False)
    if output.stat().st_dev == Path("/dev/shm").stat().st_dev:
        raise ValueError("Ordinary disk required, not tmpfs")
    for name in ("fit", "records", "logs"):
        (output / name).mkdir()
    fitting = [r for r in value["rows"] if r["role"] == "fit"]
    validation = [r for r in value["rows"] if r["role"] == "validation"]
    if (len(fitting), len(validation)) != (5262, 376):
        raise ValueError("Different fixed pools")
    forecast = sum(r["bytes"] for r in fitting)
    if forecast + 64 * 1024**2 > LIMIT:
        raise ValueError("Whole fitting cache+metadata forecast exceeds100GB")
    stats = os.statvfs(output)
    if stats.f_bavail * stats.f_frsize < forecast + 64 * 1024**2 + 32 * 1024**3:
        raise ValueError("Insufficient disk forecast/reserve")
    stage = DiskPreparation(value, output)
    stage.guard()

    def terminated(signum: int, frame: object) -> None:
        stage.stop.set()
        raise TimeoutError("Preparation terminated")

    signal.signal(signal.SIGTERM, terminated)
    pool = ThreadPoolExecutor(max_workers=4, initializer=stage.initialize)
    results = {}
    try:
        for row in validation:
            stage.validation(row)
        futures = {pool.submit(stage.one, row, i): row["id"] for i, row in enumerate(fitting, 1)}
        for future in as_completed(futures):
            result = future.result()
            results[result["id"]] = result
            stage.rows.append(result)
            from tools.run_cqt_expansion import write_json

            write_json(output / "progress.json", stage.progress(False))
        pool.shutdown(wait=True)
        stage.guard(True)
        rows = [results[r["id"]] if r["role"] == "fit" else r for r in value["rows"]]
        if len(results) != 5262:
            raise ValueError("Incomplete fitting cache")
        save(
            output / "manifest.json",
            {
                "complete": True,
                "allow_training": False,
                "provider": "readonly-disk-full-f32",
                "config_sha256": sha(config_path),
                "canonical_manifest_sha256": CANONICAL,
                "physical_overlay_sha256": OVERLAY,
                "plan_sha256": load(Path(value["canonical"]))["plan_sha256"],
                "labels_sha256": load(Path(value["canonical"]))["labels_sha256"],
                "rows": rows,
                "resources": stage.progress(True),
                "fit_bytes": sum(r["bytes"] for r in fitting),
                "validation_reused": 376,
                "no_seek_quantization_role_or_label_changes": True,
            },
        )
        stage.guard(True)
    except BaseException as error:
        stage.stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        save(
            output / "failure.json",
            {
                "complete": False,
                "error": repr(error),
                "fit_completed": len(results),
                "resources": stage.progress(False),
                "no_resume_or_repair": True,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    plan = commands.add_parser("plan")
    for name in ("canonical", "overlay", "reference", "output"):
        plan.add_argument("--" + name, type=Path, required=True)
    sealing = commands.add_parser("seal")
    for name in ("config", "output"):
        sealing.add_argument("--" + name, type=Path, required=True)
    for name in ("begin", "end"):
        sealing.add_argument("--" + name, required=True)
    run = commands.add_parser("run")
    for name in ("config", "approval", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "plan":
        save(args.output, config(args.canonical, args.overlay, args.reference))
    elif args.mode == "seal":
        save(args.output, seal(args.config, args.begin, args.end))
    else:
        execute(args.config, args.approval, args.output)


if __name__ == "__main__":
    main()
