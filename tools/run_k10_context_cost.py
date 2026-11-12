"""Finite CPU cost-only guardian; shared thread-aware resource and cleanup helpers."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from tools.musicfm_crop_io import load, save, sha
from tools.prepare_musicfm_ram_cache import capacity_guard, save_or_replace
from tools.run_musicfm_ram_cache import alive, descendants, proc, stop_and_wait
from tools.run_musicfm_teacher import owned_resources, resources


def disk_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except FileNotFoundError:
            continue  # Ready-heartbeat atomic rename may finish during this walk.
    return total


def environment(value: dict[str, Any]) -> None:
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is None or affinity(0) != set(value["cpus"]):
        raise ValueError("Exact CPU14-15 affinity required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise ValueError("CPU-only CUDA mask and nice19 required")
    if any(
        os.environ.get(k) != "1"
        for k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    ):
        raise ValueError("Numeric environment threads must remain1")
    cores = [
        int(Path(f"/sys/devices/system/cpu/cpu{x}/topology/core_id").read_text())
        for x in value["cpus"]
    ]
    if len(set(cores)) != 2:
        raise ValueError("Require two distinct physical cores")


def worker_watch(
    value: dict[str, Any], root: Path, guardian: int, digest: str
) -> Callable[[], None]:
    last = [0.0]

    def check() -> None:
        if time.monotonic() - last[0] < 0.2:
            return
        last[0] = time.monotonic()
        ready = load(root / "ready.json")
        if (
            ready["pid"] != guardian
            or not alive(guardian, ready["start_ticks"])
            or ready["config_sha256"] != digest
            or not 0 <= time.time() - ready["epoch"] <= 10
        ):
            raise RuntimeError("No fresh identity-bound CPU cost guardian")
        if time.time() >= value["deadline_epoch"]:
            raise TimeoutError("Frozen CPU cost cutoff")
        capacity_guard(value, root)

    return check


def supervise(config: Path, approval: Path, output: Path) -> None:
    from experiments.k10_context_cost import contract

    value, permit = load(config), load(approval)
    digest = sha(config)
    if (
        permit.get("authorized") is not True
        or permit.get("allow_cost") is not True
        or permit.get("allow_training") is not False
        or permit.get("config_sha256") != digest
    ):
        raise ValueError("Exact cost-only approval required; no task training allowed")
    contract(value)
    if not value["not_before_epoch"] <= time.time() < value["deadline_epoch"]:
        raise ValueError("Outside root-sealed CPU cost window")
    if value["deadline_epoch"] - value["not_before_epoch"] > value["caps"]["wall_seconds"]:
        raise ValueError("Window exceeds reviewed wall cap")
    root = Path(__file__).resolve().parents[1]
    if any(sha(root / n) != expected for n, expected in value["source_sha256"].items()):
        raise ValueError("Reviewed source differs")
    environment(value)
    output.mkdir(exist_ok=False)
    identity = proc(os.getpid())
    if identity is None:
        raise RuntimeError("Missing own start identity")
    start = time.monotonic()
    peaks: dict[str, float] = {"rss": 0.0, "cpu": 0.0}
    child: subprocess.Popen[bytes] | None = None
    tracked: dict[int, int] = {}
    error = None
    last_gpu = last_disk = 0.0
    complete = False
    try:
        resources(value, output)
        owned_resources(value, peaks)
        save_or_replace(
            output / "ready.json",
            {
                "pid": os.getpid(),
                "start_ticks": identity[0],
                "config_sha256": digest,
                "epoch": time.time(),
            },
        )
        command = [
            sys.executable,
            "-m",
            "experiments.k10_context_cost",
            "worker",
            "--config",
            str(config),
            "--approval",
            str(approval),
            "--output",
            str(output / "worker"),
            "--guardian",
            str(os.getpid()),
        ]
        with (output / "worker.log").open("xb") as log:
            child = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        tracked = descendants(child.pid)
        save(
            output / "launch.json",
            {
                "command": command,
                "worker": child.pid,
                "guardian": os.getpid(),
                "epoch": time.time(),
                "config_sha256": digest,
            },
        )
        while True:
            owned_resources(value, peaks)
            capacity_guard(value, output)
            if time.monotonic() - last_gpu >= 1:
                resources(value, output)
                last_gpu = time.monotonic()
                owned_resources(value, peaks)
            if (
                time.time() >= value["deadline_epoch"]
                or time.monotonic() - start > value["caps"]["wall_seconds"]
            ):
                raise TimeoutError("Frozen UTC/wall cap")
            if time.monotonic() - last_disk >= 1:
                if disk_bytes(output) > value["caps"]["output_bytes"]:
                    raise RuntimeError("Owned output cap")
                last_disk = time.monotonic()
            save_or_replace(
                output / "ready.json",
                {
                    "pid": os.getpid(),
                    "start_ticks": identity[0],
                    "config_sha256": digest,
                    "epoch": time.time(),
                },
            )
            tracked.update(descendants(child.pid))
            if child.poll() is not None:
                child.wait()
                owned_resources(value, peaks)
                if child.returncode != 0:
                    raise RuntimeError(f"Cost worker exit={child.returncode}")
                result = load(output / "worker/result.json")
                if result.get("status") != "COMPLETE_DISPOSABLE_CPU_COST_NO_TASK_FIT":
                    raise ValueError("No complete cost result")
                resources(value, output)
                if disk_bytes(output) > value["caps"]["output_bytes"]:
                    raise RuntimeError("Final owned output cap")
                complete = True
                break
            time.sleep(0.05)
    except BaseException as exc:
        error = repr(exc)
        if child is not None:
            stop_and_wait(child, tracked)
        raise
    finally:
        save(
            output / "controller.json",
            {
                "complete": complete,
                "error": error,
                "worker_exit": child.returncode if child else None,
                "peaks": peaks,
                "terminal_resources": owned_resources(value, peaks, enforce=False),
                "resource_scope": "guardian plus all thread-spawned live/reaped descendants",
                "wall_seconds": time.monotonic() - start,
                "allow_training": False,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "approval", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    signal.signal(
        signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError("Guardian stopped"))
    )
    supervise(args.config, args.approval, args.output_root)


if __name__ == "__main__":
    main()
