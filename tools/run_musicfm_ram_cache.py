"""Retained outer RAM-cache guardian: prep, supervised hold, explicit overlapping handoff."""

from __future__ import annotations

import argparse
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tools.musicfm_crop_io import load, save, sha
from tools.prepare_musicfm_ram_cache import (
    FIT_LIMIT,
    capacity_guard,
    cleanup_from_config,
    rounded,
    save_or_replace,
)
from tools.run_cqt_expansion import disk_bytes, process_tree


def proc(pid: int) -> tuple[int, str] | None:
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19]), fields[0]
    except FileNotFoundError:
        return None


def descendants(pid: int) -> dict[int, int]:
    result: dict[int, int] = {}
    pending = [pid]
    while pending:
        current = pending.pop()
        info = proc(current)
        if info is None:
            continue
        result[current] = info[0]
        try:
            text = (Path("/proc") / str(current) / "task" / str(current) / "children").read_text()
            pending.extend(int(value) for value in text.split())
        except FileNotFoundError:
            pass
    return result


def alive(pid: int, start: int) -> bool:
    info = proc(pid)
    return info is not None and info[0] == start and info[1] not in ("Z", "X")


def stop_and_wait(worker: subprocess.Popen[bytes], tracked: dict[int, int]) -> None:
    if worker.poll() is None:
        tracked.update(descendants(worker.pid))
    for sig, seconds in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 2.0)):
        for pid, start in tracked.items():
            if alive(pid, start):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        until = time.monotonic() + seconds
        while any(alive(pid, start) for pid, start in tracked.items()) and time.monotonic() < until:
            time.sleep(0.05)
    worker.wait(timeout=1)
    if any(alive(pid, start) for pid, start in tracked.items()):
        raise RuntimeError("Own worker/decoder still live; refuse cleanup until unmapped")


def handoff(config: dict[str, Any], approval_path: Path) -> dict[str, Any]:
    approval = load(approval_path)
    if (
        approval.get("authorized") is not True
        or approval.get("config_sha256") != config["_executed_config_sha256"]
        or approval.get("cache_token") != config["cache_token"]
        or sha(Path(approval["guardian_source_path"])) != approval["guardian_source_sha256"]
    ):
        raise ValueError("Handoff is not source/config/cache bound")
    pid, start = approval["guardian_pid"], approval["guardian_start_ticks"]
    ready = load(Path(approval["guardian_ready_path"]))
    if (
        not alive(pid, start)
        or ready.get("ready") is not True
        or ready.get("pid") != pid
        or ready.get("start_ticks") != start
        or ready.get("cache_token") != config["cache_token"]
        or ready.get("config_sha256") != config["_executed_config_sha256"]
        or not 0 <= time.time() - ready["last_guard_epoch"] <= 1.0
    ):
        raise RuntimeError("Successor guardian not live/fresh/ready; no unmonitored handoff")
    return {
        "guardian_pid": pid,
        "guardian_start_ticks": start,
        "guardian_source_sha256": approval["guardian_source_sha256"],
        "handoff_approval_sha256": sha(approval_path),
        "ready_receipt": ready,
    }


def supervise(
    config_path: Path, approval_path: Path, output_root: Path, handoff_path: Path | None
) -> None:
    config, approval = load(config_path), load(approval_path)
    if (
        sha(Path(__file__)) != config["ram_supervisor_source_sha256"]
        or approval.get("authorized") is not True
        or approval.get("allow_training") is not False
        or approval.get("config_sha256") != sha(config_path)
    ):
        raise ValueError("Exact preprocessing-only source/config approval required")
    if not config["not_before_epoch"] <= time.time() < config["deadline_epoch"]:
        raise ValueError("Outside approved outer window")
    config["_executed_config_sha256"] = sha(config_path)
    output_root.mkdir(exist_ok=False)
    prepared = output_root / "prepared"
    command = [
        sys.executable,
        "-m",
        "tools.prepare_musicfm_ram_cache",
        "run",
        "--config",
        str(config_path),
        "--approval",
        str(approval_path),
        "--output",
        str(prepared),
    ]
    with (output_root / "worker.log").open("xb") as log:
        worker = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        tracked = descendants(worker.pid)
        save(
            output_root / "launch.json",
            {
                "pid": worker.pid,
                "command": command,
                "source_sha256": sha(Path(__file__)),
                "config_sha256": sha(config_path),
                "utc_epoch": time.time(),
                "initial_tracked_starts": tracked,
            },
        )
        started = time.monotonic()
        last_disk = -1.0
        last_owned = 0
        peak_rss = 0
        peak_cpu = 0.0
        state = "PREPARING"
        outcome: dict[str, Any] = {}
        try:
            while True:
                if worker.poll() is None:
                    tracked.update(descendants(worker.pid))
                capacity_guard(config, output_root)
                if time.time() >= config["deadline_epoch"] or time.monotonic() - started > 7200:
                    raise TimeoutError("Outer prep/hold cutoff; cleanup owned cache")
                if worker.poll() is None:
                    rss, cpu = process_tree(worker.pid)
                    peak_rss, peak_cpu = max(peak_rss, rss), max(peak_cpu, cpu)
                    if rss > 2 * 1024**3 or cpu > 6000:
                        raise RuntimeError("Outer sampled worker resource cap")
                if time.monotonic() - last_disk >= 1:
                    last_disk = time.monotonic()
                    last_owned = disk_bytes(output_root)
                    cache = Path(config["ram_root"])
                    if cache.exists():
                        total = sum(
                            rounded(p.lstat().st_size, os.sysconf("SC_PAGE_SIZE"))
                            for p in cache.iterdir()
                        )
                        if total > FIT_LIMIT:
                            raise RuntimeError("Outer owned tmpfs cap")
                    if last_owned > 8 * 1024**3:
                        raise RuntimeError("Outer ordinary-output cap")
                    save_or_replace(
                        output_root / "guardian-progress.json",
                        {
                            "state": state,
                            "epoch": time.time(),
                            "sampled_peak_worker_rss": peak_rss,
                            "sampled_peak_worker_cpu": peak_cpu,
                            "ordinary_owned_bytes": last_owned,
                        },
                    )
                if worker.poll() is not None:
                    worker.wait()
                    if worker.returncode != 0:
                        raise RuntimeError(f"Preparation worker exit={worker.returncode}")
                    if state == "PREPARING":
                        manifest = load(prepared / "manifest.json")
                        if (
                            manifest.get("complete") is not True
                            or manifest.get("allow_training") is not False
                        ):
                            raise ValueError("No complete preprocessing-only cache")
                        state = "HOLDING_CONTINUOUS_WATCH"
                        save(
                            output_root / "prepared-held.json",
                            {
                                "complete": True,
                                "allow_training": False,
                                "config_sha256": sha(config_path),
                                "epoch": time.time(),
                            },
                        )
                    if handoff_path is not None and handoff_path.exists():
                        outcome = handoff(config, handoff_path)
                        capacity_guard(config, output_root)
                        save(
                            output_root / "handoff-ack.json",
                            {"status": "SUCCESSOR_WATCH_LIVE", "epoch": time.time(), **outcome},
                        )
                        state = "HANDED_OFF"
                        break
                time.sleep(0.05)
        except BaseException as error:
            state = "FAILED_OR_EXPIRED_CLEANUP"
            outcome = {"error": repr(error)}
            stop_and_wait(worker, tracked)
            if (prepared / "cache-ownership.json").exists() and Path(config["ram_root"]).exists():
                cleanup_from_config(config, prepared)
            raise
        finally:
            save(
                output_root / "controller.json",
                {
                    "state": state,
                    "worker_exit": worker.returncode,
                    "wall_seconds": time.monotonic() - started,
                    "sampled_peak_worker_rss": peak_rss,
                    "sampled_peak_worker_cpu": peak_cpu,
                    "reaped_child_cpu": resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
                    + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
                    **outcome,
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "approval", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--handoff-approval", type=Path)
    args = parser.parse_args()
    signal.signal(
        signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError("Supervisor terminated"))
    )
    supervise(args.config, args.approval, args.output_root, args.handoff_approval)


if __name__ == "__main__":
    main()
