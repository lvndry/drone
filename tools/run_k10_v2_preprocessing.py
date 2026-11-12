"""Linux outer process-group guard for the reviewed first16 preprocessing batch."""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tools.prepare_k10_v2_pilot import digest, read_json


def proc_fields(text: str, ticks: int, pages: int) -> tuple[int, float, int, str]:
    tail = text.rsplit(")", 1)[1].split()
    group = int(tail[2])
    cpu = sum(max(0, int(tail[i])) for i in (11, 12, 13, 14)) / ticks
    rss = max(0, int(tail[21])) * pages
    return group, cpu, rss, tail[0]


def usage(group: int) -> dict[str, Any]:
    cpu = 0.0
    rss = 0
    pids = []
    ticks = os.sysconf("SC_CLK_TCK")
    pages = os.sysconf("SC_PAGE_SIZE")
    for path in sorted(
        Path("/proc").iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1
    ):
        if not path.name.isdigit():
            continue
        try:
            g, c, r, state = proc_fields((path / "stat").read_text(), ticks, pages)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if g == group:
            cpu += c
            rss += r
            if state != "Z":
                pids.append(int(path.name))
    return {"cpu_seconds": cpu, "rss_bytes": rss, "live_pids": pids}


def storage(root: Path) -> tuple[int, int]:
    cache = 0
    wav = 0
    if root.exists():
        for path in root.rglob("*"):
            try:
                if path.is_file():
                    if path.suffix == ".npy":
                        cache += path.stat().st_size
                    if path.suffix == ".wav":
                        wav += path.stat().st_size
            except FileNotFoundError:
                continue
    return cache, wav


def violation(sample: dict[str, Any], caps: dict[str, Any]) -> str | None:
    for name, limit in (
        ("wall_seconds", "wall_seconds"),
        ("cpu_seconds", "cpu_seconds"),
        ("rss_bytes", "aggregate_rss_bytes"),
        ("cache_bytes", "new_cache_bytes"),
        ("wav_bytes", "wav_scratch_bytes"),
    ):
        if sample[name] > caps[limit]:
            return name
    return None


def terminate(group: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            return
        if sig == signal.SIGTERM:
            time.sleep(1)


def main() -> None:
    if not Path("/proc/self/stat").is_file():
        raise RuntimeError("Linux/proc required;no job launched")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "batch", "permit", "reuse-manifest", "fma-audio", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    batch = read_json(args.batch)
    permit = read_json(args.permit)
    if (
        permit.get("allow_preprocessing") is not True
        or permit.get("plan_sha256") != digest(args.plan)
        or permit.get("batch_sha256") != digest(args.batch)
    ):
        raise ValueError("Hash-bound root preprocessing permit required")
    if len(batch["first_batch"]) != 16 or args.output.exists():
        raise ValueError("Fresh first16 stage required")
    caps = batch["first_batch_caps"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "taskset",
        "-c",
        ",".join(map(str, caps["cpus"])),
        sys.executable,
        "-m",
        "tools.prepare_k10_v2_pilot",
    ]
    for name in ("plan", "batch", "permit", "reuse_manifest", "fma_audio", "output"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
    )
    log = args.output.with_name(args.output.name + ".log")
    report = args.output.with_name(args.output.name + ".controller.json")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    reason = None
    peak = {"cpu_seconds": 0.0, "rss_bytes": 0, "cache_bytes": 0, "wav_bytes": 0}
    with log.open("wb") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        group = process.pid
        report.write_text(
            json.dumps(
                {
                    "status": "running",
                    "pid": process.pid,
                    "pgid": group,
                    "log": str(log),
                    "command": command,
                    "caps": caps,
                },
                indent=2,
            )
            + "\n"
        )
        try:
            while True:
                sample = usage(group)
                cache, wav = storage(args.output)
                sample.update(
                    wall_seconds=time.monotonic() - start, cache_bytes=cache, wav_bytes=wav
                )
                for k in peak:
                    peak[k] = max(peak[k], sample[k])
                reason = violation(sample, caps)
                if reason is not None:
                    terminate(group)
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.2)
        finally:
            if process.poll() is None or usage(group)["live_pids"]:
                terminate(group)
            code = process.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    peak["cpu_seconds"] = max(peak["cpu_seconds"], cpu)
    cache, wav = storage(args.output)
    peak["cache_bytes"] = max(peak["cache_bytes"], cache)
    peak["wav_bytes"] = max(peak["wav_bytes"], wav)
    peak["rss_bytes"] = max(peak["rss_bytes"], after.ru_maxrss * 1024)
    final = {**peak, "wall_seconds": time.monotonic() - start}
    reason = reason or violation(final, caps)
    report.write_text(
        json.dumps(
            {
                "status": "bounded_batch_complete"
                if code == 0 and reason is None
                else "FAILED_NOT_TRAINABLE",
                "pid": process.pid,
                "pgid": group,
                "exit": code,
                "cap_violation": reason,
                "log": str(log),
                "measured": final,
                "caps": caps,
                "poll_seconds": 0.2,
                "rss_semantics": "maximum sampled sum of process-group RSS;shared pages counted perprocess",
                "cpu_semantics": "process-group own+reaped-child CPU;terminalwait resource usage included",
                "source_sha256": digest(Path(__file__)),
                "plan_sha256": digest(args.plan),
                "batch_sha256": digest(args.batch),
                "training_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if code or reason is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
