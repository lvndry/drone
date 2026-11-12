"""Outer Linux process-group guard for normal full-pilot preprocessing only."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tools.prepare_k10_v2_pilot import digest, read_json
from tools.run_k10_v2_preprocessing import storage, terminate, usage, violation


def owned_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def bulk_violation(sample: dict[str, Any], caps: dict[str, Any]) -> str | None:
    reason = violation(sample, caps)
    if reason is not None:
        return reason
    return (
        "total_owned_bytes"
        if sample["total_owned_bytes"] > caps["total_owned_output_bytes"]
        else None
    )


def main() -> None:
    if not Path("/proc/self/stat").is_file():
        raise RuntimeError("Linux/proc required;no job launched")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plan",
        "labels",
        "config",
        "permit",
        "reuse-manifest",
        "first16-manifest",
        "first16-controller",
        "fma-audio",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    permit = read_json(args.permit)
    if (
        permit.get("allow_full_pilot_preprocessing") is not True
        or permit.get("allow_training") is not False
        or permit.get("config_sha256") != digest(args.config)
        or permit.get("plan_sha256") != digest(args.plan)
    ):
        raise ValueError("Hash-bound bulk-only root permit required")
    if config["guard_source_sha256"] != digest(Path(__file__)):
        raise ValueError("Guard source changed")
    if args.output.exists():
        raise ValueError("Fresh bulk output required")
    caps = config["caps"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "taskset",
        "-c",
        ",".join(map(str, caps["cpus"])),
        sys.executable,
        "-m",
        "tools.prepare_k10_v2_bulk",
    ]
    for name in (
        "plan",
        "labels",
        "config",
        "permit",
        "reuse_manifest",
        "first16_manifest",
        "first16_controller",
        "fma_audio",
        "output",
    ):
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
    started = time.monotonic()
    reason = None
    peak = {
        "cpu_seconds": 0.0,
        "rss_bytes": 0,
        "cache_bytes": 0,
        "wav_bytes": 0,
        "total_owned_bytes": 0,
    }
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
                    wall_seconds=time.monotonic() - started,
                    cache_bytes=cache,
                    wav_bytes=wav,
                    total_owned_bytes=owned_bytes(args.output.parent),
                )
                for k in peak:
                    peak[k] = max(peak[k], sample[k])
                reason = bulk_violation(sample, caps)
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
    peak["cpu_seconds"] = max(
        peak["cpu_seconds"], after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    )
    peak["rss_bytes"] = max(peak["rss_bytes"], after.ru_maxrss * 1024)
    cache, wav = storage(args.output)
    peak["cache_bytes"] = max(peak["cache_bytes"], cache)
    peak["wav_bytes"] = max(peak["wav_bytes"], wav)
    peak["total_owned_bytes"] = max(peak["total_owned_bytes"], owned_bytes(args.output.parent))
    final = {**peak, "wall_seconds": time.monotonic() - started}
    reason = reason or bulk_violation(final, caps)
    manifest = args.output / "manifest.json"
    complete = manifest.exists() and read_json(manifest).get("complete") is True
    report.write_text(
        json.dumps(
            {
                "status": "full_preparation_complete"
                if code == 0 and reason is None and complete
                else "FAILED_NOT_TRAINABLE",
                "pid": process.pid,
                "pgid": group,
                "exit": code,
                "cap_violation": reason,
                "manifest_complete": complete,
                "manifest_sha256": digest(manifest) if manifest.exists() else None,
                "measured": final,
                "caps": caps,
                "poll_seconds": 0.2,
                "rss_semantics": "maximum sampled summed process-group RSS;shared pages counted perprocess",
                "owned_semantics": "logical bytes of all files in dedicated output parent,including hardlinks/source/logs",
                "plan_sha256": digest(args.plan),
                "config_sha256": digest(args.config),
                "source_sha256": digest(Path(__file__)),
                "training_authorized": False,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    if code or reason is not None or not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
