"""Thin sequential process-group watch; numerical K10 driver is imported unchanged."""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.prepare_k10_v2_pilot import digest, read_json
from tools.run_k10_v2_bulk import owned_bytes
from tools.run_k10_v2_preprocessing import terminate, usage


def protected_live(expected: list[dict[str, int]]) -> None:
    for row in expected:
        p = Path("/proc") / str(row["pid"])
        try:
            tail = (p / "stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            raise RuntimeError("Protected vLLM process exited") from None
        if int(tail[19]) != row["start_ticks"] or tail[0] in ("Z", "X"):
            raise RuntimeError("Protected vLLM start identity changed")


def gpu(config: dict[str, Any]) -> int:
    text = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
        text=True,
        timeout=5,
    )
    identity, free = text.strip().split(",")
    if identity.strip() != config["gpu_uuid"]:
        raise RuntimeError("GPU0 UUID changed")
    free_bytes = int(free.strip()) * 1024**2
    if free_bytes < 8 * 1024**3:
        raise RuntimeError("GPU0 free memory below8GiB;do not interrupt vLLM")
    return free_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "permit", "plan", "features", "training-source", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    permit = read_json(args.permit)
    if (
        permit.get("allow_training") is not True
        or permit.get("allow_sequential_pair") is not True
        or permit.get("plan_sha256") != digest(args.plan)
        or permit.get("pair_config_sha256") != digest(args.config)
    ):
        raise ValueError("Explicit hash-bound wholepair permit required")
    if (
        digest(args.plan) != config["plan_sha256"]
        or digest(args.features) != config["feature_manifest_sha256"]
        or digest(Path(__file__)) != config["guard_source_sha256"]
    ):
        raise ValueError("Frozen pair input/source mismatch")
    plan = read_json(args.plan)
    if digest(args.training_source / "experiments/k10_v2_pilot.py") != plan["driver_sha256"]:
        raise ValueError("Numerical driver changed")
    for name, expected in plan["bound_source_sha256"].items():
        if digest(args.training_source / name) != expected:
            raise ValueError("Bound numerical source changed")
    start = datetime.fromisoformat(permit["not_before_utc"].replace("Z", "+00:00")).timestamp()
    cutoff = datetime.fromisoformat(permit["outer_cutoff_utc"].replace("Z", "+00:00")).timestamp()
    if not start <= time.time() < cutoff:
        raise ValueError("Outside authorized UTC window")
    if args.output_root.exists():
        raise ValueError("Fresh pair output root required;no restart/resume")
    args.output_root.mkdir(parents=True)
    env = dict(
        os.environ,
        PYTHONPATH=str(args.training_source),
        CUDA_VISIBLE_DEVICES=config["gpu_uuid"],
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
    )
    results = []
    for arm in ("restricted", "expanded"):
        protected_live(config["protected_starts"])
        free = gpu(config)
        if time.time() + 1255 > cutoff:
            raise RuntimeError("Insufficient remaining approved window for fixed arm")
        output = args.output_root / arm
        log = args.output_root / (arm + ".log")
        controller = args.output_root / (arm + ".controller.json")
        command = [
            "taskset",
            "-c",
            "4,5,10,11",
            sys.executable,
            "-m",
            "experiments.k10_v2_pilot",
            "train",
            "--plan",
            str(args.plan),
            "--features",
            str(args.features),
            "--permit",
            str(args.permit),
            "--arm",
            arm,
            "--device",
            "cuda:0",
            "--output",
            str(output),
        ]
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        began = time.monotonic()
        reason = None
        minimum_free = free
        peak = {"cpu_seconds": 0.0, "rss_bytes": 0, "output_bytes": 0}
        with log.open("wb") as stream:
            process = subprocess.Popen(
                command,
                cwd=args.training_source,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            group = process.pid
            controller.write_text(
                json.dumps(
                    {
                        "status": "running",
                        "arm": arm,
                        "pid": process.pid,
                        "pgid": group,
                        "command": command,
                        "log": str(log),
                        "plan_sha256": digest(args.plan),
                        "config_sha256": digest(args.config),
                    },
                    indent=2,
                )
                + "\n"
            )
            try:
                while True:
                    sample = usage(group)
                    sample["output_bytes"] = owned_bytes(output) + (
                        log.stat().st_size if log.exists() else 0
                    )
                    for key in peak:
                        peak[key] = max(peak[key], sample[key])
                    try:
                        protected_live(config["protected_starts"])
                        minimum_free = min(minimum_free, gpu(config))
                    except Exception as error:
                        reason = str(error)
                        terminate(group)
                        break
                    if time.time() >= cutoff:
                        reason = "wholepair_UTC_cutoff"
                    elif time.monotonic() - began > 1250:
                        reason = "outer_arm_wall"
                    elif sample["cpu_seconds"] > 6000:
                        reason = "arm_CPU"
                    elif sample["rss_bytes"] > 8 * 1024**3:
                        reason = "arm_RSS"
                    elif sample["output_bytes"] > 1024**3:
                        reason = "arm_output"
                    if reason is not None:
                        terminate(group)
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(0.5)
            finally:
                if process.poll() is None or usage(group)["live_pids"]:
                    terminate(group)
                code = process.wait()
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        peak["cpu_seconds"] = max(
            peak["cpu_seconds"], after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
        )
        peak["rss_bytes"] = max(peak["rss_bytes"], after.ru_maxrss * 1024)
        elapsed = time.monotonic() - began
        if reason is None:
            if elapsed > 1250 or peak["cpu_seconds"] > 6000 or peak["rss_bytes"] > 8 * 1024**3:
                reason = "terminal_resource_cap"
        run_path = output / "run.json"
        run = read_json(run_path) if run_path.exists() else {}
        complete = (
            code == 0
            and reason is None
            and run.get("status") == "complete"
            and run.get("updates") == 8280
        )
        controller.write_text(
            json.dumps(
                {
                    "status": "complete" if complete else "FAILED_INCONCLUSIVE_NO_RETRY",
                    "arm": arm,
                    "pid": process.pid,
                    "pgid": group,
                    "exit": code,
                    "reason": reason,
                    "elapsed_seconds": elapsed,
                    "peak": peak,
                    "min_gpu0_free_bytes": minimum_free,
                    "gpu_uuid": config["gpu_uuid"],
                    "run_sha256": digest(run_path) if run_path.exists() else None,
                    "confirmation_scored": False,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        if not complete:
            raise RuntimeError(
                "Operationally incomplete arm;no unequal-horizon comparison/no retry"
            )
        results.append(run)  # No accuracy branch;expanded always follows completed restricted.
    from experiments.k10_v2_pilot import pilot_gate

    result = pilot_gate(results[0], results[1])
    (args.output_root / "gate.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    protected_live(config["protected_starts"])
    (args.output_root / "terminal.json").write_text(
        json.dumps(
            {
                "status": "BOTH_FIXED_ARMS_COMPLETE",
                "utc": datetime.now(timezone.utc).isoformat(),
                "confirmation_scored": False,
                "gpu_uuid": config["gpu_uuid"],
                "gate": result,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
