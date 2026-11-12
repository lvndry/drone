"""Finite GPU1 accuracy-pair guardian; qualified pure helpers remain unchanged."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.k10_context_pair import movements, uncertainty
from experiments.k10_gpu1_resource import POLICY, load
from experiments.k10_resolution_execution import CAPS, bindings, contract
from experiments.k10_resolution_fit import STATUS, compare
from tools.run_cqt_expansion import disk_bytes, process_tree, sha256, write_json
from tools.run_k10_gpu1_resource import (
    environment,
    sensor,
    start_identity,
    terminate_own,
    violation,
)

ADMISSION_SECONDS = 30


def supervise(config: Path, approval: Path, output: Path) -> None:
    invocation_began = time.monotonic()
    value, permit = load(config), load(approval)
    contract(value)
    bindings(value)
    digest = sha256(config)
    if (
        permit.get("authorized") is not True
        or permit.get("allow_sequential_pair") is not True
        or permit.get("allow_training") is not True
        or permit.get("config_sha256") != digest
        or permit.get("deadline_epoch") != value["deadline_epoch"]
    ):
        raise ValueError("Exact root accuracy-pair approval required")
    if (
        not value["not_before_epoch"] <= time.time() < value["deadline_epoch"]
        or value["deadline_epoch"] - value["not_before_epoch"] > CAPS["wall_seconds"]
    ):
        raise ValueError("Root-sealed thirty-minute UTC window required")
    environment(value)
    starts = list(value["protected"]["protected_starts"])
    if not {1478814, 3448680, 1481756, 3451013} <= {r["pid"] for r in starts}:
        raise ValueError("Missing four protected vLLM identities")
    if not value["protected"]["active_job_starts"]:
        raise ValueError("Bind current teacher/context identities")
    starts += value["protected"]["active_job_starts"]
    output.mkdir(exist_ok=False)
    write_json(output / "approval.json", permit)
    write_json(output / "config.json", value)
    guardian_start = start_identity(os.getpid())
    began = invocation_began
    last = next_sample = last_completion = time.monotonic()
    peaks = dict(
        rss_bytes=0,
        cpu_seconds=0.0,
        owned_gpu_bytes=0,
        min_gpu1_free_bytes=2**63,
        min_gpu0_free_bytes=2**63,
        max_poll_gap_seconds=0.0,
    )
    child: subprocess.Popen[bytes] | None = None
    child_start: int | None = None
    error = None
    complete = False

    def check(admission: bool = False) -> dict[str, Any]:
        nonlocal last, next_sample, last_completion
        time.sleep(max(0, next_sample - time.monotonic()))
        now = time.monotonic()
        next_sample = max(next_sample + POLICY["target_poll_seconds"], now)
        gap = now - last
        last = now
        peaks["max_poll_gap_seconds"] = max(peaks["max_poll_gap_seconds"], gap)
        if child is not None and child.poll() is None and gap > POLICY["poll_seconds"]:
            raise TimeoutError("Active-worker guard cadence exceeded0.2s")
        if now - began > CAPS["wall_seconds"] or time.time() >= value["deadline_epoch"]:
            raise TimeoutError("Frozen total wall/UTC cutoff")
        if any(start_identity(r["pid"]) != r["start_ticks"] for r in starts):
            raise RuntimeError("Protected service/job start changed")
        rss, cpu = process_tree(os.getpid())
        peaks["rss_bytes"], peaks["cpu_seconds"] = (
            max(peaks["rss_bytes"], rss),
            max(peaks["cpu_seconds"], cpu),
        )
        if rss > CAPS["rss_bytes"] or cpu > CAPS["cpu_seconds"]:
            raise RuntimeError("Whole owned-tree CPU/RSS cap")
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        if (
            int(memory["MemAvailable"].split()[0]) * 1024 < POLICY["host_reserve_bytes"]
            or shutil.disk_usage(output).free < POLICY["disk_reserve_bytes"]
            or disk_bytes(output) > CAPS["output_bytes"]
        ):
            raise RuntimeError("Host/disk/output reserve")
        sample = sensor(value, {child.pid} if child else set())
        completion = time.monotonic()
        sensor_gap = completion - last_completion
        last_completion = completion
        peaks["owned_gpu_bytes"] = max(peaks["owned_gpu_bytes"], sample["owned_bytes"])
        for gpu, field in (("gpu1", "free_bytes"), ("gpu0", "gpu0_free_bytes")):
            peaks[f"min_{gpu}_free_bytes"] = min(peaks[f"min_{gpu}_free_bytes"], sample[field])
        sample.update(
            rss_bytes=rss,
            cpu_seconds=cpu,
            poll_gap_seconds=gap,
            sensor_completion_gap_seconds=sensor_gap,
            observed_hard_floor_overshoot_bytes=max(
                0, POLICY["hard_floor_bytes"] - sample["free_bytes"]
            ),
        )
        with (output / "samples.jsonl").open("a") as stream:
            import json

            stream.write(json.dumps(sample) + "\n")
        failed = violation(sample, admission=admission)
        if failed:
            raise RuntimeError(failed)
        if child is not None and child.poll() is None and sensor_gap > POLICY["poll_seconds"]:
            raise TimeoutError("Active-worker sensor gap exceeded0.2s")
        write_json(
            output / "ready.json",
            dict(
                guardian=os.getpid(),
                start_ticks=guardian_start,
                epoch=time.time(),
                config_sha256=digest,
            ),
        )
        return sample

    try:
        admission_end = time.monotonic() + ADMISSION_SECONDS
        while time.monotonic() < admission_end:
            check(admission=True)
        for model in ("baseline", "candidate"):
            check(admission=True)
            command = [
                sys.executable,
                "-m",
                "experiments.k10_resolution_execution",
                "worker",
                "--config",
                str(config),
                "--approval",
                str(approval),
                "--output",
                str(output / model),
                "--model",
                model,
                "--guardian",
                str(os.getpid()),
            ]
            with (output / f"{model}.log").open("xb") as log:
                child = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
            child_start = start_identity(child.pid)
            if child_start is None:
                raise RuntimeError("Worker exited before start binding")
            write_json(
                output / f"{model}-launch.json",
                dict(
                    pid=child.pid,
                    start_ticks=child_start,
                    command=command,
                    config_sha256=digest,
                    epoch=time.time(),
                ),
            )
            while child.poll() is None:
                check()
            if child.wait() != 0 or load(output / model / "result.json").get("status") != STATUS:
                raise RuntimeError("Accuracy worker failed; no retry")
            # No other arm until own NVML allocation is absent and admission reserve has recovered.
            recovery_end = time.monotonic() + 5
            while check(admission=True)["owned_bytes"]:
                if time.monotonic() >= recovery_end:
                    raise RuntimeError("Own memory did not release")
                time.sleep(0.05)
            write_json(
                output / f"{model}-released.json",
                dict(pid=child.pid, released=True, epoch=time.time()),
            )
            child = None
        results = [load(output / model / "result.json") for model in ("baseline", "candidate")]
        historical = load(Path(value["historical_gpu_control"]))
        if sha256(Path(value["historical_gpu_control"])) != value["historical_gpu_control_sha256"]:
            raise ValueError("Historical control changed")
        gate = compare(*results, historical)
        group_rows = csv.DictReader(
            io.StringIO(gzip.decompress(Path(value["groups"]).read_bytes()).decode())
        )
        groups = {r["id"]: r["derived_uncertainty_group"] for r in group_rows}
        rows = {r["id"]: r for r in value["rows"]}
        intervals = uncertainty(*results, {i: r["ref"] for i, r in rows.items()}, groups)
        errors = movements(*results, rows, groups)
        check()
        write_json(
            output / "result.json",
            {
                "status": "COMPLETE_FRESH_GPU_RESOLUTION_PAIR_NOT_PROMOTED",
                "arms": {
                    model: sha256(output / model / "result.json")
                    for model in ("baseline", "candidate")
                },
                "continuation_gate": gate,
                "uncertainty": intervals,
                "error_movements": errors,
                "config_sha256": digest,
                "confirmation_used": False,
                "calibration_used": False,
            },
        )
        complete = True
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        cleanup_error = None
        try:
            if child is not None and child_start is not None:
                terminate_own(child, child_start)
        except BaseException as exc:
            cleanup_error = repr(exc)
        finally:
            terminal_rss, terminal_cpu = process_tree(os.getpid())
            peaks["rss_bytes"] = max(peaks["rss_bytes"], terminal_rss)
            peaks["cpu_seconds"] = max(peaks["cpu_seconds"], terminal_cpu)
            if terminal_rss > CAPS["rss_bytes"] or terminal_cpu > CAPS["cpu_seconds"]:
                cleanup_error = str(cleanup_error) + "; terminal whole-tree CPU/RSS cap"
            write_json(
                output / "controller.json",
                dict(
                    complete=complete and cleanup_error is None,
                    error=error,
                    cleanup_error=cleanup_error,
                    peaks=peaks,
                    wall_seconds=time.monotonic() - began,
                    config_sha256=digest,
                    polling_is_not_allocation_interlock=True,
                    task_fit=True,
                ),
            )
        if cleanup_error is not None and error is None:
            raise RuntimeError(cleanup_error)


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
