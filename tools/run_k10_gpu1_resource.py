"""Prospective GPU1-only synthetic guardian. No launch without a sealed permit."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.k10_gpu1_resource import POLICY, contract, load, verify_bindings
from tools.run_cqt_expansion import disk_bytes, process_tree, sha256, write_json


def start_identity(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] in ("Z", "X") else int(fields[19])
    except FileNotFoundError:
        return None


def environment(value: dict[str, Any]) -> None:
    affinity = getattr(os, "sched_getaffinity", None)
    if (
        affinity is None
        or set(affinity(0)) != {10, 11}
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or os.environ.get("CUDA_VISIBLE_DEVICES") != value["gpu_uuid"]
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or any(
            os.environ.get(name) != "1"
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        )
    ):
        raise ValueError("Exact isolated GPU1/CPU10-11/nice19 environment required")
    cores = [
        tuple(
            Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/{name}").read_text().strip()
            for name in ("physical_package_id", "core_id")
        )
        for cpu in (10, 11)
    ]
    if cores[0] == cores[1]:
        raise ValueError("Two distinct physical cores required")


def parse_sensor(gpus: str, processes: str, value: dict[str, Any], own: set[int]) -> dict[str, int]:
    devices = {}
    for line in gpus.splitlines():
        identity, total, free = [part.strip() for part in line.split(",")]
        devices[identity] = (int(total) * 1024**2, int(free) * 1024**2)
    one, zero = devices[value["gpu_uuid"]], devices[value["gpu0_uuid"]]
    owned = 0
    for line in processes.splitlines():
        if not line.strip():
            continue
        identity, pid, memory = [part.strip() for part in line.split(",")]
        if int(pid) in own:
            if identity != value["gpu_uuid"]:
                raise RuntimeError("Owned worker allocated on wrong GPU")
            owned += int(memory) * 1024**2
    return dict(total_bytes=one[0], free_bytes=one[1], gpu0_free_bytes=zero[1], owned_bytes=owned)


def violation(sample: dict[str, int], *, admission: bool) -> str | None:
    if sample["gpu0_free_bytes"] < POLICY["gpu0_reserve_bytes"]:
        return "GPU0 existing8GiB reserve"
    if sample["free_bytes"] < (
        POLICY["admission_bytes"] if admission else POLICY["early_abort_bytes"]
    ):
        return "GPU1 admission/early-abort reserve"
    if sample["owned_bytes"] > POLICY["owned_gpu_bytes"]:
        return "Owned NVML ceiling"
    return None


def sensor(value: dict[str, Any], own: set[int]) -> dict[str, Any]:
    binary = value["runtime"]["nvidia_smi"]["path"]
    began = time.monotonic()

    def query(fields: str) -> str:
        return subprocess.check_output(
            [binary, fields, "--format=csv,noheader,nounits"],
            text=True,
            timeout=POLICY["query_timeout_seconds"],
        )

    sample = parse_sensor(
        query("--query-gpu=uuid,memory.total,memory.free"),
        query("--query-compute-apps=gpu_uuid,pid,used_gpu_memory"),
        value,
        own,
    )
    elapsed = time.monotonic() - began
    if elapsed > POLICY["poll_seconds"]:
        raise TimeoutError("Stale bounded GPU sensor")
    return {**sample, "query_seconds": elapsed, "epoch": time.time()}


def terminate_own(worker: subprocess.Popen[bytes], expected: int) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if worker.poll() is not None:
            return
        try:
            if start_identity(worker.pid) != expected or os.getpgid(worker.pid) != worker.pid:
                if worker.poll() is not None:
                    return
                raise RuntimeError("Own process-group identity changed; refuse signal")
            os.killpg(worker.pid, sig)
        except ProcessLookupError:
            worker.wait(timeout=2)
            return
        try:
            worker.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            pass
    raise RuntimeError("Own worker did not terminate")


def supervise(config: Path, approval: Path, output: Path) -> None:
    invocation_began = time.monotonic()
    value, permit = load(config), load(approval)
    contract(value)
    verify_bindings(value)
    digest = sha256(config)
    if (
        permit.get("authorized") is not True
        or permit.get("allow_resource_probe") is not True
        or permit.get("allow_training") is not False
        or permit.get("config_sha256") != digest
    ):
        raise ValueError("Exact resource-only approval required")
    if (
        not value["not_before_epoch"] <= time.time() < value["deadline_epoch"]
        or value["deadline_epoch"] - value["not_before_epoch"] > POLICY["wall_seconds"]
    ):
        raise ValueError("Root-sealed five-minute UTC window required")
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
        if now - began > 300 or time.time() >= value["deadline_epoch"]:
            raise TimeoutError("Frozen total wall/UTC cutoff")
        if any(start_identity(r["pid"]) != r["start_ticks"] for r in starts):
            raise RuntimeError("Protected service/job start changed")
        rss, cpu = process_tree(os.getpid())
        peaks["rss_bytes"], peaks["cpu_seconds"] = (
            max(peaks["rss_bytes"], rss),
            max(peaks["cpu_seconds"], cpu),
        )
        if rss > POLICY["rss_bytes"] or cpu > POLICY["cpu_seconds"]:
            raise RuntimeError("Whole owned-tree CPU/RSS cap")
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        if (
            int(memory["MemAvailable"].split()[0]) * 1024 < POLICY["host_reserve_bytes"]
            or shutil.disk_usage(output).free < POLICY["disk_reserve_bytes"]
            or disk_bytes(output) > POLICY["output_bytes"]
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
        admission_end = time.monotonic() + 30
        while time.monotonic() < admission_end:
            check(admission=True)
        for model in ("baseline", "candidate"):
            check(admission=True)
            command = [
                sys.executable,
                "-m",
                "experiments.k10_gpu1_resource",
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
            if (
                child.wait() != 0
                or load(output / model / "result.json").get("complete") is not True
            ):
                raise RuntimeError("Synthetic worker failed; no retry")
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
            if terminal_rss > POLICY["rss_bytes"] or terminal_cpu > POLICY["cpu_seconds"]:
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
                    task_fit=False,
                ),
            )
        if cleanup_error is not None and error is None:
            raise RuntimeError(cleanup_error)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "approval", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    signal.signal(
        signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError("Guardian stopped"))
    )
    supervise(args.config, args.approval, args.output)


if __name__ == "__main__":
    main()
