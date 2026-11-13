"""Bound one tensor-only numerical qualification on an explicitly assigned GPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import signal
import subprocess
import time
from pathlib import Path
from types import FrameType
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interrupted(signum: int, frame: FrameType | None) -> None:
    raise InterruptedError(f"Guardian received signal {signum}")


def supervise(config_path: Path) -> None:
    config: dict[str, Any] = json.loads(config_path.read_text())
    for item in config["source_files"]:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError("Qualification source changed")
    spec = importlib.util.spec_from_file_location("myna_runtime_guard", config["guard_helper"])
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load verified guardian helper")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    root = Path(config["output_root"])
    root.mkdir()
    guard.protected(config)
    affinity = getattr(os, "sched_setaffinity", None)
    if affinity is None:
        raise RuntimeError("Linux affinity required")
    affinity(0, config["cpus"])
    os.nice(19)
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "LD_PRELOAD", "LD_LIBRARY_PATH"):
        env.pop(key, None)
    env.update(
        CUDA_VISIBLE_DEVICES=config.get("cuda_visible_devices", config["gpu_uuid"]),
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HOME=str(root / "hf-cache"),
        TORCH_HOME=str(root / "torch-cache"),
        CUDA_CACHE_PATH=str(root / "cuda-cache"),
    )
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    start = time.monotonic()
    base_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    base_self = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = 0
    peak_size = 0
    min_gpu_free: int | None = None
    max_owned_gpu = 0
    failure: str | None = None
    with (root / "worker.log").open("w") as log:
        child = subprocess.Popen(
            [
                str(Path(config["environment"]) / "bin/python"),
                "-I",
                "-B",
                config["worker_source"],
                "--config",
                str(config_path),
            ],
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        guard.write_json(
            root / "launch.json",
            {
                "guardian_pid": os.getpid(),
                "guardian_start": guard.process_start(os.getpid()),
                "worker_pid": child.pid,
                "worker_start": guard.process_start(child.pid),
            },
        )
        try:
            while child.poll() is None:
                guard.protected(config)
                rss, live_cpu = guard.tree_usage(child.pid)
                own_stat = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
                rss += int(own_stat[21]) * os.sysconf("SC_PAGE_SIZE")
                size = guard.owned_bytes(root)
                self_usage = resource.getrusage(resource.RUSAGE_SELF)
                child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                cpu = (
                    self_usage.ru_utime
                    + self_usage.ru_stime
                    - base_self.ru_utime
                    - base_self.ru_stime
                    + child_usage.ru_utime
                    + child_usage.ru_stime
                    - base_children.ru_utime
                    - base_children.ru_stime
                    + live_cpu
                )
                peak_rss, peak_size = max(peak_rss, rss), max(peak_size, size)
                query = subprocess.run(
                    [
                        "nvidia-smi",
                        "--id=" + config["gpu_uuid"],
                        "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                free = int(query.stdout.strip()) * 1024**2
                min_gpu_free = min(free, min_gpu_free) if min_gpu_free is not None else free
                apps = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in apps.stdout.splitlines():
                    pid, used = [x.strip() for x in line.split(",")]
                    if int(pid) == child.pid:
                        max_owned_gpu = max(max_owned_gpu, int(used) * 1024**2)
                if (
                    time.monotonic() - start > config["wall_seconds"]
                    or cpu > config["cpu_seconds"]
                    or rss > config["rss_bytes"]
                    or size > config["output_bytes"]
                    or free < config["min_free_gpu_bytes"]
                ):
                    raise RuntimeError("Qualification resource limit exceeded")
                time.sleep(0.5)
            if child.returncode:
                raise RuntimeError(f"Qualification worker exited {child.returncode}")
            guard.protected(config)
            if not Path(config["result_path"]).is_file():
                raise RuntimeError("Worker completed without result")
        except BaseException as error:
            failure = f"{type(error).__name__}: {error}"
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        child.wait()
    final_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    final_self = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (
        final_children.ru_utime
        + final_children.ru_stime
        - base_children.ru_utime
        - base_children.ru_stime
        + final_self.ru_utime
        + final_self.ru_stime
        - base_self.ru_utime
        - base_self.ru_stime
    )
    wall = time.monotonic() - start
    final_size = guard.owned_bytes(root)
    if failure is None and (
        wall > config["wall_seconds"]
        or cpu > config["cpu_seconds"]
        or final_size > config["output_bytes"]
    ):
        failure = "Terminal resource limit exceeded"
    receipt = {
        "status": "failed" if failure else "complete",
        "error": failure,
        "wall_seconds": wall,
        "owned_cpu_seconds": cpu,
        "sampled_peak_host_rss_bytes": peak_rss,
        "sampled_peak_output_bytes": peak_size,
        "final_output_bytes": final_size,
        "sampled_min_gpu_free_bytes": min_gpu_free,
        "sampled_peak_worker_nvml_bytes": max_owned_gpu,
        "worker_exit": child.returncode,
        "config_sha256": digest(config_path),
    }
    guard.write_json(root / "controller.json", receipt)
    print(json.dumps(receipt))
    if failure:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    supervise(args.config.resolve())


if __name__ == "__main__":
    main()
