"""Standalone bounded teacher guardian; streaming inputs, no large RAM ownership."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from tools.musicfm_crop_io import load, save, sha
from tools.preflight_musicfm import gpu_metadata
from tools.prepare_musicfm_ram_cache import capacity_guard, save_or_replace
from tools.run_cqt_expansion import disk_bytes, process_tree
from tools.run_musicfm_ram_cache import alive, descendants, proc, stop_and_wait


def resources(config: dict[str, Any], root: Path) -> None:
    capacity_guard(config, root)
    gpu = gpu_metadata(config["gpu_uuid"])
    if int(gpu["free_mib"]) * 1024**2 < 8 * 1024**3:
        raise RuntimeError("GPU free below8GiB; preserve vLLM")


def worker_watch(config: dict[str, Any], root: Path, guardian_pid: int) -> Any:  # noqa: ANN401
    last = [0.0]
    statistics = {"max_observed_heartbeat_age_seconds": 0.0, "allowance_seconds": 10.0}
    lock = threading.Lock()

    def check() -> None:
        if time.monotonic() - last[0] < 0.2:
            return
        last[0] = time.monotonic()
        ready = load(root / "ready.json")
        age = time.time() - ready["last_guard_epoch"]
        statistics["max_observed_heartbeat_age_seconds"] = max(
            statistics["max_observed_heartbeat_age_seconds"], age
        )
        if (
            ready["pid"] != guardian_pid
            or not alive(guardian_pid, ready["start_ticks"])
            or ready.get("teacher_config_sha256") != config["_config_sha256"]
            or not 0 <= time.time() - ready["last_guard_epoch"] <= 10.0
        ):
            raise RuntimeError("No fresh live guardian")
        if time.time() >= config["deadline_epoch"]:
            raise TimeoutError("Worker lease cutoff")
        capacity_guard(config, root)

    def watch() -> None:
        with lock:
            check()

    return watch, statistics


def owned_resources(
    config: dict[str, Any], peaks: dict[str, float], *, enforce: bool = True
) -> dict[str, float]:
    """Cap the guardian, its live workers/queries, and all reaped descendant CPU."""
    rss, cpu = process_tree(os.getpid())
    peaks["rss"] = max(peaks["rss"], rss)
    peaks["cpu"] = max(peaks["cpu"], cpu)
    if enforce and (rss > config["caps"]["rss_bytes"] or cpu > config["caps"]["cpu_seconds"]):
        raise RuntimeError("Owned teacher process-tree finite CPU/RSS cap")
    return dict(rss_bytes=rss, cpu_seconds=cpu)


def supervise(config_path: Path, approval_path: Path, root: Path) -> None:
    config, approval = load(config_path), load(approval_path)
    mode = config["mode"]
    if mode not in {"probe", "fit"}:
        raise ValueError("Unknown execution mode")
    cfgsha = sha(config_path)
    if (
        approval.get("authorized") is not True
        or approval.get("config_sha256") != cfgsha
        or (mode == "fit" and approval.get("allow_training") is not True)
        or (
            mode == "probe"
            and (
                approval.get("allow_probe") is not True
                or approval.get("allow_training") is not False
            )
        )
    ):
        raise ValueError("Exact mode/source-bound root approval required")
    if config["eval_batch_size"] not in (1, 16):
        raise ValueError("Only qualified1/16 FP32 encoder paths implemented")
    if (
        not config["not_before_epoch"] <= time.time() < config["deadline_epoch"]
        or config["deadline_epoch"] - config["not_before_epoch"] > config["caps"]["wall_seconds"]
    ):
        raise ValueError("Invalid finite root-approved UTC lease")
    source = Path(__file__).resolve().parents[1]
    if any(sha(source / name) != value for name, value in config["source_sha256"].items()):
        raise ValueError("Reviewed source changed")
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is None or affinity(0) != set(config["cpus"]):
        raise ValueError("Different CPU affinity")
    if any(
        os.environ.get(k) != "1"
        for k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    ):
        raise ValueError("Require unchanged isolated numeric thread environment")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != config["gpu_uuid"]:
        raise ValueError("Only pinned GPU0 UUID may be visible")
    root.mkdir(exist_ok=False)
    identity = proc(os.getpid())
    if identity is None:
        raise RuntimeError("Missing guardian start identity")
    started = time.monotonic()
    worker: subprocess.Popen[bytes] | None = None
    tracked: dict[int, int] = {}
    transferred = False
    state = "GUARDING_BEFORE_WORKER"
    failure: dict[str, Any] = {}
    lastdisk = lastgpu = 0.0
    peaks: dict[str, float] = {"rss": 0, "cpu": 0.0}
    config["_config_sha256"] = cfgsha
    try:
        while True:
            owned_resources(config, peaks)
            capacity_guard(config, root)
            if time.monotonic() - lastgpu >= 0.5:
                resources(config, root)
                lastgpu = time.monotonic()
                owned_resources(config, peaks)
            if time.time() >= config["deadline_epoch"]:
                raise TimeoutError("Guardian cutoff")
            if time.monotonic() - started > config["caps"]["wall_seconds"]:
                raise TimeoutError("Wall budget")
            save_or_replace(
                root / "ready.json",
                {
                    "ready": True,
                    "pid": os.getpid(),
                    "start_ticks": identity[0],
                    "config_sha256": cfgsha,
                    "teacher_config_sha256": cfgsha,
                    "last_guard_epoch": time.time(),
                    "state": state,
                },
            )
            if not transferred:
                transferred = True
                state = "MODEL_WORKER_ACTIVE"
                save(
                    root / "ownership-accepted.json",
                    {"config_sha256": cfgsha, "epoch": time.time()},
                )
                command = [
                    sys.executable,
                    "-m",
                    "experiments.musicfm_pilot",
                    "worker",
                    "--config",
                    str(config_path),
                    "--approval",
                    str(approval_path),
                    "--output",
                    str(root / "worker"),
                    "--guardian-pid",
                    str(os.getpid()),
                ]
                log = (root / "worker.log").open("xb")
                env = dict(
                    os.environ,
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    PYTHONDONTWRITEBYTECODE="1",
                    CUDA_CACHE_PATH=str(root / "cuda-cache"),
                    TORCHINDUCTOR_CACHE_DIR=str(root / "inductor-cache"),
                    TRITON_CACHE_DIR=str(root / "triton-cache"),
                    TMPDIR=str(root / "tmp"),
                    CUBLAS_WORKSPACE_CONFIG=":4096:8",
                )
                for directory in ("cuda-cache", "inductor-cache", "triton-cache", "tmp"):
                    (root / directory).mkdir()
                worker = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env
                )
                log.close()
                tracked = descendants(worker.pid)
                save(
                    root / "launch.json",
                    {
                        "pid": worker.pid,
                        "command": command,
                        "epoch": time.time(),
                        "config_sha256": cfgsha,
                        "guardian_start_ticks": identity[0],
                    },
                )
            if worker is not None:
                if worker.poll() is None:
                    tracked.update(descendants(worker.pid))
                    owned_resources(config, peaks)
                elif state == "MODEL_WORKER_ACTIVE":
                    worker.wait()
                    owned_resources(config, peaks)
                    if worker.returncode != 0:
                        raise RuntimeError(f"Model worker exit={worker.returncode}")
                    result = load(root / "worker/result.json")
                    expected = (
                        "DISPOSABLE_COST_ONLY"
                        if mode == "probe"
                        else "COMPLETE_FIXED_PILOT_NOT_RELEASE_EVIDENCE"
                    )
                    if result["status"] != expected:
                        raise ValueError("Incomplete worker result")
                    state = "WORKER_COMPLETE_PENDING_CHECKS"
            if state == "WORKER_COMPLETE_PENDING_CHECKS" or time.monotonic() - lastdisk >= 1:
                lastdisk = time.monotonic()
                if disk_bytes(root) > config["caps"]["output_bytes"]:
                    raise RuntimeError("Output disk cap")
                save_or_replace(
                    root / "progress.json", {"state": state, "epoch": time.time(), "peaks": peaks}
                )
            if state == "WORKER_COMPLETE_PENDING_CHECKS":
                owned_resources(config, peaks)
                save(
                    root / "held-success.json",
                    {
                        "mode": mode,
                        "result_sha256": sha(root / "worker/result.json"),
                        "epoch": time.time(),
                        "training_approved": False,
                    },
                )
                state = "COMPLETE_NO_RAM_OWNERSHIP"
                break
            time.sleep(0.05)
    except BaseException as error:
        state = "FAILED"
        failure = {"state": state, "error": repr(error), "worker_started": transferred}
        try:
            if worker is not None:
                stop_and_wait(worker, tracked)
        except BaseException as cleanup_error:
            failure["cleanup_error"] = repr(cleanup_error)
        save(root / "failure.json", failure)
        raise
    finally:
        save(
            root / "controller.json",
            {
                "state": state,
                "error": failure.get("error"),
                "cleanup_error": failure.get("cleanup_error"),
                "worker_started": transferred,
                "peaks": peaks,
                "terminal_resources": owned_resources(config, peaks, enforce=False),
                "resource_scope": "guardian and all owned live/reaped descendants, including GPU queries",
                "wall_seconds": time.monotonic() - started,
                "worker_exit": worker.returncode if worker else None,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "approval", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    signal.signal(
        signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError("Guardian terminated"))
    )
    supervise(args.config, args.approval, args.output_root)


if __name__ == "__main__":
    main()
