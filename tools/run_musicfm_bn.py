"""Bounded owner for one prospective BN diagnostic; no default execution permit.

Reuses the original teacher's resource, identity, heartbeat and owned-termination
helpers. The original training worker and its source are not changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.musicfm_bn_terminal import FILES, verify_terminal_teacher
from tools.musicfm_crop_io import load, save, sha
from tools.prepare_musicfm_ram_cache import capacity_guard, save_or_replace
from tools.run_cqt_expansion import disk_bytes
from tools.run_musicfm_ram_cache import alive, descendants, proc, stop_and_wait
from tools.run_musicfm_teacher import owned_resources

GIB = 1024**3
ORIGINAL = Path("/home/lysk-admin/drone-experiments/musicfm-key-pilot-20261112-seed13-051e4ad")
ORIGINAL_FILES = {
    "config": "sealed-config.json",
    "result": "run-v1/worker/result.json",
    "controller": "run-v1/controller.json",
    "success": "run-v1/held-success.json",
    "launch": "run-v1/launch.json",
}
CAPS = dict(wall_seconds=3600, cpu_seconds=10800, rss_bytes=8 * GIB, output_bytes=8 * GIB)
STATUSES = {
    "COMPLETE_SINGLE_BN_DIAGNOSTIC_NOT_PROMOTED",
    "BN_DIAGNOSTIC_BUDGET_INCONCLUSIVE_NO_SCORING",
}
REQUIRED_SOURCE = {
    "experiments/studies/musicfm-key-pilot/BN_PROTOCOL.md",
    "tools/run_musicfm_bn.py",
    "experiments/musicfm_bn_execution.py",
    "experiments/musicfm_bn_terminal.py",
    "experiments/musicfm_bn.py",
    "experiments/musicfm_bn_panel.py",
    "experiments/musicfm_bn_state.py",
    "experiments/musicfm_bn_calibration.py",
    "experiments/musicfm_bn_qualification.py",
}


def live_start(pid: int) -> int | None:
    identity = proc(pid)
    return identity[0] if identity is not None and identity[1] not in {"Z", "X"} else None


def original_failures_absent() -> bool:
    return all(
        not path.exists() and not path.is_symlink()
        for path in (ORIGINAL / "run-v1/failure.json", ORIGINAL / "run-v1/worker/failure.json")
    )


def read_original(config: dict[str, Any]) -> dict[str, bytes]:
    expected = config["original_files_sha256"]
    if set(expected) != FILES:
        raise ValueError("All five original metadata hashes must be sealed")
    result = {}
    for name, relative in ORIGINAL_FILES.items():
        path = ORIGINAL / relative
        if path.is_symlink() or not 0 < path.stat().st_size <= 16 * 1024**2:
            raise ValueError("Unexpected original metadata file")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected[name]:
            raise ValueError("Original terminal metadata hash changed: " + name)
        result[name] = raw
    return result


def guard_view(config: dict[str, Any]) -> dict[str, Any]:
    """Drop only the explicitly authorised stopped services from the live guard."""
    stopped = set(config.get("authorized_stopped_pids", ()))
    if not stopped:
        return config
    view = dict(config)
    view["protected_starts"] = [
        row for row in config["protected_starts"] if row["pid"] not in stopped
    ]
    return view


def resources(config: dict[str, Any], root: Path) -> None:
    """Same GPU-free guard as the teacher, but accepts the sealed UUID at any index."""
    capacity_guard(config, root)
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            config["gpu_uuid"],
            "--query-gpu=uuid,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=5,
    )
    fields = [field.strip() for field in raw.strip().split(",")]
    if len(fields) != 2 or fields[0] != config["gpu_uuid"]:
        raise RuntimeError("GPU identity changed")
    if int(fields[1]) * 1024**2 < 8 * 1024**3:
        raise RuntimeError("GPU free below 8GiB; preserve vLLM")


def validate_launch(config: dict[str, Any], approval: dict[str, Any], config_sha: str) -> None:
    # The exact device is bound by the sealed config/approval hashes. Public
    # source only requires a physical GPU UUID, so no machine-specific ID is
    # embedded here.
    if (
        config["mode"] != "single-bn-diagnostic"
        or approval.get("authorized") is not True
        or approval.get("allow_bn_diagnostic") is not True
        or approval.get("allow_training") is not False
        or approval.get("config_sha256") != config_sha
        or config["caps"] != CAPS
        or config["allocator_bytes"] != 16 * GIB
        or not str(config["gpu_uuid"]).startswith("GPU-")
        or config["torch_threads"] != 2
        or len(config["cpus"]) != 2
        or len(set(config["cpus"])) != 2
        or not 0 < config["deadline_epoch"] - config["not_before_epoch"] <= CAPS["wall_seconds"]
        or not config["not_before_epoch"] <= time.time() < config["deadline_epoch"]
    ):
        raise ValueError("Require an exact finite, source-bound diagnostic-only permit")
    stopped = config.get("authorized_stopped_pids")
    if (
        not isinstance(stopped, list)
        or stopped != approval.get("authorized_stopped_pids")
        or not set(stopped) <= {row["pid"] for row in config["protected_starts"]}
    ):
        raise ValueError("Authorized-stopped services must match the approval")
    for pid in stopped:
        path = Path("/proc") / str(pid)
        try:
            tail = (path / "stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        if tail[0] not in ("Z", "X"):
            raise ValueError("Authorized-stopped service is still alive")
    if not REQUIRED_SOURCE.issubset(config["source_sha256"]):
        raise ValueError("Missing diagnostic implementation source bindings")
    source = Path(__file__).resolve().parents[1]
    if (
        Path.cwd().resolve() != source
        or os.environ.get("PYTHONPATH")
        or os.environ.get("PYTHONHOME")
    ):
        raise ValueError("Run from the verified source root without Python path overrides")
    for name, expected in config["source_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or sha(source / relative) != expected:
            raise ValueError("Reviewed source changed: " + name)
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is None or affinity(0) != set(config["cpus"]):
        raise ValueError("Different isolated CPU affinity")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != config["gpu_uuid"] or any(
        os.environ.get(key) != "1"
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    ):
        raise ValueError("Different physical GPU visibility or numeric thread environment")


def verify_import_paths(source_sha256: dict[str, str]) -> None:
    source = Path(__file__).resolve().parents[1]
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in {"tools", "experiments", "drone"}:
            continue
        path = getattr(module, "__file__", None)
        if path is None:
            continue  # Namespace package, not an executable module file.
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(source):
            raise ValueError("Local module imported outside the verified source root: " + name)
        relative = str(resolved.relative_to(source))
        if relative not in source_sha256 or sha(resolved) != source_sha256[relative]:
            raise ValueError("Imported local module is not source-bound: " + name)


def check_deadline(config: dict[str, Any], started: float) -> None:
    if (
        time.monotonic() - started > config["caps"]["wall_seconds"]
        or time.time() >= config["deadline_epoch"]
    ):
        raise TimeoutError("Diagnostic wall/lease cutoff")


def verify_original_interpreter(launch: dict[str, Any]) -> None:
    """Preserve the pilot's virtual environment, not just its underlying binary.

    Resolving symlinks would erase the distinction between the provisioned overlay
    and the shared environment: both may point to the same system Python.
    """
    command = launch.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not isinstance(command[0], str)
        or not Path(command[0]).is_absolute()
        or os.path.abspath(sys.executable) != os.path.abspath(command[0])
    ):
        raise ValueError("Use the original pilot's isolated Python interpreter")


def supervise(config_path: Path, approval_path: Path, root: Path) -> None:
    config_path = config_path.resolve()
    approval_path = approval_path.resolve()
    root = root.absolute()
    config_raw = config_path.read_bytes()
    approval_raw = approval_path.read_bytes()
    config = json.loads(config_raw)
    approval = json.loads(approval_raw)
    config_sha = hashlib.sha256(config_raw).hexdigest()
    validate_launch(config, approval, config_sha)
    verify_import_paths(config["source_sha256"])
    source = Path(__file__).resolve().parents[1]
    if (
        root.is_symlink()
        or root.resolve() != root
        or str(root) != config["output_root"]
        or root.is_relative_to(ORIGINAL)
        or root.is_relative_to(source)
    ):
        raise ValueError("Output must be a fresh sealed path outside preserved inputs/source")
    root.mkdir(exist_ok=False)
    identity = proc(os.getpid())
    if identity is None:
        raise RuntimeError("Missing guardian start identity")
    config["_config_sha256"] = config_sha
    started = time.monotonic()
    peaks: dict[str, float] = {"rss": 0, "cpu": 0.0}
    worker: subprocess.Popen[bytes] | None = None
    tracked: dict[int, int] = {}
    complete = False
    error = None
    cleanup_error = None
    try:
        original = read_original(config)
        verify_original_interpreter(json.loads(original["launch"]))
        terminal = verify_terminal_teacher(
            original,
            config["selected_checkpoint_sha256"],
            live_start,
            failure_receipts_absent=original_failures_absent(),
        )
        original_config = json.loads(original["config"])
        if config["protected_starts"] != original_config["protected_starts"]:
            raise ValueError("Do not weaken the original vLLM protected identities")
        if any(
            config["source_sha256"].get(name) != expected
            for name, expected in original_config["source_sha256"].items()
        ):
            raise ValueError("Do not change the original teacher/evaluator numerical source")
        save(
            root / "original-terminal.json",
            dict(
                result_sha256=terminal.result_sha256,
                checkpoint_sha256=terminal.checkpoint_sha256,
                original_gate_earned=terminal.original_gate_earned,
                best_update=terminal.best_update,
            ),
        )
        last_gpu = 0.0
        last_disk = 0.0
        while True:
            owned_resources(config, peaks)
            capacity_guard(guard_view(config), root)
            now = time.monotonic()
            check_deadline(config, started)
            if now - last_gpu >= 0.5:
                resources(guard_view(config), root)
                last_gpu = time.monotonic()
            save_or_replace(
                root / "ready.json",
                dict(
                    pid=os.getpid(),
                    start_ticks=identity[0],
                    teacher_config_sha256=config_sha,
                    last_guard_epoch=time.time(),
                    state="BN_DIAGNOSTIC_ACTIVE",
                ),
            )
            if worker is None:
                environment = dict(
                    os.environ,
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    PYTHONDONTWRITEBYTECODE="1",
                    PYTHONNOUSERSITE="1",
                )
                for variable, directory in (
                    ("CUDA_CACHE_PATH", "cuda-cache"),
                    ("TORCHINDUCTOR_CACHE_DIR", "inductor-cache"),
                    ("TRITON_CACHE_DIR", "triton-cache"),
                    ("TMPDIR", "tmp"),
                ):
                    (root / directory).mkdir()
                    environment[variable] = str(root / directory)
                environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                command = [
                    sys.executable,
                    "-m",
                    "experiments.musicfm_bn_execution",
                    "--config",
                    str(config_path),
                    "--approval",
                    str(approval_path),
                    "--output",
                    str(root / "worker"),
                    "--guardian-pid",
                    str(os.getpid()),
                ]
                with (root / "worker.log").open("xb") as log:
                    worker = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        cwd=source,
                        start_new_session=True,
                    )
                tracked = descendants(worker.pid)
                save(
                    root / "launch.json",
                    dict(
                        pid=worker.pid, command=command, epoch=time.time(), config_sha256=config_sha
                    ),
                )
            elif worker.poll() is None:
                tracked.update(descendants(worker.pid))
            else:
                worker.wait()
                owned_resources(config, peaks)
                if worker.returncode != 0:
                    raise RuntimeError(f"Diagnostic worker exited {worker.returncode}")
                if disk_bytes(root) > CAPS["output_bytes"]:
                    raise RuntimeError("Owned diagnostic output cap at completion")
                result = load(root / "worker/result.json")
                if result["status"] not in STATUSES or result["promotion_authorized"] is not False:
                    raise ValueError("Incomplete diagnostic result or unexpected promotion")
                if any(alive(pid, start) for pid, start in tracked.items()):
                    raise RuntimeError("Owned descendants remain after diagnostic worker exit")
                resources(guard_view(config), root)
                owned_resources(config, peaks)
                check_deadline(config, started)
                complete = True
                break
            if now - last_disk >= 1:
                if disk_bytes(root) > CAPS["output_bytes"]:
                    raise RuntimeError("Owned diagnostic output cap")
                last_disk = now
            time.sleep(0.05)
    except BaseException as failure:
        error = repr(failure)
        if worker is not None:
            try:
                stop_and_wait(worker, tracked)
            except BaseException as cleanup_failure:
                cleanup_error = repr(cleanup_failure)
        save(
            root / "failure.json",
            dict(
                error=error,
                cleanup_error=cleanup_error,
                worker_started=worker is not None,
                remaining_owned={pid: start for pid, start in tracked.items() if alive(pid, start)},
            ),
        )
        raise
    finally:
        save(
            root / "controller.json",
            dict(
                complete=complete,
                error=error,
                cleanup_error=cleanup_error,
                worker_started=worker is not None,
                worker_exit=worker.returncode if worker is not None else None,
                peaks=peaks,
                terminal_resources=owned_resources(config, peaks, enforce=False),
                wall_seconds=time.monotonic() - started,
                config_sha256=config_sha,
            ),
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
