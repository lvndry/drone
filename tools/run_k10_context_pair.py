"""Finite matched CPU context-pair guardian; verified natural teacher completion only."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tools.musicfm_crop_io import load, save, sha
from tools.prepare_musicfm_ram_cache import capacity_guard, save_or_replace
from tools.run_k10_context_cost import disk_bytes, environment
from tools.run_k10_context_cost import worker_watch as worker_watch
from tools.run_musicfm_ram_cache import alive, descendants, proc, stop_and_wait
from tools.run_musicfm_teacher import owned_resources, resources


class TeacherVerifier:
    """Never signal a teacher; allow only its pinned complete-horizon terminal receipt."""

    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value
        self.missing_since: float | None = None
        self.terminal: dict[str, Any] | None = None

    def completion(self) -> dict[str, Any]:
        root = Path(self.value["root"])
        config_path = root / "sealed-config.json"
        if sha(config_path) != self.value["config_sha256"]:
            raise ValueError("Protected teacher configuration changed")
        config = load(config_path)
        run = root / "run-v1"
        controller = load(run / "controller.json")
        if (
            controller.get("state") != "COMPLETE_NO_RAM_OWNERSHIP"
            or controller.get("worker_exit") != 0
        ):
            raise ValueError("Protected teacher did not terminate normally")
        success = load(run / "held-success.json")
        result_path = run / "worker/result.json"
        if success.get("mode") != "fit" or success.get("result_sha256") != sha(result_path):
            raise ValueError("Protected teacher terminal result is unbound")
        result = load(result_path)
        provenance = result.get("provenance", {})
        expected = {
            "config_sha256": self.value["config_sha256"],
            "source_sha256": config["source_sha256"],
            "plan_sha256": config["plan_sha256"],
            "labels_sha256": config["labels_sha256"],
            "initial_state_sha256": config["basis_initial_state_sha256"],
        }
        if (
            result.get("status") != "COMPLETE_FIXED_PILOT_NOT_RELEASE_EVIDENCE"
            or result.get("updates") != 8280
            or result.get("events") != 120
            or result.get("confirmation_used") is not False
            or result.get("calibration_used") is not False
            or any(provenance.get(k) != v for k, v in expected.items())
        ):
            raise ValueError("Protected teacher terminal horizon/provenance differs")
        # Deliberately do not inspect accuracy or continuation: a negative study can finish normally.
        return {
            "state": "VERIFIED_LEGITIMATE_TERMINAL",
            "config_sha256": sha(config_path),
            "controller_sha256": sha(run / "controller.json"),
            "success_sha256": sha(run / "held-success.json"),
            "result_sha256": sha(result_path),
            "updates": 8280,
            "events": 120,
        }

    def __call__(self) -> dict[str, Any]:
        if self.terminal is not None:
            return self.terminal
        if all(alive(x["pid"], x["start_ticks"]) for x in self.value["starts"]):
            return {"state": "ACTIVE_STARTS_VERIFIED"}
        if self.missing_since is None:
            self.missing_since = time.monotonic()
        try:
            self.terminal = self.completion()
            return self.terminal
        except (FileNotFoundError, json.JSONDecodeError):
            # Normal guardian publication follows worker exit; bounded grace never accepts completion.
            if time.monotonic() - self.missing_since <= 10:
                return {"state": "PENDING_TERMINAL_RECEIPT", "grace_seconds": 10}
            raise RuntimeError(
                "Protected teacher exited without verified terminal receipt"
            ) from None


def supervise(config: Path, approval: Path, output: Path) -> None:
    from experiments.k10_context_pair import contract

    value, permit = load(config), load(approval)
    digest = sha(config)
    if (
        permit.get("authorized") is not True
        or permit.get("allow_context_pair") is not True
        or permit.get("allow_training") is not True
        or permit.get("config_sha256") != digest
    ):
        raise ValueError("Exact separately approved matched pair required")
    contract(value)
    teacher = TeacherVerifier(value["teacher"])
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
    teacher_state = {"state": "UNVERIFIED"}
    try:
        teacher_state = teacher()
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
            "experiments.k10_context_pair",
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
            teacher_state = teacher()
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
                    "teacher": teacher_state,
                },
            )
            tracked.update(descendants(child.pid))
            if child.poll() is not None:
                child.wait()
                owned_resources(value, peaks)
                if child.returncode != 0:
                    raise RuntimeError(f"Cost worker exit={child.returncode}")
                result = load(output / "worker/result.json")
                if result.get("status") != "COMPLETE_MATCHED_CONTEXT_PAIR_NOT_PROMOTED":
                    raise ValueError("No complete cost result")
                resources(value, output)
                if disk_bytes(output) > value["caps"]["output_bytes"]:
                    raise RuntimeError("Final owned output cap")
                if teacher_state["state"] == "PENDING_TERMINAL_RECEIPT":
                    time.sleep(0.05)
                    continue
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
                "teacher_state": teacher_state,
                "completed_arm_records": {
                    str(n): sha(output / f"worker/crop-{n}/result.json")
                    for n in (320, 65)
                    if (output / f"worker/crop-{n}/result.json").exists()
                },
                "resource_scope": "guardian plus all thread-spawned live/reaped descendants",
                "wall_seconds": time.monotonic() - start,
                "promotion_authorized": False,
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
