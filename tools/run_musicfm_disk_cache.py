"""Retained finite outer disk-preparation controller; no idle RAM or model guardian."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

from tools.musicfm_crop_io import load, save, sha
from tools.prepare_musicfm_ram_cache import capacity_guard
from tools.run_cqt_expansion import process_tree
from tools.run_musicfm_ram_cache import descendants, stop_and_wait


def supervise(config_path: Path, approval_path: Path, output: Path) -> None:
    value, approval = load(config_path), load(approval_path)
    if (
        approval.get("authorized") is not True
        or approval.get("allow_training") is not False
        or approval.get("config_sha256") != sha(config_path)
        or value["source_sha256"]["tools/run_musicfm_disk_cache.py"] != sha(Path(__file__))
    ):
        raise ValueError("Exact source/config preprocessing-only approval required")
    if not value["not_before_epoch"] <= time.time() < value["deadline_epoch"]:
        raise ValueError("Outside finite approved window")
    output.mkdir(exist_ok=False)
    began = time.monotonic()
    capacity_guard(value, output)
    command = [
        sys.executable,
        "-m",
        "tools.prepare_musicfm_disk_cache",
        "run",
        "--config",
        str(config_path),
        "--approval",
        str(approval_path),
        "--output",
        str(output / "prepared"),
    ]
    with (output / "worker.log").open("xb") as log:
        worker = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        tracked = descendants(worker.pid)
        peak_rss = 0
        peak_cpu = 0.0
        last_disk = 0.0
        owned = 0
        save(
            output / "launch.json",
            {
                "pid": worker.pid,
                "command": command,
                "epoch": time.time(),
                "config_sha256": sha(config_path),
                "source_sha256": sha(Path(__file__)),
            },
        )
        error = None
        try:
            while worker.poll() is None:
                tracked.update(descendants(worker.pid))
                capacity_guard(value, output)
                rss, cpu = process_tree(worker.pid)
                peak_rss = max(peak_rss, rss)
                peak_cpu = max(peak_cpu, cpu)
                if time.monotonic() - last_disk >= 1:
                    last_disk = time.monotonic()
                    owned = 0
                    for path in output.rglob("*"):
                        try:
                            if path.is_file() and not path.is_symlink():
                                owned += path.stat().st_size
                        except FileNotFoundError:
                            pass
                caps = value["caps"]
                if (
                    time.time() >= value["deadline_epoch"]
                    or time.monotonic() - began > caps["wall_seconds"]
                    or rss > caps["rss_bytes"]
                    or cpu > caps["cpu_seconds"]
                    or owned > caps["owned_bytes"]
                ):
                    raise RuntimeError("Outer finite resource/UTC cap")
                time.sleep(0.05)
            worker.wait()
            capacity_guard(value, output)
            if worker.returncode != 0:
                raise RuntimeError(f"Preparation exit={worker.returncode}")
            manifest = load(output / "prepared/manifest.json")
            if manifest.get("complete") is not True or len(manifest["rows"]) != 5638:
                raise ValueError("No complete fixed cache")
            save(
                output / "success.json",
                {
                    "complete": True,
                    "allow_training": False,
                    "manifest_sha256": sha(output / "prepared/manifest.json"),
                    "epoch": time.time(),
                },
            )
        except BaseException as exc:
            error = repr(exc)
            stop_and_wait(worker, tracked)
            raise
        finally:
            save(
                output / "controller.json",
                {
                    "exit": worker.returncode,
                    "error": error,
                    "wall_seconds": time.monotonic() - began,
                    "sampled_peak_worker_rss": peak_rss,
                    "sampled_peak_worker_cpu": peak_cpu,
                    "owned_bytes_last_sample": owned,
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "approval", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    signal.signal(
        signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError("Controller terminated"))
    )
    supervise(args.config, args.approval, args.output)


if __name__ == "__main__":
    main()
