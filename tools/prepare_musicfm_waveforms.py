"""Prospective bounded24k preparation; never train, score, acquire audio or repair pools."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.musicfm_crop_io import (
    RATE,
    WINDOW,
    compare,
    decode_command,
    file_identity,
    load,
    runtime_files,
    save,
    seek_geometry,
    sha,
)
from tools.run_cqt_expansion import child_cpu, disk_bytes, process_tree, write_json

PLAN_SHA = "c3e59ffb764ca5f51444447b0a18a69543228e60a9e3723d925ddd82b0fc8373"
LABEL_SHA = "b8942b5913b8ebacdf2fc9e0ebca72a934aaa7c56d4e1a97d0f60fac84354323"
MANIFEST_SHA = "fcbc61081166f1845efe8af1ec2d7f807bb2d0a906dd93728b735e287c855291"
DECODER_SHA = "3efaf109c547acaf9479a40d23055680454234560eb17ffad762151a756114ac"
RUNTIME_CONFIG_SHA = "30144ca9abf7a8258983856c13cad1be544acad4ad556269769039ec2033721b"


def fixed_start(identity: str, samples: int) -> int:
    """One predeclared uniform whole-track start; no label, duration stratum or model."""
    if samples <= 0:
        raise ValueError("Empty decoded waveform")
    seed = int(hashlib.sha256(("musicfm24k-one-crop-v1:" + identity).encode()).hexdigest()[:16], 16)
    return random.Random(seed).randrange(max(0, samples - WINDOW) + 1)


def identity_check(previous: list[dict[str, Any]], current: dict[str, Any]) -> None:
    if any(
        row["role"] != current["role"] or row["target"] != current["target"] for row in previous
    ):
        raise ValueError("24k canonical cross-role or conflicting-target family; no repair allowed")


def make_config(plan: Path, labels: Path, manifest: Path, runtime: Path) -> dict[str, Any]:
    if tuple(sha(p) for p in (plan, labels, manifest, runtime)) != (
        PLAN_SHA,
        LABEL_SHA,
        MANIFEST_SHA,
        RUNTIME_CONFIG_SHA,
    ):
        raise ValueError("Different approved inputs")
    p, exported, m, rt = load(plan), load(labels), load(manifest), load(runtime)
    if m.get("complete") is not True or len(m["rows"]) != 5638:
        raise ValueError("Require checked5638 derivative")
    pools = {
        "fit": p["bp_fit"] + p["expanded_fma_fit"],
        "validation": p["bp_selection_guard"] + p["fma_inner_selection"],
    }
    if (len(p["bp_fit"]), len(p["expanded_fma_fit"]), len(pools["validation"])) != (
        1229,
        4033,
        376,
    ):
        raise ValueError("Eligibility counts differ")
    ids = pools["fit"] + pools["validation"]
    if len(ids) != len(set(ids)) or set(ids) != set(exported["labels"]):
        raise ValueError("Pool/target membership mismatch")
    by_id = {r["id"]: r for r in m["rows"]}
    if set(by_id) != set(ids) or len(by_id) != len(m["rows"]):
        raise ValueError("Prepared membership differs")
    rows = []
    for role in ("fit", "validation"):
        for identity in sorted(pools[role]):
            source = by_id[identity]
            target = exported["labels"][identity]
            if type(target) is not int or not 0 <= target < 24:
                raise ValueError("Invalid target encoding")
            rows.append(
                {
                    "id": identity,
                    "role": role,
                    "target": target,
                    **{k: source[k] for k in ("path", "sha256", "canonical_samples")},
                }
            )
    return {
        "format": "musicfm24k-preparation-v1",
        "status": "draft_requires_root_window_approval",
        "tool_sha256": sha(Path(__file__)),
        "decoder_source_sha256": DECODER_SHA,
        "supervisor_source_sha256": sha(Path(__file__).with_name("run_cqt_expansion.py")),
        "plan_sha256": PLAN_SHA,
        "labels_sha256": LABEL_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "prior_runtime_config_sha256": RUNTIME_CONFIG_SHA,
        "runtime_file_sha256": rt["runtime_file_sha256"],
        "ffmpeg": rt["ffmpeg"],
        "ffmpeg_version": rt["ffmpeg_version"],
        "affinity": [8, 9],
        "caps": {
            "cpu_seconds": 6000,
            "wall_seconds": 7200,
            "rss_bytes": 2 * 1024**3,
            "owned_bytes": 8 * 1024**3,
            "reference_bytes": 256 * 1024**2,
        },
        "not_before_epoch": None,
        "deadline_epoch": None,
        "rows": rows,
    }


def seal(config: Path, begin: str, end: str) -> dict[str, Any]:
    value = load(config)
    first, last = datetime.fromisoformat(begin), datetime.fromisoformat(end)
    if first.tzinfo is None or last.tzinfo is None:
        raise ValueError("Explicit UTC offset required")
    value.update(
        not_before_epoch=first.timestamp(),
        deadline_epoch=last.timestamp(),
        draft_sha256=sha(config),
        status="sealed_pending_root_approval",
    )
    if not 0 < last.timestamp() - first.timestamp() <= value["caps"]["wall_seconds"]:
        raise ValueError("Invalid prospective window")
    return value


def full_identity(path: Path) -> tuple[str, int]:
    """Hash exactF32LE bytes and check finite values in bounded blocks."""
    import numpy as np

    if path.stat().st_size == 0 or path.stat().st_size % 4:
        raise ValueError("Empty/misaligned waveform")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 65536), b""):
            if not np.isfinite(np.frombuffer(block, dtype="<f4")).all():
                raise ValueError("Nonfinite decoded waveform")
            digest.update(block)
    return digest.hexdigest(), path.stat().st_size // 4


class Preparation:
    """One sequential worker; reuse frozen commands and existing resource helpers."""

    def __init__(self, config: dict[str, Any], output: Path) -> None:
        self.config = config
        self.output = output
        self.began = time.monotonic()
        self.peak_rss = 0
        self.peak_cpu = 0.0
        self.last_disk_time = 0.0
        self.last_disk_bytes = 0
        self.rows: list[dict[str, Any]] = []
        self.families: dict[str, list[dict[str, Any]]] = {}

    def disk_usage(self, force: bool = False) -> int:
        if force or time.monotonic() - self.last_disk_time >= 1.0:
            self.last_disk_bytes = disk_bytes(self.output)
            self.last_disk_time = time.monotonic()
        return self.last_disk_bytes

    def guard(self, force_disk: bool = False) -> None:
        rss, cpu = process_tree(os.getpid())
        self.peak_rss, self.peak_cpu = max(self.peak_rss, rss), max(self.peak_cpu, cpu)
        c = self.config["caps"]
        if (
            rss > c["rss_bytes"]
            or cpu > c["cpu_seconds"]
            or time.monotonic() - self.began > c["wall_seconds"]
            or time.time() >= self.config["deadline_epoch"]
            or self.disk_usage(force_disk) > c["owned_bytes"]
        ):
            raise RuntimeError("Fixed preparation resource/UTC cutoff")

    def decode(self, command: list[str], path: Path, stderr: Path, maximum: int) -> dict[str, Any]:
        self.guard()
        began, cpu = time.monotonic(), child_cpu()
        with path.open("xb") as out, stderr.open("xb") as err:
            worker = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
            try:
                while worker.poll() is None:
                    self.guard()
                    if path.stat().st_size > maximum:
                        raise ValueError("Decoded file size cap")
                    time.sleep(0.05)
                if worker.returncode or path.stat().st_size > maximum:
                    raise ValueError(f"Decode exit/size failure: {worker.returncode}")
            except BaseException:
                if worker.poll() is None:
                    os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
                raise
        self.guard()
        return {
            "command": command,
            "exit": worker.returncode,
            "wall_seconds": time.monotonic() - began,
            "child_cpu_seconds": child_cpu() - cpu,
            "stderr": str(stderr.relative_to(self.output)),
            "stderr_sha256": sha(stderr),
            "stderr_bytes": stderr.stat().st_size,
        }

    def prepare(self, row: dict[str, Any], ordinal: int) -> None:
        import numpy as np

        source = Path(row["path"])
        initial = file_identity(source)
        self.guard()
        if sha(source) != row["sha256"]:
            raise ValueError("Source SHA mismatch")
        full, crop = self.output / "reference.f32le", self.output / "crop.f32le"
        ffmpeg = Path(self.config["ffmpeg"])
        full_exec = self.decode(
            decode_command(ffmpeg, source, None, WINDOW),
            full,
            self.output / "logs" / f"{ordinal:04}-full.stderr",
            self.config["caps"]["reference_bytes"],
        )
        self.guard()
        waveform_sha, samples = full_identity(full)
        start, length = fixed_start(row["id"], samples), min(WINDOW, samples)
        crop_exec = self.decode(
            decode_command(ffmpeg, source, start, length),
            crop,
            self.output / "logs" / f"{ordinal:04}-crop.stderr",
            2 * 1024**2,
        )
        if crop.stat().st_size % 4:
            raise ValueError("Misaligned cropped F32 stream")
        reference = np.memmap(full, mode="r", dtype="<f4")
        _, trim, _ = seek_geometry(start, length)
        candidate = np.fromfile(crop, dtype="<f4")[trim : trim + length]
        comparison = compare(reference[start : start + length], candidate)
        result: dict[str, Any] = {
            **row,
            "waveform_sha256": waveform_sha,
            "samples": samples,
            "bytes": samples * 4,
            "projected_samples_from_canonical22050": (row["canonical_samples"] * RATE + 22049)
            // 22050,
            "sample_delta_from_projection": samples
            - (row["canonical_samples"] * RATE + 22049) // 22050,
            "source_stable_identity": initial,
            "crop_start": start,
            "crop": comparison,
            "full_execution": full_exec,
            "crop_execution": crop_exec,
        }
        if file_identity(source) != initial:
            raise ValueError("Source changed during preparation")
        if row["role"] == "validation":
            result["validation_cache"] = f"validation/{ordinal:04}-{waveform_sha}.f32le"
        family = self.families.setdefault(waveform_sha, [])
        # Record before any invariant failure; never repair pools/targets or resume silently.
        save(self.output / "records" / f"{ordinal:04}.json", result)
        identity_check(family, result)
        if not comparison["passed"]:
            reference[start : start + length].tofile(self.output / "failed-reference-crop.f32le")
            candidate.tofile(self.output / "failed-candidate-crop.f32le")
            raise ValueError("Fixed waveform-bit/finite/length gate failed")
        family.append(result)
        del reference
        if row["role"] == "validation":
            cached = self.output / result["validation_cache"]
            full.rename(cached)
            cached.chmod(0o444)
        else:
            full.unlink()
        crop.unlink()
        self.rows.append(result)
        self.guard()
        write_json(self.output / "progress.json", self.progress(False))

    def progress(self, complete: bool) -> dict[str, Any]:
        return {
            "complete": complete,
            "tracks": len(self.rows),
            "validation_tracks": sum(r["role"] == "validation" for r in self.rows),
            "wall_seconds": time.monotonic() - self.began,
            "sampled_peak_tree_rss_bytes": self.peak_rss,
            "sampled_tree_cpu_seconds": self.peak_cpu,
            "child_cpu_seconds": child_cpu(),
            "owned_bytes": self.last_disk_bytes,
            "disk_sample_age_seconds": time.monotonic() - self.last_disk_time,
            "disk_requested_interval_seconds": 1.0,
        }


def execute(config_path: Path, approval_path: Path, output: Path) -> None:
    config, approval = load(config_path), load(approval_path)
    if approval.get("authorized") is not True or approval.get("config_sha256") != sha(config_path):
        raise ValueError("Missing exact root approval")
    if (
        config["tool_sha256"] != sha(Path(__file__))
        or sha(Path(__file__).with_name("musicfm_crop_io.py")) != DECODER_SHA
        or config["supervisor_source_sha256"]
        != sha(Path(__file__).with_name("run_cqt_expansion.py"))
        or runtime_files(Path(config["ffmpeg"])) != config["runtime_file_sha256"]
    ):
        raise ValueError("Execution source/runtime changed")
    if sys.platform != "linux" or os.sched_getaffinity(0) != {8, 9}:
        raise ValueError("Require fixed Linux CPUs8,9")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(
        os.environ.get(k) != "1"
        for k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    ):
        raise ValueError("Require CUDA masked/native threads1")
    if not config["not_before_epoch"] <= time.time() < config["deadline_epoch"]:
        raise ValueError("Outside prospective window")
    if len(config["rows"]) != 5638 or len({r["id"] for r in config["rows"]}) != 5638:
        raise ValueError("Require unique approved5638")
    output.mkdir(exist_ok=False)
    for name in ("logs", "records", "validation"):
        (output / name).mkdir()
    stage = Preparation(config, output)
    try:
        for number, row in enumerate(config["rows"], 1):
            stage.prepare(row, number)
        if len(stage.rows) != 5638 or sum(r["role"] == "validation" for r in stage.rows) != 376:
            raise ValueError("Incomplete fixed cache")
        validation = [r for r in stage.rows if r["role"] == "validation"]
        for row in validation:
            cached = output / row["validation_cache"]
            if not cached.is_file() or cached.is_symlink() or cached.stat().st_size != row["bytes"]:
                raise ValueError("Incomplete validation cache file")
        stage.guard(force_disk=True)
        write_json(
            output / "validation-cache-complete.json",
            {
                "complete": True,
                "config_sha256": sha(config_path),
                "tracks": 376,
                "bytes": sum(r["bytes"] for r in validation),
                "rows": validation,
            },
        )
        write_json(
            output / "manifest.json",
            {
                "complete": True,
                "config_sha256": sha(config_path),
                "approval_sha256": sha(approval_path),
                "tool_sha256": sha(Path(__file__)),
                "runtime_file_sha256": config["runtime_file_sha256"],
                "plan_sha256": PLAN_SHA,
                "labels_sha256": LABEL_SHA,
                "resources": stage.progress(True),
                "rows": stage.rows,
            },
        )
        stage.guard(force_disk=True)
        write_json(output / "progress.json", stage.progress(True))
    except BaseException as error:
        save(
            output / "failure.json",
            {"complete": False, "error": repr(error), "resources": stage.progress(False)},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    plan = commands.add_parser("plan")
    for name in ("plan", "labels", "manifest", "runtime", "output"):
        plan.add_argument("--" + name, required=True, type=Path)
    sealing = commands.add_parser("seal")
    for name in ("config", "output"):
        sealing.add_argument("--" + name, required=True, type=Path)
    for name in ("begin", "end"):
        sealing.add_argument("--" + name, required=True)
    run = commands.add_parser("run")
    for name in ("config", "approval", "output"):
        run.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "plan":
        save(args.output, make_config(args.plan, args.labels, args.manifest, args.runtime))
    elif args.mode == "seal":
        save(args.output, seal(args.config, args.begin, args.end))
    else:
        execute(args.config, args.approval, args.output)


if __name__ == "__main__":
    main()
