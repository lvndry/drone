"""Prospective full-decode-once cache. No allocation without sealed root approval."""

from __future__ import annotations

import argparse
import hashlib
import mmap
import os
import signal
import stat
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tools.musicfm_crop_io import decode_command, file_identity, load, runtime_files, save, sha
from tools.prepare_musicfm_waveforms import (
    DECODER_SHA,
    LABEL_SHA,
    PLAN_SHA,
    Preparation,
    identity_check,
    make_config,
    seal,
)
from tools.run_k10_v2_pair import protected_live

GiB = 1024**3
FIT_LIMIT = 99_000_000_000
MEMORY_RESERVE = 40 * GiB
MOUNT_RESERVE = GiB
DISK_RESERVE = 32 * GiB
RAM_PARENT = Path("/dev/shm")


def rounded(size: int, page: int) -> int:
    return (size + page - 1) // page * page


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Missing MemAvailable")


def available(path: Path) -> int:
    value = os.statvfs(path)
    return value.f_bavail * value.f_frsize


def forecast(rows: list[dict[str, Any]], page: int) -> dict[str, int]:
    result = {"fit": 0, "validation": 0}
    for row in rows:
        size = 4 * ((row["canonical_samples"] * 24000 + 22049) // 22050)
        result[row["role"]] += rounded(size, page)
    return result


def capacity_guard(config: dict[str, Any], output: Path, remaining: int = 0) -> None:
    protected_live(config["protected_starts"])
    if mem_available() < MEMORY_RESERVE + remaining:
        raise RuntimeError("Insufficient MemAvailable; preserve serving priority")
    if available(RAM_PARENT) < MOUNT_RESERVE + remaining:
        raise RuntimeError("Insufficient tmpfs reserve")
    if available(output) < DISK_RESERVE:
        raise RuntimeError("Insufficient ordinary-disk reserve")


class OwnedCache:
    """Only this witnessed directory and explicitly enumerated filenames can be unlinked."""

    def __init__(self, root: Path, output: Path, names: list[str], token: str) -> None:
        if root.parent != RAM_PARENT or root.name != "drone-musicfm-fit-" + token:
            raise ValueError("Unexpected dedicated cache path")
        if len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
            raise ValueError("Invalid ownership token")
        self.root, self.output, self.names, self.token = root, output, set(names), token
        root.mkdir(mode=0o700, exist_ok=False)
        info = root.lstat()
        self.witness = {
            "token": token,
            "device": info.st_dev,
            "inode": info.st_ino,
            "uid": os.getuid(),
            "root": str(root),
            "names": sorted(self.names),
        }
        save(root / "OWNER.json", self.witness)
        save(output / "cache-ownership.json", self.witness)

    def cleanup(self) -> None:
        info = self.root.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_dev != self.witness["device"]
            or info.st_ino != self.witness["inode"]
            or info.st_uid != self.witness["uid"]
            or load(self.root / "OWNER.json") != self.witness
            or load(self.output / "cache-ownership.json") != self.witness
        ):
            raise RuntimeError("Ownership mismatch; refuse cleanup")
        members = list(self.root.iterdir())
        for path in members:
            entry = path.lstat()
            if (
                path.name not in self.names | {"OWNER.json"}
                or not stat.S_ISREG(entry.st_mode)
                or entry.st_uid != os.getuid()
                or entry.st_nlink != 1
                or entry.st_dev != self.witness["device"]
            ):
                raise RuntimeError("Unknown/symlink cache member; refuse cleanup")
        for path in members:
            if path.name != "OWNER.json":
                path.unlink()
        (self.root / "OWNER.json").unlink()
        self.root.rmdir()


def cleanup_from_config(config: dict[str, Any], output: Path) -> None:
    """Outer supervisor calls only AFTER all own workers have exited/unmapped."""
    root = Path(config["ram_root"])
    witness = load(output / "cache-ownership.json")
    if (
        root.parent != RAM_PARENT
        or root.name != "drone-musicfm-fit-" + config["cache_token"]
        or witness["root"] != str(root)
        or witness["token"] != config["cache_token"]
    ):
        raise RuntimeError("Configuration/ownership mismatch; refuse cleanup")
    cache = OwnedCache.__new__(OwnedCache)
    cache.root, cache.output, cache.token = root, output, config["cache_token"]
    cache.names, cache.witness = set(witness["names"]), witness
    cache.cleanup()


def make_ram_config(plan: Path, labels: Path, manifest: Path, runtime: Path) -> dict[str, Any]:
    value = make_config(plan, labels, manifest, runtime)
    value["format"] = "musicfm24k-full-ram-cache-v1"
    value["tool_sha256"] = sha(Path(__file__))
    value["preparation_source_sha256"] = sha(
        Path(__file__).with_name("prepare_musicfm_waveforms.py")
    )
    value["guard_source_sha256"] = sha(Path(__file__).with_name("run_k10_v2_pair.py"))
    value["cache_token"] = uuid.uuid4().hex
    value["ram_root"] = str(RAM_PARENT / ("drone-musicfm-fit-" + value["cache_token"]))
    value["protected_starts"] = [
        {"pid": 1478814, "start_ticks": 35698640},
        {"pid": 3448680, "start_ticks": 111966827},
        {"pid": 1481756, "start_ticks": 35700162},
        {"pid": 3451013, "start_ticks": 111967770},
    ]
    value["caps"].update(
        fit_page_bytes=FIT_LIMIT,
        mem_available_bytes=MEMORY_RESERVE,
        tmpfs_reserve_bytes=MOUNT_RESERVE,
        disk_reserve_bytes=DISK_RESERVE,
        owned_bytes=8 * GiB,
    )
    value["status"] = "PROSPECTIVE_NO_ALLOCATION_OR_TRAINING_AUTHORIZATION"
    return value


class RAMPreparation(Preparation):
    def __init__(self, config: dict[str, Any], output: Path, cache: OwnedCache) -> None:
        super().__init__(config, output)
        self.cache = cache
        self.page = os.sysconf("SC_PAGE_SIZE")
        self.completed_fit_bytes = rounded((cache.root / "OWNER.json").stat().st_size, self.page)
        self.active_fit: Path | None = None
        self.peak_fit_bytes = 0

    def guard(self, force_disk: bool = False) -> None:
        super().guard(force_disk)
        capacity_guard(self.config, self.output)
        active = (
            rounded(self.active_fit.stat().st_size, self.page)
            if self.active_fit and self.active_fit.exists()
            else 0
        )
        self.peak_fit_bytes = max(self.peak_fit_bytes, self.completed_fit_bytes + active)
        if self.peak_fit_bytes > FIT_LIMIT:
            raise RuntimeError("Owned page-rounded fitting cache exceeds cap")

    def prepare(self, row: dict[str, Any], ordinal: int) -> None:
        import numpy as np

        self.guard()
        source = Path(row["path"])
        initial = file_identity(source)
        if sha(source) != row["sha256"]:
            raise ValueError("Source SHA mismatch")
        name = f"wave-{ordinal:04d}.f32le"
        target = (self.cache.root if row["role"] == "fit" else self.output / "validation") / name
        if row["role"] == "fit":
            self.active_fit = target
        execution = self.decode(
            decode_command(Path(self.config["ffmpeg"]), source, None, 12 * 24000),
            target,
            self.output / "logs" / f"{ordinal:04d}-full.stderr",
            self.config["caps"]["reference_bytes"],
        )
        digest = hashlib.sha256()
        size = 0
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 65536), b""):
                self.guard()
                if len(block) % 4 or not np.isfinite(np.frombuffer(block, dtype="<f4")).all():
                    raise ValueError("Nonfinite/misaligned full waveform")
                digest.update(block)
                size += len(block)
        if size == 0 or file_identity(source) != initial:
            raise ValueError("Empty waveform or unstable source")
        record = {
            **row,
            "waveform_sha256": digest.hexdigest(),
            "samples": size // 4,
            "bytes": size,
            "cache": str(target),
            "full_execution": execution,
            "source_stable_identity": initial,
        }
        target.chmod(0o444)
        record["cache_identity"] = file_identity(target)
        save(self.output / "records" / f"{ordinal:04d}.json", record)
        family = self.families.setdefault(record["waveform_sha256"], [])
        identity_check(family, record)
        family.append(record)
        self.rows.append(record)
        if row["role"] == "fit":
            self.completed_fit_bytes += rounded(size, self.page)
            self.active_fit = None
        self.guard()
        save_or_replace(self.output / "progress.json", self.progress(False))

    def progress(self, complete: bool) -> dict[str, Any]:
        return {
            **super().progress(complete),
            "fit_page_bytes": self.completed_fit_bytes,
            "peak_fit_page_bytes": self.peak_fit_bytes,
            "mem_available_bytes": mem_available(),
            "tmpfs_available_bytes": available(RAM_PARENT),
            "rss_scope": "sampled process RSS; owned tmpfs pages accounted separately",
        }


def save_or_replace(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".next")
    save(temporary, value)
    temporary.replace(path)


def read_crop(row: dict[str, Any], start: int, length: int, *, watch: Callable[[], None]) -> Any:  # noqa: ANN401
    """Caller MUST maintain capacity/protected-start watch for the whole future fit."""
    import numpy as np

    if start < 0 or length <= 0 or start + length > row["samples"]:
        raise ValueError("Crop outside actual complete waveform")
    watch()
    path = Path(row["cache"])
    if path.is_symlink() or list(file_identity(path)) != list(row["cache_identity"]):
        raise ValueError("Immutable waveform cache identity changed")
    with (
        path.open("rb") as stream,
        mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
    ):
        result = np.frombuffer(mapped, dtype="<f4", offset=4 * start, count=length).copy()
    watch()
    return result


def execute(config_path: Path, approval_path: Path, output: Path) -> None:
    config, approval = load(config_path), load(approval_path)
    if (
        approval.get("authorized") is not True
        or approval.get("config_sha256") != sha(config_path)
        or approval.get("allow_training") is not False
    ):
        raise ValueError("Require source-bound preprocessing-only root approval")
    expected = {
        "tool_sha256": Path(__file__),
        "preparation_source_sha256": Path(__file__).with_name("prepare_musicfm_waveforms.py"),
        "guard_source_sha256": Path(__file__).with_name("run_k10_v2_pair.py"),
        "decoder_source_sha256": Path(__file__).with_name("musicfm_crop_io.py"),
        "supervisor_source_sha256": Path(__file__).with_name("run_cqt_expansion.py"),
    }
    if any(sha(path) != config[name] for name, path in expected.items()):
        raise ValueError("Bound source changed")
    if (
        config["decoder_source_sha256"] != DECODER_SHA
        or runtime_files(Path(config["ffmpeg"])) != config["runtime_file_sha256"]
    ):
        raise ValueError("Frozen full decoder/runtime changed")
    if sys.platform != "linux" or os.sched_getaffinity(0) != {8, 9}:
        raise ValueError("Require Linux CPUs8,9")
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
        raise ValueError("Outside authorized window")
    rows = config["rows"]
    if (
        len(rows) != 5638
        or len({r["id"] for r in rows}) != 5638
        or sum(r["role"] == "fit" for r in rows) != 5262
    ):
        raise ValueError("Fixed approved membership differs")
    mounts = Path("/proc/mounts").read_text().splitlines()
    if not any(line.split()[1:3] == [str(RAM_PARENT), "tmpfs"] for line in mounts):
        raise ValueError("Dedicated parent must be the existing tmpfs mount")
    output.mkdir(exist_ok=False)
    for name in ("logs", "records", "validation"):
        (output / name).mkdir()
    prediction = forecast(rows, os.sysconf("SC_PAGE_SIZE"))
    save(output / "forecast.json", prediction)
    if prediction["fit"] + 1024**2 > FIT_LIMIT:
        raise ValueError("Page-rounded fitting forecast exceeds cap")
    capacity_guard(config, output, prediction["fit"] + 1024**2)
    if available(output) < DISK_RESERVE + prediction["validation"] + 512 * 1024**2:
        raise ValueError("Validation/log forecast violates ordinary-disk reserve")
    cache = OwnedCache(
        Path(config["ram_root"]),
        output,
        [f"wave-{i:04d}.f32le" for i, r in enumerate(rows, 1) if r["role"] == "fit"],
        config["cache_token"],
    )
    stage = RAMPreparation(config, output, cache)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError("Terminated")))
    try:
        for number, row in enumerate(rows, 1):
            stage.prepare(row, number)
        stage.guard(True)
        save(
            output / "manifest.json",
            {
                "complete": True,
                "allow_training": False,
                "config_sha256": sha(config_path),
                "plan_sha256": PLAN_SHA,
                "labels_sha256": LABEL_SHA,
                "cache_ownership": cache.witness,
                "resources": stage.progress(True),
                "rows": stage.rows,
                "future_fit_contract": "Fresh root activation; verify every waveform hash once, then "
                "continuous global40GiB/tmpfs1GiB/disk32GiB/protected-start monitoring; "
                "on stop join own workers/unmap before witnessed cache cleanup",
            },
        )
    except BaseException as error:
        save_or_replace(
            output / "failure.json",
            {"complete": False, "error": repr(error), "resources": stage.progress(False)},
        )
        cache.cleanup()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    plan = commands.add_parser("plan")
    for name in ("plan", "labels", "manifest", "runtime", "output"):
        plan.add_argument("--" + name, type=Path, required=True)
    sealing = commands.add_parser("seal")
    for name in ("config", "output"):
        sealing.add_argument("--" + name, type=Path, required=True)
    sealing.add_argument("--begin", required=True)
    sealing.add_argument("--end", required=True)
    run = commands.add_parser("run")
    for name in ("config", "approval", "output"):
        run.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "plan":
        save(args.output, make_ram_config(args.plan, args.labels, args.manifest, args.runtime))
    elif args.mode == "seal":
        save(args.output, seal(args.config, args.begin, args.end))
    else:
        execute(args.config, args.approval, args.output)


if __name__ == "__main__":
    main()
