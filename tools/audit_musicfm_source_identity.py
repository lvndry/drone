"""Read-only physical-identity refresh; no decoding, models, source edits or admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

MANIFEST_SHA = "7d56908dc5e84c165c7844d1e810ed678123ae7d0a93939c60a2e600b6402b4c"
PROTECTED = [(1478814, 35698640), (3448680, 111966827), (1481756, 35700162), (3451013, 111967770)]


def identity(path: Path) -> list[int]:
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def write(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def sha(path: Path, watch: Callable[[], None]) -> str:  # noqa: ANN401
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            watch()
            digest.update(block)
    return digest.hexdigest()


def checked(
    path: Path,
    expected_sha: str,
    expected_size: int,
    watch: Callable[[], None],
    readonly: bool = False,
) -> dict[str, Any]:  # noqa: ANN401
    before = identity(path)
    mode = path.stat().st_mode
    if before[2] != expected_size or readonly and mode & 0o222:
        raise ValueError("Size/read-only contract mismatch: " + str(path))
    actual = sha(path, watch)
    after = identity(path)
    if actual != expected_sha or before != after:
        raise ValueError("Content mismatch or unstable identity: " + str(path))
    return {
        "identity": before,
        "sha256": actual,
        "bytes": before[2],
        "resolved_path": str(path.resolve(strict=True)),
        "mode": mode,
    }


class Watch:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.began = time.monotonic()
        self.lock = threading.Lock()
        self.last = 0.0
        self.stop = threading.Event()
        self.peak_rss = 0
        self.cpu = 0.0
        self.last_disk = 0.0
        self.owned = 0

    def __call__(self) -> None:
        if self.stop.is_set():
            raise RuntimeError("Identity audit stopped")
        with self.lock:
            if time.monotonic() - self.last < 0.1:
                return
            self.last = time.monotonic()
            for pid, start in PROTECTED:
                fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[19]) != start or fields[0] in ("Z", "X"):
                    raise RuntimeError("Protected process identity changed")
            rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            self.peak_rss = max(self.peak_rss, rss)
            usages = [
                resource.getrusage(who) for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
            ]
            self.cpu = sum(value.ru_utime + value.ru_stime for value in usages)
            if time.monotonic() - self.last_disk >= 1:
                self.last_disk = time.monotonic()
                self.owned = sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file())
            if (
                self.peak_rss > 512 * 1024**2
                or self.cpu > 600
                or time.monotonic() - self.began > 900
                or self.owned > 20 * 1024**2
            ):
                raise RuntimeError("Fixed audit CPU/wall/RSS/output cap")

    def receipt(self) -> dict[str, Any]:
        return {
            "wall_seconds": time.monotonic() - self.began,
            "cpu_seconds": self.cpu,
            "sampled_peak_rss": self.peak_rss,
            "owned_bytes_last_sample": self.owned,
        }


def audit(manifest: Path, output: Path) -> None:
    affinity = getattr(os, "sched_getaffinity", None)
    if (
        affinity is None
        or affinity(0) != {10, 11, 12, 13}
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        raise ValueError("Require CPUs10–13 and CUDA masked")
    if any(
        os.environ.get(k) != "1"
        for k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    ):
        raise ValueError("Require numeric threads1")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != MANIFEST_SHA:
        raise ValueError("Canonical manifest changed")
    original = json.loads(manifest.read_bytes())
    rows = original["rows"]
    if (
        original.get("complete") is not True
        or len(rows) != 5638
        or len({r["id"] for r in rows}) != 5638
    ):
        raise ValueError("Wrong complete5638 membership")
    output.mkdir(exist_ok=False)
    watch = Watch(output)
    watch()
    started = time.time()

    def terminated(signum: int, frame: object) -> None:
        watch.stop.set()
        raise TimeoutError("Audit terminated")

    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGALRM, terminated)
    signal.alarm(900)
    records: dict[str, Any] = {}
    invariant_keys = (
        "id",
        "path",
        "role",
        "target",
        "sha256",
        "waveform_sha256",
        "samples",
        "bytes",
    )

    def one(row: dict[str, Any]) -> dict[str, Any]:
        source = checked(Path(row["path"]), row["sha256"], row["source_stable_identity"][2], watch)
        result = {
            "invariants": {key: row[key] for key in invariant_keys},
            "source_identity_old": row["source_stable_identity"],
            "source_current": source,
        }
        if row["role"] == "validation":
            result["validation_cache_path"] = row["cache"]
            result["validation_identity_old"] = row["cache_identity"]
            result["validation_current"] = checked(
                Path(row["cache"]), row["waveform_sha256"], row["bytes"], watch, True
            )
        return result

    pool = ThreadPoolExecutor(max_workers=4)
    try:
        futures = {pool.submit(one, row): row["id"] for row in rows}
        for future in as_completed(futures):
            watch()
            records[futures[future]] = future.result()
        pool.shutdown(wait=True)
        watch()
        representatives = [
            Path(next(r["path"] for r in rows if r["id"].startswith("fmak:"))),
            Path(next(r["path"] for r in rows if r["id"].startswith("beatport:"))),
            Path(next(r["cache"] for r in rows if r["role"] == "validation")),
        ]
        mounts = {"proc_mountinfo": Path("/proc/self/mountinfo").read_text(), "df": {}}
        for path in representatives:
            mounts["df"][str(path)] = subprocess.check_output(
                ["df", "-B1", "-T", str(path)], text=True, timeout=10
            )
        write(output / "mount-metadata.json", mounts)
        ordered = [records[r["id"]] for r in rows]
        if any(
            record["invariants"] != {key: row[key] for key in invariant_keys}
            for record, row in zip(ordered, rows, strict=True)
        ):
            raise ValueError("Invariant preservation failure")
        write(
            output / "identity-overlay.json",
            {
                "complete": True,
                "format": "musicfm-source-physical-identity-overlay-v1",
                "canonical_manifest_sha256": MANIFEST_SHA,
                "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "rows": ordered,
                "no_decode_model_admission_or_repair": True,
            },
        )
        watch()
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == MANIFEST_SHA
        if Path(original["cache_ownership"]["root"]).exists():
            raise ValueError("Old fitting RAM root unexpectedly present")
        write(
            output / "receipt.json",
            {
                "status": "COMPLETE_READ_ONLY_IDENTITIES_REFRESHED_NOT_ACTIVATED",
                "sources": 5638,
                "validation_files": sum(r["role"] == "validation" for r in rows),
                "source_identities_changed": sum(
                    r["source_identity_old"] != r["source_current"]["identity"] for r in ordered
                ),
                "validation_identities_changed": sum(
                    "validation_current" in r
                    and r["validation_identity_old"] != r["validation_current"]["identity"]
                    for r in ordered
                ),
                "manifest_sha256": MANIFEST_SHA,
                "overlay_sha256": sha(output / "identity-overlay.json", watch),
                "mount_metadata_sha256": sha(output / "mount-metadata.json", watch),
                "started_epoch": started,
                "finished_epoch": time.time(),
                "protected_starts_unchanged": PROTECTED,
                "fit_cache_absent": True,
                "resources": watch.receipt(),
            },
        )
    except BaseException as error:
        watch.stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
        write(
            output / "failure.json",
            {
                "complete": False,
                "error": repr(error),
                "completed_sources": len(records),
                "resources": watch.receipt(),
            },
        )
        raise
    finally:
        signal.alarm(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.manifest, args.output)


if __name__ == "__main__":
    main()
