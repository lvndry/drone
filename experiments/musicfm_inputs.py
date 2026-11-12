"""Four bounded decoder threads; ordered crops, one-next-batch prefetch, lossless stderr logs."""

from __future__ import annotations

import base64
import hashlib
import json
import mmap
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from experiments.k10_v2_pilot import Draw
from experiments.musicfm_sampling import crop
from tools.musicfm_crop_io import decode_command, file_identity
from tools.musicfm_source_contract import apply_identity_overlay as apply_identity_overlay
from tools.musicfm_stream_crop import stream_crop

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def source_hash(path: Path, watch: Callable[[], None]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            watch()
            digest.update(block)
    return digest.hexdigest()


class BatchProvider(Protocol):
    def submit(self, requests: list[tuple[str, int, int]]) -> list[Future[NDArray[np.float32]]]: ...
    def resolve(self, ticket: list[Future[NDArray[np.float32]]]) -> list[NDArray[np.float32]]: ...


class StreamProvider:
    """No labels enter decoder threads; completed futures are consumed in request order."""

    def __init__(
        self,
        rows: dict[str, dict[str, Any]],
        output: Path,
        ffmpeg: Path,
        cpus: list[int],
        watch: Callable[[], None],
    ) -> None:
        if len(cpus) != 4 or len(set(cpus)) != 4:
            raise ValueError("Exactly four declared decoder CPUs required")
        fields = ("role", "path", "sha256", "samples", "source_stable_identity", "waveform_sha256")
        self.rows = {identity: {key: row[key] for key in fields} for identity, row in rows.items()}
        self.output, self.ffmpeg, self.cpus = output, ffmpeg, cpus
        output.mkdir(exist_ok=False)
        self.watch = watch
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.local = threading.local()
        self.slots = iter(range(4))
        self.ordinal = 0
        self.pending_batches = 0
        self.resolve_wait_seconds: list[float] = []
        self.logs = [(output / f"worker-{i}.jsonl").open("x") for i in range(4)]

        def initialize() -> None:
            with self.lock:
                self.local.slot = next(self.slots)

        self.pool = ThreadPoolExecutor(max_workers=4, initializer=initialize)

    def guarded(self) -> None:
        if self.stop.is_set():
            raise RuntimeError("Decoder pool stopped")
        with self.lock:
            self.watch()

    def decode(self, number: int, identity: str, start: int, length: int) -> NDArray[np.float32]:
        row = self.rows[identity]
        source = Path(row["path"])
        slot = self.local.slot
        stderr = self.output / f"pending-{number:06d}.stderr"
        began = time.monotonic()
        error = None
        try:
            self.guarded()
            initial = file_identity(source)
            if (
                list(initial) != list(row["source_stable_identity"])
                or source_hash(source, self.guarded) != row["sha256"]
            ):
                raise ValueError("Immutable compressed source changed")

            def watch() -> None:
                self.guarded()
                if stderr.exists() and stderr.stat().st_size > 8 * 1024**2:
                    raise RuntimeError("Per-request stderr exceeds8MiB; retain raw failure")

            command = [
                "taskset",
                "-c",
                str(self.cpus[slot]),
                *decode_command(self.ffmpeg, source, None, 288000),
            ]
            result = stream_crop(
                command,
                samples=row["samples"],
                waveform_sha256=row["waveform_sha256"],
                start=start,
                length=length,
                stderr=stderr,
                watch=watch,
            )
            if file_identity(source) != initial:
                raise ValueError("Source changed during full stream")
            return result
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            size = stderr.stat().st_size if stderr.exists() else 0
            raw = stderr.read_bytes() if size <= 8 * 1024**2 and stderr.exists() else None
            record = {
                "request": number,
                "id": identity,
                "start": start,
                "length": length,
                "source_sha256": row["sha256"],
                "full_waveform_sha256": row["waveform_sha256"],
                "expected_full_samples": row["samples"],
                "wall_seconds": time.monotonic() - began,
                "decoder_cpu": self.cpus[slot],
                "stderr_bytes": size,
                "error": error,
                "stderr_sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
                "stderr_base64": base64.b64encode(raw).decode() if raw is not None else None,
                "retained_raw_stderr": str(stderr)
                if stderr.exists() and (error is not None or raw is None)
                else None,
            }
            self.logs[slot].write(json.dumps(record, sort_keys=True) + "\n")
            self.logs[slot].flush()
            if error is None and raw is not None and stderr.exists():
                stderr.unlink()

    def submit(self, requests: list[tuple[str, int, int]]) -> list[Future[NDArray[np.float32]]]:
        if len(requests) != 32:
            raise ValueError("Exactly32 requests per optimizer update")
        with self.lock:
            if self.pending_batches >= 2:
                raise RuntimeError("Bounded current+next32 queue exceeded")
            self.pending_batches += 1
        result = []
        for identity, start, length in requests:
            if self.rows[identity]["role"] != "fit":
                raise ValueError("Only fitting inputs may stream")
            number = self.ordinal
            self.ordinal += 1
            result.append(self.pool.submit(self.decode, number, identity, start, length))
        return result

    def resolve(self, ticket: list[Future[NDArray[np.float32]]]) -> list[NDArray[np.float32]]:
        began = time.monotonic()
        try:
            return [future.result() for future in ticket]
        finally:
            self.resolve_wait_seconds.append(time.monotonic() - began)
            with self.lock:
                self.pending_batches -= 1

    def close(self, failed: bool = False) -> None:
        if failed:
            self.stop.set()
        self.pool.shutdown(wait=True, cancel_futures=failed)
        for log in self.logs:
            log.close()

    def __enter__(self) -> StreamProvider:
        return self

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        self.close(failed=kind is not None)


def merge_disk_cache(
    rows: dict[str, dict[str, Any]], manifest: dict[str, Any], canonical: str, overlay: str
) -> dict[str, dict[str, Any]]:
    if (
        manifest.get("complete") is not True
        or manifest.get("provider") != "readonly-disk-full-f32"
        or manifest.get("canonical_manifest_sha256") != canonical
        or manifest.get("physical_overlay_sha256") != overlay
    ):
        raise ValueError("Incomplete/unbound ordinary-disk cache")
    cached = {r["id"]: r for r in manifest["rows"]}
    if len(cached) != len(rows) or len(manifest["rows"]) != len(rows) or set(cached) != set(rows):
        raise ValueError("Disk cache changes cohort")
    keys = (
        "id",
        "path",
        "role",
        "target",
        "sha256",
        "waveform_sha256",
        "samples",
        "bytes",
        "source_stable_identity",
    )
    for identity, row in rows.items():
        if any(cached[identity][key] != row[key] for key in keys):
            raise ValueError("Disk cache changes canonical input/labels/physical binding")
    return cached


class CachedProvider:
    """One prefetch thread, exact readonly mmap copies; no decoding or auto fallback."""

    def __init__(self, rows: dict[str, dict[str, Any]], watch: Callable[[], None]) -> None:
        self.rows, self.watch = rows, watch
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.resolve_wait_seconds: list[float] = []
        self.pending = 0

    def read(self, identity: str, start: int, length: int) -> NDArray[np.float32]:
        import numpy as np

        self.watch()
        row = self.rows[identity]
        path = Path(row["cache"])
        source = Path(row["path"])
        before = file_identity(path)
        raw = file_identity(source)
        if (
            row["role"] != "fit"
            or list(before) != list(row["cache_identity"])
            or path.is_symlink()
            or path.stat().st_mode & 0o222
            or list(raw) != list(row["source_stable_identity"])
            or start < 0
            or length <= 0
            or start + length > row["samples"]
        ):
            raise ValueError("Immutable cached/source identity or crop changed")
        with (
            path.open("rb") as stream,
            mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
        ):
            result = np.frombuffer(mapped, dtype="<f4", offset=start * 4, count=length).copy()
        if file_identity(path) != before or file_identity(source) != raw:
            raise ValueError("Cache/source changed while reading")
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite cached crop")
        self.watch()
        return result

    def submit(self, requests: list[tuple[str, int, int]]) -> list[Future[NDArray[np.float32]]]:
        if len(requests) != 32 or self.pending >= 2:
            raise ValueError("Bounded32+next32 requests required")
        self.pending += 1
        return [self.pool.submit(self.read, *request) for request in requests]

    def resolve(self, ticket: list[Future[NDArray[np.float32]]]) -> list[NDArray[np.float32]]:
        began = time.monotonic()
        try:
            return [future.result() for future in ticket]
        finally:
            self.resolve_wait_seconds.append(time.monotonic() - began)
            self.pending -= 1

    def close(self, failed: bool = False) -> None:
        self.pool.shutdown(wait=True, cancel_futures=failed)


def ordered_batches(
    draws: Iterable[list[Draw]], rows: dict[str, dict[str, Any]], provider: BatchProvider
) -> Iterator[tuple[list[Draw], list[NDArray[np.float32]]]]:
    """At most current32 plus next32; caller clears consumed arrays after update."""
    iterator = iter(draws)
    current = next(iterator, None)
    if current is None:
        return

    def requests(batch: list[Draw]) -> list[tuple[str, int, int]]:
        return [(draw.identity, *crop(rows[draw.identity]["samples"], draw)) for draw in batch]

    pending = provider.submit(requests(current))
    while current is not None:
        assert pending is not None
        arrays = provider.resolve(pending)
        following = next(iterator, None)
        next_ticket = provider.submit(requests(following)) if following is not None else None
        yield current, arrays
        arrays.clear()
        current, pending = following, next_ticket
