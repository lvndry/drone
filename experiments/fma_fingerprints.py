"""CPU-only complete-posting landmark fingerprints; no classifier/model imports."""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import os
import resource
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

SR, HOP = 11025, 256
CORE_FRAMES, HALO_FRAMES = 5167, 216
MATCH_MIN, MATCH_SPAN = 5, 5.0
AMBIG_MIN, AMBIG_SPAN = 3, 2.0
MAX_PCM_BYTES = SR * 3600 * 4
VENDOR_COMMIT = "cb03ba99feafd41b8874307f0f4e808a6ce34362"
CONTRACT = dict(
    version="fma-landmark-admission-v1",
    sample_rate=SR,
    hop=HOP,
    density=20,
    reference_shifts=1,
    query_shifts=4,
    core_frames=CORE_FRAMES,
    halo_frames=HALO_FRAMES,
    complete_postings=True,
    time_mask=False,
    match_min=MATCH_MIN,
    match_span=MATCH_SPAN,
    ambiguous_min=AMBIG_MIN,
    ambiguous_span=AMBIG_SPAN,
)


def vendor_module(vendor: str) -> ModuleType:
    directory = Path(vendor).resolve()
    manifest = json.loads((directory / "vendor.json").read_text())
    if manifest["commit"] != VENDOR_COMMIT or manifest["patches"]:
        raise ValueError("Unreviewed audfprint revision/patch")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Pinned vendor source changed")
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    module = importlib.import_module("audfprint_analyze")
    if module.__file__ is None or Path(module.__file__).resolve().parent != directory:
        raise ValueError("Unexpected audfprint import origin")
    return module


def decode(
    path: str, *, offset: float | None = None, duration: float | None = None
) -> tuple[NDArray[np.float32], str]:
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-filter_threads",
        "1",
    ]
    if offset is not None:
        command += ["-ss", str(offset)]
    command += ["-i", path]
    if duration is not None:
        command += ["-t", str(duration)]
    if duration is None:
        command += ["-t", "3601"]
    command += ["-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    process = subprocess.run(command, capture_output=True, timeout=180)
    if process.returncode:
        raise ValueError("FFmpeg decode failed: " + process.stderr.decode(errors="replace")[-1000:])
    raw = process.stdout
    if len(raw) > MAX_PCM_BYTES:
        raise ValueError(
            "Reference exceeds one-hour decode guard; coverage fails, never silently truncate"
        )
    samples = np.frombuffer(raw, dtype="<f4")
    if len(samples) < SR * 3 or not np.isfinite(samples).all():
        raise ValueError("Short/nonfinite decode; ambiguous fingerprint")
    return samples, hashlib.sha256(raw).hexdigest()


def landmarks(
    samples: NDArray[np.float32], vendor: str, *, query: bool = False
) -> NDArray[np.int64]:
    module = vendor_module(vendor)
    pieces = []
    frames = (len(samples) + HOP - 1) // HOP
    for q0 in range(0, frames, CORE_FRAMES):
        q1 = min(frames, q0 + CORE_FRAMES)
        a, b = max(0, q0 - HALO_FRAMES), min(frames, q1 + HALO_FRAMES)
        for shift in (0, 64, 128, 192) if query else (0,):
            chunk = samples[a * HOP + shift : min(len(samples), b * HOP)]
            if len(chunk) < SR * 2:
                continue
            analyzer = module.Analyzer(density=20)
            peaks = analyzer.find_peaks(chunk, SR)
            hashes = module.landmarks2hashes(analyzer.peaks2landmarks(peaks)).astype(np.int64)
            if hashes.size:
                hashes[:, 0] += a
                pieces.append(hashes[(hashes[:, 0] >= q0) & (hashes[:, 0] < q1)])
    if not pieces:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.concatenate(pieces), axis=0)


def chromaprint(samples: NDArray[np.float32]) -> tuple[NDArray[np.uint32], float]:
    lib = ctypes.CDLL("libchromaprint.so.1")
    lib.chromaprint_new.argtypes = [ctypes.c_int]
    lib.chromaprint_new.restype = ctypes.c_void_p
    for name in ("chromaprint_free", "chromaprint_finish"):
        getattr(lib, name).argtypes = [ctypes.c_void_p]
    lib.chromaprint_start.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.chromaprint_feed.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16), ctypes.c_int]
    lib.chromaprint_get_raw_fingerprint.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.chromaprint_dealloc.argtypes = [ctypes.c_void_p]
    lib.chromaprint_get_item_duration.argtypes = [ctypes.c_void_p]
    lib.chromaprint_get_sample_rate.argtypes = [ctypes.c_void_p]
    context = lib.chromaprint_new(1)
    if not context:
        raise ValueError("Chromaprint allocation failed")
    try:
        if not lib.chromaprint_start(context, SR, 1):
            raise ValueError("Chromaprint start failed")
        for start in range(0, len(samples), SR * 30):
            pcm = np.clip(samples[start : start + SR * 30] * 32768, -32768, 32767).astype(np.int16)
            if not lib.chromaprint_feed(
                context, pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)), len(pcm)
            ):
                raise ValueError("Chromaprint feed failed")
        if not lib.chromaprint_finish(context):
            raise ValueError("Chromaprint finish failed")
        pointer = ctypes.POINTER(ctypes.c_uint32)()
        size = ctypes.c_int()
        if not lib.chromaprint_get_raw_fingerprint(
            context, ctypes.byref(pointer), ctypes.byref(size)
        ):
            raise ValueError("Chromaprint output failed")
        try:
            array = np.ctypeslib.as_array(pointer, shape=(size.value,)).copy()
        finally:
            lib.chromaprint_dealloc(pointer)
        return array, lib.chromaprint_get_item_duration(context) / lib.chromaprint_get_sample_rate(
            context
        )
    finally:
        lib.chromaprint_free(context)


def extract_record(
    row: dict[str, Any], vendor: str, output: str, *, query: bool = False
) -> dict[str, Any]:
    start, cpu = time.monotonic(), time.process_time()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    record = dict(ref_id=row["ref_id"], id=row["id"], path=row["path"], domain=row["domain"])
    try:
        with Path(row["path"]).open("rb") as handle:
            container_sha = hashlib.file_digest(handle, "sha256").hexdigest()
        if (
            row.get("expected_container_sha256") is not None
            and container_sha != row["expected_container_sha256"]
        ):
            raise ValueError("Frozen fitting source container differs")
        samples, pcm_sha = decode(row["path"])
        hashes = landmarks(samples, vendor, query=query)
        raw, step = chromaprint(samples)
        target = Path(output) / f"{row['ref_id']:06}.npz"
        np.savez_compressed(target, landmarks=hashes, chromaprint=raw, chromaprint_step=step)
        record.update(
            status="complete",
            frames=len(samples),
            seconds=len(samples) / SR,
            hashes=len(hashes),
            chromaprint_items=len(raw),
            chromaprint_step=step,
            pcm_sha256=pcm_sha,
            container_sha256=container_sha,
            artifact=target.name,
            bytes=target.stat().st_size,
        )
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    record.update(
        wall_seconds=time.monotonic() - start,
        cpu_seconds=time.process_time()
        - cpu
        + child_after.ru_utime
        - child_before.ru_utime
        + child_after.ru_stime
        - child_before.ru_stime,
        worker_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        decoder_peak_rss_kib=child_after.ru_maxrss,
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
    )
    return record


class PostingIndex:
    def __init__(self, path: Path, *, readonly: bool = False) -> None:
        self.db = sqlite3.connect(f"file:{path}?mode=ro" if readonly else str(path), uri=readonly)
        self.db.execute("PRAGMA cache_size=-65536")
        if not readonly:
            self.db.execute("PRAGMA journal_mode=DELETE")
            self.db.execute("PRAGMA synchronous=NORMAL")
            self.db.execute("PRAGMA temp_store=FILE")
            self.db.execute("PRAGMA max_page_count=524288")  # 2GiB database cap
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS postings(hash INTEGER NOT NULL, ref INTEGER NOT NULL, frame INTEGER NOT NULL)"
            )
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS references_done(ref INTEGER PRIMARY KEY, id TEXT NOT NULL, hashes INTEGER NOT NULL)"
            )

    def add(self, ref: int, identity: str, hashes: NDArray[np.int64]) -> None:
        if self.db.execute("SELECT 1 FROM references_done WHERE ref=?", (ref,)).fetchone():
            raise ValueError("Reference already indexed")
        with self.db:
            self.db.executemany(
                "INSERT INTO postings VALUES(?,?,?)", ((int(h), ref, int(t)) for t, h in hashes)
            )
            self.db.execute(
                "INSERT INTO references_done VALUES(?,?,?)", (ref, identity, len(hashes))
            )

    def finish(self) -> None:
        self.db.execute("CREATE INDEX IF NOT EXISTS posting_hash ON postings(hash)")
        self.db.commit()
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def match(self, hashes: NDArray[np.int64], *, deadline: float | None = None) -> dict[str, Any]:
        # Every posting is visited; no random bucket cap, packed time, top-k or
        # duration/title filter can produce an unearned negative result.
        votes: dict[tuple[int, int, int], set[int]] = defaultdict(set)
        posting_visits = 0
        for time_frame, hash_value in hashes:
            if deadline is not None and time.monotonic() > deadline:
                return dict(
                    complete=False,
                    reason="query_budget_exhausted",
                    posting_visits=posting_visits,
                    matches=[],
                )
            for ref, frame in self.db.execute(
                "SELECT ref,frame FROM postings WHERE hash=?", (int(hash_value),)
            ):
                posting_visits += 1
                if (
                    posting_visits % 65536 == 0
                    and deadline is not None
                    and time.monotonic() > deadline
                ):
                    return dict(
                        complete=False,
                        reason="query_budget_exhausted",
                        posting_visits=posting_visits,
                        matches=[],
                    )
                offset = frame - int(time_frame)
                for phase in (0, 2):
                    votes[(ref, (offset + phase) // 4, phase)].add(int(time_frame))
                if len(votes) > 2000000:
                    return dict(
                        complete=False,
                        reason="vote_memory_budget",
                        posting_visits=posting_visits,
                        matches=[],
                    )
        matches: dict[int, dict[str, Any]] = {}
        for (ref, offset_bin, phase), anchors in votes.items():
            span = (max(anchors) - min(anchors)) * HOP / SR
            count = len(anchors)
            if count < AMBIG_MIN or span < AMBIG_SPAN:
                continue
            record = dict(
                ref_id=ref,
                count=count,
                span_seconds=span,
                offset_seconds=(offset_bin * 4 - phase) * HOP / SR,
                strong=count >= MATCH_MIN and span >= MATCH_SPAN,
            )
            if ref not in matches or count > matches[ref]["count"]:
                matches[ref] = record
        return dict(
            complete=True,
            posting_visits=posting_visits,
            matches=sorted(matches.values(), key=lambda r: (-r["count"], r["ref_id"])),
        )


def chromaprint_distance(
    query: NDArray[np.uint32], reference: NDArray[np.uint32], offset: float, step: float
) -> float | None:
    if len(query) < 8 or not len(reference):
        return None
    center = round(offset / step)
    best = None
    for position in range(center - 8, center + 9):
        q0, r0 = max(0, -position), max(0, position)
        n = min(len(query) - q0, len(reference) - r0)
        if n < 8:
            continue
        xor = np.bitwise_xor(query[q0 : q0 + n], reference[r0 : r0 + n])
        value = float(np.unpackbits(xor.view(np.uint8)).mean())
        best = value if best is None else min(best, value)
    return best
