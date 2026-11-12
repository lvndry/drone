"""Metadata-only LRU projection and prospective, label-blind MusicFM crop qualification."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

RATE = 24000
WINDOW = 12 * RATE
CAPACITY = 4 * 1024**3
MANIFEST_SHA = "9f2db5efe4cbe2904508d8004f842acd29e90c6bc7f68612679c68993c439de0"
EXCLUDED = {"beatport:2997186", "fmak:051779", "fmak:096100"}
FFMPEG_SHA = "ed16af623947494a72e284b6eb8ff225f2da22b38b5d5069c2fd4b4ba3384e41"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def save(path: Path, value: dict[str, Any]) -> None:
    # Exclusive creation preserves failed and completed attempts.
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sampler(source: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("bound_k10_sampler", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import reviewed sampler")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def lru_projection(
    draws: list[str], sizes: dict[str, int], seconds: dict[str, float], capacity: int
) -> dict[str, Any]:
    cache: OrderedDict[str, int] = OrderedDict()
    used = hits = misses = written = evictions = 0
    decoded = 0.0
    domains: dict[str, dict[str, int]] = {}
    for identity in draws:
        domain = identity.split(":")[0]
        counts = domains.setdefault(domain, {"draws": 0, "hits": 0, "misses": 0})
        counts["draws"] += 1
        if identity in cache:
            hits += 1
            counts["hits"] += 1
            cache.move_to_end(identity)
            continue
        misses += 1
        counts["misses"] += 1
        decoded += seconds[identity]
        size = sizes[identity]
        # Oversized objects are decoded but cannot be admitted.
        if size <= capacity:
            while used + size > capacity:
                _, removed = cache.popitem(last=False)
                used -= removed
                evictions += 1
            cache[identity] = size
            used += size
            written += size
    return {
        "capacity_bytes": capacity,
        "draws": len(draws),
        "hits": hits,
        "misses": misses,
        "hit_fraction": hits / len(draws),
        "evictions": evictions,
        "projected_cache_write_bytes": written,
        "projected_full_decode_audio_seconds": decoded,
        "requested_crop_audio_seconds": len(draws) * 12,
        "domains": domains,
        "resident_bytes_at_end": used,
    }


def simulate(plan: Path, manifest: Path, source: Path) -> dict[str, Any]:
    if sha(manifest) != MANIFEST_SHA:
        raise ValueError("Different canonical duration inventory")
    p, m = load(plan), load(manifest)
    rows = {r["id"]: r for r in m["rows"]}
    bp, fm = p["bp_fit"], p["expanded_fma_fit"]
    if (len(bp), len(fm)) != (1229, 4033) or set(bp + fm) & EXCLUDED:
        raise ValueError("Reviewed eligibility counts differ")
    driver = sampler(source)
    if (driver.TOTAL_UPDATES, driver.BATCH, driver.BP_NUMERATOR, driver.REFERENCE_ROWS) != (
        8280,
        32,
        1230,
        2206,
    ):
        raise ValueError("Different schedule")
    draws = [draw.identity for batch in driver.paired_batches(bp, fm) for draw in batch]
    seconds = {i: rows[i]["canonical_samples"] / 22050 for i in bp + fm}
    sizes = {i: 4 * math.ceil(rows[i]["canonical_samples"] * RATE / 22050) for i in bp + fm}
    reports = [lru_projection(draws, sizes, seconds, CAPACITY)]
    reports.append(
        lru_projection(draws, {i: n + 16384 for i, n in sizes.items()}, seconds, CAPACITY)
    )
    total_seconds = sum(r["canonical_samples"] / 22050 for r in m["rows"])
    for report in reports:
        # Historical preparation includes CQT and runs in parallel. These are scaling
        # proxies, not measurements of 24kHz decode CPU or a fit-time prediction.
        report["historical_1905_cpu_seconds_duration_scaled_proxy"] = (
            1905 * report["projected_full_decode_audio_seconds"] / total_seconds
        )
    return {
        "status": "metadata_projection_only_not_training_activation",
        "plan_sha256": sha(plan),
        "duration_manifest_sha256": sha(manifest),
        "sampler_source_sha256": sha(source),
        "tool_sha256": sha(Path(__file__)),
        "eligibility": "amended plan supplied",
        "excluded_ids": sorted(EXCLUDED),
        "bp_fit": len(bp),
        "fma_fit": len(fm),
        "projected_24k_f32_working_set_bytes": sum(sizes.values()),
        "duration_projection": "ceil(canonical22050_samples*24000/22050); actual24k not measured",
        "header_sensitivity_bytes_per_file": [0, 16384],
        "projections": reports,
        "limitations": "Serial LRU; no worker-prefetch duplication. 1905CPU proxy includes CQT; "
        "neither decode24k CPU nor model cost has been measured. No labels read for decisions.",
    }


def crop_positions(samples: int, identity: str) -> list[int]:
    if samples < WINDOW:
        return [0]
    last = samples - WINDOW
    seed = int(hashlib.sha256(("musicfm-crop-io-v1:" + identity).encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    return sorted(
        {
            0,
            min(1, last),
            min(RATE + 1, last),
            last // 2,
            max(0, last - 1),
            last,
            *(rng.randrange(last + 1) for _ in range(3)),
        }
    )


def seek_geometry(start: int, length: int) -> tuple[int, int, int]:
    if start < 0 or not 0 < length <= WINDOW:
        raise ValueError("Invalid crop")
    anchor = max(0, start // RATE - 2)
    trim = start - anchor * RATE
    return anchor, trim, trim + length + RATE


def decode_command(ffmpeg: Path, audio: Path, start: int | None, length: int) -> list[str]:
    command = [
        str(ffmpeg),
        "-nostdin",
        "-v",
        "error",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-filter_complex_threads",
        "1",
    ]
    if start is not None:
        anchor, _, wanted = seek_geometry(start, length)
        if anchor:
            command.extend(["-ss", str(anchor), "-accurate_seek"])
    command.extend(
        [
            "-i",
            str(audio),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            str(RATE),
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
        ]
    )
    if start is not None:
        command.extend(["-t", f"{wanted / RATE:.9f}"])
    return command + ["pipe:1"]


def compare(reference: NDArray[np.float32], actual: NDArray[np.float32]) -> dict[str, Any]:
    import numpy as np

    if (
        reference.ndim != 1
        or actual.ndim != 1
        or len(reference) == 0
        or reference.shape != actual.shape
    ):
        return {"passed": False, "expected_samples": len(reference), "actual_samples": len(actual)}
    finite = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    exact = int(np.count_nonzero(reference.view("<u4") != actual.view("<u4")))
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    return {
        "passed": finite and exact == 0,
        "finite": finite,
        "samples": len(reference),
        "bitwise_mismatch_samples": exact,
        "mismatch_fraction": exact / max(1, len(reference)),
        "max_absolute_error": float(np.max(np.abs(difference))) if finite else None,
        "rms_error": float(np.sqrt(np.mean(difference**2))) if finite else None,
    }


def runtime_files(ffmpeg: Path) -> dict[str, str]:
    result = subprocess.run(
        ["ldd", str(ffmpeg)], capture_output=True, text=True, timeout=10, check=True
    )
    paths = {ffmpeg.resolve()}
    for line in result.stdout.splitlines():
        if "not found" in line:
            raise ValueError("Unresolved FFmpeg library")
        for token in line.split():
            if token.startswith("/"):
                paths.add(Path(token).resolve())
    return {str(path): sha(path) for path in sorted(paths)}


def seal_config(draft: Path, begin: str, deadline: str) -> dict[str, Any]:
    config = load(draft)
    ffmpeg = Path(config["ffmpeg"])
    if sha(ffmpeg) != FFMPEG_SHA or config["tool_sha256"] != sha(Path(__file__)):
        raise ValueError("Different source or FFmpeg; needs prospective review")
    config["runtime_file_sha256"] = runtime_files(ffmpeg)
    config["ffmpeg_version"] = subprocess.run(
        [str(ffmpeg), "-version"], capture_output=True, text=True, check=True, timeout=10
    ).stdout
    config["not_before_epoch"] = datetime.fromisoformat(begin).timestamp()
    config["deadline_epoch"] = datetime.fromisoformat(deadline).timestamp()
    if not 0 < config["deadline_epoch"] - config["not_before_epoch"] <= 900:
        raise ValueError("Window must be positive and at most900s")
    config["status"] = "sealed_pending_explicit_root_approval"
    config["draft_sha256"] = sha(draft)
    return config


def bounded_decode(
    command: list[str],
    target: Path,
    config: dict[str, Any],
    stage: Path,
    began: float,
    maximum: int,
) -> dict[str, Any]:
    from tools.run_cqt_expansion import child_cpu, disk_bytes, process_tree

    call_started, initial_cpu = time.monotonic(), child_cpu()

    with target.open("xb") as stream, target.with_suffix(".stderr").open("xb") as error:
        process = subprocess.Popen(command, stdout=stream, stderr=error, start_new_session=True)
        peak = 0
        try:
            while process.poll() is None:
                rss, cpu = process_tree(os.getpid())
                peak = max(peak, rss)
                if (
                    rss > 2 * 1024**3
                    or cpu > 600
                    or time.monotonic() - began > 900
                    or time.time() >= config["deadline_epoch"]
                    or target.stat().st_size > maximum
                    or disk_bytes(stage) > 512 * 1024**2
                ):
                    raise RuntimeError("Fixed decode resource/window limit")
                time.sleep(0.05)
            if process.returncode != 0 or target.stat().st_size > maximum:
                raise RuntimeError(f"Decode failed: exit={process.returncode}; retained stderr")
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    return {
        "command": command,
        "exit": process.returncode,
        "sampled_peak_tree_rss_bytes": peak,
        "wall_seconds": time.monotonic() - call_started,
        "child_cpu_seconds": child_cpu() - initial_cpu,
    }


def make_config(first: Path, manifest: Path, plan: Path) -> dict[str, Any]:
    if sha(manifest) != MANIFEST_SHA or sha(plan) != (
        "c3e59ffb764ca5f51444447b0a18a69543228e60a9e3723d925ddd82b0fc8373"
    ):
        raise ValueError("Different reviewed metadata")
    original = load(first)["first_batch"]
    if len(original) != 16 or len({r["id"] for r in original}) != 16:
        raise ValueError("Require original first16")
    p = load(plan)
    eligible = set(p["bp_fit"] + p["expanded_fma_fit"])
    rows = {r["id"]: r for r in load(manifest)["rows"]}
    selected = []
    for item in original:
        row = rows[item["id"]]
        if item["id"] not in eligible or item["sha256"] != row["sha256"]:
            raise ValueError("First16 eligibility/source changed; no substitution")
        selected.append({k: row[k] for k in ("id", "sha256", "path", "canonical_samples")})
    return {
        "status": "prospective_requires_root_window_and_runtime_receipt",
        "tool_sha256": sha(Path(__file__)),
        "plan_sha256": sha(plan),
        "first16_plan_sha256": sha(first),
        "duration_manifest_sha256": sha(manifest),
        "supervisor_source_sha256": sha(Path(__file__).with_name("run_cqt_expansion.py")),
        "ffmpeg": "/usr/bin/ffmpeg",
        "ffmpeg_sha256": FFMPEG_SHA,
        "runtime_file_sha256": {},
        "affinity": [8, 9],
        "threads": 1,
        "not_before_epoch": None,
        "deadline_epoch": None,
        "rows": selected,
    }


def file_identity(path: Path) -> tuple[int, int, int, int, int]:
    """Ignore atime, which may legitimately change when this worker reads audio."""
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def qualify(config_path: Path, output: Path, approval: Path) -> None:
    """Only execute after a separately retained, source-bound root approval."""
    import numpy as np

    config, permit = load(config_path), load(approval)
    if permit.get("config_sha256") != sha(config_path) or permit.get("authorized") is not True:
        raise ValueError("Missing exact root approval")
    if config["tool_sha256"] != sha(Path(__file__)) or sha(Path(config["ffmpeg"])) != FFMPEG_SHA:
        raise ValueError("Execution source/binary differs")
    if sys.platform != "linux" or os.sched_getaffinity(0) != {8, 9}:
        raise ValueError("Require fixed Linux CPUs8,9")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(name) != "1":
            raise ValueError("Require numeric threads1")
    if (
        not config["runtime_file_sha256"]
        or runtime_files(Path(config["ffmpeg"])) != config["runtime_file_sha256"]
    ):
        raise ValueError("Missing/mismatched complete FFmpeg shared-library receipt")
    if config["supervisor_source_sha256"] != sha(Path(__file__).with_name("run_cqt_expansion.py")):
        raise ValueError("Supervisor source changed")
    if len(config["rows"]) != 16 or len({r["id"] for r in config["rows"]}) != 16:
        raise ValueError("Require fixed16")
    if not config["not_before_epoch"] <= time.time() < config["deadline_epoch"]:
        raise ValueError("Outside approved window")
    output.mkdir(exist_ok=False)
    reports: list[dict[str, Any]] = []
    began = time.monotonic()
    for row in config["rows"]:
        if time.time() >= config["deadline_epoch"] or time.monotonic() - began >= 900:
            raise TimeoutError("Qualification deadline")
        audio, ffmpeg = Path(row["path"]), Path(config["ffmpeg"])
        if sha(audio) != row["sha256"]:
            raise ValueError("Source audio integrity failure")
        identity = file_identity(audio)
        # One full reference at a time; mmap avoids an extra complete waveform copy.
        reference_path = output / "reference.f32le"
        full_execution = bounded_decode(
            decode_command(ffmpeg, audio, None, WINDOW),
            reference_path,
            config,
            output,
            began,
            256 * 1024**2,
        )
        if reference_path.stat().st_size > 256 * 1024**2 or reference_path.stat().st_size % 4:
            raise ValueError("Reference size limit/alignment")
        reference = np.memmap(reference_path, mode="r", dtype="<f4")
        length = min(WINDOW, len(reference))
        record: dict[str, Any] = {
            "id": row["id"],
            "audio_sha256": row["sha256"],
            "full_samples": len(reference),
            "full_sha256": sha(reference_path),
            "full_execution": full_execution,
            "decoder_profile": "ffmpeg-default-recovery-no-xerror-v1",
            "cases": [],
        }
        for start in crop_positions(len(reference), row["id"]):
            if file_identity(audio) != identity:
                raise ValueError("Audio changed after integrity check")
            command = decode_command(ffmpeg, audio, start, length)
            candidate_path = output / "crop.f32le"
            execution = bounded_decode(command, candidate_path, config, output, began, 2 * 1024**2)
            _, trim, _ = seek_geometry(start, length)
            decoded = np.fromfile(candidate_path, dtype="<f4")
            candidate_path.unlink()
            candidate_path.with_suffix(".stderr").rename(
                output / f"track-{len(reports) + 1:02}-crop-{len(record['cases']) + 1:02}.stderr"
            )
            crop = decoded[trim : trim + length]
            report = compare(reference[start : start + length], crop)
            record["cases"].append({"start_sample": start, "execution": execution, **report})
            save(
                output / f"case-{len(reports) + 1:02}-{len(record['cases']):02}.json",
                {"id": row["id"], **record["cases"][-1]},
            )
            if not report["passed"] and not (output / "first-failure-reference.f32le").exists():
                reference[start : start + length].tofile(output / "first-failure-reference.f32le")
                crop.tofile(output / "first-failure-candidate.f32le")
        reports.append(record)
        del reference
        reference_path.unlink()
        reference_path.with_suffix(".stderr").rename(
            output / f"track-{len(reports):02}-full.stderr"
        )
        save(output / f"track-{len(reports):02}.json", record)
    save(
        output / "report.json",
        {
            "complete": True,
            "config_sha256": sha(config_path),
            "approval_sha256": sha(approval),
            "tool_sha256": sha(Path(__file__)),
            "rows": reports,
            "passed": all(c["passed"] for r in reports for c in r["cases"]),
            "wall_seconds": time.monotonic() - began,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sim = sub.add_parser("simulate")
    for name in ("plan", "manifest", "sampler", "output"):
        sim.add_argument("--" + name, type=Path, required=True)
    prepare = sub.add_parser("plan")
    for name in ("first", "manifest", "plan", "output"):
        prepare.add_argument("--" + name, type=Path, required=True)
    seal = sub.add_parser("seal")
    for name in ("config", "output"):
        seal.add_argument("--" + name, type=Path, required=True)
    seal.add_argument("--begin", required=True)
    seal.add_argument("--deadline", required=True)
    run = sub.add_parser("qualify")
    for name in ("config", "output", "approval"):
        run.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "simulate":
        save(args.output, simulate(args.plan, args.manifest, args.sampler))
    elif args.mode == "seal":
        save(args.output, seal_config(args.config, args.begin, args.deadline))
    elif args.mode == "plan":
        save(args.output, make_config(args.first, args.manifest, args.plan))
    else:
        qualify(args.config, args.output, args.approval)


if __name__ == "__main__":
    main()
