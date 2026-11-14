"""Bounded public-audio CPU timing; never scores key labels or selects models."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drone.checkpoints import load_checkpoint
from drone.datasets import Track, prepare
from drone.export import cpu_session
from drone.features import HOP, SR, audio_digest
from drone.model import params_count


def cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def measure(call: Callable[[], object], iterations: int) -> dict[str, float]:
    durations = []
    cpu_start = cpu_seconds()
    for _ in range(iterations):
        start = time.perf_counter()
        call()
        durations.append(time.perf_counter() - start)
    return {
        "p50_seconds": float(np.median(durations)),
        "p95_seconds": float(np.percentile(durations, 95)),
        "parent_plus_child_cpu_seconds": cpu_seconds() - cpu_start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 10:
        parser.error("Repeated timing is capped at ten iterations")
    selection = json.loads(args.selection.read_bytes())
    if selection["private_audio_used"] or selection["accuracy_scored"]:
        raise ValueError("Use the fixed public performance-only selection")
    rows = selection["rows"]
    if not 1 <= len(rows) <= 20 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Need at most 20 unique public validation tracks")
    if not all(row["id"].startswith("fmak:") for row in rows):
        raise ValueError("Only public FMAK rows are allowed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    model, _ = load_checkpoint(args.checkpoint)
    torch_load = time.perf_counter() - started
    started = time.perf_counter()
    session = cpu_session(args.onnx, threads=1)
    onnx_load = time.perf_counter() - started
    report: dict[str, Any] = {
        "source_commit": args.source_commit,
        "tool_sha256": audio_digest(Path(__file__)),
        "selection_sha256": audio_digest(args.selection),
        "checkpoint_sha256": audio_digest(args.checkpoint),
        "onnx_sha256": audio_digest(args.onnx),
        "parameters": params_count(model),
        "hardware": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "model_threads": 1,
        "iterations": args.iterations,
        "pytorch_load_seconds": torch_load,
        "onnx_load_seconds": onnx_load,
        "accuracy_scored": False,
        "private_audio_used": False,
        "records": [],
        "status": "running",
        "limitations": "Python reference implementation on server hardware, not Rust/Euphoria or target laptop. Cold means empty feature cache, not cold OS pages. First frontend timing includes library/JIT initialization. Warm-cache timings rerun inference, not a cached key-result lookup. Peak RSS includes both runtimes and scientific libraries. No ground-truth keys read or scored.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage_start = time.perf_counter()
    rss_scale = 1 / 1024 if platform.system() == "Darwin" else 1
    for number, row in enumerate(rows):
        if time.perf_counter() - stage_start > 540:
            raise TimeoutError("Nine-minute benchmark budget exhausted")
        track = Track(row["id"], row["path"], ("C", "major"))
        if audio_digest(track.audio) != row["sha256"]:
            raise ValueError(f"Selected audio changed: {track.id}")
        with tempfile.TemporaryDirectory(dir=args.output.parent) as cache:
            cpu_start = cpu_seconds()
            wall_start = time.perf_counter()
            features, _ = prepare(track, cache)
            frontend_wall = time.perf_counter() - wall_start
            frontend_cpu = cpu_seconds() - cpu_start
            batch = torch.from_numpy(features[None])
            inputs = {"features": features[None]}
            with torch.inference_mode():
                expected = model(batch).numpy()
                actual = np.asarray(session.run(["logits"], inputs)[0], dtype=np.float32)
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
                if actual.argmax(axis=1).tolist() != expected.argmax(axis=1).tolist():
                    raise ValueError(f"Backend key decision differs: {track.id}")
                torch_time = measure(lambda: model(batch), args.iterations)
                onnx_time = measure(lambda: session.run(["logits"], inputs), args.iterations)

                def warm_cache_onnx() -> object:
                    cached, _ = prepare(track, cache)
                    return session.run(["logits"], {"features": cached[None]})

                warm_time = measure(warm_cache_onnx, args.iterations)
            report["records"].append(
                dict(
                    id=track.id,
                    frames=features.shape[1],
                    duration_proxy_seconds=features.shape[1] * HOP / SR,
                    frontend_seconds=frontend_wall,
                    frontend_parent_plus_child_cpu_seconds=frontend_cpu,
                    first_frontend_call=number == 0,
                    pytorch_full_forward=torch_time,
                    onnx_full_forward=onnx_time,
                    warm_feature_cache_plus_onnx=warm_time,
                    parity_max_abs_error=float(np.max(np.abs(actual - expected))),
                    parity_key_agrees=True,
                    process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * rss_scale
                    / 1024,
                )
            )
        report["elapsed_seconds"] = time.perf_counter() - stage_start
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"K10 CPU {number + 1}/{len(rows)} frontend={frontend_wall:.3f}s "
            f"ORT={onnx_time['p50_seconds']:.3f}s warm={warm_time['p50_seconds']:.3f}s",
            flush=True,
        )
    report["status"] = "complete"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
