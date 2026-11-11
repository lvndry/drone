"""Measure checkpoint CPU latency with an explicit thread budget (no accuracy claims)."""

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from drone.checkpoints import load_checkpoint
from drone.datasets import Track, prepare
from drone.features import CROP_FRAMES, N_CORE
from drone.model import params_count


def timed(call, iterations):
    elapsed = []
    cpu_start = time.process_time()
    for _ in range(iterations):
        start = time.perf_counter()
        call()
        elapsed.append(1000 * (time.perf_counter() - start))
    return {
        "p50_ms": float(np.percentile(elapsed, 50)),
        "p95_ms": float(np.percentile(elapsed, 95)),
        "parent_cpu_ms_total": 1000 * (time.process_time() - cpu_start),
        "iterations": iterations,
    }


def benchmark(checkpoint, *, threads=1, iterations=100, audio=None, onnx=None):
    if threads < 1 or iterations < 1:
        raise ValueError("threads and iterations must be positive")
    torch.set_num_threads(threads)
    start = time.perf_counter()
    model, payload = load_checkpoint(checkpoint)
    load_ms = 1000 * (time.perf_counter() - start)
    x = torch.zeros(1, N_CORE, CROP_FRAMES)
    with torch.inference_mode():
        for _ in range(5):
            model(x)
        result = {
            "hardware": platform.platform(),
            "machine": platform.machine(),
            "torch": str(torch.__version__),
            "model_config": payload.get("model_config"),
            "parameters": params_count(model),
            "checkpoint_bytes": Path(checkpoint).stat().st_size,
            "threads": threads,
            "batch_size": 1,
            "feature_crop": payload.get("crop"),
            "model_load_ms": load_ms,
            "pytorch_forward": timed(lambda: model(x), iterations),
            "note": "Synthetic tensor latency; no accuracy measurement. Parent CPU timing excludes ffmpeg.",
        }
        if onnx:
            from drone.export import cpu_session

            session = cpu_session(onnx, threads)
            inputs = {"features": x.numpy()}
            for _ in range(5):
                session.run(None, inputs)
            result["onnx_forward"] = timed(lambda: session.run(None, inputs), iterations)
            result["onnx_bytes"] = Path(onnx).stat().st_size
        if audio:
            with tempfile.TemporaryDirectory() as scratch:
                track = Track("benchmark", audio, ("C", "major"))
                start = time.perf_counter()
                features, _ = prepare(track, scratch)
                model(torch.from_numpy(features[None]).transpose(1, 2))
                result["cold_audio_decode_features_forward_ms"] = 1000 * (
                    time.perf_counter() - start
                )
                result["warm_feature_cache_plus_forward"] = timed(
                    lambda: model(
                        torch.from_numpy(prepare(track, scratch)[0][None]).transpose(1, 2)
                    ),
                    min(10, iterations),
                )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--audio", help="Optional representative file for cold/warm full-path timing"
    )
    parser.add_argument("--onnx", help="Also benchmark a verified ONNX export")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(
        args.checkpoint,
        threads=args.threads,
        iterations=args.iterations,
        audio=args.audio,
        onnx=args.onnx,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
