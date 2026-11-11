"""Isolated ONNX/NumPy CPU benchmark, deliberately without importing PyTorch.

Load the pure feature source directly to isolate runtime overhead from the current
training package's eager torch import. This is a measurement tool, not the native
Euphoria integration or a replacement public inference API.
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def cpu_seconds():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--feature-source", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    rss_scale = 1 / 1024 if sys.platform == "darwin" else 1
    spec = importlib.util.spec_from_file_location("isolated_features", args.feature_source)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load feature source: {args.feature_source}")
    features = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(features)
    metadata = json.loads(args.model.with_suffix(".json").read_text())
    if metadata["feature_spec"] != features.FEATURE_SPEC:
        raise ValueError("ONNX metadata and feature source disagree")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    runtime_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale
    records = []
    extensions = {".mp3", ".m4a", ".flac", ".wav", ".aiff", ".aif", ".ogg", ".opus"}
    audio_files = sorted(
        p for p in args.audio.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )
    for audio in audio_files[: args.limit]:
        duration = float(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    str(audio),
                ],
                text=True,
            ).strip()
        )
        with tempfile.TemporaryDirectory() as scratch:
            start, cpu = time.monotonic(), cpu_seconds()
            raw = features.extract(audio.stem, str(audio), scratch)
            x = features.normalize(features.resample_time(raw, metadata["frames"]))
            session.run(None, {"features": np.ascontiguousarray(x.T[None])})
            records.append(
                {
                    "id": audio.stem,
                    "extension": audio.suffix.lower(),
                    "duration_seconds": duration,
                    "wall_seconds": time.monotonic() - start,
                    "cpu_seconds_including_decoder": cpu_seconds() - cpu,
                }
            )
        if len(records) % 20 == 0:
            print(f"Timed {len(records)} files", flush=True)
    if not records:
        raise ValueError("No benchmark files found")
    if "torch" in sys.modules:
        raise RuntimeError("Isolation failed: PyTorch was imported")
    result = {
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "torch_imported": False,
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "feature_source_sha256": hashlib.sha256(args.feature_source.read_bytes()).hexdigest(),
        "threads": 1,
        "worker_concurrency": 1,
        "files": records,
        "runtime_rss_kib_before_audio": runtime_rss,
        "wall_p50_seconds": float(np.percentile([r["wall_seconds"] for r in records], 50)),
        "wall_p95_seconds": float(np.percentile([r["wall_seconds"] for r in records], 95)),
        "peak_parent_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_scale,
        "note": "Process RSS in KiB (Darwin bytes converted); includes Python/NumPy/ORT. Fresh feature cache; filesystem cache uncontrolled. Native Rust integration remains unmeasured.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
