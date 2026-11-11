"""Measure a frozen model's complete CPU audio path with empty feature caches."""

import argparse
import hashlib
import json
import resource
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from drone.checkpoints import load_checkpoint
from drone.datasets import Track, prepare


def cpu_seconds():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.count <= 100:
        parser.error("This benchmark is capped at 100 files")
    torch.set_num_threads(1)
    model, checkpoint = load_checkpoint(args.checkpoint)
    records = []
    for audio in sorted(args.audio.glob("*.mp3"))[: args.count]:
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
            features, _ = prepare(
                Track(audio.stem, str(audio), ("C", "major")), scratch, T=checkpoint["frames"]
            )
            with torch.inference_mode():
                model(torch.from_numpy(features[None]).transpose(1, 2))
            records.append(
                {
                    "id": audio.stem,
                    "duration_seconds": duration,
                    "wall_seconds": time.monotonic() - start,
                    "cpu_seconds_including_decoder": cpu_seconds() - cpu,
                }
            )
        if len(records) % 20 == 0:
            print(f"Timed {len(records)} files", flush=True)
    if not records:
        raise ValueError("No MP3 files found")
    result = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "threads": 1,
        "worker_concurrency": 1,
        "files": records,
        "wall_p50_seconds": float(np.percentile([r["wall_seconds"] for r in records], 50)),
        "wall_p95_seconds": float(np.percentile([r["wall_seconds"] for r in records], 95)),
        "peak_parent_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "note": "Linux ru_maxrss. Fresh feature cache per file; filesystem cache uncontrolled. Model loaded once. No accuracy measurement.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
