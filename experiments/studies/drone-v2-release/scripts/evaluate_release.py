"""Run the packaged Predictor on original audio files and save one prediction per track."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

predictor = None


def initialize(weights: str) -> None:
    global predictor
    torch.set_num_threads(1)
    from drone import Predictor

    predictor = Predictor(weights, workdir=tempfile.mkdtemp(prefix="drone-release-"))


def predict(item: tuple[str, str]) -> dict:
    track_id, audio_path = item
    assert predictor is not None, "initialize() runs in every worker before predict()"
    started = time.process_time()
    tonic, mode, probability = predictor.predict(audio_path)
    return {
        "track_id": track_id,
        "tonic": tonic,
        "mode": mode,
        "probability": probability,
        "cpu_seconds": time.process_time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--items", required=True, help="JSON list of [track_id, audio_path]")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    items = [tuple(item) for item in json.loads(Path(arguments.items).read_text())]
    with ProcessPoolExecutor(
        max_workers=arguments.workers, initializer=initialize, initargs=(arguments.weights,)
    ) as executor:
        predictions = list(executor.map(predict, items, chunksize=4))
    Path(arguments.out).write_text(json.dumps(predictions))
    print(f"predicted {len(predictions)} tracks with {os.path.basename(arguments.weights)}")


if __name__ == "__main__":
    main()
