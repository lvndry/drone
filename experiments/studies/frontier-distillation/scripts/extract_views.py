"""Extract drone-contract CQTs, plus an optional drum-suppressed (harmonic-only) view.

The harmonic view applies librosa's harmonic/percussive separation to the decoded
22.05 kHz signal and then the unchanged drone CQT, so both views share one grid.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from environment import required_path

sys.path.insert(0, str(required_path("DRONE_SOURCE")))
import numpy as np  # noqa: E402

from drone.features import N_TOTAL, cqt, decode_mp3, read_wav_mono, resample_linear  # noqa: E402
from drone.features import SR as SAMPLE_RATE  # noqa: E402

MINIMUM_FRAMES = 100


def save(features: np.ndarray, destination: str) -> None:
    pending = destination + ".partial.npy"
    np.save(pending, features.astype(np.float16))
    Path(pending).replace(destination)


def extract(job: tuple[str, str, str]) -> str:
    audio, plain_destination, harmonic_destination = job
    need_plain = bool(plain_destination) and not Path(plain_destination).is_file()
    need_harmonic = bool(harmonic_destination) and not Path(harmonic_destination).is_file()
    if not need_plain and not need_harmonic:
        return "reused"
    try:
        with tempfile.TemporaryDirectory() as scratch:
            wav = str(Path(scratch) / "decoded.wav")
            decode_mp3(audio, wav)
            samples, rate = read_wav_mono(wav)
        samples = resample_linear(samples, rate, SAMPLE_RATE).astype(np.float32)
        plain = cqt(samples) if need_plain else None
        harmonic = None
        if need_harmonic:
            import librosa

            harmonic = cqt(librosa.effects.harmonic(samples, margin=2.0))
    except Exception as error:  # noqa: BLE001
        return f"failed:{type(error).__name__}"
    for features in (plain, harmonic):
        if features is None:
            continue
        if features.shape[0] != N_TOTAL or features.shape[1] < MINIMUM_FRAMES:
            return "too_short"
        if not np.isfinite(features).all():
            return "nonfinite"
    if plain is not None:
        save(plain, plain_destination)
    if harmonic is not None:
        save(harmonic, harmonic_destination)
    return "extracted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        required=True,
        help="JSON list of [audio, plain_out_or_empty, harmonic_out_or_empty]",
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--summary", required=True)
    arguments = parser.parse_args()
    jobs = [tuple(job) for job in json.loads(Path(arguments.jobs).read_text())]
    for _, plain, harmonic in jobs:
        for destination in (plain, harmonic):
            if destination:
                Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        outcomes = Counter(pool.map(extract, jobs, chunksize=8))
    summary = {"jobs": len(jobs), "outcomes": dict(outcomes)}
    Path(arguments.summary).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
