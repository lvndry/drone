"""Extract drone-contract CQTs for unlabeled FMA clips, skipping evaluation tracks."""

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

PREPARED_DIRECTORY = required_path("DRONE_PREP")
MINIMUM_FRAMES = 100


def extract(job: tuple[str, str]) -> str:
    audio, destination = job
    if Path(destination).is_file():
        return "reused"
    try:
        with tempfile.TemporaryDirectory() as scratch:
            wav = str(Path(scratch) / "decoded.wav")
            decode_mp3(audio, wav)
            samples, rate = read_wav_mono(wav)
            features = cqt(resample_linear(samples, rate, SAMPLE_RATE)).astype(np.float16)
    except Exception as error:  # noqa: BLE001
        return f"failed:{type(error).__name__}"
    if features.shape[0] != N_TOTAL or features.shape[1] < MINIMUM_FRAMES:
        return "too_short"
    if not np.isfinite(features).all():
        return "nonfinite"
    pending = destination + ".partial.npy"
    np.save(pending, features)
    Path(pending).replace(destination)
    return "extracted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default=str(PREPARED_DIRECTORY / "manifest.json"))
    parser.add_argument("--workers", type=int, default=24)
    arguments = parser.parse_args()
    rows = json.loads(Path(arguments.manifest).read_text())["tracks"]
    evaluation_ids = {int(row["track_id"]) for row in rows if row["panel"] in ("valid", "holdout")}
    output = Path(arguments.out)
    output.mkdir(parents=True, exist_ok=True)
    audio_files = sorted(
        path for path in Path(arguments.audio_root).rglob("*.mp3") if not path.name.startswith("._")
    )
    kept = [path for path in audio_files if int(path.stem) not in evaluation_ids]
    jobs = [(str(path), str(output / f"{int(path.stem)}.npy")) for path in kept]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        outcomes = Counter(pool.map(extract, jobs, chunksize=16))
    summary = {
        "audio_files": len(audio_files),
        "excluded_evaluation_tracks": len(audio_files) - len(kept),
        "outcomes": dict(outcomes),
    }
    (output.parent / "extract-summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
