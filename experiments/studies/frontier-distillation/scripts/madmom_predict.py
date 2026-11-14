"""Run madmom's pretrained KEY_CNN on every GiantSteps track and save its key probabilities."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

processor = None


def predict(audio_path: str) -> tuple[str, list[float] | None, str | None]:
    global processor
    if processor is None:
        from madmom.features.key import CNNKeyRecognitionProcessor  # ty: ignore[import-not-found]

        processor = CNNKeyRecognitionProcessor()
    track_id = Path(audio_path).name.split(".")[0]
    try:
        probabilities = np.asarray(processor(audio_path))[0]
        return track_id, probabilities.astype(float).tolist(), None
    except Exception as error:
        return track_id, None, f"{type(error).__name__}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()
    from madmom.features.key import KEY_LABELS  # ty: ignore[import-not-found]

    audio_paths = sorted(
        str(path) for path in Path(arguments.audio).glob("*.mp3") if not path.name.startswith("._")
    )
    probabilities_by_track: dict[str, list[float]] = {}
    errors: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        for track_id, probabilities, error in executor.map(predict, audio_paths, chunksize=4):
            if probabilities is not None:
                probabilities_by_track[track_id] = probabilities
            else:
                errors[track_id] = error or "no probabilities returned"
    result = {"labels": list(KEY_LABELS), "probabilities": probabilities_by_track, "errors": errors}
    Path(arguments.out).write_text(json.dumps(result))
    print(f"predicted {len(probabilities_by_track)} tracks, {len(errors)} errors")


if __name__ == "__main__":
    main()
