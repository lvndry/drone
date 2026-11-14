"""Score existing Drone checkpoints on GS604 against original and GiantSteps+ labels.

The scoring policy is fixed here before any prediction is compared:
- original: first listed key of the published GiantSteps annotation.
- gsplus_primary: the GiantSteps+ key covering the longest stretch of the excerpt,
  first alternative when several are separated by "|".
- gsplus_any: a prediction is exact when it equals any key GiantSteps+ lists.
- gsplus_clean: tracks with one segment and one listed key only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from environment import required_path

sys.path.insert(0, str(required_path("DRONE_SOURCE")))
import torch  # noqa: E402

from drone.checkpoints import load_checkpoint  # noqa: E402
from drone.features import extract_cqt, window  # noqa: E402
from drone.keys import score_mirex  # noqa: E402

TONIC_INDEX = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "F": 5,
    "E#": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}
MINOR_WORDS = {"minor", "min", "aeolian", "aeonian", "dorian", "phrygian", "locrian"}
MAJOR_WORDS = {"major", "maj", "ionian", "mixolydian", "lydian"}


def parse_single_key(text: str) -> int | None:
    tokens = text.strip().split()
    if len(tokens) < 2:
        return None
    tonic_name = tokens[0].rstrip("^_")
    tonic_name = tonic_name[:1].upper() + tonic_name[1:]
    if tonic_name not in TONIC_INDEX:
        return None
    mode_word = tokens[1].lower()
    if mode_word in MINOR_WORDS:
        return TONIC_INDEX[tonic_name] * 2 + 1
    if mode_word in MAJOR_WORDS:
        return TONIC_INDEX[tonic_name] * 2
    return None


def segment_length(start: float, end: float, total: float) -> float:
    if end > start:
        return end - start
    return max(total - start, 0.0)


def load_gsplus(directory: Path) -> tuple[dict[str, dict], list[str]]:
    labels: dict[str, dict] = {}
    unparsed: list[str] = []
    for path in sorted(
        (path for path in directory.glob("*.json") if not path.name.startswith("._"))
    ):
        record = json.loads(path.read_text())
        keys = record.get("key") or []
        starts = record.get("start_times") or [0] * len(keys)
        ends = record.get("end_times") or [120] * len(keys)
        segments = []
        for position, key_text in enumerate(keys):
            alternatives = [parse_single_key(part) for part in key_text.split("|")]
            parsed = [value for value in alternatives if value is not None]
            if not parsed:
                continue
            start = float(starts[position]) if position < len(starts) else 0.0
            end = float(ends[position]) if position < len(ends) else 120.0
            segments.append((segment_length(start, end, 120.0), parsed))
        if not segments:
            unparsed.append(f"{path.stem}: {keys}")
            continue
        longest = max(segments, key=lambda segment: segment[0])
        every_key = sorted({value for _, parsed in segments for value in parsed})
        labels[path.stem] = {
            "primary": longest[1][0],
            "any": every_key,
            "clean": len(keys) == 1 and len(every_key) == 1,
            "confidence": record.get("confidence"),
        }
    return labels, unparsed


def load_original(directory: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    for path in sorted(
        (path for path in directory.glob("*.key") if not path.name.startswith("._"))
    ):
        first = path.read_text().strip().split("|")[0]
        parsed = parse_single_key(first)
        if parsed is not None:
            labels[path.name.split(".")[0]] = parsed
    return labels


def extract_one(job: tuple[str, str]) -> tuple[str, str]:
    audio, cache = job
    extract_cqt(audio, Path(cache))
    return audio, "cached"


def summarize(predictions: list[int], references: list[int]) -> dict:
    exact, mirex = score_mirex(predictions, references)
    return {"n": len(predictions), "exact": round(100 * exact, 2), "mirex": round(100 * mirex, 2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--gsplus", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="name=path")
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()

    audio_files = sorted(
        path for path in Path(arguments.audio).glob("*.mp3") if not path.name.startswith("._")
    )
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        jobs = [(str(path), arguments.cache) for path in audio_files]
        extracted = dict(pool.map(extract_one, jobs, chunksize=4))
    assert len(extracted) == len(audio_files)
    track_ids = [path.name.split(".")[0] for path in audio_files]

    original = load_original(Path(arguments.original))
    gsplus, unparsed = load_gsplus(Path(arguments.gsplus))

    torch.set_num_threads(16)
    report: dict = {
        "tracks_with_audio": len(track_ids),
        "original_labels": len(original),
        "gsplus_labels": len(gsplus),
        "gsplus_unparsed": unparsed,
        "gsplus_clean": sum(1 for label in gsplus.values() if label["clean"]),
        "gsplus_confidence_counts": dict(
            Counter(str(label["confidence"]) for label in gsplus.values())
        ),
        "models": {},
        "per_track": {},
    }
    shared = [track for track in track_ids if track in original and track in gsplus]
    report["label_agreement_original_vs_gsplus_primary"] = summarize(
        [gsplus[track]["primary"] for track in shared], [original[track] for track in shared]
    )

    for specification in arguments.checkpoint:
        name, checkpoint_path = specification.split("=", 1)
        model, _ = load_checkpoint(checkpoint_path, "cpu")
        model.eval()
        predictions: dict[str, int] = {}
        with torch.inference_mode():
            for path, track in zip(audio_files, track_ids):
                features = window(extract_cqt(str(path), Path(arguments.cache)), 0)
                predictions[track] = int(model(torch.from_numpy(features).unsqueeze(0)).argmax())
        original_tracks = [track for track in track_ids if track in original]
        gsplus_tracks = [track for track in track_ids if track in gsplus]
        clean_tracks = [track for track in gsplus_tracks if gsplus[track]["clean"]]
        any_references = [
            predictions[track]
            if predictions[track] in gsplus[track]["any"]
            else gsplus[track]["primary"]
            for track in gsplus_tracks
        ]
        model_report = {
            "original": summarize(
                [predictions[track] for track in original_tracks],
                [original[track] for track in original_tracks],
            ),
            "original_on_gsplus_tracks": summarize(
                [predictions[track] for track in shared], [original[track] for track in shared]
            ),
            "gsplus_primary": summarize(
                [predictions[track] for track in gsplus_tracks],
                [gsplus[track]["primary"] for track in gsplus_tracks],
            ),
            "gsplus_any": summarize(
                [predictions[track] for track in gsplus_tracks], any_references
            ),
            "gsplus_clean": summarize(
                [predictions[track] for track in clean_tracks],
                [gsplus[track]["primary"] for track in clean_tracks],
            ),
        }
        for confidence in sorted({str(label["confidence"]) for label in gsplus.values()}):
            subset = [
                track for track in gsplus_tracks if str(gsplus[track]["confidence"]) == confidence
            ]
            model_report[f"gsplus_primary_confidence_{confidence}"] = summarize(
                [predictions[track] for track in subset],
                [gsplus[track]["primary"] for track in subset],
            )
        report["models"][name] = model_report
        report["per_track"][name] = predictions
        print(name, json.dumps(model_report))

    Path(arguments.out).write_text(json.dumps(report, indent=1) + "\n")
    print(
        "unparsed",
        len(unparsed),
        "clean",
        report["gsplus_clean"],
        "agreement",
        report["label_agreement_original_vs_gsplus_primary"],
    )


if __name__ == "__main__":
    main()
