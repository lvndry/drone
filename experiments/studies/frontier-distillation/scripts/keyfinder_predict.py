"""Run the corrected libkeyfinder wrapper on every GiantSteps track, decoded as in the 4-way benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DECODE_COMMAND = [
    "ffmpeg",
    "-loglevel",
    "error",
    "-y",
    "-ac",
    "1",
    "-ar",
    "44100",
    "-c:a",
    "pcm_s16le",
]


def predict(audio_path: Path, wrapper: Path, scratch: Path) -> tuple[str, dict]:
    track_id = audio_path.name.split(".")[0]
    wav_path = scratch / f"{track_id}.wav"
    decode = [*DECODE_COMMAND[:4], "-i", str(audio_path), *DECODE_COMMAND[4:], str(wav_path)]
    try:
        subprocess.run(decode, check=True, capture_output=True)
        output = subprocess.run(
            [str(wrapper), str(wav_path)], check=True, capture_output=True, text=True
        ).stdout
        return track_id, json.loads(output.strip().splitlines()[-1])
    except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError) as error:
        return track_id, {"status": "error", "error": f"{type(error).__name__}: {error}"}
    finally:
        wav_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--wrapper", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()
    audio_paths = sorted(
        path for path in Path(arguments.audio).glob("*.mp3") if not path.name.startswith("._")
    )
    wrapper = Path(arguments.wrapper)
    with tempfile.TemporaryDirectory() as scratch_directory:
        scratch = Path(scratch_directory)
        with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
            results = dict(
                executor.map(lambda audio_path: predict(audio_path, wrapper, scratch), audio_paths)
            )
    Path(arguments.out).write_text(json.dumps(results))
    statuses: dict[str, int] = {}
    for record in results.values():
        statuses[record.get("status", "unknown")] = (
            statuses.get(record.get("status", "unknown"), 0) + 1
        )
    print(f"tracks {len(results)}, statuses {statuses}")


if __name__ == "__main__":
    main()
