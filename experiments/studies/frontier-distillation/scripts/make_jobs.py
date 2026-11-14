"""Write extraction job lists for the harmonic-view experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from environment import required_path

PREPARED_DIRECTORY = required_path("DRONE_PREP")
GIANTSTEPS_DIRECTORY = required_path("DRONE_GIANTSTEPS")
WORK_DIRECTORY = required_path("DRONE_WORK")
mode = sys.argv[1]
if mode == "labelled":
    rows = json.loads((PREPARED_DIRECTORY / "manifest.json").read_text())["tracks"]
    jobs = [
        [row["audio_path"], "", str(WORK_DIRECTORY / "harmonic-cqt" / f"{row['audio_sha256']}.npy")]
        for row in rows
    ]
    audio_directory = GIANTSTEPS_DIRECTORY / "audio"
    for path in sorted(audio_directory.glob("*.mp3")):
        if path.name.startswith("._"):
            continue
        jobs.append(
            [str(path), "", str(WORK_DIRECTORY / "harmonic-gs" / f"{path.name.split('.')[0]}.npy")]
        )
    Path(sys.argv[2]).write_text(json.dumps(jobs))
else:
    folder = Path(sys.argv[2])
    rows = json.loads((PREPARED_DIRECTORY / "manifest.json").read_text())["tracks"]
    evaluation_ids = {int(row["track_id"]) for row in rows if row["panel"] in ("valid", "holdout")}
    jobs = []
    for path in sorted(folder.rglob("*.mp3")):
        if path.name.startswith("._") or int(path.stem) in evaluation_ids:
            continue
        stem = str(int(path.stem))
        jobs.append(
            [
                str(path),
                str(WORK_DIRECTORY / "fma-large" / "cqt-all" / f"{stem}.npy"),
                str(WORK_DIRECTORY / "fma-large" / "harmonic-all" / f"{stem}.npy"),
            ]
        )
    Path(sys.argv[3]).write_text(json.dumps(jobs))
print(len(jobs))
