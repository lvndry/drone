"""List the Myna views still to extract: every GiantSteps preview and the FMA evaluation tracks."""

from __future__ import annotations

import json
import os
from pathlib import Path

from environment import required_path

CACHE = required_path("DRONE_MYNA_CACHE")
GIANTSTEPS_AUDIO = required_path("DRONE_GIANTSTEPS") / "audio"
MANIFEST = required_path("DRONE_PREP") / "manifest.json"
ALL_SHIFTS = list(range(-6, 7))


def main() -> None:
    cached = set(os.listdir(CACHE))
    items = []
    for path in sorted(GIANTSTEPS_AUDIO.glob("*.mp3")):
        if not path.name.startswith("._"):
            items.append([f"giantsteps-{path.name.split('.')[0]}", str(path), ALL_SHIFTS])
    for row in json.loads(MANIFEST.read_text())["tracks"]:
        if row["source"] != "fmakv2" or row["panel"] not in ("valid", "holdout"):
            continue
        name = f"fmak-{int(row['track_id']):06d}"
        missing = [shift for shift in ALL_SHIFTS if f"{name}-shift{shift:+d}.npy" not in cached]
        if missing:
            items.append([name, row["audio_path"], missing])
    Path("myna-items.json").write_text(json.dumps(items))
    print(len(items), "items,", sum(len(item[2]) for item in items), "views")


if __name__ == "__main__":
    main()
