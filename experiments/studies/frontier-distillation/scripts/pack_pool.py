"""Pack per-track CQT files from several sources into one memory-mappable pool.

Training processes map the same file, so the operating system keeps a single
copy in RAM regardless of how many runs read it. Every item records its source,
FMA track id and whether its FMA genres fall in the DJ genre trees.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np
from environment import required_path

METADATA = required_path("DRONE_PREP") / "fma-meta" / "fma_metadata"
DJ_TOP_LEVEL_TITLES = {"Electronic", "Hip-Hop", "Soul-RnB"}
EXTRA_GENRE_IDS = {79, 81, 602}
FEATURE_BINS = 208


def dj_track_ids() -> set[int]:
    csv.field_size_limit(10**9)
    genre_rows = list(csv.DictReader(open(METADATA / "genres.csv", newline="")))
    top_level_ids = {
        row["genre_id"]
        for row in genre_rows
        if row["parent"] == "0" and row["title"] in DJ_TOP_LEVEL_TITLES
    }
    dj_genre_ids = {
        int(row["genre_id"]) for row in genre_rows if row["top_level"] in top_level_ids
    } | EXTRA_GENRE_IDS
    selected = set()
    with open(METADATA / "tracks.csv", newline="") as handle:
        reader = csv.reader(handle)
        level_one = next(reader)
        level_two = next(reader)
        next(reader)
        columns = {
            f"{first}.{second}": index
            for index, (first, second) in enumerate(zip(level_one, level_two))
        }
        for row in reader:
            if set(ast.literal_eval(row[columns["track.genres_all"]] or "[]")) & dj_genre_ids:
                selected.add(int(row[0]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", action="append", required=True, help="name=directory of <track_id>.npy"
    )
    parser.add_argument("--out", required=True, help="output prefix")
    arguments = parser.parse_args()
    dj_ids = dj_track_ids()
    items = []
    seen_ids: set[int] = set()
    for specification in arguments.source:
        name, directory = specification.split("=", 1)
        for path in sorted(Path(directory).glob("*.npy")):
            if path.name.endswith(".partial.npy"):
                continue
            track_id = int(path.stem)
            if name != "fma_full" and track_id in seen_ids:
                continue
            seen_ids.add(track_id)
            header = np.load(path, mmap_mode="r")
            if header.shape[0] != FEATURE_BINS:
                continue
            items.append((name, track_id, path, header.shape[1]))
    total_frames = sum(frames for _, _, _, frames in items)
    pool = np.lib.format.open_memmap(
        f"{arguments.out}.npy", mode="w+", dtype=np.float16, shape=(FEATURE_BINS, total_frames)
    )
    offsets = np.zeros(len(items), dtype=np.int64)
    lengths = np.zeros(len(items), dtype=np.int64)
    cursor = 0
    for index, (_, _, path, frames) in enumerate(items):
        pool[:, cursor : cursor + frames] = np.load(path)
        offsets[index] = cursor
        lengths[index] = frames
        cursor += frames
    pool.flush()
    np.savez(
        f"{arguments.out}-index.npz",
        offsets=offsets,
        lengths=lengths,
        track_ids=np.array([track_id for _, track_id, _, _ in items]),
        sources=np.array([name for name, _, _, _ in items]),
        dj=np.array([track_id in dj_ids for _, track_id, _, _ in items]),
    )
    summary = {
        "items": len(items),
        "frames": int(total_frames),
        "hours": round(total_frames * 4096 / 22050 / 3600, 1),
        "by_source": {
            name: sum(1 for item in items if item[0] == name)
            for name in {item[0] for item in items}
        },
        "dj_items": int(sum(track_id in dj_ids for _, track_id, _, _ in items)),
    }
    Path(f"{arguments.out}-summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
