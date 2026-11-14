"""Pack the harmonic view of an existing pool and carry over the teacher's labels for the same tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True, help="prefix of the plain-view pool")
    parser.add_argument("--teacher", required=True, help="teacher labels written for that pool")
    parser.add_argument(
        "--harmonic", required=True, help="directory of <track_id>.npy harmonic features"
    )
    parser.add_argument("--out-pool", required=True)
    parser.add_argument("--out-teacher", required=True)
    arguments = parser.parse_args()
    index = np.load(f"{arguments.pool}-index.npz")
    teacher = np.load(arguments.teacher)
    harmonic_directory = Path(arguments.harmonic)
    kept_positions = []
    kept_paths = []
    kept_frames = []
    for position, track_id in enumerate(index["track_ids"]):
        path = harmonic_directory / f"{int(track_id)}.npy"
        if not path.exists():
            continue
        header = np.load(path, mmap_mode="r")
        kept_positions.append(position)
        kept_paths.append(path)
        kept_frames.append(header.shape[1])
    feature_bins = np.load(kept_paths[0], mmap_mode="r").shape[0]
    pool = np.lib.format.open_memmap(
        f"{arguments.out_pool}.npy",
        mode="w+",
        dtype=np.float16,
        shape=(feature_bins, sum(kept_frames)),
    )
    offsets = np.zeros(len(kept_paths), dtype=np.int64)
    cursor = 0
    for item, (path, frames) in enumerate(zip(kept_paths, kept_frames, strict=True)):
        pool[:, cursor : cursor + frames] = np.load(path)
        offsets[item] = cursor
        cursor += frames
    pool.flush()
    positions = np.array(kept_positions)
    np.savez(
        f"{arguments.out_pool}-index.npz",
        offsets=offsets,
        lengths=np.array(kept_frames, dtype=np.int64),
        track_ids=index["track_ids"][positions],
        sources=index["sources"][positions],
        dj=index["dj"][positions],
    )
    carried = {name: teacher[name] for name in teacher.files}
    carried["unlabeled"] = teacher["unlabeled"][positions]
    if "unlabeled_ids" in teacher.files:
        carried["unlabeled_ids"] = teacher["unlabeled_ids"][positions]
    np.savez(arguments.out_teacher, **carried)
    summary = {
        "items": len(kept_paths),
        "of": len(index["track_ids"]),
        "frames": int(sum(kept_frames)),
    }
    Path(f"{arguments.out_pool}-summary.json").write_text(json.dumps(summary, indent=1))
    print(summary)


if __name__ == "__main__":
    main()
