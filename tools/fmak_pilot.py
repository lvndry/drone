"""Time a bounded FMAKv2 preparation pilot; never evaluate keys or train a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from drone.download import download
from experiments.reference.datasets import Track, prepare

ARCHIVE_URL = "https://zenodo.org/api/records/10719860/files/110-124.zip/content"
ARCHIVE_MD5 = "88d7dbcca82189ed75b7baa5aa132fc1"
ANNOTATION_URL = "https://zenodo.org/api/records/12759100/files/fmakv2.csv/content"
ANNOTATION_MD5 = "3b2d16784ffbda850c8ddf0519478bfd"


def prepare_one(args: tuple[Path, Path]) -> dict[str, Any]:
    path, cache = args
    start = time.monotonic()
    features, _ = prepare(Track(path.stem, str(path), ("C", "major")), cache)
    return {
        "id": path.stem,
        "seconds": time.monotonic() - start,
        "bytes": path.stat().st_size,
        "feature_shape": list(features.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.count <= 100 or not 1 <= args.workers <= 4:
        parser.error("Pilot is capped at 100 files and four workers")
    args.output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output).free < 55 * 1024**3:
        raise RuntimeError("Need 55 GiB free to preserve the host's 50 GiB storage headroom")
    annotations = args.output / "fmakv2.csv"
    download(ANNOTATION_URL, annotations, "md5:" + ANNOTATION_MD5)
    # Read identities only; key/mode annotations never enter the pilot calculation.
    with annotations.open() as stream:
        ids = {int(row["track_id"]) for row in csv.DictReader(stream)}
    archive = args.output / "110-124.zip"
    start = time.monotonic()
    download(ARCHIVE_URL, archive, "md5:" + ARCHIVE_MD5)
    download_seconds = time.monotonic() - start
    audio = args.output / "audio"
    audio.mkdir(exist_ok=True)
    start = time.monotonic()
    with zipfile.ZipFile(archive) as bundle:
        candidates = {}
        for member in bundle.infolist():
            name = Path(member.filename).name
            if "__MACOSX" in member.filename or member.is_dir() or not name.endswith(".mp3"):
                continue
            try:
                track_id = int(Path(name).stem)
            except ValueError:
                continue
            if track_id in ids:
                if track_id in candidates:
                    raise ValueError(f"Duplicate archive identity: {track_id}")
                candidates[track_id] = member
        selected = sorted(
            random.Random(13).sample(sorted(candidates), min(args.count, len(candidates)))
        )
        if not selected:
            raise ValueError("Archive has no matching FMAKv2 track IDs")
        paths = []
        for track_id in selected:
            target = audio / f"{track_id:06}.mp3"
            pending = target.with_suffix(".pending")
            with bundle.open(candidates[track_id]) as source, pending.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            os.replace(pending, target)
            paths.append(target)
    extraction_seconds = time.monotonic() - start
    cache = str(args.output / "features")
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(prepare_one, [(path, cache) for path in paths]))
    preparation_seconds = time.monotonic() - start
    result = {
        "purpose": "Runtime/data-availability pilot only; no key evaluation or model fitting",
        "sampling": "100 seeded IDs (or all if fewer) from the single 110-124 archive; not genre representative",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "archive": {"url": ARCHIVE_URL, "md5": ARCHIVE_MD5, "bytes": archive.stat().st_size},
        "annotations": {"url": ANNOTATION_URL, "md5": ANNOTATION_MD5},
        "workers": args.workers,
        "records": records,
        "download_seconds": download_seconds,
        "extraction_seconds": extraction_seconds,
        "preparation_seconds": preparation_seconds,
    }
    (args.output / "pilot.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
