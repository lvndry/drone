"""Stream selected full-length tracks out of the remote FMA full archive.

The archive is read with HTTP range requests, so only the chosen members are
downloaded. Each track is decoded, turned into a drone-contract CQT and deleted.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from environment import required_path

sys.path.insert(0, str(required_path("DRONE_SOURCE")))
import numpy as np  # noqa: E402

from drone.features import N_TOTAL, cqt, decode_mp3, read_wav_mono, resample_linear  # noqa: E402
from drone.features import SR as SAMPLE_RATE  # noqa: E402

ARCHIVE_URL = "https://os.unil.cloud.switch.ch/fma/fma_full.zip"
PREPARED_DIRECTORY = required_path("DRONE_PREP")
METADATA = PREPARED_DIRECTORY / "fma-meta" / "fma_metadata"
MANIFEST = PREPARED_DIRECTORY / "manifest.json"
DJ_TOP_LEVEL_TITLES = {"Electronic", "Hip-Hop", "Soul-RnB"}
EXTRA_GENRE_IDS = {79, 81, 602}
MINIMUM_FRAMES = 100
BLOCK_BYTES = 1 << 20


class RemoteFile:
    """A read-only, seekable view of a remote file backed by HTTP range requests."""

    def __init__(self, url: str) -> None:
        self.url = url
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
        self.position = 0
        self.cache: dict[int, bytes] = {}

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.position = offset
        elif whence == 1:
            self.position += offset
        else:
            self.position = self.size + offset
        return self.position

    def fetch_block(self, block: int) -> bytes:
        if block not in self.cache:
            start = block * BLOCK_BYTES
            end = min(start + BLOCK_BYTES, self.size) - 1
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(request, timeout=120) as response:
                        self.cache[block] = response.read()
                    break
                except OSError:
                    if attempt == 4:
                        raise
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[block]

    def read(self, count: int = -1) -> bytes:
        if count < 0:
            count = self.size - self.position
        count = min(count, self.size - self.position)
        chunks = []
        while count > 0:
            block = self.position // BLOCK_BYTES
            offset = self.position % BLOCK_BYTES
            data = self.fetch_block(block)[offset : offset + count]
            chunks.append(data)
            self.position += len(data)
            count -= len(data)
        return b"".join(chunks)


def select_tracks(per_artist: int, limit: int, seed: int) -> list[dict]:
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
    rows = json.loads(MANIFEST.read_text())["tracks"]
    evaluation_ids = {int(row["track_id"]) for row in rows if row["panel"] in ("valid", "holdout")}
    with open(METADATA / "tracks.csv", newline="") as handle:
        reader = csv.reader(handle)
        level_one = next(reader)
        level_two = next(reader)
        next(reader)
        columns = {
            f"{first}.{second}": index
            for index, (first, second) in enumerate(zip(level_one, level_two))
        }
        candidates = []
        for row in reader:
            track_id = int(row[0])
            genres = set(ast.literal_eval(row[columns["track.genres_all"]] or "[]"))
            if track_id in evaluation_ids or not genres & dj_genre_ids:
                continue
            candidates.append(
                {
                    "track_id": track_id,
                    "artist_id": row[columns["artist.id"]],
                    "genre_top": row[columns["track.genre_top"]] or "multi",
                    "duration": float(row[columns["track.duration"]] or 0),
                }
            )
    generator = random.Random(seed)
    generator.shuffle(candidates)
    candidates.sort(key=lambda candidate: candidate["genre_top"] == "multi")
    per_artist_count: Counter = Counter()
    selected = []
    for candidate in candidates:
        if (
            per_artist_count[candidate["artist_id"]] >= per_artist
            or not 30 <= float(candidate["duration"]) <= 900
        ):
            continue
        per_artist_count[candidate["artist_id"]] += 1
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def member_name(track_id: int) -> str:
    padded = f"{track_id:06d}"
    return f"fma_full/{padded[:3]}/{padded}.mp3"


def process_chunk(job: tuple[list[int], str]) -> dict[str, int]:
    track_ids, output = job
    outcomes: Counter = Counter()
    archive = zipfile.ZipFile(RemoteFile(ARCHIVE_URL))
    names = set(archive.namelist())
    for track_id in track_ids:
        destination = Path(output) / f"{track_id}.npy"
        if destination.is_file():
            outcomes["reused"] += 1
            continue
        name = member_name(track_id)
        if name not in names:
            outcomes["missing"] += 1
            continue
        try:
            with tempfile.TemporaryDirectory() as scratch:
                audio = Path(scratch) / "track.mp3"
                audio.write_bytes(archive.read(name))
                wav = Path(scratch) / "decoded.wav"
                decode_mp3(str(audio), str(wav))
                samples, rate = read_wav_mono(str(wav))
            features = cqt(resample_linear(samples, rate, SAMPLE_RATE)).astype(np.float16)
        except Exception as error:  # noqa: BLE001
            outcomes[f"failed:{type(error).__name__}"] += 1
            continue
        if (
            features.shape[0] != N_TOTAL
            or features.shape[1] < MINIMUM_FRAMES
            or not np.isfinite(features).all()
        ):
            outcomes["rejected"] += 1
            continue
        pending = destination.parent / f"{track_id}.partial.npy"
        np.save(pending, features)
        pending.replace(destination)
        outcomes["extracted"] += 1
    return dict(outcomes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--per-artist", type=int, default=15)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261114)
    arguments = parser.parse_args()
    output = Path(arguments.out)
    output.mkdir(parents=True, exist_ok=True)
    selected = select_tracks(arguments.per_artist, arguments.limit, arguments.seed)
    (output.parent / "fma-full-selection.json").write_text(json.dumps(selected))
    genre_counts = Counter(track["genre_top"] for track in selected)
    hours = sum(track["duration"] for track in selected) / 3600
    print(
        json.dumps(
            {
                "selected": len(selected),
                "hours": round(hours),
                "genres": genre_counts.most_common(8),
            }
        ),
        flush=True,
    )
    track_ids = [track["track_id"] for track in selected]
    chunks = [
        (track_ids[start : start + arguments.chunk], str(output))
        for start in range(0, len(track_ids), arguments.chunk)
    ]
    totals: Counter = Counter()
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for completed, outcome in enumerate(pool.map(process_chunk, chunks), start=1):
            totals.update(outcome)
            print(
                json.dumps({"chunks": completed, "of": len(chunks), "totals": dict(totals)}),
                flush=True,
            )
    (output.parent / "fma-full-summary.json").write_text(json.dumps(dict(totals), indent=1))


if __name__ == "__main__":
    main()
