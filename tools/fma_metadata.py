"""Fetch pinned official FMA metadata; audit artist overlap without model scoring."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

from drone.download import download

URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
CHECKSUM = "sha1:f0df49ffe5f2a6008d7dc83c6915b31835dfe733"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "fma_metadata.zip"
    download(URL, archive, CHECKSUM)
    with zipfile.ZipFile(archive) as bundle:
        tracks_bytes = bundle.read("fma_metadata/tracks.csv")
    rows = csv.reader(io.StringIO(tracks_bytes.decode()))
    category, names = next(rows), next(rows)
    columns = list(zip(category, names))
    next(rows)  # index name row
    fields = [("artist", "id"), ("artist", "name"), ("album", "id"), ("track", "title")]
    indices = {"_".join(field): columns.index(field) for field in fields}
    tracks = {}
    for row in rows:
        if not row:
            continue
        identity = f"{int(row[0]):06}"
        if identity in tracks:
            raise ValueError(f"Duplicate metadata identity {identity}")
        tracks[identity] = {name: row[index] for name, index in indices.items()}
    result = {
        "source": URL,
        "checksum": CHECKSUM,
        "tracks_csv_sha256": hashlib.sha256(tracks_bytes).hexdigest(),
        "tracks": tracks,
    }
    (args.output / "tracks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified FMA metadata: {len(tracks)} tracks", flush=True)


if __name__ == "__main__":
    main()
