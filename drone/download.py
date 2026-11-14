"""Download dev-time datasets with pinned annotations and verified checksums.

Uses curl for portable TLS/download retries. Preview audio is never packaged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

BEATPORT: dict[str, str] = {
    "audio.zip": "md5:f490ee6c23578482d6fcfa11b82636a1",
    "keys.zip": "md5:939abc05f36121badfac4087241ac172",
}
GIANTSTEPS_REVISION = "6bcd492c825ac9b8597bc650a5f6fd18b6c43d2b"
GIANTSTEPS_SHA256 = "9daf611028ba8bf448c14d819fe78eca86c000992b4a5c4e6f49a3d249c5d9fe"
AUDIO_BASES = (
    "https://www.cp.jku.at/datasets/giantsteps/backup",
    "https://geo-samples.beatport.com/lofi",
)


def verify(path: str | Path, checksum: str) -> bool:
    algorithm, expected = checksum.split(":", 1)
    with open(path, "rb") as stream:
        actual = hashlib.file_digest(stream, algorithm).hexdigest()
    return actual == expected


def download(url: str, target: str | Path, checksum: str) -> None:
    """Keep only complete verified files; failed downloads never poison the next run."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and verify(target, checksum):
        return
    with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
        pending = Path(scratch) / "download"
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "3",
                "--connect-timeout",
                "30",
                "--max-time",
                "7200",
                "--output",
                str(pending),
                url,
            ],
            check=True,
        )
        if not verify(pending, checksum):
            raise ValueError(f"Checksum mismatch for {url}; no unverified file was installed")
        os.replace(pending, target)


def extract_zip(archive: str | Path, target: str | Path, suffix: str) -> int:
    """Flatten trusted dataset audio/labels, excluding macOS resource files."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            name = Path(member.filename).name
            if "__MACOSX" in member.filename or name.startswith(".") or not name.endswith(suffix):
                continue
            if name in seen:
                raise ValueError(f"Duplicate dataset filename: {name}")
            seen.add(name)
            destination = target / name
            # Stage replacement so an interrupted extraction can be retried safely.
            with tempfile.TemporaryDirectory(dir=target) as scratch:
                pending = Path(scratch) / "entry"
                with bundle.open(member) as source, pending.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                os.replace(pending, destination)
    if not seen:
        raise ValueError(f"No {suffix} files in {archive}")
    return len(seen)


def extract_giantsteps(archive: str | Path, target: str | Path) -> None:
    target = Path(target)
    count = 0
    with tarfile.open(archive) as bundle:
        for member in bundle:
            relative = Path(*Path(member.name).parts[1:])
            if not member.isfile() or ".." in relative.parts or relative.is_absolute():
                continue
            if not (relative.parts[:2] == ("annotations", "key") or relative.parts[:1] == ("md5",)):
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                continue
            with source, destination.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            if relative.suffix == ".key":
                count += 1
    if count != 604:
        raise ValueError(f"Expected 604 GiantSteps key annotations, found {count}")


def fetch_datasets(data: str | Path, dataset: str = "all") -> None:
    data = Path(data)
    downloads = data / "downloads"
    if dataset in ("all", "train"):
        for name, checksum in BEATPORT.items():
            archive = downloads / f"beatport-{name}"
            download(
                f"https://zenodo.org/api/records/1101082/files/{name}/content", archive, checksum
            )
            count = extract_zip(
                archive,
                data / ("audio" if name == "audio.zip" else "labels"),
                ".mp3" if name == "audio.zip" else ".txt",
            )
            if count != 1486:
                raise ValueError(f"Expected 1486 entries in {name}, found {count}")
            print(f"Beatport {name}: {count} verified entries", flush=True)
    if dataset in ("all", "test"):
        archive = downloads / f"giantsteps-{GIANTSTEPS_REVISION}.tar.gz"
        download(
            "https://codeload.github.com/GiantSteps/giantsteps-key-dataset/tar.gz/"
            + GIANTSTEPS_REVISION,
            archive,
            "sha256:" + GIANTSTEPS_SHA256,
        )
        target = data / "giantsteps"
        extract_giantsteps(archive, target)
        for verified, label in enumerate(sorted((target / "annotations" / "key").glob("*.key")), 1):
            name = label.stem
            checksum_file = target / "md5" / f"{name}.md5"
            checksum = "md5:" + checksum_file.read_text().strip()
            for base in AUDIO_BASES:
                try:
                    download(f"{base}/{name}.mp3", target / "audio" / f"{name}.mp3", checksum)
                    break
                except (subprocess.CalledProcessError, ValueError):
                    if base == AUDIO_BASES[-1]:
                        raise
            if verified % 50 == 0:
                print(f"GiantSteps: {verified}/604 verified previews", flush=True)
    provenance_path = data / "sources.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    if dataset in ("all", "train"):
        provenance["beatport"] = {"record": "1101082", "checksums": BEATPORT}
    if dataset in ("all", "test"):
        provenance["giantsteps"] = {"revision": GIANTSTEPS_REVISION, "sha256": GIANTSTEPS_SHA256}
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("DRONE_DATA", "data")))
    parser.add_argument("--dataset", choices=("all", "train", "test"), default="all")
    args = parser.parse_args()
    try:
        fetch_datasets(args.data, args.dataset)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Dataset download failed: {exc}\n")


if __name__ == "__main__":
    main()
