"""Verify a pinned GTZAN archive without extraction, inference or score admission."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
import wave
from pathlib import Path, PurePosixPath
from typing import Any

from tools.import_gtzan_reference import INDEX_SHA256

ARCHIVE_BYTES = 1_226_192_050
ARCHIVE_SHA256 = "b28cc067ff6199bd826f5d1a6931458586d64acae7f44b2e882b7a97af057531"
SOURCE_COMMIT = "d2146561ecc7df707d9e6b8318885fe6a39668a2"
URL = (
    "https://huggingface.co/datasets/marsyas/gtzan/resolve/" + SOURCE_COMMIT + "/data/genres.tar.gz"
)
INDEX = (
    Path(__file__).resolve().parents[1]
    / "experiments/studies/v2-data-contract/data/gtzan-reference/sources/recording-index.json.gz"
)
MEMBER_LIMIT = 8 * 1024**2
TOTAL_LIMIT = 2 * 1024**3


def expected_members(index: Path = INDEX) -> dict[str, str]:
    raw = gzip.decompress(index.read_bytes())
    if hashlib.sha256(raw).hexdigest() != INDEX_SHA256:
        raise ValueError("Pinned recording index changed")
    tracks = json.loads(raw)["tracks"]
    result = {}
    for track in tracks.values():
        name, digest = track["audio"]
        if not name.startswith("gtzan_genre/genres/"):
            raise ValueError("Unexpected index path")
        result[name.removeprefix("gtzan_genre/")] = digest
    if len(result) != 1000:
        raise ValueError("Expected complete 1,000-recording index")
    return result


def inspect_wave(raw: bytes) -> dict[str, int | float]:
    with wave.open(io.BytesIO(raw), "rb") as audio:
        channels, rate, width, frames = (
            audio.getnchannels(),
            audio.getframerate(),
            audio.getsampwidth(),
            audio.getnframes(),
        )
        if (channels, rate, width, audio.getcomptype()) != (1, 22050, 2, "NONE"):
            raise ValueError("Unexpected WAV encoding")
        if not 0 < frames <= rate * 60:
            raise ValueError("Unexpected WAV duration")
        if len(audio.readframes(frames + 1)) != frames * width * channels:
            raise ValueError("Truncated or inconsistent WAV payload")
        return dict(
            channels=channels,
            sample_rate=rate,
            sample_width=width,
            frames=frames,
            duration_seconds=frames / rate,
        )


def inspect_members(archive: Path, expected: dict[str, str]) -> list[dict[str, Any]]:
    """Bound payload size and validate every member before interpreting audio bytes."""
    seen: set[str] = set()
    rows = []
    total = 0
    with tarfile.open(archive, "r|gz") as stream:
        for number, member in enumerate(stream, 1):
            path = PurePosixPath(member.name)
            if (
                number > 1100
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in member.name
                or member.name in seen
            ):
                raise ValueError("Unsafe, duplicate or excessive archive members")
            seen.add(member.name)
            if member.isdir():
                if str(path) != "genres" and str(path) not in {
                    str(PurePosixPath(name).parent) for name in expected
                }:
                    raise ValueError("Unexpected archive directory")
                continue
            if not member.isfile() or member.name not in expected:
                raise ValueError("Unexpected archive member or link")
            total += member.size
            if not 0 < member.size <= MEMBER_LIMIT or total > TOTAL_LIMIT:
                raise ValueError("Archive payload budget exceeded")
            handle = stream.extractfile(member)
            if handle is None:
                raise ValueError("Missing archive payload")
            raw = handle.read(MEMBER_LIMIT + 1)
            if len(raw) != member.size:
                raise ValueError("Truncated archive payload")
            digest = hashlib.md5(raw).hexdigest()
            if digest != expected[member.name]:
                raise ValueError("Recording MD5 mismatch: " + member.name)
            row: dict[str, Any] = {
                "member": member.name,
                "bytes": len(raw),
                "md5": digest,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "member_identity_verified": True,
                "overlap_cleared": False,
                "admitted_for_scoring": False,
            }
            try:
                row["wave"] = inspect_wave(raw)
            except (wave.Error, EOFError, ValueError) as error:
                if member.name != "genres/jazz/jazz.00054.wav":
                    raise ValueError("Unexpected WAV fault: " + member.name) from error
                row["known_fault"] = repr(error)
            rows.append(row)
    if {row["member"] for row in rows} != set(expected):
        raise ValueError("Incomplete recording universe")
    return sorted(rows, key=lambda row: row["member"])


def audit(archive: Path) -> dict[str, Any]:
    if archive.is_symlink() or archive.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("Pinned archive size/type mismatch")
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        while chunk := stream.read(1024**2):
            digest.update(chunk)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Pinned archive SHA256 mismatch")
    return {
        "scope": "Recording identity and WAV structure only; no inference or overlap clearance",
        "source_url": URL,
        "archive_sha256": digest.hexdigest(),
        "archive_bytes": ARCHIVE_BYTES,
        "index_sha256": INDEX_SHA256,
        "records": inspect_members(archive, expected_members()),
        "admitted_for_scoring": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Do not overwrite an existing audit")
    result = audit(args.archive)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"verified_recordings": len(result["records"]), "scoring_admitted": False}))


if __name__ == "__main__":
    main()
