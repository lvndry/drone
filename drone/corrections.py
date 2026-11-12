"""Key corrections exported by Euphoria, matched to local audio by content hash.

`euphoria corrections export` writes one JSON object per line. Each names its
track by the BLAKE3 hash of every byte of the file, so the export holds no
paths and is matched here by hashing the files of an audio folder the same way.
Only key corrections are kept; a line that corrects the tempo alone is counted
and skipped.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import blake3

from drone.datasets import Track
from drone.keys import TONICS, Key

SUPPORTED_SCHEMA_VERSION = 1
AUDIO_EXTENSIONS = frozenset({".mp3", ".m4a", ".flac", ".wav", ".aif", ".aiff"})
HELD_OUT_EVERY = 5
CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class KeyCorrection:
    """The key a person set by ear for one file's content."""

    content_hash: str
    key: Key
    detected_key: Key | None
    corrected_at: int


@dataclass(frozen=True)
class CorrectionsExport:
    key_corrections: list[KeyCorrection]
    tempo_only_count: int


@dataclass(frozen=True)
class MatchedCorrections:
    tracks: list[Track]
    unmatched_hashes: list[str]


def euphoria_content_hash(path: str | Path) -> str:
    """BLAKE3 of every byte of the file, tags included, as Euphoria computes it."""
    hasher = blake3.blake3()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            hasher.update(block)
    return hasher.hexdigest()


def key_from_fields(fields: object, line_number: int) -> Key:
    if not isinstance(fields, dict):
        raise ValueError(f"line {line_number}: a key must be an object")
    tonic = fields.get("tonic")
    mode = fields.get("mode")
    if not isinstance(tonic, int) or isinstance(tonic, bool) or not 0 <= tonic < 12:
        raise ValueError(f"line {line_number}: tonic must be a pitch class from 0 to 11")
    if mode not in ("major", "minor"):
        raise ValueError(f"line {line_number}: mode must be major or minor")
    return TONICS[tonic], str(mode)


def read_corrections_export(path: str | Path) -> CorrectionsExport:
    """Parse an export, refusing schema versions this code does not know."""
    key_corrections: dict[str, KeyCorrection] = {}
    tempo_only_count = 0
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            version = record.get("schema_version")
            if version != SUPPORTED_SCHEMA_VERSION:
                raise ValueError(
                    f"line {line_number}: schema_version {version!r} is not supported; "
                    f"this drone reads version {SUPPORTED_SCHEMA_VERSION}"
                )
            content_hash = record.get("content_hash")
            if (
                not isinstance(content_hash, str)
                or len(content_hash) != 64
                or content_hash != content_hash.lower()
                or any(character not in "0123456789abcdef" for character in content_hash)
            ):
                raise ValueError(f"line {line_number}: content_hash must be 64 hex digits")
            corrected_key = record.get("corrected_key")
            if corrected_key is None:
                tempo_only_count += 1
                continue
            detected_fields = record.get("detected_key")
            key_corrections[content_hash] = KeyCorrection(
                content_hash=content_hash,
                key=key_from_fields(corrected_key, line_number),
                detected_key=None
                if detected_fields is None
                else key_from_fields(detected_fields, line_number),
                corrected_at=int(record["corrected_at"]),
            )
    return CorrectionsExport(list(key_corrections.values()), tempo_only_count)


def audio_files(audio_directory: str | Path) -> list[Path]:
    found = []
    for folder, _, names in os.walk(audio_directory):
        for name in names:
            if Path(name).suffix.lower() in AUDIO_EXTENSIONS:
                found.append(Path(folder) / name)
    return sorted(found)


def match_corrections(
    corrections: list[KeyCorrection], audio_directory: str | Path
) -> MatchedCorrections:
    """Find each correction's file under `audio_directory` by its content hash.

    Tracks are identified by the content hash, so the same audio at two paths is
    one track.
    """
    wanted = {correction.content_hash: correction for correction in corrections}
    found: dict[str, Path] = {}
    for path in audio_files(audio_directory):
        if len(found) == len(wanted):
            break
        content_hash = euphoria_content_hash(path)
        if content_hash in wanted and content_hash not in found:
            found[content_hash] = path
    tracks = [
        Track(content_hash, str(found[content_hash]), wanted[content_hash].key)
        for content_hash in sorted(found)
    ]
    unmatched = sorted(set(wanted) - set(found))
    return MatchedCorrections(tracks, unmatched)


def is_held_out(content_hash: str) -> bool:
    """One file-byte identity in five; unchanged files keep their side as exports grow.

    Tags participate in the hash, so retagging can change the split even when
    decoded audio is unchanged. This is not a recording-level holdout identity.
    """
    return int(content_hash[:16], 16) % HELD_OUT_EVERY == 0


def split_corrections(tracks: list[Track]) -> tuple[list[Track], list[Track]]:
    fit = [track for track in tracks if not is_held_out(track.id)]
    held_out = [track for track in tracks if is_held_out(track.id)]
    if not fit or not held_out:
        raise ValueError(
            f"{len(tracks)} matched corrections give {len(fit)} to fit and {len(held_out)} "
            "held out; both need at least one. Correct more keys and export again."
        )
    return fit, held_out
