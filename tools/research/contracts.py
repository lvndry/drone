"""Label-blind validation inputs, separate from CLI parsing and model execution."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, TypeVar

import numpy as np
from numpy.typing import NDArray

from experiments.chunked import FEATURE_SPEC, sha256

VALIDATION_TRACKS = 376
PERFORMANCE_TRACKS = 20


class TimedTrackRecord(TypedDict):
    id: str
    sha256: str
    frames: int
    cache_sha256: str
    seconds: tuple[float, ...]
    median_seconds: float
    logits: list[list[float]]


def positive_frames(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 8:
        raise ValueError("Expected an integer frame count >= 8")
    return value


def text_field(row: Mapping[str, object], name: str) -> str:
    value = row[name]
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected nonempty {name}")
    return value


@dataclass(frozen=True)
class TrackIdentity:
    id: str
    domain: str
    frames: int
    sha256: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> TrackIdentity:
        domain = text_field(row, "domain")
        if domain not in ("beatport", "fmak"):
            raise ValueError("Only the fixed Beatport/FMAK validation domains are allowed")
        return cls(
            text_field(row, "id"), domain, positive_frames(row["frames"]), text_field(row, "sha256")
        )


@dataclass(frozen=True)
class CachedTrack(TrackIdentity):
    cache: str
    cache_sha256: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> CachedTrack:
        identity = TrackIdentity.from_mapping(row)
        return cls(
            identity.id,
            identity.domain,
            identity.frames,
            identity.sha256,
            text_field(row, "cache"),
            text_field(row, "cache_sha256"),
        )

    def as_dict(self) -> dict[str, str | int]:
        return dict(
            id=self.id,
            domain=self.domain,
            frames=self.frames,
            sha256=self.sha256,
            cache=self.cache,
            cache_sha256=self.cache_sha256,
        )


TrackT = TypeVar("TrackT", bound=TrackIdentity)


def load_validation_tracks(manifest: Path) -> list[CachedTrack]:
    payload = json.loads(manifest.read_bytes())
    if payload.get("feature_spec") != FEATURE_SPEC or payload.get("complete") is not True:
        raise ValueError("Incomplete/incompatible feature manifest")
    # No ground-truth reference enters the typed input contract.
    tracks = [
        CachedTrack.from_mapping(row) for row in payload["rows"] if row["role"] == "validation"
    ]
    if len(tracks) != VALIDATION_TRACKS:
        raise ValueError("Expected exactly 376 public validation tracks")
    if len({(track.domain, track.id) for track in tracks}) != len(tracks):
        raise ValueError("Duplicate validation identity")
    return tracks


def select_performance_tracks(tracks: Sequence[TrackT], selection: Path) -> list[TrackT]:
    payload = json.loads(selection.read_bytes())
    chosen = payload if isinstance(payload, list) else payload.get("rows", payload.get("tracks"))
    if not isinstance(chosen, list) or len(chosen) != PERFORMANCE_TRACKS:
        raise ValueError("Expected exactly 20 existing selected public tracks")
    matched = []
    for item in chosen:
        matches = [
            track for track in tracks if track.sha256 == item["sha256"] and track.domain == "fmak"
        ]
        if len(matches) != 1 or matches[0].frames != item["frames"]:
            raise ValueError("Selection does not uniquely match public FMAK validation cache")
        matched.append(matches[0])
    if len({track.sha256 for track in matched}) != PERFORMANCE_TRACKS:
        raise ValueError("Selection contains duplicate audio")
    return matched


def cache_path(manifest: Path, relative: str) -> Path:
    root = manifest.parent.resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Cache path must remain inside the prepared-data directory")
    return path


def load_cached_features(manifest: Path, track: CachedTrack) -> NDArray[np.float32]:
    path = cache_path(manifest, track.cache)
    if sha256(path) != track.cache_sha256:
        raise ValueError(f"Changed cache: {track.domain}/{track.id}")
    padded = np.load(path, mmap_mode="r", allow_pickle=False)
    if padded.shape != (208, track.frames) or padded.dtype != np.float16:
        raise ValueError("Expected unchanged padded FP16 CQT cache")
    features = np.ascontiguousarray(padded[8:200][None], dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"Nonfinite cached features: {track.domain}/{track.id}")
    return features


def legacy_rows(manifest: Path) -> list[dict[str, Any]]:
    """Preserve helper imports used by older fixture/export commands."""
    return [track.as_dict() for track in load_validation_tracks(manifest)]


def legacy_selection(rows: list[dict[str, Any]], selection: Path) -> list[dict[str, Any]]:
    identities = [TrackIdentity.from_mapping(row) for row in rows]
    selected = select_performance_tracks(identities, selection)
    lookup = {(row["domain"], row["id"]): row for row in rows}
    return [lookup[(track.domain, track.id)] for track in selected]
