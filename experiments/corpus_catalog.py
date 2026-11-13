"""Portable, prediction-free corpus metadata with explicit evidence and missingness.

The reference-panel adapter preserves recorded genre buckets and source durations.
Duration bands match tools.dj_coverage.band; equivalence is tested at boundaries.
Results remain a separate track_id-keyed relation, never fields of this catalog.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

SCHEMA = "drone.corpus-catalog.v1"
HEX = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Evidence:
    source_id: str
    pointer: str


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    path: str
    sha256: str
    input_recorded_at: str | None = None
    provenance_captured_at: str | None = None


@dataclass(frozen=True)
class Measurement:
    value: float | None
    unit: Literal["seconds", "bpm"]
    method: str
    missing_reason: str | None
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class GenreTag:
    name: str
    tag_id: str | int | None
    taxonomy: str | None
    relation: Literal["direct", "ancestor", "unspecified", "corpus_context"]
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class Genre:
    raw: str | None
    bucket: str | None
    method: str
    evidence: tuple[Evidence, ...]
    raw_tags: tuple[GenreTag, ...] = ()


@dataclass(frozen=True)
class Grouping:
    work_id: str | None
    performance_id: str | None
    method: str
    missing_reason: str | None
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class KeyLabel:
    raw: str
    class_id: int
    schema: str
    policy: str
    annotation_source: str
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class Exposure:
    benchmark_role: str
    development_use: str
    direct_fit_status: str
    overlap_audit_status: str
    evidence: tuple[Evidence, ...]
    subject_model_id: str
    other_model_exposure_status: str


@dataclass(frozen=True)
class Durations:
    source_audio: Measurement
    analyzed: Measurement
    full_recording: Measurement
    analyzed_band: str


@dataclass(frozen=True)
class TrackMetadata:
    track_id: str
    catalog_id: str
    corpus_id: str
    source_track_id: str
    audio_id: str
    source_audio_sha256: str
    canonical_wav_sha256: str
    canonical_pcm_sha256: str
    genre: Genre
    duration: Durations
    bpm: Measurement
    grouping: Grouping
    key_label: KeyLabel
    exposure: Exposure


@dataclass(frozen=True)
class Catalog:
    schema: str
    scope: str
    sources: tuple[SourceRecord, ...]
    tracks: tuple[TrackMetadata, ...]
    results_source_id: str | None
    results_join_key: str = "track_id"


def duration_band(seconds: float | None) -> str:
    if seconds is None:
        return "duration_unknown"
    return (
        "<30s"
        if seconds < 30
        else "30s–<2m"
        if seconds < 120
        else "2–<5m"
        if seconds < 300
        else "5–10m"
        if seconds <= 600
        else ">10m"
    )


def text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Nonempty text required: " + name)


def validate_catalog(catalog: Catalog) -> Catalog:
    if catalog.schema != SCHEMA or catalog.results_join_key != "track_id":
        raise ValueError("Unsupported catalog schema/join key")
    text(catalog.scope, "scope")
    sources = {source.source_id: source for source in catalog.sources}
    if len(sources) != len(catalog.sources) or not sources:
        raise ValueError("Unique evidence sources required")
    if catalog.results_source_id is not None and catalog.results_source_id not in sources:
        raise ValueError("Missing results-relation source")
    for source in sources.values():
        text(source.source_id, "source_id")
        text(source.path, "source path")
        path = Path(source.path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in source.path
            or not source.path
            or path == Path(".")
            or "://" in source.path
        ):
            raise ValueError("Evidence paths must be portable repository-relative paths")
        if not isinstance(source.sha256, str) or HEX.fullmatch(source.sha256) is None:
            raise ValueError("Invalid evidence SHA256")
        for timestamp in (source.input_recorded_at, source.provenance_captured_at):
            if timestamp is not None:
                text(timestamp, "timestamp")

    def evidence(records: tuple[Evidence, ...]) -> None:
        if not records:
            raise ValueError("Metadata must retain its evidence or explicit absence pointer")
        for record in records:
            if (
                record.source_id not in sources
                or not isinstance(record.pointer, str)
                or not record.pointer.startswith("/")
            ):
                raise ValueError("Invalid evidence source/pointer")

    def measurement(value: Measurement, unit: str) -> None:
        if value.unit != unit:
            raise ValueError("Measurement unit mismatch")
        text(value.method, "measurement method")
        evidence(value.evidence)
        if value.value is None:
            text(value.missing_reason, "measurement missing reason")
        elif (
            isinstance(value.value, bool)
            or not isinstance(value.value, (int, float))
            or not math.isfinite(value.value)
            or value.value <= 0
            or value.missing_reason is not None
        ):
            raise ValueError("Known measurement must be positive/finite with no missing reason")

    track_ids, catalog_ids = set(), set()
    for row in catalog.tracks:
        for field in ("track_id", "catalog_id", "corpus_id", "source_track_id"):
            text(getattr(row, field), field)
        if row.track_id in track_ids or row.catalog_id in catalog_ids:
            raise ValueError("Repeated track/catalog ID")
        track_ids.add(row.track_id)
        catalog_ids.add(row.catalog_id)
        if row.catalog_id != row.corpus_id + ":" + row.source_track_id:
            raise ValueError("Catalog ID disagrees with logical source identity")
        for value in (row.source_audio_sha256, row.canonical_wav_sha256, row.canonical_pcm_sha256):
            if not isinstance(value, str) or HEX.fullmatch(value) is None:
                raise ValueError("Invalid audio SHA256")
        if row.audio_id != "sha256:" + row.source_audio_sha256:
            raise ValueError("Audio ID disagrees with content identity")
        for value in (
            row.duration.source_audio,
            row.duration.analyzed,
            row.duration.full_recording,
        ):
            measurement(value, "seconds")
        measurement(row.bpm, "bpm")
        if row.duration.analyzed_band != duration_band(row.duration.analyzed.value):
            raise ValueError("Duration band disagrees with analyzed support")
        evidence(row.genre.evidence)
        text(row.genre.method, "genre method")
        for value in (row.genre.raw, row.genre.bucket):
            if value is not None:
                text(value, "genre")
        for tag in row.genre.raw_tags:
            text(tag.name, "genre tag name")
            if tag.relation not in {"direct", "ancestor", "unspecified", "corpus_context"}:
                raise ValueError("Unknown source genre-tag relation")
            if tag.tag_id is not None and (
                isinstance(tag.tag_id, bool) or not isinstance(tag.tag_id, (str, int))
            ):
                raise ValueError("Invalid native genre tag ID")
            if tag.taxonomy is not None:
                text(tag.taxonomy, "genre taxonomy")
            evidence(tag.evidence)
        evidence(row.grouping.evidence)
        text(row.grouping.method, "grouping method")
        if row.grouping.work_id is None and row.grouping.performance_id is None:
            text(row.grouping.missing_reason, "grouping missing reason")
        elif row.grouping.missing_reason is not None:
            raise ValueError("Known grouping cannot have a missing reason")
        for value in (row.grouping.work_id, row.grouping.performance_id):
            if value is not None:
                text(value, "group identity")
        if type(row.key_label.class_id) is not int or not 0 <= row.key_label.class_id < 24:
            raise ValueError("Invalid key class")
        for value in (
            row.key_label.raw,
            row.key_label.schema,
            row.key_label.policy,
            row.key_label.annotation_source,
        ):
            text(value, "key label provenance")
        evidence(row.key_label.evidence)
        evidence(row.exposure.evidence)
        for value in (
            row.exposure.benchmark_role,
            row.exposure.development_use,
            row.exposure.direct_fit_status,
            row.exposure.overlap_audit_status,
            row.exposure.subject_model_id,
            row.exposure.other_model_exposure_status,
        ):
            text(value, "exposure")
    return catalog


def validate_sources(catalog: Catalog, root: Path) -> Catalog:
    """Verify declared metadata bytes without opening track audio."""
    validate_catalog(catalog)
    resolved_root = root.resolve()
    for source in catalog.sources:
        path = (resolved_root / source.path).resolve()
        if not path.is_relative_to(resolved_root) or not path.is_file():
            raise ValueError("Source path escapes repository or is not a file")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != source.sha256:
                raise ValueError("Catalog metadata source bytes changed: " + source.source_id)
    return catalog


def catalog_bytes(catalog: Catalog) -> bytes:
    validate_catalog(catalog)
    value = asdict(catalog)
    value["sources"] = sorted(value["sources"], key=lambda row: row["source_id"])
    value["tracks"] = sorted(
        value["tracks"], key=lambda row: (row["corpus_id"], row["source_track_id"])
    )
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def load_catalog(path: Path) -> Catalog:
    raw = path.read_bytes()
    data = json.loads(gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw)

    def ev(value: dict[str, Any]) -> tuple[Evidence, ...]:
        return tuple(Evidence(**item) for item in value["evidence"])

    def measure(value: dict[str, Any]) -> Measurement:
        selected: dict[str, Any] = dict(value)
        selected["evidence"] = ev(value)
        return Measurement(**selected)

    tracks = []
    for row in data["tracks"]:
        duration = row["duration"]
        tracks.append(
            TrackMetadata(
                **(
                    row
                    | dict(
                        genre=Genre(
                            **(
                                row["genre"]
                                | {
                                    "evidence": ev(row["genre"]),
                                    "raw_tags": tuple(
                                        GenreTag(**(tag | {"evidence": ev(tag)}))
                                        for tag in row["genre"]["raw_tags"]
                                    ),
                                }
                            )
                        ),
                        duration=Durations(
                            **(
                                duration
                                | {
                                    key: measure(duration[key])
                                    for key in ("source_audio", "analyzed", "full_recording")
                                }
                            )
                        ),
                        bpm=measure(row["bpm"]),
                        grouping=Grouping(**(row["grouping"] | {"evidence": ev(row["grouping"])})),
                        key_label=KeyLabel(
                            **(row["key_label"] | {"evidence": ev(row["key_label"])})
                        ),
                        exposure=Exposure(**(row["exposure"] | {"evidence": ev(row["exposure"])})),
                    )
                )
            )
        )
    return validate_catalog(
        Catalog(
            **(
                data
                | dict(
                    sources=tuple(SourceRecord(**row) for row in data["sources"]),
                    tracks=tuple(tracks),
                )
            )
        )
    )
