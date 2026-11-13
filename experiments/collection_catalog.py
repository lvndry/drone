"""Prediction/target-free collection view; v1 scored reference catalogs stay unchanged.

External metadata is identified by hash and a portable descriptor, resolved through
caller-supplied local paths. Frozen memberships/groups are observations of existing
ledgers, not new split assignments or inferred independence guarantees.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

from experiments.corpus_catalog import Evidence, GenreTag, Measurement, SourceRecord

SCHEMA = "drone.collection-catalog.v1"
T = TypeVar("T")
HEX = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ExternalSource:
    source_id: str
    sha256: str
    descriptor: Evidence


@dataclass(frozen=True)
class OptionalIdentity:
    sha256: str | None
    method: str
    missing_reason: str | None
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class AudioSupport:
    support_kind: str
    samples: int | None
    sample_rate: int | None
    channels: int | None
    dtype: str | None
    pcm_identity: OptionalIdentity
    decode_contract: str
    interval_policy: str
    witness_kind: str
    missing_reason: str | None
    evidence: tuple[Evidence, ...]

    @property
    def seconds(self) -> float | None:
        return (
            self.samples / self.sample_rate
            if self.samples is not None and self.sample_rate
            else None
        )


@dataclass(frozen=True)
class Relation:
    kind: str
    entity_id: str
    name: str | None
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class Membership:
    plan_id: str
    role: str
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class FrozenGroup:
    scheme: str
    group_id: str
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class ScopedExposure:
    subject_kind: str
    subject_id: str
    checkpoint_sha256: str | None
    checkpoint_missing_reason: str | None
    relation: str
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class CollectionTrack:
    track_id: str
    corpus_id: str
    source_track_id: str
    source_audio: OptionalIdentity
    native_genre_top: str | None
    genre_tags: tuple[GenreTag, ...]
    genre_method: str
    source_duration: Measurement
    catalog_recording_duration: Measurement
    bpm: Measurement
    analyzed_audio: AudioSupport
    relations: tuple[Relation, ...]
    relation_missing_reasons: tuple[str, ...]
    memberships: tuple[Membership, ...]
    frozen_groups: tuple[FrozenGroup, ...]
    exposures: tuple[ScopedExposure, ...]


@dataclass(frozen=True)
class CollectionCatalog:
    schema: str
    collection_id: str
    sources: tuple[SourceRecord, ...]
    external_sources: tuple[ExternalSource, ...]
    tracks: tuple[CollectionTrack, ...]


def require_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Nonempty text required: " + field)


def validate_collection(value: CollectionCatalog) -> CollectionCatalog:
    if value.schema != SCHEMA:
        raise ValueError("Unsupported collection schema")
    require_text(value.collection_id, "collection ID")
    ids = [source.source_id for source in value.sources] + [
        source.source_id for source in value.external_sources
    ]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Unique source IDs required")

    def ev(records: tuple[Evidence, ...]) -> None:
        if not records or any(
            record.source_id not in ids
            or not isinstance(record.pointer, str)
            or not record.pointer.startswith("/")
            for record in records
        ):
            raise ValueError("Invalid evidence cross-reference")

    def sha(digest: object) -> None:
        if not isinstance(digest, str) or HEX.fullmatch(digest) is None:
            raise ValueError("Invalid SHA256")

    for source in value.sources:
        sha(source.sha256)
        path = Path(source.path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path == Path(".")
            or ":" in source.path
            or "\\" in source.path
        ):
            raise ValueError("Source paths must be portable repository-relative files")
    for source in value.external_sources:
        sha(source.sha256)
        if source.descriptor.source_id not in {source.source_id for source in value.sources}:
            raise ValueError("External sources need a bundled descriptor witness")
        ev((source.descriptor,))

    def identity(item: OptionalIdentity) -> None:
        require_text(item.method, "identity method")
        ev(item.evidence)
        if item.sha256 is None:
            require_text(item.missing_reason, "identity missing reason")
        else:
            sha(item.sha256)
            if item.missing_reason is not None:
                raise ValueError("Known identity cannot have a missing reason")

    def measurement(item: Measurement, unit: str) -> None:
        if item.unit != unit:
            raise ValueError("Measurement unit mismatch")
        require_text(item.method, "measurement method")
        ev(item.evidence)
        if item.value is None:
            require_text(item.missing_reason, "measurement missing reason")
        elif (
            isinstance(item.value, bool)
            or not isinstance(item.value, (int, float))
            or not math.isfinite(item.value)
            or item.value <= 0
            or item.missing_reason is not None
        ):
            raise ValueError("Known measurements must be positive and finite")

    tracks = set()
    for row in value.tracks:
        if row.track_id in tracks or row.track_id != row.corpus_id + ":" + row.source_track_id:
            raise ValueError("Duplicate or inconsistent track identity")
        tracks.add(row.track_id)
        identity(row.source_audio)
        measurement(row.source_duration, "seconds")
        measurement(row.catalog_recording_duration, "seconds")
        measurement(row.bpm, "bpm")
        for tag in row.genre_tags:
            require_text(tag.name, "native tag name")
            if tag.relation not in {"direct", "ancestor", "unspecified", "corpus_context"}:
                raise ValueError("Unknown tag relation")
            ev(tag.evidence)
        support = row.analyzed_audio
        require_text(support.support_kind, "support kind")
        require_text(support.witness_kind, "support witness kind")
        require_text(support.decode_contract, "decode contract")
        require_text(support.interval_policy, "interval policy")
        ev(support.evidence)
        identity(support.pcm_identity)
        if support.samples is None:
            require_text(support.missing_reason, "analyzed support missing reason")
        elif (
            type(support.samples) is not int
            or support.samples <= 0
            or type(support.sample_rate) is not int
            or support.sample_rate <= 0
            or type(support.channels) is not int
            or support.channels <= 0
            or support.dtype not in {"<f4", "<i2"}
            or support.missing_reason is not None
        ):
            raise ValueError("Invalid recorded analyzed support")
        for relation in row.relations:
            if relation.kind not in {"artist", "remixer", "album", "release"}:
                raise ValueError("Unknown metadata relation")
            require_text(relation.entity_id, "related entity")
            ev(relation.evidence)
        for membership in row.memberships:
            require_text(membership.plan_id, "frozen plan")
            require_text(membership.role, "frozen role")
            ev(membership.evidence)
        for group in row.frozen_groups:
            require_text(group.scheme, "group scheme")
            require_text(group.group_id, "group ID")
            ev(group.evidence)
        for exposure in row.exposures:
            if exposure.subject_kind not in {"model", "research_program"}:
                raise ValueError("Exposure subject required")
            require_text(exposure.subject_id, "exposure subject")
            require_text(exposure.relation, "source-declared exposure")
            ev(exposure.evidence)
            if exposure.checkpoint_sha256 is None:
                require_text(exposure.checkpoint_missing_reason, "checkpoint missing reason")
            else:
                sha(exposure.checkpoint_sha256)
                if exposure.checkpoint_missing_reason is not None:
                    raise ValueError("Known checkpoint cannot be missing")
    return value


def collection_bytes(value: CollectionCatalog) -> bytes:
    validate_collection(value)
    data = asdict(value)
    for field in ("sources", "external_sources"):
        data[field] = sorted(data[field], key=lambda row: row["source_id"])
    data["tracks"] = sorted(data["tracks"], key=lambda row: row["track_id"])
    stream = io.BytesIO()
    encoder = json.JSONEncoder(sort_keys=True, indent=2, allow_nan=False)
    for chunk in encoder.iterencode(data):
        stream.write(chunk.encode())
    stream.write(b"\n")
    return stream.getvalue()


def resolve_sources(
    value: CollectionCatalog, root: Path, external_paths: dict[str, Path]
) -> dict[str, Path]:
    """Resolve only supplied metadata files; verify each exact declared hash."""
    validate_collection(value)
    expected_external = {source.source_id for source in value.external_sources}
    if set(external_paths) != expected_external:
        raise ValueError("External resolver must cover exactly the declared metadata sources")
    resolved = {}
    for source in (*value.sources, *value.external_sources):
        path = (
            (root / source.path).resolve()
            if isinstance(source, SourceRecord)
            else external_paths[source.source_id]
        )
        if isinstance(source, SourceRecord) and not path.is_relative_to(root.resolve()):
            raise ValueError("Bundled source escapes repository")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != source.sha256:
                raise ValueError("Metadata source changed: " + source.source_id)
        resolved[source.source_id] = path
    return resolved


def load_collection(path: Path) -> CollectionCatalog:
    raw = path.read_bytes()
    data = json.loads(gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw)

    def ev(item: dict[str, Any]) -> tuple[Evidence, ...]:
        return tuple(Evidence(**record) for record in item["evidence"])

    def make(cls: type[T], item: dict[str, Any]) -> T:
        return cls(**(dict(item) | {"evidence": ev(item)}))

    tracks = []
    for row in data["tracks"]:
        converted = dict(row)
        converted["source_audio"] = make(OptionalIdentity, row["source_audio"])
        for name in ("source_duration", "catalog_recording_duration", "bpm"):
            converted[name] = make(Measurement, row[name])
        support = dict(row["analyzed_audio"])
        support["pcm_identity"] = make(OptionalIdentity, support["pcm_identity"])
        converted["analyzed_audio"] = make(AudioSupport, support)
        for name, cls in (
            ("genre_tags", GenreTag),
            ("relations", Relation),
            ("memberships", Membership),
            ("frozen_groups", FrozenGroup),
            ("exposures", ScopedExposure),
        ):
            converted[name] = tuple(make(cls, item) for item in row[name])
        converted["relation_missing_reasons"] = tuple(row["relation_missing_reasons"])
        tracks.append(CollectionTrack(**converted))
    return validate_collection(
        CollectionCatalog(
            **(
                data
                | dict(
                    sources=tuple(SourceRecord(**source) for source in data["sources"]),
                    external_sources=tuple(
                        ExternalSource(
                            **(source | {"descriptor": Evidence(**source["descriptor"])})
                        )
                        for source in data["external_sources"]
                    ),
                    tracks=tuple(tracks),
                )
            )
        )
    )
