"""Join existing FMA metadata and frozen roles; never select targets or read audio."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.collection_catalog import (
    SCHEMA,
    AudioSupport,
    CollectionCatalog,
    CollectionTrack,
    ExternalSource,
    FrozenGroup,
    Membership,
    OptionalIdentity,
    Relation,
    ScopedExposure,
    collection_bytes,
    resolve_sources,
    validate_collection,
)
from experiments.corpus_catalog import Evidence, GenreTag, Measurement, SourceRecord

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/studies/v2-data-contract/data"
PLAN = "experiments/studies/k10-v2-pilot/data/canonical-eligibility-v2/plan-v2.json.gz"
BUNDLED = {
    "coverage_inputs": "experiments/studies/v2-data-contract/data/coverage-inputs.json",
    "fma_group_input": "experiments/studies/v2-data-contract/data/input.json.gz",
    "fma_exposure_ledger": "experiments/studies/v2-data-contract/data/exposure-ledger.csv.gz",
    "frozen_plan": PLAN,
}
EXTERNAL = {
    "fma_master_metadata": ("fma", "master_metadata"),
    "fma_taxonomy": ("fma", "taxonomy"),
    "training_cache_metadata": ("bp", "canonical_counts"),
}
FIELDS = (
    "expanded_fma_fit",
    "restricted_fma_fit",
    "fma_inner_selection",
    "confirmation_reserved_container_unique",
)


def decoded(path: Path) -> bytes:
    raw = path.read_bytes()
    return gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw


def source(path: str, name: str, root: Path) -> SourceRecord:
    return SourceRecord(name, path, hashlib.sha256((root / path).read_bytes()).hexdigest())


def records(root: Path) -> tuple[tuple[SourceRecord, ...], tuple[ExternalSource, ...]]:
    bundled = tuple(source(path, name, root) for name, path in BUNDLED.items())
    index = json.loads((root / BUNDLED["coverage_inputs"]).read_bytes())
    external = tuple(
        ExternalSource(
            name, index[corpus][role]["sha256"], Evidence("coverage_inputs", f"/{corpus}/{role}")
        )
        for name, (corpus, role) in EXTERNAL.items()
    )
    # Require exact original metadata witnesses, not a newly substituted index.
    for name, corpus, role in (
        ("fma_group_input", "fma", "sanitized_group_input"),
        ("fma_exposure_ledger", "fma", "exposure_ledger"),
        ("frozen_plan", "fma", "plan"),
    ):
        if (
            next(item.sha256 for item in bundled if item.source_id == name)
            != index[corpus][role]["sha256"]
        ):
            raise ValueError("Bundled frozen role/group source changed")
    return bundled, external


def id_value(value: str) -> str | None:
    return value if value.isdigit() and int(value) > 0 else None


def build(root: Path, resolver: dict[str, Path]) -> CollectionCatalog:
    bundled, external = records(root)
    empty = CollectionCatalog(SCHEMA, "fmak5489-source-metadata", bundled, external, ())
    paths = resolve_sources(empty, root, resolver)
    safe = json.loads(decoded(paths["fma_group_input"]))
    if safe["provenance"]["no_target_labels"] is not True:
        raise ValueError("Blinded group source required")
    allowed = {
        "identity",
        "component",
        "old_role",
        "audio_sha256",
        "artist_id",
        "album_id",
        "genre",
        "duration",
    }
    if any(set(row) != allowed for row in safe["tracks"]):
        raise ValueError("Unexpected blinded metadata columns")
    groups = {row["identity"]: row for row in safe["tracks"]}
    ledger = {
        row["id"]: row
        for row in csv.DictReader(io.StringIO(decoded(paths["fma_exposure_ledger"]).decode()))
    }
    if len(groups) != 5489 or set(groups) != set(ledger):
        raise ValueError("Complete5489 group/ledger membership required")
    text = decoded(paths["frozen_plan"]).decode()
    plans = {}
    for field in FIELDS:
        match = re.search(r'"' + field + r'"\s*:\s*(\[[^\]]*\])', text)
        if match is None:
            raise ValueError("Missing frozen fitting role")
        values = json.loads(match[1])
        if len(set(values)) != len(values) or not set(values) <= set(groups):
            raise ValueError("Invalid frozen role membership")
        plans[field] = set(values)
    if [len(plans[field]) for field in FIELDS] != [4033, 976, 250, 997]:
        raise ValueError("Current eligible FMA roles changed")
    with paths["fma_taxonomy"].open() as stream:
        taxonomy: dict[int, dict[str, Any]] = {
            int(row["genre_id"]): dict(title=row["title"], parent=int(row["parent"]))
            for row in csv.DictReader(stream)
        }
    if len(taxonomy) != 163:
        raise ValueError("Pinned native taxonomy changed")
    # Select metadata columns only; the master table contains no admitted tempo field.
    master = {}
    with paths["fma_master_metadata"].open(newline="") as stream:
        for row in csv.DictReader(stream):
            identity = f"fmak:{int(row['track_id']):06}"
            if identity not in groups:
                continue
            if identity in master:
                raise ValueError("Repeated FMA master ID")
            master[identity] = {
                field: row[field]
                for field in (
                    "artist_id",
                    "artist_name",
                    "album_id",
                    "album_title",
                    "track_genre_top",
                    "track_genres_all",
                    "track_duration",
                )
            }
    if set(master) != set(groups):
        raise ValueError("Master metadata does not cover every5489 ID")
    # This metadata receipt includes target fields; never select/export those fields.
    cache = {}
    for item in json.loads(paths["training_cache_metadata"].read_bytes())["rows"]:
        identity = item["id"]
        if identity not in groups:
            continue
        if identity in cache:
            raise ValueError("Duplicate saved cache metadata ID")
        cache[identity] = {
            field: item[field]
            for field in ("samples", "bytes", "sha256", "waveform_sha256", "full_execution")
        }
    if set(cache) != plans["expanded_fma_fit"] | plans["fma_inner_selection"]:
        raise ValueError("Saved analyzed-support coverage differs from fitting/inner panel")
    tracks = []
    for identity in sorted(groups):
        group = groups[identity]
        history = ledger[identity]
        metadata = master[identity]
        origin = (Evidence("fma_group_input", f"/tracks/{identity}"),)
        master_evidence = (Evidence("fma_master_metadata", f"/rows/{identity}"),)
        history_evidence = (Evidence("fma_exposure_ledger", f"/rows/{identity}"),)
        if group["component"] != history["component"]:
            raise ValueError("Frozen component witnesses disagree")
        try:
            duration = float(metadata["track_duration"])
        except ValueError:
            duration = float("nan")
        if not math.isfinite(duration) or duration <= 0:
            duration = None
        if group["duration"] != duration:
            raise ValueError("Publisher duration/group metadata disagree")
        tags = ast.literal_eval(metadata["track_genres_all"] or "[]")
        if not isinstance(tags, list) or any(
            type(tag) is not int or tag not in taxonomy for tag in tags
        ):
            raise ValueError("Invalid native genre hierarchy IDs")
        relations = []
        missing = []
        for kind, key, name in (
            ("artist", "artist_id", "artist_name"),
            ("album", "album_id", "album_title"),
        ):
            entity = id_value(metadata[key])
            if (entity or "") != group[key]:
                raise ValueError("Source entity/group metadata disagree")
            if entity is None:
                missing.append(kind + "_not_recorded")
            else:
                relations.append(
                    Relation(kind, f"fma:{kind}:{entity}", metadata[name] or None, master_evidence)
                )
        memberships = [
            Membership("fma-exposure-ledger-v1", history["proposed_role"], history_evidence),
            Membership("fma-original-partition-v1", group["old_role"], origin),
        ]
        memberships += [
            Membership("canonical-eligibility-v2", field, (Evidence("frozen_plan", "/" + field),))
            for field in FIELDS
            if identity in plans[field]
        ]
        if identity in cache:
            item = cache[identity]
            command = item["full_execution"]["command"]
            if (
                item["sha256"] != group["audio_sha256"]
                or item["full_execution"]["exit"] != 0
                or command[command.index("-ar") + 1] != "24000"
                or command[command.index("-ac") + 1] != "1"
                or command[command.index("-f") + 1] != "f32le"
                or any(flag in command for flag in ("-ss", "-t", "-to"))
                or type(item["samples"]) is not int
                or item["bytes"] != item["samples"] * 4
            ):
                raise ValueError("Saved source identity/decode contract differs")
            evidence = (Evidence("training_cache_metadata", f"/rows/{identity}"),)
            support = AudioSupport(
                "recorded_24khz_training_cache",
                item["samples"],
                24000,
                1,
                "<f4",
                OptionalIdentity(
                    item["waveform_sha256"],
                    "saved_prepared_f32le_sha256_not_reverified",
                    None,
                    evidence,
                ),
                "saved_ffmpeg_mono_24000Hz_f32le",
                "whole_source_no_seek_or_trim_in_saved_command",
                "saved_cache_metadata_not_new_decode_verification",
                None,
                evidence,
            )
        else:
            support = AudioSupport(
                "recorded_24khz_training_cache",
                None,
                None,
                None,
                None,
                OptionalIdentity(
                    None, "not_recorded", "no_cache_identity_in_selected_receipt", origin
                ),
                "not_recorded_for_this_ID",
                "not_recorded_for_this_ID",
                "saved_metadata_only",
                "no_analyzed_support_for_reserved_or_excluded_ID",
                origin,
            )
        tracks.append(
            CollectionTrack(
                identity,
                "fmak",
                identity.split(":", 1)[1],
                OptionalIdentity(
                    group["audio_sha256"],
                    "saved_group_input_source_container_sha256_not_reverified",
                    None,
                    origin,
                ),
                metadata["track_genre_top"] or None,
                tuple(
                    GenreTag(
                        taxonomy[tag]["title"],
                        tag,
                        "FMA-native-genre-taxonomy",
                        "unspecified",
                        master_evidence,
                    )
                    for tag in tags
                ),
                "source_track_genres_all_includes_hierarchy; direct_vs_ancestor_not_disambiguated; top_field_separate",
                Measurement(
                    None,
                    "seconds",
                    "not_measured_here",
                    "source_container_duration_not_probed",
                    origin,
                ),
                Measurement(
                    duration,
                    "seconds",
                    "publisher_FMA_master_track_duration_not_reverified",
                    "missing_publisher_duration" if duration is None else None,
                    master_evidence,
                ),
                Measurement(
                    None,
                    "bpm",
                    "not_admitted",
                    "no_tempo_field_in_pinned_FMA_master_metadata",
                    master_evidence,
                ),
                support,
                tuple(relations),
                tuple(missing),
                tuple(memberships),
                (
                    FrozenGroup(
                        "existing_FMA_frozen_component_v1_not_recomputed",
                        group["component"],
                        origin,
                    ),
                ),
                (
                    ScopedExposure(
                        "research_program",
                        "FMA_historical_exposure_ledger_v1",
                        None,
                        "program_history_has_no_single_checkpoint_identity",
                        history["historical_exposure"],
                        history_evidence,
                    ),
                ),
            )
        )
    return validate_collection(
        CollectionCatalog(SCHEMA, "fmak5489-source-metadata", bundled, external, tuple(tracks))
    )


def summary(value: CollectionCatalog) -> dict[str, Any]:
    return dict(
        schema="drone.collection-coverage.v1",
        collection_id=value.collection_id,
        tracks=len(value.tracks),
        source_hash_known=sum(row.source_audio.sha256 is not None for row in value.tracks),
        source_hash_distinct=len(
            {row.source_audio.sha256 for row in value.tracks if row.source_audio.sha256}
        ),
        analyzed_support_known=sum(row.analyzed_audio.samples is not None for row in value.tracks),
        analyzed_support_unknown=sum(row.analyzed_audio.samples is None for row in value.tracks),
        bpm_known=sum(row.bpm.value is not None for row in value.tracks),
        missing_native_top=sum(row.native_genre_top is None for row in value.tracks),
        empty_native_all_tags=sum(not row.genre_tags for row in value.tracks),
        missing_artist=sum(
            not any(item.kind == "artist" for item in row.relations) for row in value.tracks
        ),
        missing_album=sum(
            not any(item.kind == "album" for item in row.relations) for row in value.tracks
        ),
        frozen_components=len(
            {group.group_id for row in value.tracks for group in row.frozen_groups}
        ),
        memberships=dict(
            sorted(
                Counter(
                    (membership.plan_id + ":" + membership.role)
                    for row in value.tracks
                    for membership in row.memberships
                ).items()
            )
        ),
        audio_read=False,
        targets_selected=False,
        inference_performed=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("master-metadata", "taxonomy", "cache-metadata"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DATA / "fmak5489-collection.json.gz")
    parser.add_argument("--coverage", type=Path, default=DATA / "fmak5489-collection-coverage.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build(
        ROOT,
        dict(
            fma_master_metadata=args.master_metadata,
            fma_taxonomy=args.taxonomy,
            training_cache_metadata=args.cache_metadata,
        ),
    )
    encoded = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=encoded, mtime=0) as stream:
        stream.write(collection_bytes(result))
    coverage = (json.dumps(summary(result), sort_keys=True, indent=2) + "\n").encode()
    for path, raw in ((args.output, encoded.getvalue()), (args.coverage, coverage)):
        if args.check:
            if path.read_bytes() != raw:
                raise ValueError("Collection artifact differs from metadata sources")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    print(json.dumps(summary(result), sort_keys=True))


if __name__ == "__main__":
    main()
