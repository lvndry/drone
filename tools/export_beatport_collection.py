"""Export label-free Beatport collection metadata from pinned existing witnesses.

Use --external-sources paths.json to relocate existing metadata. Its JSON object
must map exactly workbook, store_json_zip, and canonical_counts to local paths.
Paths are resolution hints only: original SHA256/byte pins remain mandatory and
catalog external descriptors never contain the replacement paths. Omitting the
option retains the archived local path hints for convenience.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from experiments.collection_catalog import CollectionCatalog
    from experiments.corpus_catalog import Evidence

from tools.dj_coverage_bp import NS, parse_tags, resolve_shared, resolved, selected_cell, sha

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "experiments/studies/v2-data-contract/data/coverage-inputs.json"
EXTERNAL_IDS = frozenset({"workbook", "store_json_zip", "canonical_counts"})


def pinned_inputs(
    root: Path = ROOT, external_paths: dict[str, Path] | None = None
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    if external_paths is not None and set(external_paths) != EXTERNAL_IDS:
        raise ValueError("External resolver must cover exactly: " + ", ".join(sorted(EXTERNAL_IDS)))
    records = json.loads((root / INPUTS.relative_to(ROOT)).read_bytes())["bp"]
    raws = {}
    sources = {}
    for name, record in records.items():
        original = Path(record["path"])
        marker = "/experiments/"
        path = (
            root / "experiments" / str(original).split(marker, 1)[1]
            if marker in str(original)
            else original
        )
        resolved_path = (
            external_paths[name] if external_paths is not None and name in EXTERNAL_IDS else path
        )
        raw = resolved_path.read_bytes()
        if sha(raw) != record["sha256"] or len(raw) != record["bytes"]:
            raise ValueError("Pinned BP source mismatch: " + name)
        raws[name] = raw
        sources[name] = {
            "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            "sha256": sha(raw),
            "bytes": len(raw),
        }
    return raws, sources


def plan_and_groups(raws: dict[str, bytes]) -> tuple[dict[str, str], dict[str, str]]:
    # Read only membership arrays; no key/target fields enter the adapter.
    plan = json.loads(gzip.decompress(raws["plan"]))
    roles = {}
    for field, role, expected in (
        ("bp_fit", "fit", 1229),
        ("bp_selection_guard", "inner_selection", 126),
    ):
        ids = plan[field]
        if len(ids) != expected or len(set(ids)) != expected:
            raise ValueError("Unexpected BP plan membership")
        for identity in ids:
            if not re.fullmatch(r"beatport:[1-9]\d*", identity) or identity in roles:
                raise ValueError("Invalid or duplicate BP identity")
            roles[identity] = role
    groups = {
        row["id"]: row["derived_uncertainty_group"]
        for row in csv.DictReader(io.StringIO(gzip.decompress(raws["groups"]).decode()))
        if row["domain"] == "beatport"
    }
    if set(groups) != set(roles):
        raise ValueError("Group membership differs from plan")
    return roles, groups


def workbook_genres(raw: bytes, wanted: set[str]) -> dict[str, dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(raw)) as book:
        sheet = ET.fromstring(book.read("xl/worksheets/sheet1.xml"))
        shared = book.read("xl/sharedStrings.xml")
        ids = [
            (
                row,
                selected_cell(
                    next((cell for cell in row if re.fullmatch(r"A\d+", cell.get("r", ""))), None),
                    "A",
                ),
            )
            for row in sheet.findall("m:sheetData/m:row", NS)
        ]
        strings = resolve_shared(
            shared, {int(cell[1]) for _, cell in ids if cell is not None and cell[0] == "s"}
        )
        chosen = []
        for row, cell in ids:
            identity = resolved(cell, strings)
            if not identity.isdigit() or "beatport:" + str(int(identity)) not in wanted:
                continue
            genre = selected_cell(
                next((cell for cell in row if re.fullmatch(r"F\d+", cell.get("r", ""))), None), "F"
            )
            chosen.append(("beatport:" + str(int(identity)), row.get("r"), genre))
        strings = resolve_shared(
            shared, {int(cell[1]) for _, _, cell in chosen if cell is not None and cell[0] == "s"}
        )
        result = {
            identity: {"raw": resolved(cell, strings), "row": row} for identity, row, cell in chosen
        }
        if len(result) != len(chosen) or set(result) != wanted:
            raise ValueError("Workbook membership incomplete or duplicated")
        return result


def observed_entities(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Expected native entity list")
    result = []
    for entity in value:
        if not isinstance(entity, dict):
            raise ValueError("Expected native entity object")
        result.append({field: entity.get(field) for field in ("id", "name", "slug")})
    return result


def store_metadata(raw: bytes, wanted: set[str]) -> dict[str, dict[str, Any]]:
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        for member in bundle.namelist():
            match = re.fullmatch(r"meta/(\d+) .+\.json", member)
            if match is None or "beatport:" + match[1] not in wanted:
                continue
            member_raw = bundle.read(member)
            text = member_raw.decode()
            token = re.compile(r'"(?:[^"\\]|\\.)*"|(?<![\w])nan(?![\w])')
            normalized = token.sub(lambda m: "null" if m[0] == "nan" else m[0], text)
            data = json.loads(normalized)
            identity = "beatport:" + str(data.get("id"))
            if identity != "beatport:" + match[1] or identity in result:
                raise ValueError("Store member identity mismatch or duplicate")
            duration = data.get("duration") or {}
            milliseconds = duration.get("milliseconds")
            if milliseconds is not None and (type(milliseconds) is not int or milliseconds < 0):
                raise ValueError("Invalid catalog duration")
            release = data.get("release")
            result[identity] = {
                "genres": parse_tags(data.get("genres")),
                "subgenres": parse_tags(data.get("sub_genres")),
                "artists": observed_entities(data.get("artists")),
                "remixers": observed_entities(data.get("remixers")),
                "release": observed_entities([release] if release is not None else []),
                "catalog_duration_seconds": None if milliseconds is None else milliseconds / 1000,
                "member": member,
                "sha256": sha(member_raw),
                "bare_nan_normalized": normalized != text,
            }
    return result


TEMPO = "experiments/studies/v2-data-contract/data/beatport-tempo-metadata.json.gz"
TEMPO_SHA = "e466eab331bdee28445f209d850e1cc30e20c8581a90d48bae5a6a2a0c0d8f69"
RELEASE = "experiments/studies/bp-gs-overlap/data/release-metadata.json"
RELEASE_SHA = "69432e5f6ddc0ad10a55de591e3174e6db9ef705355cfdca5a1f678a9bfa4c4f"
OUTPUT = "experiments/studies/v2-data-contract/data/beatport1355-collection.json.gz"


def export_collection(
    root: Path = ROOT, external_paths: dict[str, Path] | None = None
) -> CollectionCatalog:
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
        validate_collection,
    )
    from experiments.corpus_catalog import Evidence, GenreTag, Measurement, SourceRecord

    raws, records = pinned_inputs(root, external_paths)
    roles, groups = plan_and_groups(raws)
    wanted = set(roles)
    genres = workbook_genres(raws["workbook"], wanted)
    api = store_metadata(raws["store_json_zip"], wanted)
    tempo_raw = (root / TEMPO).read_bytes()
    release_raw = (root / RELEASE).read_bytes()
    if sha(tempo_raw) != TEMPO_SHA or sha(release_raw) != RELEASE_SHA:
        raise ValueError("Admitted tempo or release witness changed")
    tempo_doc = json.loads(gzip.decompress(tempo_raw))
    tempo = {row["track_id"]: (index, row) for index, row in enumerate(tempo_doc["tracks"])}
    if set(tempo) != wanted or len(tempo_doc["tracks"]) != len(wanted):
        raise ValueError("Tempo membership mismatch")
    if tempo_doc["membership_plan_sha256"] != records["plan"]["sha256"]:
        raise ValueError("Tempo plan mismatch")
    release_doc = json.loads(release_raw)
    release = release_doc["metadata"]
    fit = set(release["train_ids"])
    selection = set(release["validation_ids"])
    audio_exposure = set(release["training_audio_sha256"])
    sources = [
        SourceRecord(
            "coverage_inputs",
            str(INPUTS.relative_to(ROOT)),
            sha((root / INPUTS.relative_to(ROOT)).read_bytes()),
        ),
        SourceRecord("tempo", TEMPO, TEMPO_SHA),
        SourceRecord("release", RELEASE, RELEASE_SHA),
    ]
    external = []
    for name, record in records.items():
        if Path(record["path"]).is_absolute():
            external.append(
                ExternalSource(name, record["sha256"], Evidence("coverage_inputs", "/bp/" + name))
            )
        else:
            sources.append(SourceRecord(name, record["path"], record["sha256"]))

    def ev(source: str, pointer: str) -> tuple[Evidence, ...]:
        return (Evidence(source, pointer),)

    cache = {}
    for index, row in enumerate(json.loads(raws["canonical_counts"])["rows"]):
        if row["id"] not in wanted:
            continue
        if row["id"] in cache:
            raise ValueError("Duplicate cache row")
        cache[row["id"]] = (
            index,
            {key: row[key] for key in ("samples", "bytes", "sha256", "waveform_sha256")},
        )
    if set(cache) != wanted:
        raise ValueError("Cache membership mismatch")
    tracks = []
    for identity in sorted(wanted):
        numeric = identity.split(":", 1)[1]
        index, row = cache[identity]
        cache_ev = ev("canonical_counts", f"/rows/{index}")
        samples = row["samples"]
        if type(samples) is not int or samples <= 0 or row["bytes"] != samples * 4:
            raise ValueError("Invalid recorded float32 cache support")
        native = api.get(identity)
        native_ev = ev("store_json_zip", "/" + native["member"] if native else "/")
        workbook_ev = ev("workbook", "/xl/worksheets/sheet1.xml/F" + genres[identity]["row"])
        tags = [
            GenreTag(
                genres[identity]["raw"], None, "beatport_workbook_native", "direct", workbook_ev
            )
        ]
        relations = []
        missing = []
        for field, taxonomy in (
            ("genres", "beatport_api_genres"),
            ("subgenres", "beatport_api_subgenres"),
        ):
            for tag in native[field] if native else []:
                tags.append(GenreTag(tag["name"], tag["id"], taxonomy, "direct", native_ev))
        for field, kind in (("artists", "artist"), ("remixers", "remixer"), ("release", "release")):
            entities = native[field] if native else []
            for entity in entities:
                if entity["id"] is None:
                    missing.append(kind + "_native_id_missing")
                    continue
                relations.append(
                    Relation(kind, f"beatport:{kind}:{entity['id']}", entity["name"], native_ev)
                )
            if not entities:
                missing.append(kind + "_not_recorded_in_archived_store_metadata")
        tempo_index, tempo_row = tempo[identity]
        if tempo_row["role"] != roles[identity]:
            raise ValueError("Tempo role mismatch")
        if native and tempo_row["source_member_sha256"] != native["sha256"]:
            raise ValueError("Tempo/API source member mismatch")
        bpm = Measurement(
            tempo_row["bpm"],
            "bpm",
            tempo_doc["method"],
            None if tempo_row["bpm"] is not None else tempo_row["bpm_status"],
            ev("tempo", f"/tracks/{tempo_index}"),
        )
        declared_role = (
            "declared_fit_id"
            if numeric in fit
            else (
                "declared_selection_id"
                if numeric in selection
                else "id_not_in_release_lists_overlap_unknown"
            )
        )
        exposures = (
            ScopedExposure(
                "model",
                "drone-v1",
                release_doc["checkpoint_sha256"],
                None,
                declared_role,
                ev("release", "/metadata"),
            ),
            ScopedExposure(
                "model",
                "drone-v1",
                release_doc["checkpoint_sha256"],
                None,
                "source_hash_in_release_audio_list_role_not_inferred"
                if row["sha256"] in audio_exposure
                else "source_hash_not_in_release_audio_list_overlap_unknown",
                ev("release", "/metadata/training_audio_sha256") + cache_ev,
            ),
        )
        duration = native["catalog_duration_seconds"] if native else None
        tracks.append(
            CollectionTrack(
                identity,
                "beatport",
                numeric,
                OptionalIdentity(row["sha256"], "recorded_source_container_sha256", None, cache_ev),
                genres[identity]["raw"],
                tuple(tags),
                "native_workbook_and_archived_api_taxonomies_kept_separate_no_normalization",
                Measurement(
                    None,
                    "seconds",
                    "not_measured_from_source_container",
                    "cache_samples_are_not_independent_source_duration_measurement",
                    cache_ev,
                ),
                Measurement(
                    duration,
                    "seconds",
                    "archived_publisher_catalog_duration",
                    None if duration is not None else "no_archived_catalog_duration",
                    native_ev,
                ),
                bpm,
                AudioSupport(
                    "recorded24khztrainingcache",
                    samples,
                    24000,
                    1,
                    "<f4",
                    OptionalIdentity(
                        row["waveform_sha256"], "recorded_float32_pcm_sha256", None, cache_ev
                    ),
                    "ffmpeg_mono_24000_pcm_f32le_full_source",
                    "recorded_full_available_source_decode_not_verified_full_song",
                    "existing_manifest_no_audio_reinspection",
                    None,
                    cache_ev,
                ),
                tuple(relations),
                tuple(missing),
                (
                    Membership(
                        "canonical-eligibility-v2:" + records["plan"]["sha256"],
                        roles[identity],
                        ev(
                            "plan", "/bp_fit" if roles[identity] == "fit" else "/bp_selection_guard"
                        ),
                    ),
                ),
                (
                    FrozenGroup(
                        "frozen_uncertainty_proxy_not_verified_artist_component",
                        groups[identity],
                        ev("groups", "/rows/" + identity),
                    ),
                ),
                exposures,
            )
        )
    return validate_collection(
        CollectionCatalog(
            SCHEMA, "beatport1355-current-plan", tuple(sources), tuple(external), tuple(tracks)
        )
    )


def coverage(catalog: CollectionCatalog) -> dict[str, Any]:
    from collections import Counter

    from tools.dj_coverage_bp import band

    return {
        "collection_id": catalog.collection_id,
        "tracks": len(catalog.tracks),
        "roles": dict(Counter(m.role for row in catalog.tracks for m in row.memberships)),
        "bpm_known": sum(row.bpm.value is not None for row in catalog.tracks),
        "bpm_unknown": sum(row.bpm.value is None for row in catalog.tracks),
        "catalog_duration_known": sum(
            row.catalog_recording_duration.value is not None for row in catalog.tracks
        ),
        "source_duration_known": sum(
            row.source_duration.value is not None for row in catalog.tracks
        ),
        "recorded_training_cache_bands": dict(
            sorted(
                Counter(
                    band(seconds)
                    if (seconds := row.analyzed_audio.seconds) is not None
                    else "unknown"
                    for row in catalog.tracks
                ).items()
            )
        ),
        "workbook_native_genres": dict(
            sorted(Counter(row.native_genre_top for row in catalog.tracks).items())
        ),
        "relation_tracks": {
            kind: sum(any(item.kind == kind for item in row.relations) for row in catalog.tracks)
            for kind in ("artist", "remixer", "release")
        },
        "frozen_groups": len(
            {group.group_id for row in catalog.tracks for group in row.frozen_groups}
        ),
        "exposure_relations": dict(
            Counter(e.relation for row in catalog.tracks for e in row.exposures)
        ),
        "audio_read": False,
        "inference_performed": False,
        "interpretation": "Coverage only. Publisher tempo and catalog duration are unverified metadata; frozen groups remain proxies. Recorded 24kHz cache is not release input. No accuracy measurements.",
    }


def main() -> None:
    import argparse
    import os
    import resource
    import threading
    import time

    from experiments.collection_catalog import collection_bytes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--external-sources",
        type=Path,
        help="JSON mapping exactly workbook, store_json_zip, canonical_counts to local files",
    )
    args = parser.parse_args()
    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    stop = threading.Event()

    def guard() -> None:
        while not stop.wait(0.02):
            use = resource.getrusage(resource.RUSAGE_SELF)
            rss = use.ru_maxrss if __import__("sys").platform == "darwin" else use.ru_maxrss * 1024
            if time.monotonic() - started > 120 or rss > 256 * 1024**2:
                os._exit(124)

    threading.Thread(target=guard, daemon=True).start()
    external_paths = None
    if args.external_sources is not None:
        mapping = json.loads(args.external_sources.read_bytes())
        if not isinstance(mapping, dict) or any(
            not isinstance(path, str) for path in mapping.values()
        ):
            raise ValueError("External sources must be a JSON object of local path strings")
        external_paths = {name: Path(path) for name, path in mapping.items()}
    catalog = export_collection(external_paths=external_paths)
    raw = collection_bytes(catalog)
    compressed = gzip.compress(raw, mtime=0)
    summary = coverage(catalog)
    summary["catalog_sha256"] = sha(compressed)
    summary["generator_sha256"] = sha(Path(__file__).read_bytes())
    summary_raw = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode()
    if len(compressed) + len(summary_raw) > 16 * 1024**2:
        raise ValueError("Output budget exceeded")
    output = ROOT / OUTPUT
    summary_path = output.with_name("beatport1355-collection-coverage.json")
    if args.write:
        output.write_bytes(compressed)
        summary_path.write_bytes(summary_raw)
    elif output.read_bytes() != compressed or summary_path.read_bytes() != summary_raw:
        raise ValueError("Collection reproduction differs")
    use = resource.getrusage(resource.RUSAGE_SELF)
    stop.set()
    print(
        json.dumps(
            {
                "tracks": len(catalog.tracks),
                "sha256": sha(compressed),
                "wall_seconds": time.monotonic() - started,
                "cpu_seconds": use.ru_utime + use.ru_stime,
                "peak_rss_native_units": use.ru_maxrss,
                "output_bytes": len(compressed) + len(summary_raw),
            }
        )
    )


if __name__ == "__main__":
    main()
