"""Export GS604 metadata from pinned annotation and saved decode witnesses; no audio.

Pass --external-sources JSON mapping only giantsteps_annotations to an existing
metadata archive. Local paths are resolution hints and never exported identities.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import resource
import sys
import tarfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.collection_catalog import (
    SCHEMA,
    AudioSupport,
    CollectionCatalog,
    CollectionTrack,
    ExternalSource,
    Membership,
    OptionalIdentity,
    ScopedExposure,
    collection_bytes,
    resolve_sources,
    validate_collection,
)
from experiments.corpus_catalog import Evidence, GenreTag, Measurement, SourceRecord

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("experiments/studies/v2-data-contract/data")
DESCRIPTOR = DATA / "giantsteps604-collection-sources.json"
OUTPUT = DATA / "giantsteps604-collection.json.gz"
COVERAGE = DATA / "giantsteps604-collection-coverage.json"
REVISION = "6bcd492c825ac9b8597bc650a5f6fd18b6c43d2b"
ARCHIVE_SHA = "9daf611028ba8bf448c14d819fe78eca86c000992b4a5c4e6f49a3d249c5d9fe"
ARCHIVE_PREFIX = "giantsteps-key-dataset-" + REVISION


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def annotation_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Select native genre and file metadata only; key annotations are not selected."""
    genres, durations = {}, {}
    with tarfile.open(path) as archive:
        for member in archive:
            match = re.fullmatch(
                re.escape(ARCHIVE_PREFIX) + r"/annotations/(genre|jams)/(\d+)\.LOFI\.(genre|jams)",
                member.name,
            )
            if match is None:
                continue
            kind, number, extension = match.groups()
            if kind != extension or not member.isfile() or member.size > 128 * 1024:
                raise ValueError("Unexpected selected metadata member")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Missing selected metadata member")
            raw = stream.read()
            identity = "giantsteps:" + number
            destination = genres if kind == "genre" else durations
            if identity in destination:
                raise ValueError("Repeated publisher identity")
            if kind == "genre":
                value = raw.decode().strip()
                if not value or "\n" in value:
                    raise ValueError("Unexpected native genre text")
            else:
                value = json.loads(raw)["file_metadata"]["duration"]
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError("Invalid publisher preview duration")
            destination[identity] = dict(value=value, member=member.name, sha256=sha(raw))
    if len(genres) != 604 or set(genres) != set(durations):
        raise ValueError("Complete604 publisher metadata required")
    return {
        identity: dict(genre=genres[identity], duration=durations[identity]) for identity in genres
    }


def join_receipts(
    reference: dict[str, Any], plan: dict[str, Any]
) -> list[tuple[int, dict[str, Any]]]:
    """Preserve every source ID, including repeated container/PCM observations."""
    if reference["contract"]["sample_rate"] != 11025:
        raise ValueError("Unexpected fingerprint decode sample rate")
    selected = [
        (i, row) for i, row in enumerate(reference["records"]) if row["domain"] == "giantsteps"
    ]
    expected = {row["id"]: row for row in plan["gs_records"]}
    if len(selected) != 604 or len(expected) != 604 or len(plan["gs_records"]) != 604:
        raise ValueError("Complete604 receipt membership required")
    seen = set()
    for _, row in selected:
        identity = row["id"]
        if identity in seen or identity not in expected:
            raise ValueError("Duplicate or mismatched decoded identity")
        seen.add(identity)
        other = expected[identity]
        if any(
            row[field] != other[field]
            for field in ("ref_id", "artifact", "container_sha256", "pcm_sha256")
        ):
            raise ValueError("Plan and decode witnesses disagree")
        if row["status"] != "complete" or type(row["frames"]) is not int or row["frames"] <= 0:
            raise ValueError("Incomplete decoded support")
        if not math.isclose(row["seconds"], row["frames"] / 11025, abs_tol=1e-9):
            raise ValueError("Saved frames/duration disagree")
    return selected


def build(root: Path, external_paths: dict[str, Path]) -> CollectionCatalog:
    descriptor = json.loads((root / DESCRIPTOR).read_bytes())
    sources = [
        SourceRecord(
            "collection_descriptor", str(DESCRIPTOR), sha((root / DESCRIPTOR).read_bytes())
        )
    ]
    sources += [
        SourceRecord(name, row["path"], row["sha256"])
        for name, row in descriptor["bundled"].items()
    ]
    external = (
        ExternalSource(
            "giantsteps_annotations",
            ARCHIVE_SHA,
            Evidence("collection_descriptor", "/external/giantsteps_annotations"),
        ),
    )
    empty = CollectionCatalog(SCHEMA, "giantsteps604-source-metadata", tuple(sources), external, ())
    paths = resolve_sources(empty, root, external_paths)
    metadata = annotation_metadata(paths["giantsteps_annotations"])
    reference = json.loads(gzip.decompress(paths["fingerprint_receipt"].read_bytes()))
    plan = json.loads(gzip.decompress(paths["overlap_plan"].read_bytes()))
    if (
        reference["source_sha256"]["experiments/fma_fingerprints.py"]
        != descriptor["bundled"]["fingerprint_decode_source"]["sha256"]
    ):
        raise ValueError("Fingerprint decode source differs from historical receipt")
    selected = join_receipts(reference, plan)
    if {row["id"] for _, row in selected} != set(metadata):
        raise ValueError("Publisher and recording membership disagree")
    release = json.loads(paths["release_metadata"].read_bytes())
    tracks = []
    for index, row in selected:
        identity = row["id"]
        native = metadata[identity]
        origin = (Evidence("fingerprint_receipt", f"/records/{index}"),)
        genre_ev = (Evidence("giantsteps_annotations", "/" + native["genre"]["member"]),)
        duration_ev = (
            Evidence(
                "giantsteps_annotations",
                "/" + native["duration"]["member"] + "/file_metadata/duration",
            ),
        )
        tracks.append(
            CollectionTrack(
                identity,
                "giantsteps",
                identity.split(":", 1)[1],
                OptionalIdentity(
                    row["container_sha256"],
                    "saved_fingerprint_source_container_sha256_not_reverified",
                    None,
                    origin,
                ),
                native["genre"]["value"],
                (
                    GenreTag(
                        native["genre"]["value"],
                        None,
                        "GiantSteps-native-site-genre-" + REVISION,
                        "direct",
                        genre_ev,
                    ),
                ),
                "publisher_native_genre_file_no_legacy_bucket_mapping",
                Measurement(
                    None,
                    "seconds",
                    "not_probed_here",
                    "source_container_duration_not_independently_probed",
                    origin,
                ),
                Measurement(
                    None,
                    "seconds",
                    "not_established_from_preview_metadata",
                    "publisher_JAMS_describes_preview_not_full_catalog_recording",
                    duration_ev,
                ),
                Measurement(None, "bpm", "not_admitted", "no_admitted_BPM_source", genre_ev),
                AudioSupport(
                    "saved_11025Hz_fingerprint_decode",
                    row["frames"],
                    11025,
                    1,
                    "<f4",
                    OptionalIdentity(
                        row["pcm_sha256"],
                        "saved_float32_decode_sha256_not_reverified",
                        None,
                        origin,
                    ),
                    "saved_ffmpeg_mono_11025Hz_f32le_fingerprint_decode",
                    "whole_available_source_with_one_hour_rejection_guard_not_full_song",
                    "historical_fingerprint_receipt_no_new_audio_read",
                    None,
                    origin,
                ),
                (),
                ("artist_ID_not_admitted", "release_ID_not_admitted", "work_ID_not_established"),
                (Membership("GiantSteps-Key-" + REVISION, "source_inventory", genre_ev),),
                (),
                (
                    ScopedExposure(
                        "model",
                        "drone-v1",
                        release["checkpoint_sha256"],
                        None,
                        "published_full604_evaluation_development_exposed_direct_fit_not_established",
                        (
                            Evidence("release_model_card", "/GiantSteps-604-evaluation"),
                            Evidence("release_metadata", "/checkpoint_sha256"),
                        ),
                    ),
                ),
            )
        )
    return validate_collection(
        CollectionCatalog(
            SCHEMA, "giantsteps604-source-metadata", tuple(sources), external, tuple(tracks)
        )
    )


def coverage(catalog: CollectionCatalog) -> dict[str, Any]:
    durations = [
        value for row in catalog.tracks if (value := row.analyzed_audio.seconds) is not None
    ]
    return dict(
        schema="drone.collection-coverage.v1",
        collection_id=catalog.collection_id,
        tracks=len(catalog.tracks),
        distinct_source_sha256=len({r.source_audio.sha256 for r in catalog.tracks}),
        distinct_decoded_pcm_sha256=len(
            {r.analyzed_audio.pcm_identity.sha256 for r in catalog.tracks}
        ),
        native_genres=dict(sorted(Counter(r.native_genre_top for r in catalog.tracks).items())),
        publisher_full_recording_duration_known=sum(
            r.catalog_recording_duration.value is not None for r in catalog.tracks
        ),
        fingerprint_decode_support_known=sum(
            r.analyzed_audio.samples is not None for r in catalog.tracks
        ),
        fingerprint_decode_seconds_min=min(durations),
        fingerprint_decode_seconds_max=max(durations),
        bpm_known=0,
        artist_ID_known=0,
        work_group_known=0,
        full604_release_evaluation_exposed=len(catalog.tracks),
        audio_read=False,
        targets_selected=False,
        inference_performed=False,
        interpretation="Metadata coverage only; saved11025Hz fingerprint decode and publisher preview duration are distinct witnesses, neither proves full-song support. Duplicate PCM identities retained; no new independence or accuracy claim.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-sources", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    stop = threading.Event()

    def guard() -> None:
        while not stop.wait(0.02):
            native = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss = native if sys.platform == "darwin" else native * 1024
            if time.monotonic() - started > 120 or rss > 256 * 1024**2:
                os._exit(124)

    threading.Thread(target=guard, daemon=True).start()
    mapping = json.loads(args.external_sources.read_bytes())
    if not isinstance(mapping, dict) or any(
        not isinstance(value, str) for value in mapping.values()
    ):
        raise ValueError("External sources must map IDs to local path strings")
    catalog = build(ROOT, {name: Path(value) for name, value in mapping.items()})
    stream = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as archive:
        archive.write(collection_bytes(catalog))
    compressed = stream.getvalue()
    report = coverage(catalog) | dict(
        catalog_sha256=sha(compressed), generator_sha256=sha(Path(__file__).read_bytes())
    )
    report_raw = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    if len(compressed) + len(report_raw) > 16 * 1024**2:
        raise ValueError("Output budget exceeded")
    for relative, raw in ((OUTPUT, compressed), (COVERAGE, report_raw)):
        path = ROOT / relative
        if args.write:
            path.write_bytes(raw)
        elif path.read_bytes() != raw:
            raise ValueError("Collection reproduction differs")
    use = resource.getrusage(resource.RUSAGE_SELF)
    stop.set()
    print(
        json.dumps(
            dict(
                tracks=len(catalog.tracks),
                output_bytes=len(compressed) + len(report_raw),
                wall_seconds=time.monotonic() - started,
                cpu_seconds=use.ru_utime + use.ru_stime,
                peak_rss_bytes=use.ru_maxrss if sys.platform == "darwin" else use.ru_maxrss * 1024,
            )
        )
    )


if __name__ == "__main__":
    main()
