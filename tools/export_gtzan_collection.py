"""Join GTZAN's frozen memberships and recorded integrity observations, without audio.

Identity matching, WAV-header parsing, reference eligibility and scoring admission
are separate observations. The historical reference ledger is never rewritten.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
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
    Membership,
    OptionalIdentity,
    ScopedExposure,
    collection_bytes,
    validate_collection,
)
from experiments.corpus_catalog import Evidence, GenreTag, Measurement, SourceRecord

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("experiments/studies/v2-data-contract/data")
OUTPUT = DATA / "gtzan1000-collection.json.gz"
COVERAGE = DATA / "gtzan1000-collection-coverage.json"
PINS = {
    "reference_ledger": (
        DATA / "gtzan-reference/reference-ledger.json.gz",
        "35249d256681670b368ca2463673c36f87d57b19b641cf1edca558b7da354b48",
    ),
    "recording_index": (
        DATA / "gtzan-reference/sources/recording-index.json.gz",
        "f770e0b8e9200ab298c17f15c607470ece98467d5b34780507a535f589d2b8d6",
    ),
    "integrity_bundle": (
        DATA / "gtzan-audio-integrity/evidence.tar.gz",
        "0c3b991306d84d3e9c9b90be55c9b22e83d6c41a4750245b6718619f2fa0ffb0",
    ),
    "release_metadata": (
        Path("experiments/studies/bp-gs-overlap/data/release-metadata.json"),
        "69432e5f6ddc0ad10a55de591e3174e6db9ef705355cfdca5a1f678a9bfa4c4f",
    ),
}
RUNS = (
    "gtzan-audio-integrity-20261113-v1",
    "gtzan-audio-integrity-audit-20261113-v2",
    "gtzan-audio-integrity-inventory-20261113-v3",
)
INVENTORY_MEMBER = RUNS[2] + "/audio-integrity.json"
INVENTORY_SHA = "26e315689eb3ae57479a6d72337c9560eb73e1ff9cb0e1ee9bfb5b6ca413ccc5"
ARCHIVE_SHA = "b28cc067ff6199bd826f5d1a6931458586d64acae7f44b2e882b7a97af057531"
EXCLUSIONS = {
    "unknown_or_modulating_key",
    "classical_unannotated",
    "not_in_sturm_filtered_union",
    "published_loader_corruption_flag",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def load_inputs(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str]:
    values: dict[str, Any] = {}
    for name, (relative, expected) in PINS.items():
        raw = (root / relative).read_bytes()
        if sha(raw) != expected:
            raise ValueError("Pinned GTZAN metadata changed: " + name)
        if name != "integrity_bundle":
            values[name] = json.loads(gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw)
    with tarfile.open(root / PINS["integrity_bundle"][0]) as archive:

        def read(name: str) -> dict[str, Any]:
            member = archive.extractfile(name)
            if member is None:
                raise ValueError("Missing integrity evidence member")
            raw = member.read()
            if name == INVENTORY_MEMBER and sha(raw) != INVENTORY_SHA:
                raise ValueError("Diagnostic inventory member changed")
            return json.loads(raw)

        states = [read(run + "/controller.json")["state"] for run in RUNS]
        if states != ["FAILED", "FAILED", "COMPLETE_DIAGNOSTIC_INVENTORY_NOT_STRICT_PASS"]:
            raise ValueError("Original failed audits must remain failed")
        inventory = read(INVENTORY_MEMBER)
        download = read(RUNS[0] + "/download.json")
        if (download["sha256"], download["bytes"]) != (ARCHIVE_SHA, 1226192050):
            raise ValueError("Acquired archive identity differs")
    if (
        inventory["mode"] != "diagnostic_inventory"
        or inventory["admitted_for_scoring"] is not False
        or inventory["archive_sha256"] != ARCHIVE_SHA
    ):
        raise ValueError("Integrity observation cannot grant scoring admission")
    return (
        values["reference_ledger"],
        values["recording_index"],
        inventory,
        values["release_metadata"]["checkpoint_sha256"],
    )


def join_records(
    ledger: list[dict[str, Any]], index: dict[str, Any], inventory: dict[str, Any]
) -> list[tuple[int, dict[str, Any], int, dict[str, Any]]]:
    """Validate exact membership and keep identity/format outcomes independent."""
    expected = index["tracks"]
    if index["version"] != "1.0" or len(expected) != 1000 or len(ledger) != 1000:
        raise ValueError("Complete GTZAN1000 membership required")
    records = inventory["records"]
    by_member = {row["member"]: (i, row) for i, row in enumerate(records)}
    if len(records) != 1000 or len(by_member) != 1000:
        raise ValueError("Duplicate or incomplete integrity inventory")
    seen = set()
    joined = []
    for position, row in enumerate(ledger):
        identity, member = row["id"], row["audio_member"]
        if identity in seen or identity not in expected:
            raise ValueError("Duplicate or unknown reference identity")
        seen.add(identity)
        if (
            row["genre"] != identity.split(".")[0]
            or member != f"genres/{row['genre']}/{identity}.wav"
        ):
            raise ValueError("Native ID/genre/member mismatch")
        expected_path, expected_md5 = expected[identity]["audio"]
        if expected_path != "gtzan_genre/" + member or row["expected_member_md5"] != expected_md5:
            raise ValueError("Reference/index member identity mismatch")
        record_index, observed = by_member[member]
        if (
            observed["expected_md5"] != expected_md5
            or type(observed["member_identity_verified"]) is not bool
            or observed["member_identity_verified"] != (observed["md5"] == expected_md5)
        ):
            raise ValueError("Recorded identity observation disagrees with expected source")
        if (
            observed["overlap_cleared"] is not False
            or observed["admitted_for_scoring"] is not False
        ):
            raise ValueError("Integrity records cannot imply overlap clearance or admission")
        for field in (
            "raw_single_key_reference",
            "named_filtered_intersection",
            "metadata_eligible",
        ):
            if type(row[field]) is not bool:
                raise ValueError("Eligibility observations must be booleans")
        if row["named_filtered_intersection"] and not row["raw_single_key_reference"]:
            raise ValueError("Filtered references must remain within raw references")
        if (
            row["metadata_eligible"] != row["named_filtered_intersection"]
            or not set(row["exclusions"]) <= EXCLUSIONS
        ):
            raise ValueError("Frozen eligibility rule differs")
        if (
            row["audio_verified"] is not False
            or row["overlap_cleared"] is not False
            or row["admitted"] is not False
        ):
            raise ValueError("Historical reference ledger must remain metadata-only")
        wave = observed.get("wave")
        if wave is not None:
            if (
                "wave_error" in observed
                or wave["sample_rate"] != 22050
                or wave["channels"] != 1
                or wave["sample_width"] != 2
                or type(wave["frames"]) is not int
                or wave["frames"] <= 0
                or not math.isclose(
                    wave["duration_seconds"], wave["frames"] / 22050, rel_tol=0, abs_tol=1e-9
                )
            ):
                raise ValueError("Invalid or contradictory saved WAV-header support")
        elif not isinstance(observed.get("wave_error"), str):
            raise ValueError("Missing WAV support needs its recorded failure")
        joined.append((position, row, record_index, observed))
    if seen != set(expected) or {row[1]["audio_member"] for row in joined} != set(by_member):
        raise ValueError("Reference and integrity memberships disagree")
    if (
        sum(row[1]["raw_single_key_reference"] for row in joined),
        sum(row[1]["named_filtered_intersection"] for row in joined),
    ) != (837, 769):
        raise ValueError("Frozen reference cohorts changed")
    return joined


def build(root: Path = ROOT) -> CollectionCatalog:
    ledger, index, inventory, checkpoint = load_inputs(root)
    sources = tuple(SourceRecord(name, str(path), digest) for name, (path, digest) in PINS.items())
    tracks = []
    for position, row, record_index, observed in join_records(ledger, index, inventory):
        ref_ev = (Evidence("reference_ledger", f"/{position}"),)
        obs_ev = (Evidence("integrity_bundle", f"/{INVENTORY_MEMBER}/records/{record_index}"),)
        unknown_pcm = OptionalIdentity(
            None, "not_decoded_for_this_catalog", "no_admitted_model_input_PCM_witness", obs_ev
        )
        memberships = [
            Membership("gtzan-source-inventory-v1", "source_inventory", ref_ev),
            Membership(
                "gtzan-raw-single-key-v1",
                "included" if row["raw_single_key_reference"] else "excluded",
                ref_ev,
            ),
            Membership(
                "gtzan-named-filtered-reference-v1",
                "included" if row["named_filtered_intersection"] else "excluded",
                ref_ev,
            ),
            Membership(
                "gtzan-recorded-member-identity-v3",
                "expected_MD5_match"
                if observed["member_identity_verified"]
                else "expected_MD5_mismatch",
                obs_ev,
            ),
            Membership(
                "gtzan-recorded-WAV-header-v3",
                "parseable_PCM_header" if "wave" in observed else "format_failure",
                obs_ev,
            ),
            Membership("gtzan-recorded-scoring-admission-v3", "not_admitted_for_scoring", obs_ev),
        ]
        memberships.extend(
            Membership("gtzan-frozen-exclusion-reasons-v1", reason, ref_ev)
            for reason in row["exclusions"]
        )
        wave = observed.get("wave")
        tracks.append(
            CollectionTrack(
                track_id="gtzan:" + row["id"],
                corpus_id="gtzan",
                source_track_id=row["id"],
                source_audio=OptionalIdentity(
                    observed["sha256"],
                    "saved_acquired_member_SHA256_not_reverified_identity_status_is_separate",
                    None,
                    obs_ev,
                ),
                native_genre_top=row["genre"],
                genre_tags=(
                    GenreTag(
                        row["genre"],
                        None,
                        "GTZAN-native-directory-genre-v1",
                        "corpus_context",
                        ref_ev,
                    ),
                ),
                genre_method="frozen_source_directory_genre_no_modern_taxonomy_mapping",
                source_duration=Measurement(
                    wave["frames"] / wave["sample_rate"] if wave else None,
                    "seconds",
                    "saved_WAV_header_frame_count_not_newly_probed_or_decoded",
                    None if wave else "recorded_WAV_header_parse_failed",
                    obs_ev,
                ),
                catalog_recording_duration=Measurement(
                    None,
                    "seconds",
                    "not_established",
                    "source_excerpt_does_not_establish_full_recording_duration",
                    obs_ev,
                ),
                bpm=Measurement(None, "bpm", "not_admitted", "no_admitted_BPM_source", ref_ev),
                analyzed_audio=AudioSupport(
                    "no_admitted_model_analysis",
                    None,
                    None,
                    None,
                    None,
                    unknown_pcm,
                    "not_established_for_classifier",
                    "source_excerpt_header_is_not_model_analysis_support",
                    "no_classifier_execution_in_bound_sources",
                    "no_admitted_model_analysis_witness",
                    obs_ev,
                ),
                relations=(),
                relation_missing_reasons=(
                    "verified_artist_ID_not_admitted",
                    "verified_album_ID_not_admitted",
                    "work_or_version_group_not_established",
                ),
                memberships=tuple(memberships),
                frozen_groups=(),
                exposures=(
                    ScopedExposure(
                        "model",
                        "drone-v1",
                        checkpoint,
                        None,
                        "overlap_not_established_by_bundled_integrity_evidence",
                        obs_ev + (Evidence("release_metadata", "/checkpoint_sha256"),),
                    ),
                ),
            )
        )
    return validate_collection(
        CollectionCatalog(SCHEMA, "gtzan1000-source-integrity-metadata", sources, (), tuple(tracks))
    )


def coverage(catalog: CollectionCatalog) -> dict[str, Any]:
    def contains(row: CollectionTrack, plan: str, role: str) -> bool:
        return any(m.plan_id == plan and m.role == role for m in row.memberships)

    genres = sorted({r.native_genre_top for r in catalog.tracks if r.native_genre_top})

    return dict(
        schema="drone.collection-coverage.v1",
        collection_id=catalog.collection_id,
        tracks=len(catalog.tracks),
        native_genres=dict(sorted(Counter(r.native_genre_top for r in catalog.tracks).items())),
        cohorts={
            name: dict(
                tracks=sum(contains(r, plan, "included") for r in catalog.tracks),
                native_genres={
                    genre: sum(
                        r.native_genre_top == genre and contains(r, plan, "included")
                        for r in catalog.tracks
                    )
                    for genre in genres
                },
            )
            for name, plan in (
                ("raw_single_key", "gtzan-raw-single-key-v1"),
                ("named_filtered", "gtzan-named-filtered-reference-v1"),
            )
        },
        recorded_identity_matches=sum(
            contains(r, "gtzan-recorded-member-identity-v3", "expected_MD5_match")
            for r in catalog.tracks
        ),
        recorded_identity_failures=[
            r.track_id
            for r in catalog.tracks
            if contains(r, "gtzan-recorded-member-identity-v3", "expected_MD5_mismatch")
        ],
        recorded_parseable_WAV_headers=sum(
            contains(r, "gtzan-recorded-WAV-header-v3", "parseable_PCM_header")
            for r in catalog.tracks
        ),
        recorded_format_failures=[
            r.track_id
            for r in catalog.tracks
            if contains(r, "gtzan-recorded-WAV-header-v3", "format_failure")
        ],
        recorded_source_excerpt_duration_known=sum(
            r.source_duration.value is not None for r in catalog.tracks
        ),
        publisher_full_recording_duration_known=sum(
            r.catalog_recording_duration.value is not None for r in catalog.tracks
        ),
        analyzed_audio_support_known=sum(
            r.analyzed_audio.samples is not None for r in catalog.tracks
        ),
        bpm_known=sum(r.bpm.value is not None for r in catalog.tracks),
        not_admitted_for_scoring=sum(
            contains(r, "gtzan-recorded-scoring-admission-v3", "not_admitted_for_scoring")
            for r in catalog.tracks
        ),
        overlap_not_established=sum(
            any(
                e.relation == "overlap_not_established_by_bundled_integrity_evidence"
                for e in r.exposures
            )
            for r in catalog.tracks
        ),
        audio_read=False,
        targets_selected=False,
        inference_performed=False,
        interpretation="Historical reference eligibility, acquired-member identity and WAV-header parsing are separate observations; no classifier admission, overlap clearance, decoded-audio usability or full-song coverage is inferred.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    stop = threading.Event()

    def guard() -> None:
        while not stop.wait(0.025):
            use = resource.getrusage(resource.RUSAGE_SELF)
            peak = use.ru_maxrss if sys.platform == "darwin" else use.ru_maxrss * 1024
            if peak > 256 * 1024**2 or time.monotonic() - started > 120:
                os.write(2, b"GTZAN metadata export budget exceeded\n")
                os._exit(124)

    threading.Thread(target=guard, daemon=True).start()
    catalog = build()
    canonical = collection_bytes(catalog)
    output = ROOT / OUTPUT
    if args.write:
        stream = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped:
            zipped.write(canonical)
        packed = stream.getvalue()
    else:
        packed = output.read_bytes()
        if gzip.decompress(packed) != canonical:
            raise ValueError("Collection canonical payload differs")
    report = coverage(catalog) | {
        "catalog_sha256": sha(packed),
        "generator_sha256": sha(Path(__file__).read_bytes()),
    }
    report_bytes = json_bytes(report)
    if len(packed) + len(report_bytes) > 16 * 1024**2:
        raise ValueError("Metadata output cap exceeded")
    if args.write:
        output.write_bytes(packed)
        (ROOT / COVERAGE).write_bytes(report_bytes)
    elif (ROOT / COVERAGE).read_bytes() != report_bytes:
        raise ValueError("Coverage summary differs")
    use = resource.getrusage(resource.RUSAGE_SELF)
    peak = use.ru_maxrss if sys.platform == "darwin" else use.ru_maxrss * 1024
    elapsed, cpu = time.monotonic() - started, use.ru_utime + use.ru_stime
    stop.set()
    if peak > 256 * 1024**2 or elapsed > 120 or cpu > 60:
        raise RuntimeError("Completed export exceeded its resource budget")
    print(
        json.dumps(
            dict(
                tracks=len(catalog.tracks),
                wall_seconds=elapsed,
                cpu_seconds=cpu,
                os_peak_rss_bytes=peak,
                output_bytes=len(packed) + len(report_bytes),
            )
        )
    )


if __name__ == "__main__":
    main()
