"""Export existing corrected reference-panel metadata; no scan, audio, predictions or inference."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from experiments.corpus_catalog import (
    SCHEMA,
    Catalog,
    Durations,
    Evidence,
    Exposure,
    Genre,
    GenreTag,
    Grouping,
    KeyLabel,
    Measurement,
    SourceRecord,
    TrackMetadata,
    catalog_bytes,
    duration_band,
    validate_catalog,
    validate_sources,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/studies"
DEFAULT = STUDY / "v2-data-contract/data/reference-panel-catalog.json.gz"


def source_record(path: Path, source_id: str) -> SourceRecord:
    relative = path.resolve().relative_to(ROOT)
    return SourceRecord(source_id, str(relative), hashlib.sha256(path.read_bytes()).hexdigest())


def export_reference_panel(
    manifest: dict[str, Any],
    sources: tuple[SourceRecord, ...],
    result_ids: tuple[str, ...] | None = None,
) -> Catalog:
    """Adapt SHA-bound benchmark metadata without importing detector/score code."""
    records = manifest["tracks"]
    identities = [row["track_id"] for row in records]
    if len(set(identities)) != len(identities) or (
        result_ids is not None
        and (set(identities) != set(result_ids) or len(result_ids) != len(set(result_ids)))
    ):
        raise ValueError("Metadata/results track-ID join must be exact and unique")
    output = []
    for index, row in enumerate(records):
        pointer = f"/tracks/{index}"
        origin = (Evidence("reference_manifest", pointer),)
        corpus, identity = row["source"], row["track_id"]
        prefix = "swd-" if corpus == "swd-v2.1" else "gs-" if corpus == "giantsteps" else None
        if prefix is None or not identity.startswith(prefix):
            raise ValueError("Unknown corpus/track-ID convention")
        source_id = identity[len(prefix) :]
        audio = row["input_identity"]
        if audio["track_id"] != identity:
            raise ValueError("Input audio witness belongs to another track")
        wav = audio["canonical_wav"]
        parameters = wav["parameters"]
        if (
            parameters["sample_rate"],
            parameters["channels"],
            parameters["sample_width"],
            parameters["compression"],
        ) != (44100, 1, 2, "NONE"):
            raise ValueError("Unexpected analyzed canonical WAV contract")
        frames = parameters["frames"]
        if type(frames) is not int or frames <= 0 or wav["pcm_bytes"] != frames * 2:
            raise ValueError("Invalid analyzed audio support")
        seconds = frames / 44100
        source_seconds = row["duration"]
        if type(source_seconds) not in (int, float) or source_seconds <= 0:
            raise ValueError("Existing source-duration measurement missing/invalid")
        groups = (
            re.fullmatch(r"(Schubert_D911-\d{2})_(HU33|SC06)", source_id)
            if corpus == "swd-v2.1"
            else None
        )
        grouping = (
            Grouping(
                groups[1],
                groups[2],
                "SWD WorkId_PerfId filename convention recorded by benchmark adapter",
                None,
                origin,
            )
            if groups
            else Grouping(
                None,
                None,
                "not_in_recorded_metadata",
                "work_and_performance_not_established",
                origin,
            )
        )
        if type(row["dev_contaminated"]) is not bool:
            raise ValueError("Development exposure must be explicit boolean evidence")
        output.append(
            TrackMetadata(
                identity,
                corpus + ":" + source_id,
                corpus,
                source_id,
                "sha256:" + audio["source_sha256"],
                audio["source_sha256"],
                wav["container_sha256"],
                wav["pcm_sha256"],
                Genre(
                    row["genre"],
                    row["bucket"],
                    "legacy_benchmark_gs_bucket_not_normalized_taxonomy; drum-and-bass folded into dubstep; glitch-hop folded into hip-hop"
                    if corpus == "giantsteps"
                    else "recorded_corpus_classical_context_not_track_genre_annotation",
                    origin,
                    (
                        GenreTag(
                            row["genre"],
                            None,
                            None,
                            "unspecified" if corpus == "giantsteps" else "corpus_context",
                            origin,
                        ),
                    ),
                ),
                Durations(
                    Measurement(
                        float(source_seconds),
                        "seconds",
                        "recorded_ffprobe_source_file_duration",
                        None,
                        origin,
                    ),
                    Measurement(
                        seconds,
                        "seconds",
                        "canonical_wav_frames_divided_by_sample_rate",
                        None,
                        origin,
                    ),
                    Measurement(
                        None,
                        "seconds",
                        "not_verified",
                        "full_recording_duration_not_established_from_source_metadata",
                        origin,
                    ),
                    duration_band(seconds),
                ),
                Measurement(
                    None,
                    "bpm",
                    "not_measured",
                    "no_admitted_tempo_evidence_for_reference_panel",
                    origin,
                ),
                grouping,
                KeyLabel(
                    row["ref_raw"],
                    row["ref"],
                    "tonic-times-two-plus-minor-v1",
                    "first_listed_key_for_ambiguous_reference",
                    "SWD v2.1 ann_audio_globalkey.csv via recorded benchmark manifest"
                    if corpus == "swd-v2.1"
                    else "GiantSteps annotations/key via recorded benchmark manifest; exact annotation revision not recorded here",
                    origin,
                ),
                Exposure(
                    "reference_benchmark",
                    "published_development_benchmark"
                    if row["dev_contaminated"]
                    else "development_use_not_recorded_in_manifest",
                    "direct_fitting_exposure_not_established",
                    "recording_overlap_independence_not_established",
                    origin,
                ),
            )
        )
    return validate_catalog(
        Catalog(
            SCHEMA,
            f"corrected-reference-panel-{len(records)}",
            sources,
            tuple(output),
            "benchmark_results" if result_ids is not None else None,
        )
    )


def csv_bytes(catalog: Catalog) -> bytes:
    stream = io.StringIO(newline="")
    columns = (
        "track_id",
        "catalog_id",
        "corpus_id",
        "source_track_id",
        "audio_id",
        "source_audio_sha256",
        "canonical_wav_sha256",
        "canonical_pcm_sha256",
        "genre_raw",
        "genre_raw_tags",
        "genre_bucket",
        "genre_method",
        "source_audio_seconds",
        "analyzed_seconds",
        "full_recording_seconds",
        "full_recording_missing_reason",
        "analyzed_duration_band",
        "bpm",
        "bpm_method",
        "bpm_missing_reason",
        "work_id",
        "performance_id",
        "grouping_method",
        "reference_raw",
        "reference_class",
        "label_policy",
        "development_use",
        "direct_fit_status",
        "overlap_audit_status",
    )
    writer = csv.DictWriter(stream, columns, lineterminator="\n")
    writer.writeheader()
    for row in sorted(catalog.tracks, key=lambda row: (row.corpus_id, row.source_track_id)):
        writer.writerow(
            dict(
                track_id=row.track_id,
                catalog_id=row.catalog_id,
                corpus_id=row.corpus_id,
                source_track_id=row.source_track_id,
                audio_id=row.audio_id,
                source_audio_sha256=row.source_audio_sha256,
                canonical_wav_sha256=row.canonical_wav_sha256,
                canonical_pcm_sha256=row.canonical_pcm_sha256,
                genre_raw=row.genre.raw,
                genre_raw_tags=json.dumps(
                    [asdict(tag) for tag in row.genre.raw_tags], sort_keys=True
                ),
                genre_bucket=row.genre.bucket,
                genre_method=row.genre.method,
                source_audio_seconds=row.duration.source_audio.value,
                analyzed_seconds=row.duration.analyzed.value,
                full_recording_seconds=row.duration.full_recording.value,
                full_recording_missing_reason=row.duration.full_recording.missing_reason,
                analyzed_duration_band=row.duration.analyzed_band,
                bpm=row.bpm.value,
                bpm_method=row.bpm.method,
                bpm_missing_reason=row.bpm.missing_reason,
                work_id=row.grouping.work_id,
                performance_id=row.grouping.performance_id,
                grouping_method=row.grouping.method,
                reference_raw=row.key_label.raw,
                reference_class=row.key_label.class_id,
                label_policy=row.key_label.policy,
                development_use=row.exposure.development_use,
                direct_fit_status=row.exposure.direct_fit_status,
                overlap_audit_status=row.exposure.overlap_audit_status,
            )
        )
    return stream.getvalue().encode()


def verify_reference_panel(catalog: Catalog, root: Path = ROOT) -> Catalog:
    """Rebuild every metadata value from verified sources before using it in figures."""
    validate_sources(catalog, root)
    sources = {source.source_id: source for source in catalog.sources}
    manifest = json.loads((root / sources["reference_manifest"].path).read_bytes())
    ids = None
    if catalog.results_source_id is not None:
        with (root / sources[catalog.results_source_id].path).open(newline="") as stream:
            ids = tuple(row["track_id"] for row in csv.DictReader(stream))
    expected = export_reference_panel(manifest, catalog.sources, ids)
    if catalog_bytes(catalog) != catalog_bytes(expected):
        raise ValueError("Reference-panel metadata disagrees with its recorded source adapter")
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=STUDY / "key-sota-4way/data/manifest.json")
    parser.add_argument("--results", type=Path, default=STUDY / "key-sota-4way/data/bench.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--csv", type=Path, default=DEFAULT.with_suffix("").with_suffix(".csv"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--metadata-only", action="store_true", help="Do not require or read a results relation"
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_bytes())
    ids = None
    sources = (source_record(args.manifest, "reference_manifest"),)
    if not args.metadata_only:
        with args.results.open(newline="") as stream:
            # Only the identity column is consumed. Predictions stay in the existing relation.
            ids = tuple(row["track_id"] for row in csv.DictReader(stream))
        sources += (source_record(args.results, "benchmark_results"),)
    catalog = export_reference_panel(manifest, sources, ids)
    raw = catalog_bytes(catalog)
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as stream:
        stream.write(raw)
    artifacts = ((args.output, buffer.getvalue()), (args.csv, csv_bytes(catalog)))
    for path, content in artifacts:
        if args.check:
            if path.read_bytes() != content:
                raise ValueError("Catalog artifact is stale: " + str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    print(
        json.dumps(
            dict(
                tracks=len(catalog.tracks),
                bpm_known=sum(row.bpm.value is not None for row in catalog.tracks),
                catalog_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
