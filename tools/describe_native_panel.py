"""Retrospective metadata-only description of the fixed public native376 panel."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.collection_catalog import CollectionCatalog, CollectionTrack, load_collection
from experiments.corpus_catalog import GenreTag, Measurement

ROOT = Path(__file__).resolve().parents[1]
STUDY = Path("research/drone-native-cqt/composed")
MEMBERSHIP_SHA = "08a90fd7686ca354a6fecd30e65a5dca90a666d16fdd336590ba78b60b6459c2"
CATALOGS = {
    "beatport": (
        "beatport1355",
        "24a09353dc4e653bfc62898a0ca61ee2ff4838d62c3ea07ee261a2fe4f00b11f",
        1355,
    ),
    "fmak": ("fmak5489", "a626b886b61e94a6296e6160a672be54797b62a7f8f60916c82905f2c34a73b2", 5489),
}
BANDS = ("under_2_min", "2_to_under_5_min", "5_to_under_10_min", "10_min_or_longer")


@dataclass(frozen=True)
class PanelTrack:
    index: int
    track_id: str
    source: str
    source_audio_sha256: str
    decoded_samples: int
    decoded_sample_rate: int
    decoded_seconds: float
    duration_band: str
    native_genre_top: str | None
    genre_method: str
    genre_tags: tuple[GenreTag, ...]
    publisher_bpm: Measurement
    publisher_recording_duration: Measurement
    complete_song_status: str = "not_verified"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def band(seconds: float) -> str:
    """Use the native panel's predeclared boundaries, not another corpus's bins."""
    return (
        BANDS[0]
        if seconds < 120
        else BANDS[1]
        if seconds < 300
        else BANDS[2]
        if seconds < 600
        else BANDS[3]
    )


def join(members: list[dict[str, Any]], catalogs: dict[str, CollectionCatalog]) -> list[PanelTrack]:
    indexed: dict[str, CollectionTrack] = {}
    for source, catalog in catalogs.items():
        for track in catalog.tracks:
            if track.track_id in indexed or track.corpus_id != source:
                raise ValueError("Duplicate or cross-source catalog identity")
            indexed[track.track_id] = track
    result = []
    seen = set()
    for index, member in enumerate(members):
        identity = member["id"]
        if identity in seen or member["index"] != index:
            raise ValueError("Changed panel order or duplicate identity")
        seen.add(identity)
        track = indexed.get(identity)
        if (
            track is None
            or track.source_audio.sha256 != member["audio_sha256"]
            or track.corpus_id != member["domain"]
        ):
            raise ValueError("Missing catalog row or source-audio hash mismatch")
        seconds = member["pcm_samples"] / 22050
        result.append(
            PanelTrack(
                index,
                identity,
                track.corpus_id,
                member["audio_sha256"],
                member["pcm_samples"],
                22050,
                seconds,
                band(seconds),
                track.native_genre_top,
                track.genre_method,
                track.genre_tags,
                track.bpm,
                track.catalog_recording_duration,
            )
        )
    return result


def tag_counts(rows: list[PanelTrack]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str | None, str | int | None, str, str]] = Counter()
    for row in rows:
        counts.update(
            {(tag.taxonomy, tag.tag_id, tag.name, tag.relation) for tag in row.genre_tags}
        )
    return [
        dict(taxonomy=taxonomy, tag_id=tag_id, name=name, relation=relation, tracks=count)
        for (taxonomy, tag_id, name, relation), count in sorted(
            counts.items(), key=lambda item: (-item[1], str(item[0]))
        )
    ]


def summarize(rows: list[PanelTrack]) -> dict[str, Any]:
    tempo = [row.publisher_bpm.value for row in rows if row.publisher_bpm.value is not None]
    genres = sorted({row.native_genre_top for row in rows if row.native_genre_top is not None})
    if any(row.native_genre_top is None for row in rows):
        genres_with_missing: list[str | None] = [*genres, None]
    else:
        genres_with_missing = list(genres)
    return {
        "tracks": len(rows),
        "duration_counts": {
            label: sum(row.duration_band == label for row in rows) for label in BANDS
        },
        "decoded_duration_range_seconds": [
            min(row.decoded_seconds for row in rows),
            max(row.decoded_seconds for row in rows),
        ],
        "native_top_known": sum(row.native_genre_top is not None for row in rows),
        "native_top_missing": sum(row.native_genre_top is None for row in rows),
        "any_genre_tags_known": sum(bool(row.genre_tags) for row in rows),
        "any_genre_tags_missing": sum(not row.genre_tags for row in rows),
        "native_top_by_duration": [
            {
                "native_genre_top": genre,
                "tracks": sum(row.native_genre_top == genre for row in rows),
                "duration_counts": {
                    label: sum(
                        row.native_genre_top == genre and row.duration_band == label for row in rows
                    )
                    for label in BANDS
                },
            }
            for genre in genres_with_missing
        ],
        "native_tag_counts_nonexclusive": tag_counts(rows),
        "tags_by_duration_nonexclusive": {
            label: tag_counts([row for row in rows if row.duration_band == label])
            for label in BANDS
        },
        "genre_methods": dict(Counter(row.genre_method for row in rows)),
        "publisher_bpm_known": len(tempo),
        "publisher_bpm_missing": len(rows) - len(tempo),
        "publisher_bpm_range": [min(tempo), max(tempo)] if tempo else None,
        "publisher_bpm_methods": dict(Counter(row.publisher_bpm.method for row in rows)),
        "publisher_bpm_missing_reasons": dict(
            Counter(
                row.publisher_bpm.missing_reason for row in rows if row.publisher_bpm.value is None
            )
        ),
    }


def analyze(root: Path = ROOT) -> dict[str, Any]:
    path = root / STUDY / "FUSED_376_MEMBERSHIP.json"
    if sha(path) != MEMBERSHIP_SHA:
        raise ValueError("Frozen native panel membership changed")
    members = json.loads(path.read_bytes())["cases"]
    if len(members) != 376:
        raise ValueError("Fixed 376 panel required")
    catalogs = {}
    inputs = {}
    for source, (stem, expected, count) in CATALOGS.items():
        relative = Path("experiments/studies/v2-data-contract/data") / f"{stem}-collection.json.gz"
        path = root / relative
        if sha(path) != expected:
            raise ValueError("Pinned target-free catalog changed")
        catalog = load_collection(path)
        if len(catalog.tracks) != count:
            raise ValueError("Catalog count changed")
        catalogs[source] = catalog
        inputs[source] = {
            "path": str(relative),
            "sha256": expected,
            "collection_id": catalog.collection_id,
            "sources": [asdict(item) for item in catalog.sources],
            "external_sources": [asdict(item) for item in catalog.external_sources],
        }
    rows = join(members, catalogs)
    return {
        "schema": "drone.native376-metadata-description.v1",
        "join_kind": "retrospective_descriptive_track_id_and_source_audio_sha256",
        "inputs": inputs,
        "membership_sha256": MEMBERSHIP_SHA,
        "generator_sha256": sha(root / "tools/describe_native_panel.py"),
        "sources": {
            source: summarize([row for row in rows if row.source == source]) for source in catalogs
        },
        "tracks": [asdict(row) for row in rows],
        "interpretation": {
            "historical_frozen_run_packet_genre_bpm": "unknown; not retroactively amended",
            "metadata_available_for_this_later_join": True,
            "native_top_and_all_tags_kept_separate": True,
            "taxonomies_normalized": False,
            "hierarchical_or_multiple_tags_are_nonexclusive": True,
            "publisher_bpm_independently_verified": False,
            "full_song_provenance_verified": False,
            "accuracy_or_weakness_claim": False,
        },
        "audio_read": False,
        "model_executed": False,
        "key_labels_read": False,
    }


def plot(result: dict[str, Any], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "native376-metadata-v1"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), layout="constrained")
    for axis, source, title in zip(
        axes,
        ["beatport", "fmak"],
        ["Beatport: native workbook genre (126)", "FMAK: native top genre (250)"],
        strict=True,
    ):
        rows = result["sources"][source]["native_top_by_duration"]
        values = [[row["duration_counts"][label] for label in BANDS] for row in rows]
        axis.imshow(
            values, cmap="Blues", vmin=0, vmax=max(max(row) for row in values), aspect="auto"
        )
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                axis.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > max(max(r) for r in values) * 0.6 else "#173447",
                    fontsize=9,
                )
        axis.set_yticks(
            range(len(rows)), [row["native_genre_top"] or "Missing top field" for row in rows]
        )
        axis.set_xticks(range(4), ["<2 min", "2–<5 min", "5–<10 min", "≥10 min"])
        axis.set(title=title, xlabel="Duration of the retained decoded input")
        axis.tick_params(length=0)
    fig.suptitle("What is in the fixed 376-recording numerical panel?", fontsize=16)
    fig.supxlabel(
        "Retrospective ID + audio-hash join; counts describe coverage, not model accuracy.\nNative taxonomies and top fields stay separate; color scales within each source. FMAK: 249 with tags, 125 without a top field.\nBeatport publisher BPM is available on 60/126 (86–175), not independently verified; FMAK BPM and full-song status unknown.",
        fontsize=9,
    )
    fig.savefig(destination / "public376-metadata.png", dpi=180)
    svg = destination / "public376-metadata.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    result = analyze()
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path = ROOT / STUDY / "public376-metadata.json.gz"
    if args.write:
        path.write_bytes(gzip.compress(raw, mtime=0))
    elif gzip.decompress(path.read_bytes()) != raw:
        raise ValueError("Stored metadata description differs")
    if args.plot:
        plot(result, ROOT / STUDY)
    print(
        json.dumps(
            {
                "tracks": len(result["tracks"]),
                "metadata_bytes": len(raw),
                "audio_read": False,
                "model_executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
