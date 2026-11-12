"""Produce a PROPOSED, label-blind FMA component contract; never activates roles."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VERSION = "fmak-v2-proposal-20261112-v1"
SEED = 20261112
PANELS = (
    ("calibration", 159, 60),
    ("inner_selection", 250, 100),
    ("development_confirmation", 1000, 200),
)


@dataclass(frozen=True)
class Track:
    identity: str
    component: str
    old_role: str
    audio_sha256: str
    artist_id: str
    album_id: str
    genre: str
    duration: float | None


@dataclass(frozen=True)
class Group:
    identity: str
    tracks: tuple[Track, ...]

    @property
    def duration(self) -> float:
        return sum(t.duration or 0.0 for t in self.tracks)

    @property
    def genres(self) -> Counter[str]:
        return Counter(t.genre for t in self.tracks)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rank(component: str) -> str:
    return sha(f"{VERSION}:{SEED}:{component}".encode())


def load_input(path: Path) -> tuple[list[Track], dict[str, Any]]:
    raw = path.read_bytes()
    payload = json.loads(gzip.decompress(raw))
    rows = [Track(**row) for row in payload["tracks"]]
    if len({t.identity for t in rows}) != len(rows):
        raise ValueError("Duplicate metadata ID")
    if any(t.duration is not None and t.duration <= 0 for t in rows):
        raise ValueError("Nonpositive duration")
    if any(not t.audio_sha256 or not t.component for t in rows):
        raise ValueError("Missing saved audio hash or full component")
    # The strict constructor rejects undeclared fields, including keys/model scores.
    return rows, {"compressed_sha256": sha(raw), **payload["provenance"]}


def assign(tracks: list[Track]) -> dict[str, str]:
    grouped: dict[str, list[Track]] = defaultdict(list)
    for track in tracks:
        grouped[track.component].append(track)
    groups = {k: Group(k, tuple(v)) for k, v in grouped.items()}
    forced = {t.component for t in tracks if t.old_role == "fit"}
    roles = {k: "expanded_fit" for k in forced}
    remaining = {k: g for k, g in groups.items() if k not in forced}
    reservoir = [t for g in remaining.values() for t in g.tracks]
    if not reservoir:
        raise ValueError("No non-forced groups")
    reference_genres = Counter(t.genre for t in reservoir)
    seconds_per_row = sum(t.duration or 0.0 for t in reservoir) / len(reservoir)
    if seconds_per_row <= 0:
        raise ValueError("No usable reservoir durations")
    for role, target_rows, minimum_groups in PANELS:
        limit = int(target_rows * 0.05)
        total_rows = 0
        duration = 0.0
        genres: Counter[str] = Counter()
        for step in range(1, minimum_groups + 1):
            eligible = [g for g in remaining.values() if len(g.tracks) <= limit]
            if not eligible:
                raise ValueError(f"Insufficient bounded components for {role}")
            progressive_rows = target_rows * step / minimum_groups
            progressive_duration = progressive_rows * seconds_per_row

            def cost(group: Group) -> tuple[float, str]:
                size = len(group.tracks)
                row_error = (
                    (total_rows + size - progressive_rows) / (target_rows / minimum_groups)
                ) ** 2
                duration_error = (
                    (duration + group.duration - progressive_duration)
                    / (target_rows * seconds_per_row / minimum_groups)
                ) ** 2
                genre_error = sum(
                    (
                        (genres[k] + group.genres[k] - progressive_rows * n / len(reservoir))
                        / (target_rows / minimum_groups)
                    )
                    ** 2
                    for k, n in sorted(reference_genres.items())
                ) / len(reference_genres)
                return row_error + duration_error + genre_error, rank(group.identity)

            chosen = min(eligible, key=cost)
            roles[chosen.identity] = role
            remaining.pop(chosen.identity)
            total_rows += len(chosen.tracks)
            duration += chosen.duration
            genres.update(chosen.genres)
    roles.update({k: "expanded_fit" for k in remaining})
    return roles


def describe(tracks: list[Track]) -> dict[str, Any]:
    counts = Counter(t.component for t in tracks)
    n = len(tracks)
    duration = sum(t.duration or 0.0 for t in tracks)
    return {
        "rows": n,
        "unique_saved_audio_sha256": len({t.audio_sha256 for t in tracks}),
        "components": len(counts),
        "singleton_components": sum(v == 1 for v in counts.values()),
        "largest_component": max(counts.values(), default=0),
        "largest_component_row_share": max(counts.values(), default=0) / n if n else 0,
        "row_mass_concentration_equivalent": n * n / sum(v * v for v in counts.values())
        if n
        else 0,
        "duration_seconds_metadata": duration,
        "duration_missing_rows": sum(t.duration is None for t in tracks),
        "genre_top_counts": dict(sorted(Counter(t.genre for t in tracks).items())),
        "genre_top_missing_rows": sum(t.genre == "MISSING" for t in tracks),
        "primary_artist_ids": len({t.artist_id for t in tracks if t.artist_id}),
        "album_id_missing_rows": sum(not t.album_id for t in tracks),
    }


def proposal(tracks: list[Track], provenance: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    roles = assign(tracks)
    by_hash: dict[str, set[str]] = defaultdict(set)
    for t in tracks:
        by_hash[t.audio_sha256].add(t.component)
    if any(len(v) != 1 for v in by_hash.values()):
        raise ValueError("Saved identical audio crosses full components")
    panels = {
        role: [t for t in tracks if roles[t.component] == role]
        for role in ("expanded_fit", *(p[0] for p in PANELS))
    }
    restricted = [t for t in tracks if t.old_role == "fit"]
    if not {t.identity for t in restricted} <= {t.identity for t in panels["expanded_fit"]}:
        raise ValueError("Restricted cohort nesting failed")
    representatives: dict[str, Track] = {}
    for t in sorted(panels["expanded_fit"], key=lambda t: (t.old_role != "fit", t.identity)):
        representatives.setdefault(t.audio_sha256, t)
    retained = {t.identity for t in representatives.values()}
    if not {t.identity for t in restricted} <= retained:
        raise ValueError("Historical fitting IDs conflict under prospective deduplication")
    exposure = Counter((t.old_role, roles[t.component]) for t in tracks)
    summary: dict[str, Any] = {
        "status": "PROPOSED_NOT_ACTIVATED",
        "training_authorized": False,
        "version": VERSION,
        "seed": SEED,
        "assignment_uses_labels_or_scores": False,
        "source_sha256": sha(Path(__file__).read_bytes()),
        "inputs": provenance,
        "all_fma": describe(tracks),
        "restricted_fma_fit": describe(restricted),
        "proposed_panels": {k: describe(v) for k, v in panels.items()},
        "old_role_to_proposed_role": [
            {"old_role": a, "proposed_role": b, "rows": n} for (a, b), n in sorted(exposure.items())
        ],
        "prospective_unique_fma_fit_rows": len(retained),
        "shared_bp_fitting_pool_rows": 1230,
        "bp_entity_unknown_rows": 612,
        "bp_new_inner_roles_assigned": False,
        "combined_prospective_unique_fit_rows": len(retained) + 1230,
        "class_support": "NOT_ACCESSED; separate label diagnostic required before activation",
        "activation_blockers": [
            "Root review of historical role retirement and balance",
            "Root policy decision on shared BP exposure/proxy grouping;612 identities unknown",
            "Separate class/conflicting-label support diagnostic",
            "Feature/audio identity and cached overlap audit review",
        ],
    }
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "id",
            "component",
            "old_role",
            "proposed_role",
            "restricted_fit",
            "proposed_unique_fit_representative",
            "historical_exposure",
        ]
    )
    for t in sorted(tracks, key=lambda t: t.identity):
        exposure_note = {
            "fit": "supervised_fit",
            "validation": "adaptive_development",
            "calibration": "excluded_latest_fit_historical_diagnostic",
            "final": "program_exposed_historical_final",
            "quarantined_development": "legacy_development_exposure_not_cleared",
        }[t.old_role]
        writer.writerow(
            [
                t.identity,
                t.component,
                t.old_role,
                "PROPOSED_" + roles[t.component],
                int(t.old_role == "fit"),
                int(t.identity in retained),
                exposure_note,
            ]
        )
    return summary, gzip.compress(stream.getvalue().encode(), mtime=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tracks, provenance = load_input(args.input)
    summary, ledger = proposal(tracks, provenance)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "proposal.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    (args.output / "exposure-ledger.csv.gz").write_bytes(ledger)


if __name__ == "__main__":
    main()
