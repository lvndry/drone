"""Existing public genre/duration metadata only; no key references, audio or models."""

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
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

BANDS = ("<30s", "30s–<2m", "2–<5m", "5–10m", ">10m", "duration_unknown")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def band(seconds: float | None) -> str:
    if seconds is None:
        return BANDS[-1]
    return (
        BANDS[0]
        if seconds < 30
        else BANDS[1]
        if seconds < 120
        else BANDS[2]
        if seconds < 300
        else BANDS[3]
        if seconds <= 600
        else BANDS[4]
    )


def known_id(value: str) -> str | None:
    return value if value.isdigit() and int(value) > 0 else None


def checked_inputs(config: Path, section: str) -> dict[str, Path]:
    """Read existing source paths only after matching their supplied SHA256 receipts."""
    records = json.loads(config.read_bytes())[section]
    paths = {name: Path(row["path"]) for name, row in records.items()}
    for name, path in paths.items():
        if sha(path) != records[name]["sha256"]:
            raise ValueError("Existing metadata source changed: " + name)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = checked_inputs(args.inputs, "fma")
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sources = {
        name: {"sha256": sha(path), "bytes": path.stat().st_size} for name, path in paths.items()
    }
    assert (
        sources["plan"]["sha256"]
        == "c3e59ffb764ca5f51444447b0a18a69543228e60a9e3723d925ddd82b0fc8373"
    )
    taxonomy = {
        int(row["genre_id"]): {"title": row["title"], "parent": int(row["parent"])}
        for row in csv.DictReader(paths["taxonomy"].open())
    }
    titles = {row["title"]: key for key, row in taxonomy.items()}
    # Whitelist ONLY top-level identity arrays; never parse expected_refs/targets.
    text = gzip.decompress(paths["plan"].read_bytes()).decode()
    fields = (
        "bp_fit",
        "bp_selection_guard",
        "restricted_fma_fit",
        "expanded_fma_fit",
        "fma_inner_selection",
        "confirmation_reserved_container_unique",
    )
    plan = {}
    for field in fields:
        match = re.search(r'"' + field + r'"\s*:\s*(\[[^\]]*\])', text)
        assert match is not None
        ids = json.loads(match.group(1))
        assert all(
            isinstance(identity, str) and re.fullmatch(r"(fmak|beatport):[0-9]+", identity)
            for identity in ids
        )
        plan[field] = set(ids)
        assert len(ids) == len(plan[field])
    ledger = list(
        csv.DictReader(io.StringIO(gzip.decompress(paths["exposure_ledger"].read_bytes()).decode()))
    )
    exposed = {row["id"]: row for row in ledger}
    assert len(exposed) == len(ledger) == 5489
    safe = json.loads(gzip.decompress(paths["sanitized_group_input"].read_bytes()))
    assert safe["provenance"]["no_target_labels"] is True
    assert safe["provenance"]["metadata_csv"]["sha256"] == sources["master_metadata"]["sha256"]
    assert all(
        set(track)
        == {
            "identity",
            "component",
            "old_role",
            "audio_sha256",
            "artist_id",
            "album_id",
            "genre",
            "duration",
        }
        for track in safe["tracks"]
    )
    groups = {track["identity"]: track for track in safe["tracks"]}
    assert set(groups) == set(exposed)
    roles: dict[str, set[str]] = {
        "all_fmak": set(exposed),
        "expanded_fit": plan["expanded_fma_fit"],
        "restricted_fit_nested": plan["restricted_fma_fit"],
        "inner_selection": plan["fma_inner_selection"],
        "confirmation_reserved_unique": plan["confirmation_reserved_container_unique"],
        "confirmation_reserved_raw": {
            identity
            for identity, row in exposed.items()
            if row["proposed_role"] == "PROPOSED_development_confirmation"
        },
        "calibration_excluded": {
            identity
            for identity, row in exposed.items()
            if row["proposed_role"] == "PROPOSED_calibration"
        },
    }
    roles["inactive_or_quarantined_fit"] = {
        identity
        for identity, row in exposed.items()
        if row["proposed_role"] == "PROPOSED_expanded_fit"
    } - roles["expanded_fit"]
    assert tuple(
        len(roles[role])
        for role in (
            "expanded_fit",
            "restricted_fit_nested",
            "inner_selection",
            "confirmation_reserved_unique",
            "confirmation_reserved_raw",
            "calibration_excluded",
            "inactive_or_quarantined_fit",
        )
    ) == (4033, 976, 250, 997, 998, 159, 49)
    partition = [
        "expanded_fit",
        "inner_selection",
        "confirmation_reserved_raw",
        "calibration_excluded",
        "inactive_or_quarantined_fit",
    ]
    assert set.union(*(roles[role] for role in partition)) == roles["all_fmak"]
    assert sum(len(roles[role]) for role in partition) == 5489
    tracks: dict[str, dict[str, Any]] = {}
    for row in csv.DictReader(paths["master_metadata"].open(newline="")):
        identity = "fmak:" + f"{int(row['track_id']):06}"
        assert identity not in tracks
        tags = (
            frozenset(ast.literal_eval(row["track_genres_all"]))
            if row["track_genres_all"]
            else frozenset()
        )
        assert all(type(tag) is int and tag in taxonomy for tag in tags)
        try:
            duration = float(row["track_duration"])
        except ValueError:
            duration = float("nan")
        duration = duration if math.isfinite(duration) and duration > 0 else None
        tracks[identity] = {
            "artist": known_id(row["artist_id"]),
            "album": known_id(row["album_id"]),
            "tags": tags,
            "top": row["track_genre_top"],
            "duration": duration,
            "band": band(duration),
            "license": row["track_license"],
        }
    assert roles["all_fmak"] <= tracks.keys()
    for identity in roles["all_fmak"]:
        assert groups[identity]["component"] == exposed[identity]["component"]
        assert groups[identity]["duration"] == tracks[identity]["duration"]
        assert groups[identity]["artist_id"] == (tracks[identity]["artist"] or "")
        assert groups[identity]["album_id"] == (tracks[identity]["album"] or "")
    # Master metadata-only artist/album graph. No audio-identity joins are available here.
    parent = {identity: identity for identity in tracks}

    def root(identity: str) -> str:
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    links = {}
    for identity, row in tracks.items():
        for kind in ("artist", "album"):
            value = row[kind]
            if value is None:
                continue
            key = (kind, value)
            if key in links:
                own_root = root(identity)
                linked_root = root(links[key])
                parent[max(own_root, linked_root)] = min(own_root, linked_root)
            else:
                links[key] = identity
    master_components = {identity: root(identity) for identity in tracks}
    roles["master_metadata_all"] = set(tracks)
    roles["master_outside_fmak_metadata_only"] = set(tracks) - roles["all_fmak"]
    exposed_master_components = {master_components[identity] for identity in roles["all_fmak"]}
    roles["master_outside_fmak_unlinked_primary_credits"] = {
        identity
        for identity in roles["master_outside_fmak_metadata_only"]
        if master_components[identity] not in exposed_master_components
    }

    def stats(ids: set[str], scope: str) -> dict[str, Any]:
        data = [tracks[identity] for identity in ids]
        components = (
            master_components
            if scope.startswith("master_")
            else {identity: exposed[identity]["component"] for identity in ids}
        )
        return {
            "tracks": len(ids),
            "known_artists": len({track["artist"] for track in data if track["artist"]}),
            "known_albums": len({track["album"] for track in data if track["album"]}),
            "components": len({components[identity] for identity in ids}),
            "missing_artist_tracks": sum(track["artist"] is None for track in data),
            "missing_album_tracks": sum(track["album"] is None for track in data),
            "missing_top_genre_tracks": sum(not track["top"] for track in data),
            "empty_genres_all_tracks": sum(not track["tags"] for track in data),
            "unknown_duration_tracks": sum(track["duration"] is None for track in data),
        }

    specific = [
        "Hip-Hop",
        "Rap",
        "Alternative Hip-Hop",
        "Abstract Hip-Hop",
        "Hip-Hop Beats",
        "Nerdcore",
        "Wonky",
        "Afrobeat",
        "House",
        "Chill-out",
        "Techno",
        "Bigbeat",
        "Disco",
        "Dance",
        "Drum & Bass",
        "Jungle",
        "Dubstep",
        "Skweee",
        "Breakbeat",
        "Breakcore - Hard",
        "Trip-Hop",
        "Downtempo",
        "Funk",
        "Soul-RnB",
        "Reggae - Dub",
        "Reggae - Dancehall",
        "Cumbia",
        "Salsa",
        "Synth Pop",
    ]
    broad = [
        "Electronic",
        "Minimal Electronic",
        "IDM",
        "Glitch",
        "Ambient Electronic",
        "African",
        "Latin",
        "Garage",
    ]
    categories: dict[str, dict[str, Any]] = {
        name: {"ids": [titles[name]], "kind": "named_taxonomic_membership"} for name in specific
    }
    categories.update(
        {name: {"ids": [titles[name]], "kind": "broad_or_ambiguous_surrogate"} for name in broad}
    )
    club = [
        "House",
        "Techno",
        "Disco",
        "Dance",
        "Drum & Bass",
        "Jungle",
        "Dubstep",
        "Breakbeat",
        "Breakcore - Hard",
    ]
    categories["named_dance_related_union"] = {
        "ids": [titles[name] for name in club],
        "kind": "declared_union_not_a_verified_club_label",
        "members": club,
    }
    categories["House_minus_Chill-out_metadata_slice"] = {
        "ids": [titles["House"]],
        "exclude_ids": [titles["Chill-out"]],
        "kind": "declared_set_difference_not_a_new_genre",
    }
    categories["Techno_minus_Bigbeat_metadata_slice"] = {
        "ids": [titles["Techno"]],
        "exclude_ids": [titles["Bigbeat"]],
        "kind": "declared_set_difference_not_a_new_genre",
    }
    categories["Afrobeats_requested_UNVERIFIED"] = {
        "ids": [],
        "kind": "no_matching_taxonomy_tag_not_music_absence",
    }
    coverage = {}
    for name, category in categories.items():
        members = {
            identity
            for identity, row in tracks.items()
            if row["tags"] & set(category["ids"])
            and not row["tags"] & set(category.get("exclude_ids", []))
        }
        coverage[name] = {}
        for role, ids in roles.items():
            relevant = ids & members
            coverage[name][role] = {
                "all": stats(relevant, role),
                "bands": {
                    duration_band: stats(
                        {
                            identity
                            for identity in relevant
                            if tracks[identity]["band"] == duration_band
                        },
                        role,
                    )
                    for duration_band in BANDS
                },
            }
    inventory = []
    for tag, row in taxonomy.items():
        members = {identity for identity, track in tracks.items() if tag in track["tags"]}
        inventory.append(
            {
                "id": tag,
                **row,
                "roles": {role: stats(ids & members, role) for role, ids in roles.items()},
            }
        )
    role_summary = {
        role: {
            "all": stats(ids, role),
            "bands": {
                duration_band: stats(
                    {identity for identity in ids if tracks[identity]["band"] == duration_band},
                    role,
                )
                for duration_band in BANDS
            },
            "historical_exposure": dict(
                sorted(
                    Counter(
                        exposed[identity]["historical_exposure"]
                        for identity in ids
                        if identity in exposed
                    ).items()
                )
            ),
        }
        for role, ids in roles.items()
    }
    overlaps = {}
    selected = {
        "Hip-Hop": {titles["Hip-Hop"]},
        "House": {titles["House"]},
        "Techno": {titles["Techno"]},
        "Afrobeat": {titles["Afrobeat"]},
        "named_dance_related_union": set(categories["named_dance_related_union"]["ids"]),
        "Electronic": {titles["Electronic"]},
    }
    for first in selected:
        for second in selected:
            if first >= second:
                continue
            common = {
                identity
                for identity, track in tracks.items()
                if track["tags"] & selected[first] and track["tags"] & selected[second]
            }
            overlaps[first + " & " + second] = {
                role: stats(ids & common, role) for role, ids in roles.items()
            }
    role_overlap = []
    for index, first in enumerate(partition):
        for second in partition[index + 1 :]:
            first_components = {exposed[identity]["component"] for identity in roles[first]}
            second_components = {exposed[identity]["component"] for identity in roles[second]}
            first_master = {master_components[identity] for identity in roles[first]}
            second_master = {master_components[identity] for identity in roles[second]}
            role_overlap.append(
                {
                    "roles": [first, second],
                    "track_overlap": len(roles[first] & roles[second]),
                    "frozen_component_overlap": len(first_components & second_components),
                    "full_master_artist_album_component_overlap": len(first_master & second_master),
                }
            )
    genre_parent_omissions = sum(
        any(
            taxonomy[tag]["parent"] and taxonomy[tag]["parent"] not in track["tags"]
            for tag in track["tags"]
        )
        for track in tracks.values()
    )
    present_titles = set(titles)
    absent = [
        "Afrobeats",
        "Deep House",
        "Melodic House",
        "Tech House",
        "Progressive House",
        "Minimal Techno",
        "Trance",
        "UK Garage",
        "Grime",
        "Trap",
        "Amapiano",
        "Gqom",
        "Kwaito",
        "Reggaeton",
        "Jersey Club",
        "Footwork",
        "Baile Funk",
    ]
    absent = [name for name in absent if name not in present_titles]
    result = {
        "kind": "genre-duration-existing-metadata-audit-v1",
        "sources": sources,
        "source_policy": "No key target files opened. Plan parser whitelists identity arrays only; expected_refs is never parsed. Sanitized input declares no_target_labels. No audio/private files/model/decoding/acquisition.",
        "duration_policy": "FMA published track_duration seconds, not new measured decode duration; [0,30),[30,120),[120,300),[300,600],(600,infinity). Unknown/zero/nonfinite separate.",
        "genre_policy": "track_genres_all membership, including ancestral taxonomy tags, NOT necessarily a direct leaf tag or verified genre. Multi-label counts overlap. Taxonomy names reproduced verbatim.",
        "component_policy": {
            "FMAK_roles": "Frozen exposure-ledger artist/album/audio connected components; counts within each slice, components can span durations/genres.",
            "master_roles": "Fresh metadata-only connected components over known positive artist/album IDs across all master rows; no audio identity, credits only, different universe from frozen cohort.",
        },
        "role_policy": "Partition is expanded_fit+inner_selection+confirmation_reserved_raw+calibration_excluded+inactive_or_quarantined_fit. Restricted_fit is nested. Reserved_unique is997 representatives of998 raw reserved rows; neither means never exposed.",
        "out_of_scope_bp": {
            "fit": len(plan["bp_fit"]),
            "inner_selection": len(plan["bp_selection_guard"]),
            "reason": "FMA taxonomy/master cannot supply BP genre coverage; no BP genres inferred.",
        },
        "taxonomy_absent_names_not_proof_of_missing_music": absent,
        "taxonomy_ambiguities": [
            "Afrobeat81 is NOT evidence for requested Afrobeats.",
            "Garage85 descends from Rock12; do not interpret as UK garage.",
            "House182 includes taxonomy child Chill-out400; not verified deep/melodic/tech house.",
            "Techno181 includes Bigbeat401; not verified all techno subgenres.",
            "Electronic15 includes ambient/experimental/non-club material; African92 is not Afrobeats.",
        ],
        "master_metadata_genres_all_parent_omission_tracks": genre_parent_omissions,
        "categories": categories,
        "roles": role_summary,
        "coverage": coverage,
        "all_taxonomy_inventory": inventory,
        "category_overlaps": overlaps,
        "frozen_role_overlap": role_overlap,
        "reference_labels_or_scores_used": False,
        "dataset_or_pool_changes": False,
    }
    path = output_dir / "coverage.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    use = resource.getrusage(resource.RUSAGE_SELF)
    receipt = {
        "script_sha256": sha(Path(__file__)),
        "result_sha256": sha(path),
        "result_bytes": path.stat().st_size,
        "wall_seconds": time.monotonic() - started,
        "self_cpu_seconds": use.ru_utime + use.ru_stime,
        "peak_rss_bytes": use.ru_maxrss if sys.platform == "darwin" else use.ru_maxrss * 1024,
        "platform": sys.platform,
        "new_audio_model_dataset_activity": False,
    }
    (output_dir / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
