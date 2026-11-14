"""Fitting-only metadata projection for BN_PROTOCOL; no waveform or label access.

Caller must verify the original plan/overlaid cache-manifest byte hashes first.
These checks validate their roles and produce a prospective panel, not evidence
that files exist or a permission to run the diagnostic.
"""

from __future__ import annotations

import hashlib
import json
import re
from fractions import Fraction
from pathlib import PurePosixPath
from typing import Any

from experiments.musicfm_bn import DOMAIN_MASS, PER_DOMAIN, crop_bounds, recording_weight, schedule


def _ids(plan: dict[str, Any], key: str, count: int) -> list[str]:
    values = plan[key]
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != count
    ):
        raise ValueError("Different complete unique plan pool: " + key)
    return values


def _path(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Expected an absolute pinned server path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Expected an absolute pinned server path")
    return value


def panel_rows(plan: dict[str, Any], rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Project the original 5,638-row metadata into 512 deterministic fitting crops.

    Output deliberately excludes reference keys and all selection row metadata.
    Crop-byte hashes cannot be provided until bounded, verified waveform reading;
    this prospective metadata is not an executable sealed manifest.
    """
    beatport, fma, beatport_selection, fma_selection = (
        _ids(plan, key, count)
        for key, count in (
            ("bp_fit", 1229),
            ("expanded_fma_fit", 4033),
            ("bp_selection_guard", 126),
            ("fma_inner_selection", 250),
        )
    )
    expected = beatport + fma + beatport_selection + fma_selection
    if len(set(expected)) != 5638 or set(rows) != set(expected):
        raise ValueError("Overlapping pools, missing input or extra/reserved cache row")
    fitting = set(beatport + fma)
    for identity, row in rows.items():
        if row.get("id") != identity or row.get("role") != (
            "fit" if identity in fitting else "validation"
        ):
            raise ValueError("Cache ID/role disagrees with the original plan")
    clips = schedule(
        {identity: rows[identity]["samples"] for identity in beatport},
        {identity: rows[identity]["samples"] for identity in fma},
    )
    if len(clips) != 512 or len({clip.identity for clip in clips}) != 512:
        raise ValueError("Schedule must return exactly 512 unique fitting recordings")
    for domain, eligible in (("beatport", set(beatport)), ("fmak", set(fma))):
        selected = [clip for clip in clips if clip.domain == domain]
        if (
            len(selected) != PER_DOMAIN
            or any(clip.identity not in eligible for clip in selected)
            or any(
                not isinstance(clip.weight, Fraction) or clip.weight != recording_weight(domain)
                for clip in selected
            )
            or sum((clip.weight for clip in selected), Fraction()) != DOMAIN_MASS[domain]
        ):
            raise ValueError("Schedule output changed domain, fitting membership or exact masses")
    result = []
    for clip in clips:
        row = rows[clip.identity]
        if row["role"] != "fit" or clip.samples != row["samples"]:
            raise ValueError("Schedule output changed fitting role or actual sample count")
        if (clip.start, clip.length) != crop_bounds(clip.domain, clip.identity, clip.samples):
            raise ValueError("Schedule output changed the fixed crop mapping")
        digest = row.get("waveform_sha256")
        if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
            raise ValueError("Missing pinned full-waveform digest")
        identities = {}
        for key in ("cache_identity", "source_stable_identity"):
            value = row.get(key)
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 5
                or any(type(part) is not int or part < 0 for part in value)
            ):
                raise ValueError("Invalid pinned physical file identity")
            identities[key] = list(value)
        if row.get("bytes") != clip.samples * 4 or identities["cache_identity"][2] != row["bytes"]:
            raise ValueError("Cache byte length disagrees with actual sample count")
        result.append(
            dict(
                domain=clip.domain,
                id=clip.identity,
                samples=clip.samples,
                start=clip.start,
                length=clip.length,
                weight_numerator=clip.weight.numerator,
                weight_denominator=clip.weight.denominator,
                waveform_sha256=digest,
                cache=_path(row.get("cache")),
                path=_path(row.get("path")),
                **identities,
            )
        )
    return result


def metadata_bytes(plan: dict[str, Any], rows: dict[str, dict[str, Any]]) -> bytes:
    return json.dumps(
        dict(
            status="PROSPECTIVE_METADATA_ONLY_NO_CROP_HASHES_OR_EXECUTION_PERMISSION",
            rows=panel_rows(plan, rows),
        ),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def metadata_sha256(plan: dict[str, Any], rows: dict[str, dict[str, Any]]) -> str:
    return hashlib.sha256(metadata_bytes(plan, rows)).hexdigest()
