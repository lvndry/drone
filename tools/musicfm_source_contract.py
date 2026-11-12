"""Apply a separately audited physical overlay without changing canonical model inputs."""

from __future__ import annotations

from typing import Any


def apply_identity_overlay(
    rows: dict[str, dict[str, Any]], overlay: dict[str, Any], manifest_sha256: str
) -> dict[str, dict[str, Any]]:
    if (
        overlay.get("complete") is not True
        or overlay.get("canonical_manifest_sha256") != manifest_sha256
        or overlay.get("format") != "musicfm-source-physical-identity-overlay-v1"
    ):
        raise ValueError("Unbound/incomplete physical identity overlay")
    records = {r["invariants"]["id"]: r for r in overlay["rows"]}
    if len(records) != len(rows) or len(overlay["rows"]) != len(rows) or set(records) != set(rows):
        raise ValueError("Identity overlay changes membership")
    keys = ("id", "path", "role", "target", "sha256", "waveform_sha256", "samples", "bytes")
    result = {}
    for identity, row in rows.items():
        record = records[identity]
        current = record["source_current"]
        if (
            record["invariants"] != {key: row[key] for key in keys}
            or record["source_identity_old"] != list(row["source_stable_identity"])
            or current["sha256"] != row["sha256"]
            or current["bytes"] != row["source_stable_identity"][2]
            or current["identity"][2] != current["bytes"]
        ):
            raise ValueError("Identity overlay changes source/content/labels")
        result[identity] = {**row, "source_stable_identity": current["identity"]}
        if row["role"] == "validation":
            cached = record["validation_current"]
            if (
                record["validation_cache_path"] != row["cache"]
                or cached["sha256"] != row["waveform_sha256"]
                or cached["bytes"] != row["bytes"]
                or cached["mode"] & 0o222
                or record["validation_identity_old"] != list(row["cache_identity"])
            ):
                raise ValueError("Identity overlay changes immutable validation input")
            result[identity]["cache_identity"] = cached["identity"]
    return result
