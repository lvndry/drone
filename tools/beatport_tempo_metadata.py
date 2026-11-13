"""Export source-reported tempo for the frozen BP development cohort, without audio."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import TypedDict

from tools.dj_coverage_bp import EXPECTED

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "experiments/studies/k10-v2-pilot/data/canonical-eligibility-v2/plan-v2.json.gz"
DESTINATION = ROOT / "experiments/studies/v2-data-contract/data/beatport-tempo-metadata.json.gz"
BANDS = ("<90", "90–<110", "110–<130", "130–<150", "150+", "unknown")


class TempoRow(TypedDict):
    track_id: str
    role: str
    bpm: float | None
    bpm_status: str
    bpm_band: str
    source_member: str | None
    source_member_sha256: str | None
    nonstandard_nan_normalized: bool


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def selected_tempo(raw: bytes) -> tuple[int, float | None, str, bool]:
    """Select only store ID/tempo; store key fields never become reference labels."""
    text = raw.decode("utf-8")
    # The archived publisher JSON includes bare lowercase nan. Preserve the
    # source bytes/hash and record normalization; never alter quoted strings.
    token = re.compile(r'"(?:[^"\\]|\\.)*"|(?<![\w])nan(?![\w])')
    normalized = token.sub(lambda match: "null" if match[0] == "nan" else match[0], text)
    payload = json.loads(normalized)
    if not isinstance(payload, dict) or type(payload.get("id")) is not int:
        raise ValueError("Publisher track ID must be an integer")
    identity: int = payload["id"]
    if identity < 0:
        raise ValueError("Negative publisher track ID")
    value = payload.get("bpm")
    if value is None:
        return identity, None, "missing_publisher_value", normalized != text
    if type(value) not in (int, float):
        raise ValueError("Publisher BPM must be numeric or null")
    bpm = float(value)
    if not math.isfinite(bpm) or bpm < 0:
        raise ValueError("Publisher BPM must be finite and nonnegative")
    if bpm == 0:
        return identity, None, "publisher_zero_sentinel", normalized != text
    return identity, bpm, "publisher_reported_unverified", normalized != text


def tempo_band(bpm: float | None) -> str:
    if bpm is None:
        return "unknown"
    for boundary, name in zip((90, 110, 130, 150), BANDS, strict=False):
        if bpm < boundary:
            return name
    return "150+"


def extract_rows(store_zip: Path, roles: dict[str, str]) -> list[TempoRow]:
    found: dict[str, TempoRow] = {}
    with zipfile.ZipFile(store_zip) as archive:
        for member in archive.infolist():
            match = re.fullmatch(r"meta/(\d+) .+\.json", member.filename)
            if match is None:
                continue
            track_id = "beatport:" + str(int(match[1]))
            if track_id not in roles:
                continue
            raw = archive.read(member)
            identity, bpm, status, normalized = selected_tempo(raw)
            if track_id != f"beatport:{identity}" or track_id in found:
                raise ValueError("Duplicate or mismatched publisher identity")
            found[track_id] = TempoRow(
                track_id=track_id,
                role=roles[track_id],
                bpm=bpm,
                bpm_status=status,
                bpm_band=tempo_band(bpm),
                source_member=member.filename,
                source_member_sha256=sha(raw),
                nonstandard_nan_normalized=normalized,
            )
    return [
        found.get(
            track_id,
            TempoRow(
                track_id=track_id,
                role=roles[track_id],
                bpm=None,
                bpm_status="publisher_record_missing",
                bpm_band="unknown",
                source_member=None,
                source_member_sha256=None,
                nonstandard_nan_normalized=False,
            ),
        )
        for track_id in sorted(roles)
    ]


def build(store_zip: Path, plan: Path = PLAN) -> dict[str, object]:
    for path, expected in ((store_zip, EXPECTED["store_json_zip"]), (plan, EXPECTED["plan"])):
        if sha(path.read_bytes()) != expected:
            raise ValueError("Frozen metadata input differs: " + str(path))
    membership = json.loads(gzip.decompress(plan.read_bytes()))
    roles: dict[str, str] = {}
    for field, role in (("bp_fit", "fit"), ("bp_selection_guard", "inner_selection")):
        for track_id in membership[field]:
            if track_id in roles or not re.fullmatch(r"beatport:\d+", track_id):
                raise ValueError("Duplicate or invalid frozen cohort member")
            roles[track_id] = role
    if Counter(roles.values()) != {"fit": 1229, "inner_selection": 126}:
        raise ValueError("Frozen development membership changed")
    rows = extract_rows(store_zip, roles)
    summaries = {}
    for role in ("all", "fit", "inner_selection"):
        selected = [row for row in rows if role == "all" or row["role"] == role]
        counts = Counter(row["bpm_band"] for row in selected)
        known = [row["bpm"] for row in selected if row["bpm"] is not None]
        summaries[role] = {
            "n": len(selected),
            "known": len(known),
            "missing": len(selected) - len(known),
            "min_bpm": min(known) if known else None,
            "max_bpm": max(known) if known else None,
            "bands": {band: counts[band] for band in BANDS},
            "statuses": dict(sorted(Counter(row["bpm_status"] for row in selected).items())),
        }
    return {
        "schema": "drone.beatport-tempo-metadata.v1",
        "scope": "frozen-expanded-development-plan-bp-only",
        "unit": "beats_per_minute",
        "method": "archived_beatport_store_metadata_not_independently_verified",
        "interpretation": "Publisher tempo for the catalog track; not measured preview tempo, not a key target or a model score. Half/double-time ambiguity is unresolved.",
        "source_zip_sha256": EXPECTED["store_json_zip"],
        "membership_plan": PLAN.relative_to(ROOT).as_posix(),
        "membership_plan_sha256": EXPECTED["plan"],
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "inference_performed": False,
        "audio_read": False,
        "summary": summaries,
        "tracks": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-zip", required=True, type=Path)
    parser.add_argument("--plan", default=PLAN, type=Path)
    parser.add_argument("--output", default=DESTINATION, type=Path)
    parser.add_argument("--check", action="store_true", help="Compare with existing export")
    args = parser.parse_args()
    result = build(args.store_zip, args.plan)
    encoded = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    compressed = gzip.compress(encoded, mtime=0)
    if args.check:
        compressed = args.output.read_bytes()
        if gzip.decompress(compressed) != encoded:
            raise ValueError("Stored tempo export differs from source-derived values")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(compressed)
    print(json.dumps({"sha256": sha(compressed), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
