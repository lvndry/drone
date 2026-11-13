"""Freeze label-free GTZAN overlap queries and release exposure; no audio or matching."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import re
import tarfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from tools import check_gtzan_audio_report as report
from tools.check_gtzan_audio import ARCHIVE_BYTES, ARCHIVE_SHA256, URL
from tools.import_gtzan_reference import DEFAULT, encode, produce

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/studies/v2-data-contract"
OUTPUT = STUDY / "data/gtzan-overlap-manifest.json.gz"
RELEASE = ROOT / "experiments/studies/bp-gs-overlap/data/release-metadata.json"
REFERENCES = ROOT / "experiments/studies/bp-gs-overlap/data/reference-result-v1.json.gz"
RELEASE_SHA = "69432e5f6ddc0ad10a55de591e3174e6db9ef705355cfdca5a1f678a9bfa4c4f"
CHECKPOINT_SHA = "dcf8a7a292b7c12076ffff63c8c6550cb09a6c41ce4d000fd7949fc6bb4b452c"
REFERENCE_SHA = "0690f58a09be01d5ab37acbb5b845ec77a3b77fc29411227eabc33eaae266a73"
REFERENCE_DECODED_SHA = "5b3d7296f8277d8b43470736e670d35e58c2c94230b0f95dc53172d33787ed14"


@dataclass(frozen=True)
class Query:
    id: str
    member: str
    wav_md5: str
    wav_sha256: str
    bytes: int
    frames: int
    sample_rate: int
    duration_seconds: float


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require_hash(value: object, length: int) -> str:
    if not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError("Invalid recording digest")
    return value


def queries(
    ledger: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
    *,
    eligibility_field: Literal[
        "metadata_eligible", "raw_single_key_reference"
    ] = "metadata_eligible",
) -> tuple[Query, ...]:
    """Project existing eligibility, never infer it from labels or overlap outcomes."""
    ids = [r["id"] for r in ledger]
    members = [r["audio_member"] for r in ledger]
    actual = [r["member"] for r in inventory]
    if len(set(ids)) != len(ids) or len(set(members)) != len(members):
        raise ValueError("Duplicate eligibility ID or member")
    if len(set(actual)) != len(actual) or set(actual) != set(members):
        raise ValueError("Duplicate or incomplete inventory")
    by_member = {r["member"]: r for r in inventory}
    result = []
    for row in sorted(ledger, key=lambda r: r["id"]):
        if type(row[eligibility_field]) is not bool:
            raise ValueError("Eligibility must be boolean")
        if not row[eligibility_field]:
            continue
        ident, member = row["id"], row["audio_member"]
        if not isinstance(ident, str) or re.fullmatch(r"[a-z]+\.\d{5}", ident) is None:
            raise ValueError("Invalid query ID")
        if member != f"genres/{ident.split('.')[0]}/{ident}.wav":
            raise ValueError("Query member does not match ID")
        item = by_member[member]
        md5 = require_hash(row["expected_member_md5"], 32)
        if (
            item["member_identity_verified"] is not True
            or item["expected_md5"] != md5
            or item["md5"] != md5
            or item["overlap_cleared"] is not False
            or item["admitted_for_scoring"] is not False
            or "wave_error" in item
        ):
            raise ValueError("Query lacks unadmitted matching audio identity")
        wave = item["wave"]
        frames, rate, size = wave["frames"], wave["sample_rate"], item["bytes"]
        if (
            any(type(v) is not int or v <= 0 for v in (frames, rate, size))
            or (wave["channels"], rate, wave["sample_width"]) != (1, 22050, 2)
            or frames > rate * 60
            or size < frames * 2
            or not math.isfinite(wave["duration_seconds"])
            or wave["duration_seconds"] != frames / rate
        ):
            raise ValueError("Invalid inventory WAV geometry")
        result.append(
            Query(
                ident,
                member,
                md5,
                require_hash(item["sha256"], 64),
                size,
                frames,
                rate,
                frames / rate,
            )
        )
    if not result:
        raise ValueError("Empty query population")
    return tuple(result)


def release_anchor(
    release: Mapping[str, Any], references: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Retain actual release roles and all known exact container/PCM aliases."""
    if release["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Unexpected released checkpoint")
    ref_ids = [r["ref_id"] for r in references]
    identities = [r["id"] for r in references]
    if len(set(ref_ids)) != len(ref_ids) or len(set(identities)) != len(identities):
        raise ValueError("Duplicate reference identity")
    by_id = {r["id"]: r for r in references}
    for row in references:
        if row["status"] != "complete":
            raise ValueError("Incomplete reference extraction metadata")
        require_hash(row["container_sha256"], 64)
        require_hash(row["pcm_sha256"], 64)
    declared: dict[str, set[str]] = {}
    for role, field in (("fitting", "train_ids"), ("selection", "validation_ids")):
        values = release["metadata"][field]
        normalized = []
        for value in values:
            if not isinstance(value, str) or re.fullmatch(r"(?:beatport:)?[0-9]+", value) is None:
                raise ValueError("Unexpected release ID namespace")
            normalized.append("beatport:" + str(int(value.split(":")[-1])))
        if not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("Empty or duplicate release IDs")
        declared[role] = set(normalized)
    if declared["fitting"] & declared["selection"]:
        raise ValueError("Release fitting/selection IDs overlap")
    exposed = declared["fitting"] | declared["selection"]
    if not exposed.issubset(by_id):
        raise ValueError("Release exposure not covered by reference metadata")
    expected = {by_id[i]["container_sha256"] for i in exposed}
    witness = release["metadata"]["training_audio_sha256"]
    if set(witness) != expected:
        raise ValueError("Release audio witness differs from declared IDs")
    role_hashes = {
        role: {
            field: {by_id[i][field] for i in names} for field in ("container_sha256", "pcm_sha256")
        }
        for role, names in declared.items()
    }
    rows = []
    for row in sorted(references, key=lambda r: r["ref_id"]):
        roles = {}
        for role, names in declared.items():
            evidence = []
            if row["id"] in names:
                evidence.append("declared_id")
            for field in ("container_sha256", "pcm_sha256"):
                if row[field] in role_hashes[role][field]:
                    evidence.append(field)
            if evidence:
                roles[role] = evidence
        if roles:
            rows.append(
                {
                    "ref_id": row["ref_id"],
                    "id": row["id"],
                    "domain": row["domain"],
                    "container_sha256": row["container_sha256"],
                    "pcm_sha256": row["pcm_sha256"],
                    "release_roles": roles,
                }
            )
    return {
        "checkpoint_sha256": CHECKPOINT_SHA,
        "declared_ids": {r: sorted(v) for r, v in declared.items()},
        "declared_counts": {r: len(v) for r, v in declared.items()},
        "known_exact_exposure_aliases": rows,
        "alias_scope": "Existing full reference metadata only; not exhaustive work/artist identity",
    }


def pinned(path: Path, expected: str) -> bytes:
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError("Pinned metadata changed: " + path.name)
    return raw


def manifest() -> dict[str, Any]:
    """Reproduce all metadata witnesses before creating any query-manifest bytes."""
    summary = report.summary()
    if summary["cohorts"]["named_filtered_intersection"] != {
        "recordings": 769,
        "matching_identities": 769,
        "parseable_wav": 769,
    }:
        raise ValueError("Terminal named filtered cohort differs")
    metadata, ledger = produce(DEFAULT / "sources")
    with tarfile.open(report.EVIDENCE) as archive:
        stream = archive.extractfile(report.RUNS[-1] + "/audio-integrity.json")
        if stream is None:
            raise ValueError("Missing terminal inventory")
        inventory = json.load(stream)
    if (inventory["archive_bytes"], inventory["archive_sha256"], inventory["source_url"]) != (
        ARCHIVE_BYTES,
        ARCHIVE_SHA256,
        URL,
    ):
        raise ValueError("Inventory archive provenance differs")
    selected = queries(ledger, inventory["records"])
    sensitivity = queries(
        ledger, inventory["records"], eligibility_field="raw_single_key_reference"
    )
    if len(sensitivity) != 837 or not {q.id for q in selected}.issubset(q.id for q in sensitivity):
        raise ValueError("Frozen sensitivity population differs")
    counts = dict(sorted(Counter(q.id.split(".")[0] for q in selected).items()))
    if len(selected) != 769 or counts.get("hiphop") != 73:
        raise ValueError("Frozen eligibility population differs")
    release = json.loads(pinned(RELEASE, RELEASE_SHA))
    raw = gzip.decompress(pinned(REFERENCES, REFERENCE_SHA))
    if digest(raw) != REFERENCE_DECODED_SHA:
        raise ValueError("Decoded reference metadata changed")
    reference = json.loads(raw)
    records = reference["records"]
    scopes = dict(Counter(r["domain"] for r in records))
    if (
        reference["references"] != 7579
        or reference["complete_coverage"] is not True
        or scopes != {"beatport": 1486, "fmak": 5489, "giantsteps": 604}
    ):
        raise ValueError("Full reference coverage differs")
    anchor = release_anchor(release, records)
    if anchor["declared_counts"] != {"fitting": 1237, "selection": 126}:
        raise ValueError("Released exposure counts differ")
    return {
        "schema": "gtzan-overlap-query-manifest-v1",
        "status": "FROZEN_METADATA_PREPARATION_NOT_OVERLAP_ADMITTED",
        "eligibility_rule": metadata["eligibility_rule"],
        "queries": [asdict(q) for q in selected],
        "query_count": len(selected),
        "sensitivity_queries": [asdict(q) for q in sensitivity],
        "sensitivity_query_count": len(sensitivity),
        "query_counts_by_genre": counts,
        "audio_archive": {"url": URL, "bytes": ARCHIVE_BYTES, "sha256": ARCHIVE_SHA256},
        "sources": {
            "audio_integrity_evidence_sha256": report.EVIDENCE_SHA,
            "release_metadata_sha256": RELEASE_SHA,
            "reference_result_gzip_sha256": REFERENCE_SHA,
            "reference_result_decoded_sha256": REFERENCE_DECODED_SHA,
            "eligibility": metadata["sources"],
            "implementation_sha256": {
                name: digest((ROOT / name).read_bytes())
                for name in (
                    "tools/prepare_gtzan_overlap.py",
                    "tools/import_gtzan_reference.py",
                    "tools/check_gtzan_audio_report.py",
                    "tools/check_gtzan_audio.py",
                )
            },
        },
        "release_exposure": anchor,
        "intended_reference_scopes": scopes,
        "reference_index_sha256": "b2609bd2b7184291e7604c74887feae6dd399c1bd4ca696c6f616ec27d155148",
        "fingerprint_admission": False,
        "overlap_cleared": False,
        "admitted_for_scoring": False,
        "inference_performed": False,
        "limitations": [
            "No query fingerprints or overlap results exist in this manifest",
            "Overlap outcomes do not alter the frozen primary769 or sensitivity837 cohorts",
            "Unresolved exposure limits independence claims, not fixed-benchmark arithmetic",
            "Old BP-to-GS review flags do not clear these GTZAN queries",
            "Reference roles are release-declared exposure, not current candidate fits",
            "No resource permit, audio acquisition or execution authority is granted",
        ],
    }


def compressed(value: object) -> bytes:
    """Serialize the unchanged JSON deterministically without filename or timestamp."""
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=buffer, mode="wb", mtime=0) as stream:
        stream.write(encode(value))
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    raw = compressed(manifest())
    if args.check:
        if args.output.read_bytes() != raw:
            raise ValueError("Frozen overlap query manifest changed")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(raw)
    print(
        json.dumps(
            {
                "queries": 769,
                "sha256": digest(raw),
                "decoded_sha256": digest(gzip.decompress(raw)),
                "admitted_for_scoring": False,
            }
        )
    )


if __name__ == "__main__":
    main()
