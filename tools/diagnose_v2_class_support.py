"""Post-assignment metadata diagnostic; never assigns or repairs component roles."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import re
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def diagnose(
    prepared: Path,
    assignment: bytes,
    ledger: bytes,
    expected_ledger_sha256: str,
    expected_metadata_sha256: str,
) -> dict[str, Any]:
    if hashlib.sha256(ledger).hexdigest() != expected_ledger_sha256:
        raise ValueError("Before-label assignment witness mismatch")
    rows = json.loads(gzip.decompress(assignment))["tracks"]
    roles = {
        r["id"]: r["proposed_role"]
        for r in csv.DictReader(io.StringIO(gzip.decompress(ledger).decode()))
    }
    if set(roles) != {r["identity"] for r in rows}:
        raise ValueError("Assignment membership mismatch")
    with zipfile.ZipFile(prepared) as archive:
        raw = archive.read("metadata.npy")
    if hashlib.sha256(raw).hexdigest() != expected_metadata_sha256:
        raise ValueError("Prepared metadata source hash changed")
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("Expected NPY metadata")
    start = 10 if raw[6] == 1 else 12
    size = struct.unpack("<H" if start == 10 else "<I", raw[8:start])[0]
    header = ast.literal_eval(raw[start : start + size].decode())
    if header["shape"] != () or not header["descr"].startswith("<U"):
        raise ValueError("Expected scalar little-endian Unicode metadata")
    text = raw[start + size :].decode("utf-32-le").rstrip("\0")

    def field(name: str) -> list[Any]:
        match = re.search('"' + name + r'"\s*:', text)
        if match is None:
            raise ValueError(f"Missing metadata field {name}")
        return json.JSONDecoder().raw_decode(text[match.end() :].lstrip())[0]

    identities, refs = field("ids"), field("refs")
    if len(identities) != len(refs) or any(type(v) is not int or not 0 <= v < 24 for v in refs):
        raise ValueError("Expected canonical24 integer labels")
    labels = {"fmak:" + i: v for i, v in zip(identities, refs, strict=True)}
    if set(labels) != set(roles):
        raise ValueError("Prepared metadata membership mismatch")
    hash_labels: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        hash_labels[row["audio_sha256"]].add(labels[row["identity"]])
    conflicts = sorted(k for k, v in hash_labels.items() if len(v) > 1)
    totals: dict[str, Counter[int]] = defaultdict(Counter)
    unique: dict[str, Counter[int]] = defaultdict(Counter)
    eligible_raw: dict[str, Counter[int]] = defaultdict(Counter)
    seen: dict[str, set[str]] = defaultdict(set)
    raw_hashes: dict[str, set[str]] = defaultdict(set)
    for r in sorted(rows, key=lambda r: (r["old_role"] != "fit", r["identity"])):
        identity = r["identity"]
        role = roles[identity]
        label = labels[identity]
        digest = r["audio_sha256"]
        panels = [role] + (["restricted_fit"] if r["old_role"] == "fit" else [])
        for panel in panels:
            totals[panel][label] += 1
            raw_hashes[panel].add(digest)
            if digest in conflicts:
                continue
            eligible_raw[panel][label] += 1
            if digest not in seen[panel]:
                unique[panel][label] += 1
                seen[panel].add(digest)
    return {
        "status": "PROPOSED_POST_ASSIGNMENT_SOURCE_INTEGRITY_OVERLAY",
        "eligibility_rule": "quarantine_all_records_in_any_conflicting_exact_container_SHA_group_across_all_FMA_roles",
        "no_replacements_or_role_reassignment": True,
        "expected_before_label_ledger_sha256": expected_ledger_sha256,
        "assignment_compressed_sha256": hashlib.sha256(assignment).hexdigest(),
        "prepared_metadata_npy_sha256": hashlib.sha256(raw).hexdigest(),
        "prepared_path": str(prepared),
        "labels_used_for_assignment": False,
        "label_schema": "tonic-times-two-plus-minor-v1",
        "label_provenance": "Inherited prepared FMAK-v2 annotation targets;not human-consensus proof",
        "conflicting_saved_audio_sha256_count": len(conflicts),
        "conflicting_saved_audio_sha256": conflicts,
        "conflict_records": [
            {
                "id": r["identity"],
                "old_role": r["old_role"],
                "proposed_role": roles[r["identity"]],
                "component": r["component"],
                "reason": "conflicting_keys_for_identical_container_SHA",
            }
            for r in rows
            if r["audio_sha256"] in conflicts
        ],
        "panels": {
            k: {
                "raw_counts24": [totals[k][i] for i in range(24)],
                "eligible_raw_counts24": [eligible_raw[k][i] for i in range(24)],
                "eligible_unique_counts24": [unique[k][i] for i in range(24)],
                "raw_rows": sum(totals[k].values()),
                "raw_unique_audio": len(raw_hashes[k]),
                "eligible_raw_rows": sum(eligible_raw[k].values()),
                "eligible_unique_audio": len(seen[k]),
                "missing_classes_unique": [i for i in range(24) if not unique[k][i]],
                "minimum_unique_class_support": min(unique[k][i] for i in range(24)),
            }
            for k in sorted(totals)
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def write_overlay(
    assignment: bytes, ledger: bytes, diagnostic: dict[str, Any], output: Path
) -> None:
    """Full-row proposed eligibility ledger; assignment remains byte-for-byte unchanged."""
    rows = json.loads(gzip.decompress(assignment))["tracks"]
    roles = {
        r["id"]: r["proposed_role"]
        for r in csv.DictReader(io.StringIO(gzip.decompress(ledger).decode()))
    }
    conflicts = set(diagnostic["conflicting_saved_audio_sha256"])
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["id", "old_role", "proposed_role", "source_integrity_status", "reason"])
    for row in sorted(rows, key=lambda r: r["identity"]):
        flagged = row["audio_sha256"] in conflicts
        writer.writerow(
            [
                row["identity"],
                row["old_role"],
                roles[row["identity"]],
                "PROPOSED_QUARANTINE" if flagged else "PROPOSED_SOURCE_CHECK_ELIGIBLE",
                "conflicting_keys_for_identical_container_SHA"
                if flagged
                else "no_observed_exact_SHA_label_conflict",
            ]
        )
    output.write_bytes(gzip.compress(stream.getvalue().encode(), mtime=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--expected-ledger-sha256", required=True)
    parser.add_argument("--expected-metadata-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay-output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(
        args.prepared,
        args.input.read_bytes(),
        args.ledger.read_bytes(),
        args.expected_ledger_sha256,
        args.expected_metadata_sha256,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_overlay(args.input.read_bytes(), args.ledger.read_bytes(), result, args.overlay_output)


if __name__ == "__main__":
    main()
