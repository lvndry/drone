"""Reproduce recording-integrity counts from metadata; never opens audio."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

from tools.check_gtzan_audio import ARCHIVE_BYTES, ARCHIVE_SHA256, expected_members
from tools.import_gtzan_reference import DEFAULT, produce

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/studies/v2-data-contract/data/gtzan-audio-integrity/evidence.tar.gz"
EVIDENCE_SHA = "0c3b991306d84d3e9c9b90be55c9b22e83d6c41a4750245b6718619f2fa0ffb0"
RUNS = (
    "gtzan-audio-integrity-20261113-v1",
    "gtzan-audio-integrity-audit-20261113-v2",
    "gtzan-audio-integrity-inventory-20261113-v3",
)


def summary() -> dict[str, Any]:
    if hashlib.sha256(EVIDENCE.read_bytes()).hexdigest() != EVIDENCE_SHA:
        raise ValueError("Terminal evidence archive changed")
    with tarfile.open(EVIDENCE) as archive:

        def read(name: str) -> dict[str, Any]:
            stream = archive.extractfile(name)
            if stream is None:
                raise ValueError("Missing terminal evidence: " + name)
            return json.load(stream)

        download = read(RUNS[0] + "/download.json")
        if (download["bytes"], download["sha256"]) != (ARCHIVE_BYTES, ARCHIVE_SHA256):
            raise ValueError("Acquired archive does not match publisher pin")
        controllers = [read(run + "/controller.json") for run in RUNS]
        if [c["state"] for c in controllers] != [
            "FAILED",
            "FAILED",
            "COMPLETE_DIAGNOSTIC_INVENTORY_NOT_STRICT_PASS",
        ]:
            raise ValueError("Do not discard or reinterpret failed audits")
        result = read(RUNS[-1] + "/audio-integrity.json")
    if result["mode"] != "diagnostic_inventory" or result["admitted_for_scoring"] is not False:
        raise ValueError("Inventory is not a strict audit pass or scoring admission")
    records = result["records"]
    expected = expected_members()
    by_member = {r["member"]: r for r in records}
    if len(records) != 1000 or set(by_member) != set(expected):
        raise ValueError("Incomplete or duplicate inventory")
    for name, row in by_member.items():
        if (
            row["expected_md5"] != expected[name]
            or row["member_identity_verified"] != (row["md5"] == expected[name])
            or row["overlap_cleared"] is not False
            or row["admitted_for_scoring"] is not False
        ):
            raise ValueError("Inconsistent identity or admission record")
    _, ledger = produce(DEFAULT / "sources")
    cohorts = {}
    for field in ("raw_single_key_reference", "named_filtered_intersection"):
        selected = [by_member[r["audio_member"]] for r in ledger if r[field]]
        cohorts[field] = {
            "recordings": len(selected),
            "matching_identities": sum(r["member_identity_verified"] for r in selected),
            "parseable_wav": sum("wave" in r for r in selected),
        }
    return {
        "total_recordings": len(records),
        "matching_identities": sum(r["member_identity_verified"] for r in records),
        "parseable_wav": sum("wave" in r for r in records),
        "identity_mismatches": [r["member"] for r in records if not r["member_identity_verified"]],
        "format_faults": [r["member"] for r in records if "wave_error" in r],
        "cohorts": cohorts,
        "controllers": controllers,
        "download_seconds": download["seconds"],
        "admitted_for_scoring": False,
        "inference_performed": False,
    }


if __name__ == "__main__":
    print(json.dumps(summary(), indent=2, sort_keys=True))
