"""Metadata-only cached-PCM joins against a proposed role ledger; never changes roles."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.audit_bp_gs import FULL_SHA, PREPARED_SHA, digest, normalize, read


def audit(
    references: Path, ledger: Path, proposal: Path, prepared: Path, release: Path
) -> dict[str, Any]:
    raw = gzip.decompress(references.read_bytes())
    if hashlib.sha256(raw).hexdigest() != FULL_SHA or digest(prepared) != PREPARED_SHA:
        raise ValueError("Frozen descriptor/prepared identity differs")
    records = json.loads(raw)["records"]
    roles = list(csv.DictReader(io.StringIO(gzip.decompress(ledger.read_bytes()).decode())))
    by_id = {int(row["id"].split(":")[1]): row for row in roles}
    fma = {int(row["id"].split(":")[1]): row for row in records if row["domain"] == "fmak"}
    if len(roles) != len(by_id) or len(by_id) != 5489 or set(fma) != set(by_id):
        raise ValueError("Complete unique FMA role/descriptor join required")
    expected = {
        "PROPOSED_expanded_fit",
        "PROPOSED_inner_selection",
        "PROPOSED_calibration",
        "PROPOSED_development_confirmation",
    }
    if {row["proposed_role"] for row in roles} != expected:
        raise ValueError("Unreviewed proposed role schema")
    families: dict[str, list[int]] = defaultdict(list)
    for identity, row in fma.items():
        families[row["pcm_sha256"]].append(identity)
    if any(not row.get("component") for row in roles):
        raise ValueError("Saved component identity is required for PCM-family diagnostics")
    duplicate_families: list[dict[str, Any]] = [
        dict(
            pcm_sha256=pcm,
            proposed_roles=sorted({by_id[i]["proposed_role"] for i in ids}),
            saved_components=sorted({by_id[i]["component"] for i in ids}),
            bridges_saved_components=len({by_id[i]["component"] for i in ids}) > 1,
            records=[by_id[i] for i in sorted(ids)],
        )
        for pcm, ids in sorted(families.items())
        if len(ids) > 1
    ]
    cross_roles = []
    for pcm, ids in sorted(families.items()):
        if len({by_id[identity]["proposed_role"] for identity in ids}) > 1:
            cross_roles.append(
                dict(pcm_sha256=pcm, records=[by_id[identity] for identity in sorted(ids)])
            )
    bp = {row["id"]: row for row in records if row["domain"] == "beatport"}
    cohorts = dict(
        current_fitting={
            row["id"]
            for row in read(prepared)["rows"]
            if row["domain"] == "beatport" and row["role"] == "train"
        },
        release_fitting=normalize(read(release)["metadata"]["train_ids"]),
    )
    cross_domain = {}
    for name, ids in cohorts.items():
        joined = []
        for identity in sorted(ids):
            pcm = bp[identity]["pcm_sha256"]
            matches = [
                by_id[i]
                for i in families.get(pcm, [])
                if by_id[i]["proposed_role"] != "PROPOSED_expanded_fit"
            ]
            if matches:
                joined.append(
                    dict(bp_id=identity, pcm_sha256=pcm, proposed_fma_evaluation_records=matches)
                )
        cross_domain[name] = dict(fitting_ids=len(ids), matching_bp_ids=len(joined), matches=joined)
    return dict(
        version="bp-gs-cached-pcm-role-check-v1",
        reference_result_decoded_sha256=FULL_SHA,
        inputs={
            name: dict(path=str(path), sha256=digest(path))
            for name, path in (
                ("references", references),
                ("ledger", ledger),
                ("proposal", proposal),
                ("prepared", prepared),
                ("release", release),
            )
        },
        proposal_status=read(proposal)["status"],
        fma_records=5489,
        fma_unique_cached_pcm_hashes=len(families),
        fma_role_counts={
            role: dict(
                rows=sum(row["proposed_role"] == role for row in roles),
                unique_cached_pcm_hashes=len(
                    {
                        fma[identity]["pcm_sha256"]
                        for identity, row in by_id.items()
                        if row["proposed_role"] == role
                    }
                ),
            )
            for role in sorted(expected)
        },
        pcm_hash_contract=dict(
            sample_rate_hz=11025,
            channels=1,
            dtype="little-endian float32 (<f4)",
            hash="SHA256 of exact ffmpeg stdout bytes",
            decode_path="ffmpeg -nostdin -hide_banner -loglevel error -threads 1 -filter_threads 1 -i SOURCE -t 3601 -vn -ac 1 -ar 11025 -f f32le -",
            coverage_guard="Reject output exceeding 3600 seconds; no silent truncation",
            frozen_engine_sha256="a0d315f3f6f6f62b33b7453119350807702eec35a17723b9079fca30c4328f71",
            canonical_22050_model_input=False,
            limitation="Equality after mono downmix/downsampling is not equality of original stereo or canonical model-input samples",
        ),
        within_role_duplicate_pcm_families=duplicate_families,
        within_role_duplicate_pcm_summary={
            role: dict(
                families=sum(family["proposed_roles"] == [role] for family in duplicate_families),
                excess_rows=sum(
                    len(family["records"]) - 1
                    for family in duplicate_families
                    if family["proposed_roles"] == [role]
                ),
                bridges_saved_components=sum(
                    family["bridges_saved_components"]
                    for family in duplicate_families
                    if family["proposed_roles"] == [role]
                ),
            )
            for role in sorted(expected)
        },
        fma_cross_role_pcm_groups=len(cross_roles),
        fma_cross_role_groups=cross_roles,
        bp_fitting_to_fma_evaluation=cross_domain,
        labels_read_or_scored=False,
        audio_read=False,
        roles_changed=False,
        fingerprint_queries_performed=False,
        limitations="Exact cached decoded-PCM identity only; no near-duplicate/work clearance. Proposed rows are reported without changing eligibility or unioning components.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("references", "ledger", "proposal", "prepared", "release", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.references, args.ledger, args.proposal, args.prepared, args.release)
    result["source_sha256"] = digest(Path(__file__))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key
                not in ("fma_cross_role_groups", "within_role_duplicate_pcm_families", "inputs")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
