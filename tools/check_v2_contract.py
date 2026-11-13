"""Reproduce the metadata proposal and verify its separate eligibility overlay."""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.diagnose_v2_class_support import write_overlay
from tools.propose_v2_contract import describe, load_input, proposal, sha

STUDY = Path(__file__).resolve().parents[1] / "experiments/studies/v2-data-contract"


def amended_hash(data: Path, artifact: str, pinned: str, current: str) -> bool:
    """Verify witness-bound bytes and every append-only protocol transition.

    Receipt assertions describe the amendment; snapshot hashes and byte prefixes
    establish that the original before-label protocol was retained unchanged.
    """
    chain = json.loads((data / "protocol-amendments.json").read_text())
    if artifact != "PROTOCOL.md" or chain.get("assignment_artifacts_changed") is not False:
        return False
    study = data.parent.resolve()

    def snapshot(relative: str) -> bytes:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative:
            raise ValueError("Protocol snapshot must be study-relative")
        resolved = (study / path).resolve()
        if not resolved.is_relative_to(study):
            raise ValueError("Protocol snapshot escapes study")
        return resolved.read_bytes()

    digest = pinned
    previous: bytes | None = None
    steps = chain["amendments"]
    if not steps:
        return False
    for step in steps:
        if (
            step["artifact"] != artifact
            or step["from_sha256"] != digest
            or step.get("fma_assignment_artifacts_changed") is not False
        ):
            return False
        before = snapshot(step["from_snapshot"])
        after = snapshot(step["to_snapshot"])
        if (
            sha(before) != digest
            or sha(after) != step["to_sha256"]
            or (previous is not None and before != previous)
            or not after.startswith(before)
        ):
            return False
        previous = after
        digest = step["to_sha256"]
    return digest == current and previous == (study / artifact).read_bytes()


def check(study: Path = STUDY) -> dict[str, Any]:
    data = study / "data"
    tracks, provenance = load_input(data / "input.json.gz")
    proposed, ledger = proposal(tracks, provenance)
    if json.loads((data / "proposal.json").read_text()) != proposed:
        raise ValueError("Proposal does not reproduce")
    if (data / "exposure-ledger.csv.gz").read_bytes() != ledger:
        raise ValueError("Exposure ledger does not reproduce")
    witness = json.loads((data / "before-label-witness.json").read_text())
    for relative, digest in witness["artifacts"].items():
        if sha((study / relative).read_bytes()) != digest:
            if relative == "PROTOCOL.md" and amended_hash(
                data, relative, digest, sha((study / relative).read_bytes())
            ):
                continue
            if relative != "data/input.json.gz":
                raise ValueError("Before-label assignment artifact changed")
            sanitized = json.loads((data / "input-provenance-sanitization.json").read_text())
            if digest != sanitized["before_label_original_compressed_sha256"]:
                raise ValueError("Original before-label input witness differs")
            current = (data / "input.json.gz").read_bytes()
            payload = json.loads(gzip.decompress(current))
            track_digest = sha(
                json.dumps(payload["tracks"], sort_keys=True, separators=(",", ":")).encode()
            )
            if (
                sha(current) != sanitized["published_compressed_sha256"]
                or track_digest != sanitized["canonical_tracks_sha256_before_and_after"]
            ):
                raise ValueError("Sanitized track/source proof differs")
            if any("headers" in row for row in payload["provenance"]["bp_metadata_downloads"]):
                raise ValueError("HTTP response headers must not be published")
    diagnostic = json.loads((data / "class-support.json").read_text())
    if diagnostic["expected_before_label_ledger_sha256"] != sha(ledger):
        raise ValueError("Class support is not bound to fixed assignment")
    if diagnostic["labels_used_for_assignment"]:
        raise ValueError("Unexpected label-informed assignment")
    if diagnostic["source_sha256"] != sha(
        Path(__file__).with_name("diagnose_v2_class_support.py").read_bytes()
    ):
        raise ValueError("Diagnostic source identity changed")
    if (
        diagnostic["prepared_metadata_npy_sha256"]
        != "94f65bb54329b239028dfffc1e8bf37f7d037f6e1e7c25f6721e73efd2c037bc"
    ):
        raise ValueError("Unexpected prepared metadata source")
    role_rows = list(csv.DictReader(io.StringIO(gzip.decompress(ledger).decode())))
    overlay = list(
        csv.DictReader(
            io.StringIO(
                gzip.decompress((data / "eligibility-overlay.csv.gz").read_bytes()).decode()
            )
        )
    )
    if {r["id"] for r in role_rows} != {r["id"] for r in overlay}:
        raise ValueError("Overlay coverage differs")
    raw_by_id = {r["id"]: r for r in role_rows}
    conflicts = set(diagnostic["conflicting_saved_audio_sha256"])
    track_by_id = {t.identity: t for t in tracks}
    links: dict[str, dict[str, set[str]]] = {
        k: defaultdict(set) for k in ("artist", "album", "audio")
    }
    for row in overlay:
        original = raw_by_id[row["id"]]
        if any(row[k] != original[k] for k in ("old_role", "proposed_role")):
            raise ValueError("Overlay reassigned a role")
        track = track_by_id[row["id"]]
        flagged = track.audio_sha256 in conflicts
        if (row["source_integrity_status"] == "PROPOSED_QUARANTINE") != flagged:
            raise ValueError("Overlay does not apply general conflict rule")
        for kind, identity in (
            ("artist", track.artist_id),
            ("album", track.album_id),
            ("audio", track.audio_sha256),
        ):
            if identity:
                links[kind][identity].add(row["proposed_role"])
    violations = {k: sum(len(v) > 1 for v in values.values()) for k, values in links.items()}
    if any(violations.values()):
        raise ValueError("Observed positive-ID or saved-audio cross-role violation")
    # Recreate the overlay without reading labels again or touching the frozen study.
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "overlay.gz"
        write_overlay((data / "input.json.gz").read_bytes(), ledger, diagnostic, target)
        if target.read_bytes() != (data / "eligibility-overlay.csv.gz").read_bytes():
            raise ValueError("Eligibility overlay does not reproduce")
    eligible_panels = {}
    for role in sorted({r["proposed_role"] for r in role_rows}):
        panel_rows = [
            track_by_id[r["id"]]
            for r in overlay
            if r["proposed_role"] == role and r["source_integrity_status"] != "PROPOSED_QUARANTINE"
        ]
        eligible_panels[role] = describe(panel_rows)
        class_counts = diagnostic["panels"][role]
        if sum(class_counts["eligible_raw_counts24"]) != len(panel_rows):
            raise ValueError("Diagnostic eligible row count differs")
        if sum(class_counts["eligible_unique_counts24"]) != len(
            {r.audio_sha256 for r in panel_rows}
        ):
            raise ValueError("Diagnostic eligible unique count differs")
    return {
        "eligible_panels": eligible_panels,
        "metadata_checks_passed": True,
        "training_authorized": False,
        "rows": len(tracks),
        "ledger_sha256": sha(ledger),
        "quarantined_rows": sum(
            r["source_integrity_status"] == "PROPOSED_QUARANTINE" for r in overlay
        ),
        "old_fit_quarantined": sum(
            r["old_role"] == "fit" and r["source_integrity_status"] == "PROPOSED_QUARANTINE"
            for r in overlay
        ),
        "observed_cross_role_violations": violations,
        "old_roles": dict(Counter(t.old_role for t in tracks)),
    }


def main() -> None:
    print(json.dumps(check(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
