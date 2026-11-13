"""Label-blind GTZAN overlap worker; numerical fingerprint code stays frozen."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.bp_gs_scope import outcome
from experiments.fma_alignment_matcher import AllOffsetIndex
from experiments.fma_fingerprints import CONTRACT, chromaprint, decode, landmarks
from experiments.gtzan_overlap_continuation import cohort_ledger, population_status, prior_ledger
from tools.audit_bp_gs import FULL_SHA, INDEX_SHA, match_query, verify_dictionary
from tools.run_cqt_expansion import sha256, write_json

VERSION = "gtzan-overlap-v1"
ARCHIVE_SHA = "b28cc067ff6199bd826f5d1a6931458586d64acae7f44b2e882b7a97af057531"
MEMBER_LIMIT = 8 * 1024**2
LOCAL_ABSTENTIONS = {"vote_memory_budget", "query_budget_exhausted"}


def load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw)


def control_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [r for r in rows if r["panel"] != "negative"]
    negative = [r for r in rows if r["panel"] == "negative"]
    return dict(
        complete=all(r["complete"] for r in rows),
        expected_positive=len(positive),
        retrieved=sum(r["expected_retrieved"] for r in positive),
        negative_identity_flags=sum(r["identity_flagged"] for r in negative),
        negative_review_flags=sum(r["review_flagged"] for r in negative),
        passed=bool(positive and negative)
        and all(r["complete"] for r in rows)
        and all(r["expected_retrieved"] for r in positive)
        and not any(r["identity_flagged"] for r in negative),
    )


def owned_directory(path: Path) -> None:
    """Only a newly-created, process-owned directory may hold temporary audio."""
    path.mkdir(mode=0o700, exist_ok=False)
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise ValueError("Unsafe owned staging directory")


def save_features(samples: np.ndarray, vendor: str, path: Path) -> dict[str, Any]:
    began = time.monotonic()
    hashes = landmarks(samples, vendor, query=True)
    cp, step = chromaprint(samples)
    if path.exists() or path.is_symlink():
        raise ValueError("Query artifact already exists")
    np.savez_compressed(path, landmarks=hashes, chromaprint=cp, chromaprint_step=step)
    return dict(
        artifact=path.name,
        sha256=sha256(path),
        seconds=len(samples) / 11025,
        hashes=len(hashes),
        chromaprint_items=len(cp),
        extraction_seconds=time.monotonic() - began,
        query_shifts=4,
    )


def fresh_pcm(row: dict[str, Any], target: Path, ffmpeg: str) -> dict[str, Any]:
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-ss",
        str(row["offset_seconds"]),
        "-i",
        row["source_path"],
        "-t",
        "30",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "22050",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, timeout=180)
    if result.returncode != 0 or not 44 <= len(result.stdout) <= 2 * 1024**2:
        raise ValueError("Fresh-control PCM crop failed or exceeded bound")
    with target.open("xb") as stream:
        stream.write(result.stdout)
    return dict(command=command, wav_sha256=sha256(target), wav_bytes=target.stat().st_size)


def write_evidence(
    index: AllOffsetIndex,
    query: Path,
    references: Path,
    output: Path,
    number: int,
    identity: str,
    deadline: float,
    expected_ref: int | None = None,
    panel: str = "audit",
    expected_offset: float | None = None,
) -> dict[str, Any]:
    began = time.monotonic()
    matched, informative = match_query(index, query, references, deadline)
    raw = json.dumps(dict(id=identity, match=matched), sort_keys=True, allow_nan=False).encode()
    path = output / f"{number:06}.json.gz"
    if path.exists():
        raise ValueError("Evidence already exists")
    path.write_bytes(gzip.compress(raw, mtime=0))
    decisions = [r["decision"] for r in matched["matches"]]
    return dict(
        id=identity,
        panel=panel,
        complete=matched["complete"],
        reason=matched.get("reason"),
        informative=informative,
        expected_ref=expected_ref,
        expected_retrieved=any(
            r["ref_id"] == expected_ref
            and r["decision"] == "identity_evidence"
            and (expected_offset is None or abs(r["offset_seconds"] - expected_offset) <= 0.25)
            for r in matched["matches"]
        ),
        identity_flagged="identity_evidence" in decisions,
        review_flagged="review" in decisions,
        outcome=outcome(
            exact=False, complete=matched["complete"], informative=informative, decisions=decisions
        ),
        evidence=path.name,
        evidence_sha256=sha256(path),
        decoded_sha256=hashlib.sha256(raw).hexdigest(),
        query_sha256=sha256(query),
        alignments=len(matched["matches"]),
        match_seconds=time.monotonic() - began,
    )


def validate_plan(plan: dict[str, Any], mode: str) -> None:
    if plan["version"] != VERSION or mode not in {"qualification", "audit"}:
        raise ValueError("Unknown plan or mode")
    if plan["index_sha256"] != INDEX_SHA or plan["descriptor_sha256"] != FULL_SHA:
        raise ValueError("Reference identity differs")
    if len(plan["controls"]) != 244 or len({r["id"] for r in plan["controls"]}) != 244:
        raise ValueError("Require original244 fixed controls")
    if sum(r["panel"] == "negative" for r in plan["controls"]) != 64:
        raise ValueError("Require original180 positive/64 negative controls")
    fresh = plan["fresh_controls"]
    if not fresh or len(fresh) > 24 or len({r["id"] for r in fresh}) != len(fresh):
        raise ValueError("Require bounded fixed fresh controls")
    if any(r["duration_seconds"] != 30 or r["offset_seconds"] <= 0 for r in fresh):
        raise ValueError("Fresh controls must be30-second interior excerpts")
    if {r["domain"] for r in fresh} != {"beatport", "fmak", "giantsteps"}:
        raise ValueError("Fresh controls must exercise all three reference domains")
    cohort = plan.get("cohort", [])
    if len(cohort) != 837 or len({r["id"] for r in cohort}) != 837:
        raise ValueError("Require frozen837 sensitivity union")
    primary = plan["primary_ids"]
    if (
        len(primary) != 769
        or len(set(primary)) != 769
        or not set(primary) <= {r["id"] for r in cohort}
    ):
        raise ValueError("Require frozen769 primary membership within837")
    members = [r["member"] for r in cohort]
    if len(set(members)) != 837 or any(
        not name.startswith("genres/") or ".." in Path(name).parts for name in members
    ):
        raise ValueError("Unsafe or repeated cohort member")
    if any("jazz.00054" in name for name in members):
        raise ValueError("Prospectively excluded recording in cohort")
    # No labels, logits, predictions or checkpoint/model objects belong in this worker.
    forbidden = {"labels", "key", "target", "predictions", "logits"}
    if any(forbidden & set(row) for row in cohort + fresh):
        raise ValueError("Label/prediction fields forbidden in worker rows")


def run(plan: dict[str, Any], config: dict[str, Any], output: Path) -> None:
    mode = config["mode"]
    validate_plan(plan, mode)
    if output.is_symlink() or output.stat().st_uid != os.getuid():
        raise ValueError("Unsafe output owner")
    stage, queries, evidence = output / "stage", output / "queries", output / "evidence"
    for directory in (stage, queries, evidence):
        owned_directory(directory)
    root = Path(plan["cache_root"])
    descriptor_path = root / "references-v1/result.json"
    if sha256(descriptor_path) != FULL_SHA:
        raise ValueError("Reference descriptor differs")
    fp = root / "references-v1/fingerprints"
    index_path = root / "references-v1/index.sqlite"
    index = AllOffsetIndex.from_verified_artifacts(
        index_path,
        descriptor=load(descriptor_path),
        query_extraction_contract=CONTRACT,
        expected_index_sha256=INDEX_SHA,
        fingerprint_root=fp,
        artifact_sha256=plan["artifact_sha256"],
    )
    completed: list[dict[str, Any]] = []
    unvisited = {row["id"] for row in plan["cohort"]}
    if config.get("continuation"):
        if mode != "audit":
            raise ValueError("Continuation is audit-only")
        try:
            completed, unvisited = prior_ledger(plan, config)
        except BaseException:
            index.db.close()
            raise
        write_json(
            output / "imported-prefix.json",
            dict(
                results=completed,
                unvisited_ids=[row["id"] for row in plan["cohort"] if row["id"] in unvisited],
            ),
        )
        write_json(
            output / "partial.json",
            dict(
                results=completed,
                remaining_are_not_negatives=True,
                cohort_ledger=cohort_ledger(plan, completed),
                population_status=population_status(plan, completed),
            ),
        )
    began = time.monotonic()

    def match(
        path: Path,
        identity: str,
        panel: str,
        expected: int | None = None,
        expected_offset: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        deadline = min(config["deadline_epoch"], time.time() + config["query_seconds"])
        # Matcher uses monotonic time, permit uses UTC.
        monotonic_deadline = time.monotonic() + max(0, deadline - time.time())
        row = write_evidence(
            index,
            path,
            fp,
            evidence,
            len(completed),
            identity,
            monotonic_deadline,
            expected,
            panel,
            expected_offset,
        )
        row.update(metadata or {})
        completed.append(row)
        write_json(
            output / "partial.json",
            dict(
                results=completed,
                remaining_are_not_negatives=True,
                cohort_ledger=cohort_ledger(plan, completed) if mode == "audit" else None,
                population_status=population_status(plan, completed) if mode == "audit" else None,
            ),
        )
        if mode == "qualification" and not row["complete"]:
            raise RuntimeError("Incomplete query; remaining queries are unresolved")
        if mode == "audit" and not row["complete"] and row["reason"] not in LOCAL_ABSTENTIONS:
            raise RuntimeError("Unknown/nonlocal incomplete-query failure; stop audit")

    try:
        verify_dictionary(index, plan["dictionary"])
        if mode == "qualification":
            for row in plan["controls"]:
                path = Path(row["path"])
                if (
                    not path.resolve().is_relative_to(root.resolve())
                    or sha256(path) != row["sha256"]
                ):
                    raise ValueError("Cached control path/hash differs")
                match(path, row["id"], row["panel"], row["expected_ref"])
            historical = control_summary(completed)
            write_json(output / "cached-controls.json", historical)
            if not historical["passed"]:
                raise ValueError("Cached operational control gate failed")
            for number, row in enumerate(plan["fresh_controls"]):
                path = Path(row["source_path"])
                if sha256(path) != row["source_container_sha256"]:
                    raise ValueError("Fresh control source differs")
                crop_path = stage / "fresh.wav"
                crop_began = time.monotonic()
                crop_receipt = fresh_pcm(row, crop_path, config["ffmpeg_path"])
                try:
                    samples, _ = decode(str(crop_path))
                finally:
                    crop_path.unlink()
                if abs(len(samples) / 11025 - 30) > 1 / 11025:
                    raise ValueError("Fresh control does not contain complete30seconds")
                query = queries / f"{number:06}.npz"
                feature = save_features(samples, plan["vendor_path"], query)
                feature["crop_receipt"] = crop_receipt
                feature["full_extraction_seconds"] = time.monotonic() - crop_began
                match(query, row["id"], "fresh", row["expected_ref"], row["offset_seconds"])
                completed[-1]["extraction"] = feature
            fresh_results = completed[244:]
            passed = all(r["expected_retrieved"] and r["complete"] for r in fresh_results)
            if not passed:
                raise ValueError("Fresh30-second extraction controls failed")
        else:
            archive_path = Path(plan["archive_path"])
            if sha256(archive_path) != ARCHIVE_SHA or plan["archive_sha256"] != ARCHIVE_SHA:
                raise ValueError("GTZAN archive differs")
            members = {r["member"]: r for r in plan["cohort"] if r["id"] in unvisited}
            all_members = {r["member"]: r["id"] for r in plan["cohort"]}
            observed_order: list[str] = []
            seen: set[str] = set()
            with tarfile.open(archive_path, "r|gz") as archive:
                for member in archive:
                    if config.get("continuation") and member.name in all_members:
                        observed_order.append(all_members[member.name])
                        if (
                            observed_order[-1]
                            != config["continuation"]["archive_order_ids"][len(observed_order) - 1]
                        ):
                            raise ValueError(
                                "Actual archive order differs from the frozen continuation"
                            )
                    if member.name not in members:
                        continue
                    if (
                        member.name in seen
                        or not member.isfile()
                        or not 0 < member.size <= MEMBER_LIMIT
                    ):
                        raise ValueError("Repeated/unsafe selected archive member")
                    seen.add(member.name)
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError("Missing selected member payload")
                    raw = stream.read(MEMBER_LIMIT + 1)
                    row = members[member.name]
                    if (
                        len(raw) != member.size
                        or hashlib.sha256(raw).hexdigest() != row["container_sha256"]
                    ):
                        raise ValueError("Selected archive member differs")
                    pcm_path = stage / "query.wav"
                    with pcm_path.open("xb") as target:
                        target.write(raw)
                    query = queries / f"{len(completed):06}.npz"
                    try:
                        samples, pcm_sha = decode(str(pcm_path))
                        feature = save_features(samples, plan["vendor_path"], query)
                        feature["pcm_sha256"] = pcm_sha
                        feature["container_sha256"] = row["container_sha256"]
                        match(
                            query,
                            row["id"],
                            "audit",
                            metadata=dict(
                                extraction=feature,
                                primary769=row["id"] in set(plan["primary_ids"]),
                                provenance="current_audit",
                            ),
                        )
                        del samples
                    finally:
                        pcm_path.unlink()
            if seen != set(members):
                raise ValueError("Missing frozen cohort members")
            if (
                config.get("continuation")
                and observed_order != config["continuation"]["archive_order_ids"]
            ):
                raise ValueError("Incomplete archive-order population")
        if sha256(index_path) != INDEX_SHA:
            raise ValueError("Reference index changed")
        write_json(
            output / "result.json",
            dict(
                version=VERSION,
                mode=mode,
                status="complete"
                if all(row["complete"] for row in completed)
                else "complete_with_unresolved",
                all_queries_attempted=True,
                all_queries_complete=all(row["complete"] for row in completed),
                population_status=population_status(plan, completed) if mode == "audit" else None,
                cohort_ledger=cohort_ledger(plan, completed) if mode == "audit" else None,
                results=completed,
                wall_seconds=time.monotonic() - began,
                index_sha256=INDEX_SHA,
                classifier_outputs_used=False,
                labels_scored=False,
                qualification_passed=mode == "qualification",
                qualification_binding_sha256=config["qualification_binding_sha256"],
                admitted_records=[],
                primary_ids=plan["primary_ids"],
            ),
        )
    finally:
        index.db.close()
