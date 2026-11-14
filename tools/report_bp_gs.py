"""Summarize frozen cached-query evidence; never matches audio or changes outcomes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    return gzip.decompress(raw) if path.suffix == ".gz" else raw


def read_payload(path: Path) -> dict[str, Any]:
    return json.loads(payload_bytes(path))


def review_category(match: dict[str, Any]) -> str:
    tier = (
        "identity_landmarks"
        if match["count"] >= 20 and match["span_seconds"] >= 8
        else "review_landmarks"
    )
    chromaprint_distance = match["chromaprint_distance"]
    corroboration = (
        "missing"
        if chromaprint_distance is None
        else "corroborates"
        if chromaprint_distance <= 0.15
        else "disagrees"
    )
    return tier + ":" + corroboration


def summarize(plan: dict[str, Any], result: dict[str, Any], evidence: Path) -> dict[str, Any]:
    rows = [row for row in result["results"] if row["kind"] == "bp-audit"]
    if (
        result["status"] != "complete"
        or len(rows) != 1486
        or any(not row["complete"] for row in rows)
    ):
        raise ValueError("Only the complete fixed audit can produce this report")
    queries = {row["record"]["id"]: row for row in plan["queries"]}
    if {row["id"] for row in rows} != set(queries):
        raise ValueError("Scientific population differs")
    targets = {row["ref_id"]: row["id"] for row in plan["gs_records"]}
    outcomes = Counter(row["outcome"] for row in rows)
    exposure = {}
    for model in plan["model_witnesses"]:
        name = model["model_id"]
        counts: dict[str, Any] = {"checkpoint_sha256": model["checkpoint_sha256"]}
        for state in (
            "definite_detected_overlap",
            "review",
            "insufficient_information",
            "no_detected_overlap",
        ):
            for role in ("fitting", "validation"):
                for relation in ("direct_id", "container_alias_only", "pcm_alias_only"):
                    selected = []
                    for row in rows:
                        witness = queries[row["id"]]["model_exposure"][name][role]
                        present = (
                            witness["direct_id"]
                            if relation == "direct_id"
                            else not witness["direct_id"]
                            and bool(
                                witness["container_aliases"]
                                if relation == "container_alias_only"
                                else witness["pcm_aliases"]
                            )
                        )
                        if row["outcome"] == state and present:
                            selected.append(row["id"])
                    counts[f"{state}:{role}:{relation}"] = dict(
                        count=len(selected), ids=sorted(selected)
                    )
        exposure[name] = counts
    categories: Counter[str] = Counter()
    category_queries: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pairs: set[tuple[str, str]] = set()
    all_alignments = 0
    review_alignments = 0
    for row in sorted(rows, key=lambda value: value["id"]):
        path = evidence / row["evidence"]
        compressed = path.read_bytes()
        if hashlib.sha256(compressed).hexdigest() != row["evidence_sha256"]:
            raise ValueError("Query evidence SHA differs")
        raw = gzip.decompress(compressed)
        if hashlib.sha256(raw).hexdigest() != row["decoded_sha256"]:
            raise ValueError("Decoded evidence SHA differs")
        matches = json.loads(raw)["match"]["matches"]
        all_alignments += len(matches)
        for match in matches:
            if match["ref_id"] not in targets:
                raise ValueError("Scientific alignment outside GS604")
            if match["decision"] != "review":
                continue
            review_alignments += 1
            category = review_category(match)
            categories[category] += 1
            category_queries[category].add(row["id"])
            pairs.add((row["id"], targets[match["ref_id"]]))
            # Deterministic illustrative examples, never ranking or selecting new decisions.
            if len(examples[category]) < 3 and all(
                example["query_id"] != row["id"] for example in examples[category]
            ):
                examples[category].append(
                    dict(
                        query_id=row["id"],
                        target_id=targets[match["ref_id"]],
                        evidence=row["evidence"],
                        evidence_sha256=row["evidence_sha256"],
                        offset_bin=match["offset_bin"],
                        phase=match["phase"],
                        count=match["count"],
                        span_seconds=match["span_seconds"],
                        chromaprint_distance=match["chromaprint_distance"],
                    )
                )
    return dict(
        version="bp-gs-frozen-report-v1",
        scientific_queries=1486,
        outcomes={
            name: outcomes[name]
            for name in (
                "definite_detected_overlap",
                "review",
                "insufficient_information",
                "no_detected_overlap",
            )
        },
        review_fraction=outcomes["review"] / 1486,
        model_exposure=exposure,
        scientific_alignment_records=all_alignments,
        review_alignment_records=review_alignments,
        distinct_review_recording_pairs=len(pairs),
        review_alignment_categories=dict(categories),
        review_query_categories={name: len(ids) for name, ids in category_queries.items()},
        examples=dict(examples),
        category_limitation="Categories count retained offset/phase alignments, not independent observations; query categories overlap. CP disagreement does not clear a pair.",
        example_policy="First three distinct query IDs per category in lexical order, first retained alignment each; illustrative, not representative or new decisions",
        no_false_positive_rate_estimated=True,
        outcomes_unchanged=True,
        audio_read=False,
        labels_scored=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "result", "evidence", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(gzip.decompress(args.plan.read_bytes()))
    result = read_payload(args.result)
    report = summarize(plan, result, args.evidence)
    report.update(
        source_sha256=digest(Path(__file__)),
        result_sha256=hashlib.sha256(payload_bytes(args.result)).hexdigest(),
        plan_compressed_sha256=digest(args.plan),
        index_sha256=result["index_sha256_after"],
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("model_exposure", "examples")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
