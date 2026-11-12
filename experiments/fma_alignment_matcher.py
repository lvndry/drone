"""Read-only all-offset correction; frozen extraction/index and decision thresholds unchanged."""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from experiments.fma_fingerprints import (
    AMBIG_MIN,
    AMBIG_SPAN,
    CONTRACT,
    HOP,
    MATCH_MIN,
    MATCH_SPAN,
    SR,
    PostingIndex,
)

MATCHER_CONTRACT = dict(
    version="all-qualifying-offsets-v1",
    extraction_contract=CONTRACT,
    offset_bin_frames=4,
    phases=[0, 2],
    distinct_query_anchors=True,
    minimum_returned_anchors=AMBIG_MIN,
    minimum_returned_span=AMBIG_SPAN,
    per_reference_alignment_pruning=False,
    top_k=False,
    complete_postings=True,
    wrapped_time=False,
    maximum_vote_cells=2000000,
    incomplete_queries_are_negatives=False,
)


class AllOffsetIndex(PostingIndex):
    """Open immutable index bytes and retain every qualifying offset/phase alignment."""

    def __init__(self, path: Path) -> None:
        super().__init__(path, readonly=True)
        self.db.execute("PRAGMA query_only=ON")

    @classmethod
    def from_verified_artifacts(
        cls,
        path: Path,
        *,
        descriptor: dict[str, Any],
        query_extraction_contract: dict[str, Any],
        expected_index_sha256: str,
        fingerprint_root: Path,
        artifact_sha256: dict[str, str],
        expected_references: int = 7579,
    ) -> AllOffsetIndex:
        """Production reuse requires the original descriptor and immutable-byte witnesses."""
        frozen_engine = "a0d315f3f6f6f62b33b7453119350807702eec35a17723b9079fca30c4328f71"
        engine_path = Path(__file__).with_name("fma_fingerprints.py")
        if (
            descriptor["contract"] != CONTRACT
            or query_extraction_contract != CONTRACT
            or descriptor["source_sha256"]["experiments/fma_fingerprints.py"] != frozen_engine
            or hashlib.sha256(engine_path.read_bytes()).hexdigest() != frozen_engine
            or not descriptor["complete_coverage"]
            or descriptor["references"] != expected_references
            or len(artifact_sha256) != expected_references
        ):
            raise ValueError("Extraction descriptor/source/reference coverage identity differs")
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected_index_sha256:
                raise ValueError("Immutable index SHA differs")
        for name, sha in artifact_sha256.items():
            if Path(name).name != name:
                raise ValueError("Unsafe reference artifact name")
            with (fingerprint_root / name).open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != sha:
                    raise ValueError("Immutable fingerprint SHA differs")
        result = cls(path)
        if (
            result.db.execute("SELECT count(*) FROM references_done").fetchone()[0]
            != expected_references
            or result.db.execute("SELECT count(*) FROM postings").fetchone()[0]
            != descriptor["postings"]
        ):
            result.db.close()
            raise ValueError("Index coverage counters differ from verified descriptor")
        return result

    def add(self, ref: int, identity: str, hashes: NDArray[np.int64]) -> None:
        raise PermissionError("Corrected matcher cannot mutate the reference index")

    def finish(self) -> None:
        raise PermissionError("Corrected matcher cannot build or mutate the reference index")

    def match(
        self,
        hashes: NDArray[np.int64],
        *,
        deadline: float | None = None,
        maximum_vote_cells: int = 2000000,
    ) -> dict[str, Any]:
        if maximum_vote_cells < 1:
            raise ValueError("Vote-cell budget must be positive")
        votes: dict[tuple[int, int, int], set[int]] = defaultdict(set)
        posting_visits = 0
        for time_frame, hash_value in hashes:
            if deadline is not None and time.monotonic() > deadline:
                return dict(
                    complete=False,
                    reason="query_budget_exhausted",
                    matches=[],
                    posting_visits=posting_visits,
                )
            for ref, frame in self.db.execute(
                "SELECT ref,frame FROM postings WHERE hash=?", (int(hash_value),)
            ):
                posting_visits += 1
                if (
                    posting_visits % 65536 == 0
                    and deadline is not None
                    and time.monotonic() > deadline
                ):
                    return dict(
                        complete=False,
                        reason="query_budget_exhausted",
                        matches=[],
                        posting_visits=posting_visits,
                    )
                offset = frame - int(time_frame)
                for phase in (0, 2):
                    votes[(ref, (offset + phase) // 4, phase)].add(int(time_frame))
                if len(votes) > maximum_vote_cells:
                    return dict(
                        complete=False,
                        reason="vote_memory_budget",
                        matches=[],
                        posting_visits=posting_visits,
                    )
        matches = []
        for (ref, offset_bin, phase), anchors in votes.items():
            if deadline is not None and time.monotonic() > deadline:
                return dict(
                    complete=False,
                    reason="query_budget_exhausted",
                    matches=[],
                    posting_visits=posting_visits,
                )
            count = len(anchors)
            span = (max(anchors) - min(anchors)) * HOP / SR
            if count < AMBIG_MIN or span < AMBIG_SPAN:
                continue
            # Every original 3-anchor/2s qualifying bin reaches downstream policy.
            # In particular a short high-count bin cannot hide a long lower-count bin.
            matches.append(
                dict(
                    ref_id=ref,
                    offset_bin=offset_bin,
                    phase=phase,
                    count=count,
                    span_seconds=span,
                    offset_seconds=(offset_bin * 4 - phase) * HOP / SR,
                    strong=count >= MATCH_MIN and span >= MATCH_SPAN,
                )
            )
        return dict(
            complete=True,
            posting_visits=posting_visits,
            matches=sorted(matches, key=lambda r: (r["ref_id"], r["offset_bin"], r["phase"])),
        )
