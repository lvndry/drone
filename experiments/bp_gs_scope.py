"""Target-scope adapter around the unchanged frozen all-offset voting algorithm."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from experiments.fma_alignment_matcher import AllOffsetIndex

POSTING_QUERY = "SELECT ref,frame FROM postings WHERE hash=?"
RULE: dict[str, Any] = dict(
    version="fma-control-only-v2",
    identity_anchors=20,
    identity_span=8.0,
    identity_chromaprint_max=0.15,
    review_anchors=10,
    review_span=5.0,
    chromaprint_scope="corroboration_at_landmark_offsets_only",
    exact_container_or_pcm_identity="reject",
    no_candidate_queries=True,
)


def classify(match: dict[str, Any], cp: float | None) -> str:
    """Literal frozen v2 decision rule; its historical scope label is preserved."""
    if match["count"] < RULE["review_anchors"] or match["span_seconds"] < RULE["review_span"]:
        return "below_screening_support"
    if (
        match["count"] >= RULE["identity_anchors"]
        and match["span_seconds"] >= RULE["identity_span"]
        and cp is not None
        and cp <= RULE["identity_chromaprint_max"]
    ):
        return "identity_evidence"
    return "review"


class Bindings(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int, /) -> object: ...


class TargetConnection(sqlite3.Connection):
    """Filter postings before votes; no database, dictionary or matcher changes."""

    low: int
    high: int

    def execute(
        self, sql: str, parameters: Bindings | Mapping[str, object] = (), /
    ) -> sqlite3.Cursor:
        if sql == POSTING_QUERY:
            if isinstance(parameters, Mapping) or len(parameters) != 1:
                raise ValueError("Unexpected frozen postings query bindings")
            return super().execute(
                sql + " AND ref BETWEEN ? AND ?", (parameters[0], self.low, self.high)
            )
        if "postings" in sql.lower():
            raise ValueError("Unexpected postings query; refuse unscoped matching")
        return super().execute(sql, parameters)


def scope(index: AllOffsetIndex, path: Path, target_ids: list[int]) -> None:
    """Called only after the original complete-index factory and dictionary check."""
    ids = sorted(set(target_ids))
    if not ids or len(ids) != len(target_ids) or ids != list(range(ids[0], ids[-1] + 1)):
        raise ValueError("Require an explicit unique contiguous target scope")
    if isinstance(index.db, TargetConnection):
        index.db.low, index.db.high = ids[0], ids[-1]
        return
    connection = sqlite3.connect(
        f"file:{path.resolve()}?mode=ro", uri=True, factory=TargetConnection
    )
    connection.low, connection.high = ids[0], ids[-1]
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA cache_size=-65536")
    index.db.close()
    index.db = connection


def outcome(*, exact: bool, complete: bool, informative: bool, decisions: list[str]) -> str:
    if exact or "identity_evidence" in decisions:
        return "definite_detected_overlap"
    if not complete or not informative:
        return "insufficient_information"
    if "review" in decisions:
        return "review"
    return "no_detected_overlap"
