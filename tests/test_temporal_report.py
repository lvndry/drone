"""Audit the packaged temporal evidence independently of machine-local checkpoints."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/studies/k10-temporal/data"


def test_temporal_evidence_bytes_and_decisions() -> None:
    provenance = json.loads((DATA / "provenance.json").read_bytes())
    result = json.loads((DATA / "result13.json").read_bytes())
    for name, record in provenance["artifacts"].items():
        payload = (DATA / name).read_bytes()
        assert len(payload) == record["bytes"]
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        if name.endswith(".gz"):
            decoded = gzip.decompress(payload)
            assert len(decoded) == record["decoded_bytes"]
            assert hashlib.sha256(decoded).hexdigest() == record["decoded_sha256"]
    for relative, digest in provenance["executed_files_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    for arm in "MOS":
        history = json.loads(gzip.decompress((DATA / f"history-{arm}.json.gz").read_bytes()))
        selected = min(
            history,
            key=lambda row: (-row["selection"]["exact"], row["selection"]["ce"], row["epoch"]),
        )
        assert selected["epoch"] == result["arms"][arm]["best_epoch"]
        assert history[-1]["epoch"] == result["arms"][arm]["stop_epoch"]
        for domain in ("beatport", "fmak"):
            assert selected["metrics"][domain] == result["arms"][arm]["metrics"][domain]
    reconstruction = json.loads((DATA / "reconstruction.json").read_bytes())
    assert reconstruction["bit_identical"] == 2582
    assert reconstruction["max_abs_error"] == 0
    assert not result["gates"]["candidate_pass"]
    assert not result["gates"]["mechanism_pass"]
    assert not result["promoted"]
