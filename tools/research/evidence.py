"""Artifact and implementation identities required before resource measurement."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from experiments.chunked import ATOL, RTOL, sha256
from tools.research.contracts import CachedTrack
from tools.research.measurements import runtime_environment

ROOT = Path(__file__).resolve().parents[2]


def support_sources() -> dict[str, str]:
    """Hash implementation helpers, not only their command-line entry point."""
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }


def source_provenance() -> dict[str, object]:
    return {
        "runtime_source_sha256": sha256(ROOT / "experiments/chunked.py"),
        "evaluator_source_sha256": sha256(ROOT / "tools/evaluate_chunked.py"),
        "support_source_sha256": support_sources(),
    }


def require_fields(
    report: Mapping[str, object], expected: Mapping[str, object], message: str
) -> None:
    for key, value in expected.items():
        actual = report.get(key)
        # JSON true and 1 must not be interchangeable for the passing gate.
        if actual != value or (isinstance(value, bool) and type(actual) is not bool):
            raise ValueError(f"{message}: {key}")


@dataclass(frozen=True)
class ArtifactSet:
    manifest: Path
    bundle: Path
    full_model: Path
    core_outputs: int = 64

    def identity(self) -> dict[str, object]:
        return {
            "manifest_sha256": sha256(self.manifest),
            "full_model_sha256": sha256(self.full_model),
            "contract_sha256": sha256(self.bundle / "contract.json"),
            "prefix_sha256": sha256(self.bundle / "prefix.onnx"),
            "head_sha256": sha256(self.bundle / "head.onnx"),
            "core_outputs": self.core_outputs,
            **source_provenance(),
        }

    def require_validation(self, path: Path) -> None:
        expected = dict(
            mode="validate",
            passed=True,
            tracks=376,
            atol=ATOL,
            rtol=RTOL,
            **self.identity(),
            **runtime_environment(),
        )
        require_fields(
            json.loads(path.read_bytes()),
            expected,
            "Validation report does not pass for these exact artifacts/sources",
        )


def positive_measurement(record: Mapping[str, object], name: str) -> float:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid full benchmark measurement: {name}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid full benchmark measurement: {name}")
    return float(value)


@dataclass(frozen=True)
class FullBenchmark:
    peak_rss_kib: float
    median_seconds: Mapping[str, float]
    sha256: str


def require_full_benchmark(
    path: Path, artifacts: ArtifactSet, selection: Path, chosen: list[CachedTrack]
) -> FullBenchmark:
    """Reject incompatible or failed pair evidence before loading a model."""
    report = json.loads(path.read_bytes())
    expected = {
        "mode": "benchmark",
        "bench_mode": "full",
        "torch_imported": False,
        "tracks": 20,
        "repetitions": 20,
        "warmups_per_track": 1,
        "selection_sha256": sha256(selection),
        **artifacts.identity(),
        **runtime_environment(),
    }
    require_fields(report, expected, "Incompatible full benchmark")
    if "error" in report or report.get("passed") is False:
        raise ValueError("Cannot pair a failed full benchmark")
    records = report.get("records")
    if not isinstance(records, list) or len(records) != len(chosen):
        raise ValueError("Benchmark identities differ")
    durations = {}
    for record, track in zip(records, chosen, strict=True):
        require_fields(
            record,
            dict(
                id=track.id,
                sha256=track.sha256,
                cache_sha256=track.cache_sha256,
                frames=track.frames,
            ),
            "Benchmark identities differ",
        )
        durations[track.sha256] = positive_measurement(record, "median_seconds")
    return FullBenchmark(
        positive_measurement(report, "peak_process_rss_kib"), durations, sha256(path)
    )
