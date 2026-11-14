"""The memory probe cannot reuse parity evidence for changed inputs/code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from experiments.chunked import sha256
from tools.benchmark_k10_memory_curve import check_validation
from tools.research.evidence import source_provenance
from tools.research.measurements import runtime_environment


def validation_fixture(tmp_path: Path) -> argparse.Namespace:
    root = Path(__file__).parents[1]
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("contract.json", "prefix.onnx", "head.onnx"):
        (bundle / name).write_text(name)
    manifest = tmp_path / "manifest.json"
    model = tmp_path / "full.onnx"
    manifest.write_text("fixture manifest")
    model.write_text("fixture model")
    report = json.loads(
        (root / "experiments/studies/k10-chunked/data/v2/validation.json").read_bytes()
    )
    report.update(source_provenance())
    report.update(runtime_environment())
    for key, path in (
        ("manifest_sha256", manifest),
        ("full_model_sha256", model),
        ("contract_sha256", bundle / "contract.json"),
        ("prefix_sha256", bundle / "prefix.onnx"),
        ("head_sha256", bundle / "head.onnx"),
    ):
        report[key] = sha256(path)
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(report))
    return argparse.Namespace(
        manifest=manifest, full_model=model, bundle=bundle, validation_report=path
    )


def test_memory_probe_requires_matching_parity_artifacts(tmp_path: Path) -> None:
    args = validation_fixture(tmp_path)
    check_validation(args)
    args.full_model.write_text("changed model")
    with pytest.raises(ValueError, match="exact artifacts/sources"):
        check_validation(args)


@pytest.mark.parametrize("field", ["passed", "runtime_source_sha256", "evaluator_source_sha256"])
def test_memory_probe_rejects_failed_or_stale_parity(tmp_path: Path, field: str) -> None:
    args = validation_fixture(tmp_path)
    report = json.loads(args.validation_report.read_bytes())
    report[field] = False if field == "passed" else "changed source"
    args.validation_report.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="exact artifacts/sources"):
        check_validation(args)
