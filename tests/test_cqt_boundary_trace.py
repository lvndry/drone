"""Tiny observation/storage checks only: no CQT, decoder, model or native binary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools import cqt_boundary_trace as trace


def test_complete_complex_capture_preserves_layout_and_deduplicates(tmp_path: Path) -> None:
    values = np.asfortranarray(np.array([[1 + 2j, -3 + 4j]], dtype=np.complex64))
    before = values.tobytes()
    writer = trace.Writer(tmp_path)
    first = writer.array(values, "<c8")
    second = writer.array(values.copy(), "<c8")
    assert first == second
    assert first["shape"] == [1, 2]
    assert (tmp_path / first["path"]).read_bytes() == before
    assert values.tobytes() == before and values.flags.f_contiguous
    assert len(list((tmp_path / "objects").iterdir())) == 1


def test_writer_rejects_corrupted_existing_blob(tmp_path: Path) -> None:
    writer = trace.Writer(tmp_path)
    values = np.array([2], dtype="<f4")
    record = writer.array(values, "<f4")
    (tmp_path / record["path"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="Bound artifact"):
        writer.array(values, "<f4")


def test_writer_cap_and_nonfinite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writer = trace.Writer(tmp_path)
    with pytest.raises(ValueError, match="Nonfinite"):
        writer.array(np.array([np.nan]), "<f4")
    monkeypatch.setattr(trace, "LIMIT", 262147)
    with pytest.raises(ValueError, match="cap"):
        writer.array(np.array([1], dtype=np.float32), "<f4")
    assert not list((tmp_path / "objects").iterdir())


def test_output_symlink_rejected(tmp_path: Path) -> None:
    (tmp_path / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        trace.Writer(tmp_path)


def test_terminal_hooks_return_unmodified_actual_numpy_results(tmp_path: Path) -> None:
    records: dict[str, Any] = {}
    writer = trace.Writer(tmp_path)
    proxy = trace.NumpyProxy(writer, records, False)
    values = np.asfortranarray(np.full((208, 399), 3 + 4j, dtype=np.complex64))
    before = values.tobytes()
    magnitude = proxy.abs(values)
    assert magnitude.tobytes() == np.abs(values).tobytes()
    assert magnitude.flags.f_contiguous
    logged = proxy.log1p(magnitude)
    assert logged.tobytes() == np.log1p(magnitude).tobytes()
    assert values.tobytes() == before
    normalized = records["terminal.normalized"]
    assert normalized["shape"] == [192, 399]
    assert (tmp_path / normalized["path"]).read_bytes() == values[8:200].tobytes()


def test_normalization_observes_actual_float64_sqrt_once(tmp_path: Path) -> None:
    records: dict[str, Any] = {}
    proxy = trace.NumpyProxy(trace.Writer(tmp_path), records, True)
    values = np.arange(1, 209, dtype=np.float64).reshape(208, 1)
    expected = np.sqrt(values)
    result = proxy.sqrt(values)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64 and result.tobytes() == expected.tobytes()
    artifact = records["terminal.scales"]
    assert (tmp_path / artifact["path"]).read_bytes() == expected[8:200, 0].tobytes()
    with pytest.raises(ValueError, match="Repeated"):
        proxy.sqrt(values)


def test_runtime_proof_admission_fails_before_loading_feature_source(tmp_path: Path) -> None:
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps({"format": "cqt-boundary-runtime-proof-v1", "historical_gaps": []}))
    with pytest.raises(ValueError, match="closure gaps"):
        trace.runtime(proof, trace.digest(proof.read_bytes()))


def test_cqt_stage_scalar_sqrt_is_passed_through(tmp_path: Path) -> None:
    records: dict[str, Any] = {}
    proxy = trace.NumpyProxy(trace.Writer(tmp_path), records, True)
    result = proxy.sqrt(2.0)
    assert isinstance(result, np.floating)
    assert result.tobytes() == np.sqrt(2.0).tobytes()
    assert records == {}
