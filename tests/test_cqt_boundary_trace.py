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


def support_proof() -> dict[str, Any]:
    root = Path(trace.__file__).absolute().parents[1]
    files = {
        "experiments": "experiments/__init__.py",
        "tools.research": "tools/research/__init__.py",
        "experiments.chunked": "experiments/chunked.py",
        "tools.research.contracts": "tools/research/contracts.py",
        "tools.research.measurements": "tools/research/measurements.py",
        "tools.export_cqt_goldens": "tools/export_cqt_goldens.py",
    }
    return {
        "interpreter_path": trace.sys.executable,
        "trace_support_modules": {
            "tools": {"namespace_paths": [str(root / "tools")]},
            **{
                name: {"path": str(root / file), "sha256": trace.digest((root / file).read_bytes())}
                for name, file in files.items()
            },
        },
    }


def isolate_support_search(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(trace.__file__).absolute().parents[1]
    # The shared test venv has another editable checkout. Do not weaken production
    # admission: isolate only this test's module search, leaving the venv untouched.
    paths = [value for value in trace.sys.path if not (Path(value) / "tools").is_dir()]
    monkeypatch.setattr(trace.sys, "path", [str(root), *paths])


def test_support_closure_checks_sources_without_importing_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("PYTHONHOME", raising=False)
    isolate_support_search(monkeypatch)
    before = set(trace.sys.modules)
    trace.support_origins(support_proof())
    assert set(trace.sys.modules) == before


@pytest.mark.parametrize("name", ["PYTHONPATH", "PYTHONHOME"])
def test_support_rejects_python_environment_even_empty(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setenv(name, "")
    with pytest.raises(ValueError, match="must be absent"):
        trace.support_origins(support_proof())


def test_support_rejects_other_interpreter_and_incomplete_or_changed_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("PYTHONHOME", raising=False)
    isolate_support_search(monkeypatch)
    proof = support_proof()
    proof["interpreter_path"] += ".different"
    with pytest.raises(ValueError, match="interpreter"):
        trace.support_origins(proof)
    proof = support_proof()
    del proof["trace_support_modules"]["tools.research"]
    with pytest.raises(ValueError, match="Incomplete"):
        trace.support_origins(proof)
    proof = support_proof()
    proof["trace_support_modules"]["experiments"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Bound artifact"):
        trace.support_origins(proof)
    proof = support_proof()
    proof["trace_support_modules"]["tools"]["namespace_paths"].append("/unbound")
    with pytest.raises(ValueError, match="namespace"):
        trace.support_origins(proof)


def test_support_rejects_loaded_module_from_another_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import ModuleType

    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("PYTHONHOME", raising=False)
    isolate_support_search(monkeypatch)
    wrong = ModuleType("experiments.chunked")
    wrong.__file__ = "/unbound/chunked.py"
    monkeypatch.setitem(trace.sys.modules, "experiments.chunked", wrong)
    with pytest.raises(ValueError, match="Loaded support origin"):
        trace.support_origins(support_proof())
