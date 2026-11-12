"""Observe actual private calls without changing their numerical outputs."""

from pathlib import Path

import numpy as np
import pytest

from tools.export_cqt_goldens import HOP, Trace, load_features_module, synthetic_cases


def test_fixed_synthetic_inputs() -> None:
    cases = synthetic_cases()
    assert [name for name, _ in cases] == ["silence", "impulses", "tone", "major", "minor", "noise"]
    assert [len(values) for _, values in cases] == [
        7 * HOP,
        8 * HOP + 1,
        15 * HOP - 1,
        16 * HOP,
        17 * HOP + 17,
        31 * HOP + 127,
    ]
    for (_, first), (_, second) in zip(cases, synthetic_cases(), strict=True):
        np.testing.assert_array_equal(first, second)


def test_trace_is_observational_and_restores(tmp_path: Path) -> None:
    features = load_features_module(Path("drone/features.py"))
    samples = synthetic_cases()[1][1].astype(np.float32)
    expected = features.cqt(samples)
    with Trace(tmp_path, "impulse", False) as trace:
        actual = features.cqt(samples)
    np.testing.assert_array_equal(actual, expected)
    assert len(trace.stages) == 9
    assert len(trace.resamplings) == 7
    assert [stage["n_fft"] for stage in trace.stages] == [
        256,
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        256,
    ]
    with pytest.raises(RuntimeError), Trace(tmp_path, "exception", False):
        raise RuntimeError("restoration check")
    np.testing.assert_array_equal(features.cqt(samples), expected)
