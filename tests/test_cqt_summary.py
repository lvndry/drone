"""Independent evidence-gate checks preserve combined tolerances and first ties."""

from __future__ import annotations

import pytest

from tools.summarize_cqt_expansion import compare_logits


def test_combined_gate_and_first_argmax() -> None:
    expected = [1.0] * 24
    actual = expected.copy()
    assert compare_logits(actual, expected)["argmax"] == 0
    actual[2] += 1.5e-5
    assert compare_logits(actual, expected)["allclose"]
    assert not compare_logits(actual, expected)["argmax_agrees"]
    actual[2] += 1e-4
    assert not compare_logits(actual, expected)["allclose"]


def test_invalid_logits_do_not_become_plot_points() -> None:
    with pytest.raises(ValueError):
        compare_logits([0.0] * 23, [0.0] * 24)
    with pytest.raises(ValueError):
        compare_logits([float("nan")] * 24, [0.0] * 24)
