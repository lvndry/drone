"""Contract tests without frontend/model execution."""

import math

from tools.cqt_mac_stage import compare


def test_comparison_requires_finite_and_same_key() -> None:
    reference = [0.0] * 24
    reference[2] = 1.0
    assert compare(reference, reference)["passed"]
    wrong = reference.copy()
    wrong[2] = math.nan
    assert not compare(wrong, reference)["passed"]
    wrong = reference.copy()
    wrong[3] = 1.000001
    assert not compare(wrong, reference)["passed"]


def test_comparison_preserves_declared_tolerance() -> None:
    reference = [0.0] * 24
    reference[0] = 2.0
    good = reference.copy()
    good[1] = 0.000009
    assert compare(good, reference)["passed"]
    good[1] = 0.000011
    assert not compare(good, reference)["passed"]
