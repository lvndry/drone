"""Control identity preservation and whole-song temporal support."""

from __future__ import annotations

from collections import Counter

import pytest

from experiments.k10_v2_pilot import Draw, paired_batches
from experiments.musicfm_sampling import batches, crop


def test_interleaving_preserves_both_control_subsequences_and_counts() -> None:
    beatport = ["b1", "b2", "b3"]
    fma = ["f1", "f2", "f3", "f4"]
    original = list(paired_batches(beatport, fma, updates=100))
    interleaved = list(batches(beatport, fma, updates=100))
    assert list(batches(beatport, fma, updates=100)) == interleaved
    for left, right in zip(original, interleaved, strict=True):
        assert Counter(left) == Counter(right)
        for domain in ("beatport", "fmak"):
            assert [draw for draw in left if draw.domain == domain] == [
                draw for draw in right if draw.domain == domain
            ]
    assert {batch[-1].domain for batch in interleaved} == {"beatport", "fmak"}


def test_crop_uses_ten_minutes_and_is_independent_of_request_order() -> None:
    draws = [Draw("song", "fmak", index) for index in range(100)]
    result = {draw: crop(600 * 24000, draw) for draw in draws}
    assert {draw: crop(600 * 24000, draw) for draw in reversed(draws)} == result
    assert all(
        0 <= start <= 588 * 24000 and length == 12 * 24000 for start, length in result.values()
    )
    assert any(start > 480 * 24000 for start, _ in result.values())
    assert any(start < 120 * 24000 for start, _ in result.values())


def test_short_tracks_use_actual_samples_without_padding() -> None:
    assert crop(1025, Draw("short", "fmak", 0)) == (0, 1025)
    assert crop(288000, Draw("exact", "fmak", 1)) == (0, 288000)
    with pytest.raises(ValueError, match="exceed 1024"):
        crop(1024, Draw("invalid", "fmak", 2))
