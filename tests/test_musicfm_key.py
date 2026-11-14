"""Whole-song coverage and prediction boundaries, without model/audio execution."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.musicfm_key import (
    MIN_SAMPLES,
    WINDOW_SAMPLES,
    SongLogits,
    SongWindow,
    song_windows,
)


@pytest.mark.parametrize("samples", [0, 1, MIN_SAMPLES, -1, True, 1025.0])
def test_reject_unsupported_song_lengths(samples: int) -> None:
    with pytest.raises(ValueError, match="integer sample count"):
        song_windows(samples)


@pytest.mark.parametrize(
    "samples",
    [
        MIN_SAMPLES + 1,
        WINDOW_SAMPLES - 1,
        WINDOW_SAMPLES,
        WINDOW_SAMPLES + 1,
        WINDOW_SAMPLES + MIN_SAMPLES,
        WINDOW_SAMPLES + MIN_SAMPLES + 1,
        300 * 24000,
        600 * 24000 + 17,
    ],
)
def test_every_sample_supported_once_with_valid_actual_context(samples: int) -> None:
    windows = song_windows(samples)
    cursor = 0
    for window in windows:
        assert (
            0
            <= window.start
            <= window.support_start
            < window.support_stop
            <= window.stop
            <= samples
        )
        assert MIN_SAMPLES < window.stop - window.start <= WINDOW_SAMPLES
        assert window.support_start == cursor
        cursor = window.support_stop
    assert cursor == samples
    assert sum(window.weight for window in windows) == samples


def test_tiny_tail_has_context_but_only_its_own_weight() -> None:
    window = song_windows(2 * WINDOW_SAMPLES + 1)[-1]
    assert window == SongWindow(
        WINDOW_SAMPLES + 1, 2 * WINDOW_SAMPLES + 1, 2 * WINDOW_SAMPLES, 2 * WINDOW_SAMPLES + 1
    )
    assert window.weight == 1
    own_tail = song_windows(2 * WINDOW_SAMPLES + MIN_SAMPLES + 1)[-1]
    assert own_tail.start == own_tail.support_start == 2 * WINDOW_SAMPLES


def test_full_ten_minute_song_is_not_capped_to_preview() -> None:
    windows = song_windows(600 * 24000)
    assert len(windows) == 50
    assert windows[-1].stop == 600 * 24000
    accumulator = SongLogits(600 * 24000)
    for index, window in enumerate(windows):
        logits = np.zeros(24)
        logits[0 if index < 10 else 1] = 1
        accumulator.add(window, logits)
    # The final eight minutes must influence the result, not just the preview.
    result = accumulator.finish()
    assert result[0] == pytest.approx(0.2)
    assert result[1] == pytest.approx(0.8)
    assert np.argmax(result) == 1


def test_duration_weighted_logits_match_explicit_reference() -> None:
    samples = WINDOW_SAMPLES + WINDOW_SAMPLES // 2
    accumulator = SongLogits(samples)
    first = np.arange(24, dtype=np.float32)
    second = first[::-1].copy()
    accumulator.add(accumulator.windows[0], first)
    with pytest.raises(ValueError, match="complete song"):
        accumulator.finish()
    accumulator.add(accumulator.windows[1], second)
    expected = (
        first.astype(np.float64) * WINDOW_SAMPLES
        + second.astype(np.float64) * (WINDOW_SAMPLES // 2)
    ) / samples
    np.testing.assert_array_equal(accumulator.finish(), expected)
    assert accumulator.finish().dtype == np.float64


def test_reject_reordered_repeated_and_invalid_predictions_without_advancing() -> None:
    accumulator = SongLogits(WINDOW_SAMPLES + 1025)
    first, last = accumulator.windows
    with pytest.raises(ValueError, match="next fixed song window"):
        accumulator.add(last, np.zeros(24))
    for values in (np.zeros(23), np.zeros((1, 24)), [float("nan")] * 24, [float("inf")] * 24):
        with pytest.raises(ValueError, match="24 finite"):
            accumulator.add(first, values)
    accumulator.add(first, np.zeros(24))
    with pytest.raises(ValueError, match="next fixed song window"):
        accumulator.add(first, np.zeros(24))
    accumulator.add(last, np.ones(24))
    np.testing.assert_array_equal(accumulator.finish(), np.full(24, 1025 / (WINDOW_SAMPLES + 1025)))
    with pytest.raises(ValueError, match="next fixed song window"):
        accumulator.add(last, np.ones(24))
