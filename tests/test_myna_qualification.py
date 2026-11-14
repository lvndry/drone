"""Boundary planning regressions without model downloads or CUDA."""

from __future__ import annotations

import pytest

from experiments.myna_qualification import mel_blocks


@pytest.mark.parametrize("samples", [1025, 2048, 99999, 100000, 100352, 16000 * 600 + 137])
def test_streamed_frames_have_exact_global_centers(samples: int) -> None:
    centers: list[int] = []
    for left, right, offset, count in mel_blocks(samples):
        assert left % 512 == 0
        assert 0 <= left < right <= samples
        assert right - left > 1024
        assert right - left <= 196 * 512 + 2048
        for index in range(count):
            center = left + (offset + index) * 512
            assert left == 0 or center - 1024 >= left
            assert right == samples or center + 1024 <= right
            centers.append(center)
    assert centers == list(range(0, samples + 1, 512))


@pytest.mark.parametrize("samples", [0, 1, 1024, -1])
def test_too_short_reflection_fails_without_padding(samples: int) -> None:
    with pytest.raises(ValueError):
        mel_blocks(samples)


def test_invalid_block_size_fails() -> None:
    with pytest.raises(ValueError):
        mel_blocks(4096, 0)
