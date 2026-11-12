"""No audio/model execution: geometry, cache accounting, and immutable result gates."""

from pathlib import Path

import numpy as np
import pytest

from tools.musicfm_crop_io import (
    RATE,
    WINDOW,
    compare,
    crop_positions,
    decode_command,
    lru_projection,
    seek_geometry,
)


def test_lru_recency_and_oversized_objects() -> None:
    report = lru_projection(
        ["bp:a", "bp:b", "bp:a", "bp:c", "bp:b", "bp:big"],
        {"bp:a": 4, "bp:b": 4, "bp:c": 4, "bp:big": 9},
        {"bp:a": 1.0, "bp:b": 2.0, "bp:c": 3.0, "bp:big": 10.0},
        8,
    )
    assert report["hits"] == 1
    assert report["misses"] == 5
    assert report["evictions"] == 2
    assert report["projected_cache_write_bytes"] == 16
    assert report["projected_full_decode_audio_seconds"] == 18
    assert report["resident_bytes_at_end"] == 8


@pytest.mark.parametrize("start", [0, 1, RATE + 1, 2 * RATE, 19 * RATE + 23999])
def test_seek_geometry_preserves_exact_global_offset(start: int) -> None:
    anchor, trim, wanted = seek_geometry(start, WINDOW)
    assert anchor * RATE + trim == start
    assert wanted == trim + WINDOW + RATE
    assert 0 <= trim < 3 * RATE
    if anchor > 0:
        assert trim >= 2 * RATE


def test_fixed_positions_cover_endpoints_and_are_reproducible() -> None:
    samples = 331 * RATE + 123
    positions = crop_positions(samples, "public:fixed")
    assert positions == crop_positions(samples, "public:fixed")
    assert {0, 1, RATE + 1, samples - WINDOW - 1, samples - WINDOW} <= set(positions)
    assert all(0 <= start <= samples - WINDOW for start in positions)
    assert crop_positions(WINDOW - 1, "short") == [0]


def test_command_bounds_output_after_resampling_and_retains_context() -> None:
    command = decode_command(Path("/usr/bin/ffmpeg"), Path("input.mp3"), 100 * RATE + 1, WINDOW)
    assert command.index("-ss") < command.index("-i") < command.index("-ar")
    assert command[command.index("-ss") + 1] == "98"
    assert command[command.index("-t") + 1] == "15.000041667"
    assert "pcm_f32le" in command
    assert "-ss" not in decode_command(Path("ffmpeg"), Path("input"), 1, WINDOW)
    assert "-t" not in decode_command(Path("ffmpeg"), Path("input"), None, WINDOW)


def test_gate_requires_bits_finiteness_and_length_without_tolerance() -> None:
    reference = np.array([0.0, 1.0, -1.0], dtype="<f4")
    assert compare(reference, reference.copy())["passed"] is True
    changed = reference.copy()
    changed[1] = np.nextafter(changed[1], np.float32(2))
    assert compare(reference, changed)["passed"] is False
    changed[0] = np.nan
    assert compare(reference, changed)["finite"] is False
    assert compare(reference, reference[:-1])["passed"] is False
    changed = reference.copy()
    changed[0] = -0.0
    assert compare(reference, changed)["bitwise_mismatch_samples"] == 1
