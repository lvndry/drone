"""Coverage counters never substitute publisher duration for observed audio support."""

from dataclasses import replace
from pathlib import Path

import pytest

from experiments.collection_catalog import load_collection
from tools.report_collection_coverage import INPUTS, STUDY, analyze, summarize


def test_unknown_audio_support_is_not_filled_from_publisher_duration() -> None:
    catalog = load_collection(STUDY / "data" / INPUTS["Beatport"][0])
    row = catalog.tracks[0]
    unknown = replace(
        row.analyzed_audio,
        samples=None,
        sample_rate=None,
        channels=None,
        dtype=None,
        missing_reason="not_in_receipt",
    )
    assert row.catalog_recording_duration.value is not None
    result = summarize(replace(catalog, tracks=(replace(row, analyzed_audio=unknown),)))
    assert result["availability"]["publisher_duration"]["known"] == 1
    assert result["availability"]["recorded_cache_duration"] == {"known": 0, "unknown": 1}
    assert result["recorded_cache_duration_bands"]["unknown"] == 1
    assert sum(result["recorded_cache_duration_bands"].values()) == 1


def test_shared_duration_boundaries_and_complete_denominators() -> None:
    catalog = load_collection(STUDY / "data" / INPUTS["Beatport"][0])
    row = catalog.tracks[0]
    seconds = (29, 30, 119, 120, 299, 300, 600, 601)
    rows = tuple(
        replace(
            row,
            track_id=f"beatport:{i + 1}",
            source_track_id=str(i + 1),
            analyzed_audio=replace(row.analyzed_audio, samples=duration * 24000),
        )
        for i, duration in enumerate(seconds)
    )
    result = summarize(replace(catalog, tracks=rows))
    assert list(result["recorded_cache_duration_bands"].values()) == [1, 2, 2, 2, 1, 0]
    assert all(
        cell["known"] + cell["unknown"] == len(rows) for cell in result["availability"].values()
    )
    assert result["distinct_recorded_source_hashes"] == 1  # rows are not unique songs


def test_catalog_hash_change_stops_before_interpretation(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / INPUTS["Beatport"][0]).write_bytes(b"substituted catalog")
    with pytest.raises(ValueError, match="Pinned collection changed: Beatport"):
        analyze(tmp_path)
