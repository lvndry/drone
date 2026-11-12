"""Aggregate-only source/coverage checks; no private inputs, audio, models or downloads."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tools.dj_coverage import band
from tools.dj_coverage_bp import selected_cell, selected_store_fields
from tools.dj_coverage_report import load, verify


def test_native_taxonomies_and_actual_preview_support_remain_separate() -> None:
    data = load()
    fma, bp = data["fma"], data["bp"]
    assert fma["coverage"]["Hip-Hop"]["inner_selection"]["all"]["tracks"] == 16
    assert (
        fma["coverage"]["House_minus_Chill-out_metadata_slice"]["inner_selection"]["all"]["tracks"]
        == 0
    )
    assert bp["workbook_genres"]["Deep House"]["inner_selection"]["tracks"] == 12
    assert bp["workbook_genres"]["Hip-Hop / R&B"]["inner_selection"]["tracks"] == 6
    assert bp["roles"]["inner_selection"]["actual_preview_seconds_max"] < 121
    assert bp["roles"]["inner_selection"]["store_duration_bands"]["5–10m"] == 46
    assert not bp["roles"]["inner_selection"]["artist_album_components_verified"]
    assert "per_track" not in bp and "wider_graph_bridge_witnesses" not in fma


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (29.999, "<30s"),
        (30, "30s–<2m"),
        (119.999, "30s–<2m"),
        (120, "2–<5m"),
        (300, "5–10m"),
        (600, "5–10m"),
        (600.001, ">10m"),
    ],
)
def test_declared_duration_boundaries(seconds: float, expected: str) -> None:
    assert band(seconds) == expected


@pytest.mark.parametrize("damage", ["band-count", "preview-store", "source-plan", "raw-rows"])
def test_bad_aggregate_or_scope_is_rejected(damage: str) -> None:
    data = copy.deepcopy(load())
    if damage == "band-count":
        data["fma"]["roles"]["inner_selection"]["bands"]["<30s"]["tracks"] += 1
    elif damage == "preview-store":
        data["bp"]["roles"]["inner_selection"]["actual_preview_seconds_max"] = 600
    elif damage == "source-plan":
        data["bp"]["sources"]["plan"]["sha256"] = "different"
    else:
        data["bp"]["per_track"] = []
    with pytest.raises(ValueError):
        verify(data)


def test_bp_json_allowlist_preserves_quoted_nan_and_never_exports_store_key() -> None:
    raw = b'{"id":123,"genres":[{"name":"nan"}],"duration":{"milliseconds":nan},"key":"NOT_A_REFERENCE_EXPORT","artists":[]}'
    chosen, normalized = selected_store_fields(raw)
    assert normalized and set(chosen) == {"id", "genres", "sub_genres", "duration"}
    assert chosen["genres"][0]["name"] == "nan"
    assert chosen["duration"]["milliseconds"] is None
    assert "key" not in chosen and "artists" not in chosen


def test_bp_xml_rejects_nonallowlisted_columns_and_formulas() -> None:
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    cell = ET.fromstring(f'<c xmlns="{namespace}" r="F2"><v>42</v></c>')
    assert selected_cell(cell, "F") == ("n", "42")
    with pytest.raises(ValueError):
        selected_cell(cell, "G")
    with pytest.raises(ValueError):
        selected_cell(cell, "A")
    formula = ET.fromstring(f'<c xmlns="{namespace}" r="F2"><f>1+1</f><v>2</v></c>')
    with pytest.raises(AssertionError):
        selected_cell(formula, "F")


def test_packed_evidence_damage_is_detected(tmp_path: Path) -> None:
    from tools.dj_coverage_report import STUDY

    target = tmp_path / "data"
    target.mkdir()
    (target / "coverage-index.json").write_bytes((STUDY / "data/coverage-index.json").read_bytes())
    packed = bytearray((STUDY / "data/coverage-aggregate.json.gz").read_bytes())
    packed[-10] ^= 1
    (target / "coverage-aggregate.json.gz").write_bytes(packed)
    with pytest.raises(ValueError, match="compressed evidence"):
        load(tmp_path)


def test_aggregate_does_not_embed_source_rows_keys_or_predictions() -> None:
    forbidden = {
        "per_track",
        "wider_graph_bridge_witnesses",
        "raw_utf8",
        "reference_logits",
        "ref",
        "target",
        "key",
    }

    def check(value: object) -> None:
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    check(load())
