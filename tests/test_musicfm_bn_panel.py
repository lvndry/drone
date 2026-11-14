"""Synthetic role/binding checks; never read a real plan, cache file or waveform."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from fractions import Fraction
from typing import Any

import pytest

from experiments import musicfm_bn_panel as panel


@pytest.fixture
def inputs() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    plan: dict[str, Any] = {}
    rows = {}
    for key, count, role in (
        ("bp_fit", 1229, "fit"),
        ("expanded_fma_fit", 4033, "fit"),
        ("bp_selection_guard", 126, "validation"),
        ("fma_inner_selection", 250, "validation"),
    ):
        plan[key] = [f"{key}:{index}" for index in range(count)]
        for identity in plan[key]:
            rows[identity] = dict(
                id=identity,
                role=role,
                samples=300000,
                bytes=1200000,
                waveform_sha256="a" * 64,
                cache_identity=[1, 2, 1200000, 4, 5],
                source_stable_identity=[1, 3, 400000, 4, 5],
                cache="/pinned/cache/" + identity,
                path="/pinned/source/" + identity,
                target=7,
            )
    return plan, rows


def test_metadata_is_deterministic_fitting_only_and_not_executable(
    inputs: tuple[dict[str, Any], dict[str, dict[str, Any]]],
) -> None:
    plan, rows = inputs
    raw = panel.metadata_bytes(plan, rows)
    assert raw == panel.metadata_bytes(plan, dict(reversed(list(rows.items()))))
    result = json.loads(raw)
    assert "NO_CROP_HASHES_OR_EXECUTION_PERMISSION" in result["status"]
    assert len(result["rows"]) == 512
    assert all(row["id"] in plan["bp_fit"] + plan["expanded_fma_fit"] for row in result["rows"])
    assert not any("target" in row or "ref" in row for row in result["rows"])
    assert len(panel.metadata_sha256(plan, rows)) == 64
    changed = copy.deepcopy(rows)
    for row in changed.values():
        row.pop("target")
    assert panel.metadata_bytes(plan, changed) == raw


@pytest.mark.parametrize("defect", ["extra", "missing", "overlap", "role", "identity"])
def test_role_or_membership_changes_fail_before_projection(
    inputs: tuple[dict[str, Any], dict[str, dict[str, Any]]], defect: str
) -> None:
    plan, rows = inputs
    identity = plan["bp_fit"][0]
    if defect == "extra":
        rows["reserved"] = rows[identity]
    elif defect == "missing":
        rows.pop(identity)
    elif defect == "overlap":
        plan["bp_selection_guard"][0] = identity
    elif defect == "role":
        rows[identity]["role"] = "validation"
    else:
        rows[identity]["id"] = "other"
    with pytest.raises(ValueError):
        panel.metadata_bytes(plan, rows)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("waveform_sha256", "invalid"),
        ("cache_identity", [1, 2, 3]),
        ("source_stable_identity", [True, 2, 3, 4, 5]),
        ("bytes", 4),
        ("cache", "relative/path"),
        ("path", "/pinned/../other"),
    ],
)
def test_selected_metadata_requires_complete_physical_bindings(
    inputs: tuple[dict[str, Any], dict[str, dict[str, Any]]], field: str, value: object
) -> None:
    plan, rows = inputs
    identity = panel.panel_rows(plan, rows)[0]["id"]
    rows[identity][field] = value
    with pytest.raises(ValueError):
        panel.metadata_bytes(plan, rows)


@pytest.mark.parametrize(
    "defect", ["selection", "duplicate", "domain", "mass", "samples", "crop", "count"]
)
def test_schedule_regressions_cannot_bypass_fitting_boundary(
    inputs: tuple[dict[str, Any], dict[str, dict[str, Any]]],
    defect: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, rows = inputs
    clips = list(
        panel.schedule(
            {identity: rows[identity]["samples"] for identity in plan["bp_fit"]},
            {identity: rows[identity]["samples"] for identity in plan["expanded_fma_fit"]},
        )
    )
    if defect == "selection":
        clips[0] = replace(clips[0], identity=plan["bp_selection_guard"][0])
    elif defect == "duplicate":
        clips[0] = clips[1]
    elif defect == "domain":
        clips[0] = replace(clips[0], domain="fmak")
    elif defect == "mass":
        clips[0] = replace(clips[0], weight=Fraction(1, 512))
    elif defect == "samples":
        clips[0] = replace(clips[0], samples=300001)
    elif defect == "crop":
        clips[0] = replace(clips[0], start=clips[0].start + 1)
    else:
        clips.pop()
    monkeypatch.setattr(panel, "schedule", lambda *_: tuple(clips))
    with pytest.raises(ValueError):
        panel.metadata_bytes(plan, rows)
