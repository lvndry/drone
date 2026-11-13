"""Portable evidence validation: no decoder, classifier or plotting execution."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import plot_original_file as plot
else:
    plot = importlib.import_module("research.drone-native-cqt.composed.plot_original_file")

SUMMARY = (
    Path(__file__).parents[1]
    / "research/drone-native-cqt/composed/original-file-timing-summary.json"
)
SUMMARY_SHA = "455b1fe99e1b2bc79159ff9c124e2019669d945f057e12b6c83eda2eaa9bdf8a"


def test_portable_actual_summary_retains_failed_numeric_cases() -> None:
    hosts = plot.read_summary(SUMMARY, SUMMARY_SHA)
    assert len(hosts) == 2
    assert [row["historical_reference_qualified"] for row in hosts[0]["rows"]] == [
        True,
        False,
        False,
        True,
        True,
    ]
    assert all(row["historical_reference_qualified"] for row in hosts[1]["rows"])
    with pytest.raises(ValueError, match="identity/schema"):
        plot.read_summary(SUMMARY, "0" * 64)


@pytest.mark.parametrize(
    "corruption", ["missing_repeat", "negative_time", "wrong_total", "wrong_member"]
)
def test_portable_summary_rejects_misleading_measurements(tmp_path: Path, corruption: str) -> None:
    data = json.loads(SUMMARY.read_text())
    row = data["measurements"][0]["rows"][0]
    if corruption == "missing_repeat":
        row["warm_instrumented_seconds"].pop()
    elif corruption == "negative_time":
        row["cold_stage_seconds"][0] = -1
    elif corruption == "wrong_total":
        row["cold_instrumented_seconds"] += 1
    else:
        row["id"] = "different-recording"
    path = tmp_path / "corrupt.json"
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        plot.read_summary(path, hashlib.sha256(raw).hexdigest())


def test_incomplete_run_never_becomes_a_figure(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text('{"complete": false, "failure": "identity mismatch"}')
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="Only completed"):
        plot.read_run(tmp_path)


@pytest.mark.parametrize("value", [None, -1, float("nan"), float("inf"), True, "3"])
def test_invalid_measurements_are_not_coerced(value: object) -> None:
    with pytest.raises(ValueError):
        plot.finite(value)
