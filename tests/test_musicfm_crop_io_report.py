"""Publication checker revalidates every immutable waveform case without audio."""

import gzip
import json
from pathlib import Path

import pytest

from tools.musicfm_crop_io_report import verify


def evidence() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments/studies/musicfm-crop-io/data/runtime-raw.json.gz"
    return json.loads(gzip.decompress(path.read_bytes()))


def test_all_frozen_cases_are_bound_and_exact() -> None:
    result = verify(evidence())
    assert result["cases"] == 144
    assert result["tracks"] == 16
    assert result["nonempty_stderr_files"] == 0


@pytest.mark.parametrize("name", ["sealed-config.json", "qualification-v1/report.json"])
def test_changed_raw_bytes_are_rejected(name: str) -> None:
    files = evidence()
    files[name] += " "
    with pytest.raises(ValueError, match="Frozen raw"):
        verify(files)


def test_missing_per_case_evidence_is_rejected() -> None:
    files = evidence()
    del files["qualification-v1/case-01-01.json"]
    with pytest.raises(KeyError):
        verify(files)
