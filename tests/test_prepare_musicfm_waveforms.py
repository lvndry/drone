"""Pure synthetic byte/metadata checks; no audio subprocess or GPU."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tools.prepare_musicfm_waveforms import fixed_start, full_identity, identity_check, seal


def test_fixed_start_is_label_blind_reproducible_and_within_whole_track() -> None:
    samples = 24000 * 600 + 121
    start = fixed_start("fmak:000001", samples)
    assert start == fixed_start("fmak:000001", samples)
    assert 0 <= start <= samples - 12 * 24000
    assert fixed_start("short", 100) == 0
    with pytest.raises(ValueError, match="Empty"):
        fixed_start("empty", 0)


def test_finite_waveform_hashes_exact_raw_le_bytes(tmp_path: Path) -> None:
    path = tmp_path / "raw"
    raw = np.array([0.0, -0.0, 1.0, -1.0, 1e20], dtype="<f4").tobytes()
    path.write_bytes(raw)
    assert full_identity(path) == (hashlib.sha256(raw).hexdigest(), 5)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"123",
        np.array([np.nan], dtype="<f4").tobytes(),
        np.array([np.inf], dtype="<f4").tobytes(),
    ],
)
def test_bad_waveforms_fail_without_repair(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "raw"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        full_identity(path)


def test_identity_families_preserve_same_role_same_target_aliases() -> None:
    family = [{"role": "fit", "target": 2}]
    identity_check(family, {"role": "fit", "target": 2})
    with pytest.raises(ValueError, match="no repair"):
        identity_check(family, {"role": "validation", "target": 2})
    with pytest.raises(ValueError, match="no repair"):
        identity_check(family, {"role": "fit", "target": 3})


def test_sealing_requires_aware_finite_prospective_window(tmp_path: Path) -> None:
    path = tmp_path / "draft.json"
    path.write_text('{"caps":{"wall_seconds":7200}}')
    with pytest.raises(ValueError, match="UTC"):
        seal(path, "2026-11-12T12:00:00", "2026-11-12T13:00:00")
    with pytest.raises(ValueError, match="window"):
        seal(path, "2026-11-12T12:00:00Z", "2026-11-12T15:00:00Z")
    config = seal(path, "2026-11-12T12:00:00Z", "2026-11-12T13:00:00Z")
    assert config["deadline_epoch"] - config["not_before_epoch"] == 3600
