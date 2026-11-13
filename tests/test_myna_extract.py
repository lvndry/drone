"""Extraction admission and ordered cache publication, without encoder or audio execution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.myna_extract import (
    ViewBuffer,
    atomic_json,
    digest,
    owned_path,
    projected_output_bound,
    verify,
)


def test_embedding_batches_preserve_all_windows_and_publish_npy(tmp_path: Path) -> None:
    buffer = ViewBuffer(19)
    first = np.arange(16 * 384, dtype=np.float32).reshape(16, 384)
    second = np.full((3, 384), -7, dtype=np.float32)
    buffer.append(0, first)
    buffer.append(16, second)
    target = tmp_path / "complete.npy"
    receipt = buffer.save(target)
    assert receipt == {"bytes": target.stat().st_size, "sha256": digest(target)}
    actual = np.load(target, allow_pickle=False)
    np.testing.assert_array_equal(actual, np.concatenate([first, second]))
    assert not target.with_name("complete.npy.pending").exists()


@pytest.mark.parametrize("start", [1, 15, 17])
def test_reordered_or_skipped_batches_rejected(start: int) -> None:
    buffer = ViewBuffer(20)
    buffer.append(0, np.zeros((16, 384), dtype=np.float32))
    with pytest.raises(ValueError, match="order/coverage"):
        buffer.append(start, np.zeros((1, 384), dtype=np.float32))


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((17, 384), dtype=np.float32),
        np.zeros((1, 383), dtype=np.float32),
        np.zeros((1, 384), dtype=np.float64),
        np.full((1, 384), np.nan, dtype=np.float32),
    ],
)
def test_invalid_encoder_outputs_rejected(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        ViewBuffer(20).append(0, values)


def test_incomplete_view_never_creates_blob(tmp_path: Path) -> None:
    buffer = ViewBuffer(20)
    buffer.append(0, np.zeros((16, 384), dtype=np.float32))
    with pytest.raises(ValueError, match="Incomplete"):
        buffer.save(tmp_path / "partial.npy")
    assert list(tmp_path.iterdir()) == []


def test_existing_blob_is_not_replaced(tmp_path: Path) -> None:
    target = tmp_path / "existing.npy"
    target.write_bytes(b"preserved")
    buffer = ViewBuffer(1)
    buffer.append(0, np.zeros((1, 384), dtype=np.float32))
    with pytest.raises(ValueError, match="replace"):
        buffer.save(target)
    assert target.read_bytes() == b"preserved"


def test_owned_output_rejects_escape_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "outside").symlink_to(tmp_path, target_is_directory=True)
    for path in ["../other.npy", str(tmp_path / "absolute.npy"), "outside/other.npy"]:
        with pytest.raises(ValueError, match="Escaping"):
            owned_path(root, path)
    assert owned_path(root, "embeddings/good.npy") == root / "embeddings/good.npy"


def test_source_binding_and_atomic_receipt(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    atomic_json(path, {"complete": False, "windows": 3})
    expected = digest(path)
    assert verify({"path": str(path), "sha256": expected}) == path
    assert json.loads(path.read_text()) == {"complete": False, "windows": 3}
    path.write_text("{}")
    with pytest.raises(ValueError, match="identity"):
        verify({"path": str(path), "sha256": expected})


def test_output_budget_accounts_for_every_window_and_blob() -> None:
    small = {"tracks": [{"views": [{"windows": 3}, {"windows": 7}]}]}
    large = {"tracks": [{"views": [{"windows": 3}, {"windows": 8}]}]}
    extra = {"tracks": [{"views": [{"windows": 3}, {"windows": 7}, {"windows": 1}]}]}
    assert projected_output_bound(large) - projected_output_bound(small) == 384 * 4
    assert projected_output_bound(extra) - projected_output_bound(small) == 384 * 4 + 256 + 2048
    with pytest.raises(ValueError, match="geometry"):
        projected_output_bound({"tracks": [{"views": [{"windows": 0}]}]})
