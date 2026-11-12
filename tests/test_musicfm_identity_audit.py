"""Synthetic files: identity/content/read-only errors stop without editing inputs."""

import hashlib
from pathlib import Path

import pytest

from tools.audit_musicfm_source_identity import checked


def test_equal_content_relocated_identity_is_recorded_without_edit(tmp_path: Path) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"fixed bytes")
    path.chmod(0o444)
    before = path.stat()
    result = checked(path, hashlib.sha256(b"fixed bytes").hexdigest(), 11, lambda: None, True)
    assert result["bytes"] == 11 and path.stat().st_ino == before.st_ino


def test_wrong_content_or_writable_validation_rejected(tmp_path: Path) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"fixed bytes")
    with pytest.raises(ValueError, match="Content"):
        checked(path, "a" * 64, 11, lambda: None)
    with pytest.raises(ValueError, match="read-only"):
        checked(path, hashlib.sha256(b"fixed bytes").hexdigest(), 11, lambda: None, True)


def test_mutation_while_hashing_is_not_repaired(tmp_path: Path) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"fixed bytes")

    def mutate() -> None:
        path.write_bytes(b"other bytes")

    with pytest.raises(ValueError):
        checked(path, hashlib.sha256(b"fixed bytes").hexdigest(), 11, mutate)
