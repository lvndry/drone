"""Native research relocation preserves evidence and only declared adaptations."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from research.verify_migration import verify

ROOT = Path(__file__).resolve().parents[1]


def test_imported_research_provenance_and_local_dependency() -> None:
    assert verify(ROOT) == dict(imported_files=90, byte_identical=77, declared_adjustments=13)


def test_original_evidence_corruption_cannot_hide_in_migration(tmp_path: Path) -> None:
    shutil.copytree(
        ROOT / "research",
        tmp_path / "research",
        ignore=shutil.ignore_patterns("target", "__pycache__"),
    )
    workflow = tmp_path / ".github/workflows"
    workflow.mkdir(parents=True)
    shutil.copyfile(
        ROOT / ".github/workflows/drone-native-research.yml", workflow / "drone-native-research.yml"
    )
    path = tmp_path / "research/drone-native-cqt/composed/evidence/mac376-raw.json.gz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Imported bytes/mode changed"):
        verify(tmp_path)
