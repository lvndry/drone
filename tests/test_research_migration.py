"""Native relocation evidence stays fixed while declared Drone source/docs evolve."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from research.verify_migration import verify

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def migrated(tmp_path: Path) -> Path:
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
    return tmp_path


def evolution(root: Path) -> dict[str, Any]:
    return json.loads((root / "research/NATIVE_EVOLUTION.json").read_bytes())


def write_evolution(root: Path, value: dict[str, Any]) -> None:
    (root / "research/NATIVE_EVOLUTION.json").write_text(json.dumps(value))


def test_imported_research_provenance_and_local_dependency() -> None:
    assert verify(ROOT) == dict(
        imported_files=90,
        byte_identical=73,
        declared_adjustments=13,
        declared_evolutions=5,
        declared_additions=39,
    )


def test_original_evidence_corruption_cannot_hide_in_migration(migrated: Path) -> None:
    path = migrated / "research/drone-native-cqt/composed/evidence/mac376-raw.json.gz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Imported bytes/mode changed"):
        verify(migrated)


def test_original_evidence_cannot_be_rebound_as_an_evolution(migrated: Path) -> None:
    name = "research/drone-native-cqt/composed/evidence/mac376-raw.json.gz"
    path = migrated / name
    baseline = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(path.read_bytes() + b"changed")
    data = evolution(migrated)
    data["changes"].append(
        dict(
            path=name,
            migrated_sha256=baseline,
            current_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            git_mode="100644",
            reason="override",
        )
    )
    write_evolution(migrated, data)
    with pytest.raises(ValueError, match="evidence/notice cannot evolve"):
        verify(migrated)


def test_initial_manifest_remains_immutable_even_with_matching_evolution_pointer(
    migrated: Path,
) -> None:
    path = migrated / "research/MIGRATION.json"
    path.write_bytes(path.read_bytes() + b"\n")
    data = evolution(migrated)
    data["migration_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_evolution(migrated, data)
    with pytest.raises(ValueError, match="Original migration manifest changed"):
        verify(migrated)


@pytest.mark.parametrize(
    "mutation", ["baseline", "duplicate", "undeclared", "mode", "added_bytes", "symlink"]
)
def test_evolution_requires_complete_anchored_regular_inventory(
    migrated: Path, mutation: str
) -> None:
    data = evolution(migrated)
    if mutation == "baseline":
        data["changes"][0]["migrated_sha256"] = "0" * 64
        write_evolution(migrated, data)
    elif mutation == "duplicate":
        data["additions"].append(data["additions"][0])
        write_evolution(migrated, data)
    elif mutation == "undeclared":
        (migrated / "research/drone-native-cqt/new-unrecorded.txt").write_text("unrecorded")
    elif mutation == "mode":
        path = migrated / data["changes"][0]["path"]
        path.chmod(0o755)
    elif mutation == "added_bytes":
        path = migrated / data["additions"][0]["path"]
        path.write_bytes(path.read_bytes() + b"changed")
    else:
        path = migrated / data["additions"][0]["path"]
        raw = path.read_bytes()
        path.unlink()
        outside = migrated / "outside-copy"
        outside.write_bytes(raw)
        path.symlink_to(outside)
    with pytest.raises(ValueError):
        verify(migrated)
