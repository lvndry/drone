"""Verify the frozen native import and explicitly recorded subsequent Drone work."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MIGRATION_SHA256 = "b44969c2ac90402126907982913d847e314b84b502581185ff12fcadc32fb159"
FOLDERS = ("research/drone-native-cqt", "research/drone-native-cached")


def entries(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for record in records:
        name = record["path"]
        if not isinstance(name, str):
            raise ValueError("Invalid/repeated migration path")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name or name in result:
            raise ValueError("Invalid/repeated migration path")
        result[name] = record
    return result


def current_identity(root: Path, name: str) -> tuple[str, str]:
    path = root / name
    if (
        path.is_symlink()
        or not path.is_file()
        or any(
            parent.is_symlink()
            for parent in path.parents
            if parent != root and root in parent.parents
        )
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError("Missing/nonregular imported path: " + name)
    return hashlib.sha256(path.read_bytes()).hexdigest(), (
        "100755" if path.stat().st_mode & 0o111 else "100644"
    )


def may_evolve(name: str) -> bool:
    """Allow explicit source/docs/CI updates, never rebinding recorded evidence or notices."""
    path = PurePosixPath(name)
    return not (
        any(part in {"evidence", "figures", "vendor"} for part in path.parts)
        or path.name.startswith("LICENSE")
        or path.suffix in {".gz", ".json", ".png", ".svg"}
    )


def verify(root: Path = ROOT) -> dict[str, int]:
    raw = (root / "research/MIGRATION.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MIGRATION_SHA256:
        raise ValueError("Original migration manifest changed")
    manifest = json.loads(raw)
    adjustments = entries(manifest["integration_adjustments"])
    recorded = entries(manifest["files"])
    if not set(adjustments) <= set(recorded):
        raise ValueError("Adjustment has no imported identity")
    evolution = json.loads((root / "research/NATIVE_EVOLUTION.json").read_bytes())
    if (
        evolution["format"] != "drone-native-research-evolution-v1"
        or evolution["migration_sha256"] != MIGRATION_SHA256
    ):
        raise ValueError("Evolution loses frozen migration identity")
    changes, additions = entries(evolution["changes"]), entries(evolution["additions"])
    if not set(changes) <= set(recorded) or set(additions) & set(recorded):
        raise ValueError("Evolution changes/additions disagree with imported inventory")
    byte_identical = 0
    for name, item in recorded.items():
        actual, mode = current_identity(root, name)
        adjustment = adjustments.get(name)
        migrated = adjustment["current_sha256"] if adjustment else item["sha256"]
        if adjustment and adjustment["original_sha256"] != item["sha256"]:
            raise ValueError("Adjustment loses original identity: " + name)
        expected = migrated
        if name in changes:
            change = changes[name]
            if not may_evolve(name):
                raise ValueError("Imported evidence/notice cannot evolve: " + name)
            if change["migrated_sha256"] != migrated or change["git_mode"] != item["git_mode"]:
                raise ValueError("Evolution loses migrated identity/mode: " + name)
            if not change["reason"].strip():
                raise ValueError("Evolution reason required: " + name)
            expected = change["current_sha256"]
        if actual != expected or mode != item["git_mode"]:
            raise ValueError("Imported bytes/mode changed: " + name)
        byte_identical += actual == item["sha256"]
    for name, addition in additions.items():
        if not any(name.startswith(folder + "/") for folder in FOLDERS):
            raise ValueError("Addition outside native studies: " + name)
        if not addition["reason"].strip():
            raise ValueError("Addition reason required: " + name)
        if current_identity(root, name) != (addition["sha256"], addition["git_mode"]):
            raise ValueError("Added bytes/mode changed: " + name)
    for folder in FOLDERS:
        paths = {
            str(path.relative_to(root))
            for path in (root / folder).rglob("*")
            if (path.is_file() or path.is_symlink())
            and "target" not in path.relative_to(root / folder).parts
            and "__pycache__" not in path.parts
        }
        expected_paths = {name for name in (*recorded, *additions) if name.startswith(folder + "/")}
        if paths != expected_paths:
            raise ValueError("Unexpected/missing imported study file")
    cqt = (root / "research/drone-native-cqt/Cargo.toml").read_text()
    if 'drone-native-cached = { path = "../drone-native-cached" }' not in cqt:
        raise ValueError("Standalone dependency moved outside Drone")
    return dict(
        imported_files=len(recorded),
        byte_identical=byte_identical,
        declared_adjustments=len(adjustments),
        declared_evolutions=len(changes),
        declared_additions=len(additions),
    )


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
