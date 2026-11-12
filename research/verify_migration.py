"""Verify exact imported native-research identities and declared relocation edits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root: Path = ROOT) -> dict[str, int]:
    manifest = json.loads((root / "research/MIGRATION.json").read_bytes())
    adjustments = {item["path"]: item for item in manifest["integration_adjustments"]}
    recorded = {item["path"]: item for item in manifest["files"]}
    for name, item in recorded.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing/nonregular imported path: " + name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        change = adjustments.get(name)
        expected = change["current_sha256"] if change else item["sha256"]
        if actual != expected or bool(path.stat().st_mode & 0o111) != (
            item["git_mode"] == "100755"
        ):
            raise ValueError("Imported bytes/mode changed: " + name)
        if change and change["original_sha256"] != item["sha256"]:
            raise ValueError("Adjustment loses original identity: " + name)
    for folder in ("research/drone-native-cqt", "research/drone-native-cached"):
        paths = {
            str(path.relative_to(root))
            for path in (root / folder).rglob("*")
            if path.is_file()
            and "target" not in path.relative_to(root / folder).parts
            and "__pycache__" not in path.parts
        }
        if paths != {name for name in recorded if name.startswith(folder + "/")}:
            raise ValueError("Unexpected/missing imported study file")
    # Preserve the exact local dependency geometry; no path to another checkout.
    cqt = (root / "research/drone-native-cqt/Cargo.toml").read_text()
    if 'drone-native-cached = { path = "../drone-native-cached" }' not in cqt:
        raise ValueError("Standalone dependency moved outside Drone")
    return dict(
        imported_files=len(recorded),
        byte_identical=len(recorded) - len(adjustments),
        declared_adjustments=len(adjustments),
    )


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
