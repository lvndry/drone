"""Metadata-only rejection tests for the published BN v4 provenance closure."""

from __future__ import annotations

import copy
import difflib
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools.check_musicfm_bn_v4 import EVIDENCE, load_evidence, reconstruct_source, sha, verify


def test_terminal_receipt_and_fixed_negative_gate() -> None:
    result = verify(load_evidence())
    assert result == {
        "resources": {
            "wall_seconds": 707.6067512661684,
            "rss_bytes": 6219206656,
            "cpu_seconds": 843.28,
        },
        "exact_delta": {"beatport": -7, "fmak": 3},
        "useful_signal": False,
        "source_files": 132,
    }


@pytest.mark.parametrize("damage", ["missing", "tampered", "unlisted", "manifest"])
def test_archive_members_bound_to_frozen_manifest(tmp_path: Path, damage: str) -> None:
    archive = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, archive)
    if damage == "missing":
        (archive / "controller.json").unlink()
    elif damage == "tampered":
        (archive / "controller.json").write_text("{}")
    elif damage == "unlisted":
        (archive / "extra.json").write_text("{}")
    else:
        manifest = archive / "SHA256SUMS"
        manifest.write_text("\n".join(manifest.read_text().splitlines()[1:]) + "\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        load_evidence(archive)


@pytest.mark.parametrize(
    "damage",
    [
        "exit",
        "cleanup",
        "binding",
        "wall",
        "rss",
        "nan",
        "terminal",
        "gate",
        "confirmation",
        "panel",
        "denominator",
        "receipt",
        "cap",
        "source",
    ],
)
def test_semantic_contradictions_rejected(damage: str) -> None:
    docs = copy.deepcopy(load_evidence())
    if damage == "exit":
        docs["controller"]["worker_exit"] = 1
    elif damage == "cleanup":
        docs["controller"]["cleanup_error"] = "failed"
    elif damage == "binding":
        docs["controller"]["config_sha256"] = "0" * 64
    elif damage == "wall":
        docs["controller"]["wall_seconds"] = 3601
    elif damage == "rss":
        docs["controller"]["peaks"]["rss"] = 8 * 1024**3 + 1
    elif damage == "nan":
        docs["controller"]["peaks"]["cpu"] = float("nan")
    elif damage == "terminal":
        docs["controller"]["terminal_resources"]["cpu_seconds"] = 900
    elif damage == "gate":
        docs["result"]["useful_signal"] = True
    elif damage == "confirmation":
        docs["result"]["confirmation_used"] = True
    elif damage == "panel":
        docs["result"]["candidate_metrics"]["fmak"]["ids"].reverse()
    elif damage == "denominator":
        docs["result"]["candidate_metrics"]["fmak"]["exact"] = 0.9
    elif damage == "receipt":
        docs["admission"]["fits_budget"] = False
    elif damage == "cap":
        docs["run"]["caps"]["rss_bytes"] *= 2
    else:
        docs["source-lock"]["base_commit"] = "0" * 40
    with pytest.raises(ValueError):
        verify(docs)


def git(repository: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments], cwd=repository, stderr=subprocess.DEVNULL)


@pytest.mark.parametrize("damage", [None, "patch", "pin", "path"])
def test_source_reconstruction_is_exact_and_does_not_modify_checkout(
    tmp_path: Path, damage: str | None
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    git(repository, "init")
    before = "original\n"
    after = "historical override\n"
    (repository / "source.py").write_text(before)
    git(repository, "add", "source.py")
    git(
        repository,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    base = git(repository, "rev-parse", "HEAD").decode().strip()
    patch = tmp_path / "overrides.patch"
    patch.write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/source.py",
                tofile="b/source.py",
            )
        )
    )
    lock: dict[str, Any] = {
        "base_commit": base,
        "override_patch_sha256": sha(patch.read_bytes()),
        "source_sha256": {"source.py": sha(after.encode())},
    }
    if damage == "patch":
        patch.write_text("tampered")
    elif damage == "pin":
        lock["source_sha256"]["source.py"] = sha(before.encode())
    elif damage == "path":
        lock["source_sha256"] = {"../escape.py": "0" * 64}
    if damage is None:
        reconstruct_source(lock, patch, repository)
    else:
        with pytest.raises(ValueError):
            reconstruct_source(lock, patch, repository)
    assert (repository / "source.py").read_text() == before
    assert git(repository, "status", "--porcelain") == b""
    assert not (tmp_path / "escape.py").exists()
