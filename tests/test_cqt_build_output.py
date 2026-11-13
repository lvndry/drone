"""Test output accounting without importing the macOS guardian or launching Cargo."""

from __future__ import annotations

import ast
import errno
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "research/drone-native-cqt/composed/qualify_boundary_build.py.txt"
)


def scanner(root: Path) -> tuple[Callable[[], int], dict[str, Any]]:
    tree = ast.parse(SCRIPT.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"files", "owned_bytes"}
    ]
    assert {node.name for node in functions} == {"files", "owned_bytes"}
    namespace: dict[str, Any] = {"OUT": root, "errno": errno}
    body: list[ast.stmt] = list(functions)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return cast(Callable[[], int], namespace["owned_bytes"]), namespace


def test_removed_object_is_ignored_and_remaining_outputs_are_counted(tmp_path: Path) -> None:
    scan, namespace = scanner(tmp_path)
    transient, retained = tmp_path / "temporary.rcgu.o", tmp_path / "retained.o"
    transient.write_bytes(b"temporary")
    retained.write_bytes(b"1234567")

    def disappear_after_enumeration(root: Path) -> list[Path]:
        assert root == tmp_path
        transient.unlink()
        return [transient, retained]

    namespace["files"] = disappear_after_enumeration
    assert scan() == 7
    namespace["files"] = lambda root: [retained]
    retained.write_bytes(b"12345678901")
    assert scan() == 11  # Subsequent sampling sees growth rather than reusing a stale total.


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EIO, errno.ENOTDIR])
def test_non_enoent_stat_failures_are_not_suppressed(tmp_path: Path, error_number: int) -> None:
    scan, namespace = scanner(tmp_path)

    class FaultyFile:
        def stat(self) -> SimpleNamespace:
            raise OSError(error_number, "injected stat failure")

    namespace["files"] = lambda root: [FaultyFile()]
    with pytest.raises(OSError) as caught:
        scan()
    assert caught.value.errno == error_number


def test_owned_symlinks_remain_rejected(tmp_path: Path) -> None:
    target = tmp_path / "actual.o"
    target.write_bytes(b"123")
    (tmp_path / "alias.o").symlink_to(target)
    scan, _ = scanner(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        scan()


def test_zero_length_and_nested_files_contribute_to_same_total(tmp_path: Path) -> None:
    (tmp_path / "empty.log").write_bytes(b"")
    nested = tmp_path / "release/deps"
    nested.mkdir(parents=True)
    (nested / "one.o").write_bytes(b"abc")
    (nested / "two.o").write_bytes(b"12345")
    scan, _ = scanner(tmp_path)
    assert scan() == 8
