"""Synthetic process supervision; no audio, model, GPU or corpus access."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from tools import run_cqt_expansion as expansion
from tools import run_musicfm_ram_cache as ram
from tools import run_musicfm_teacher as teacher


def fake_proc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "proc"
    root.mkdir()
    original = Path

    def mapped(value: str) -> Path:
        text = str(value)
        return (
            root / text.removeprefix("/proc").lstrip("/")
            if text.startswith("/proc")
            else original(value)
        )

    monkeypatch.setattr(expansion, "Path", mapped)
    monkeypatch.setattr(ram, "Path", mapped)
    original_sysconf = os.sysconf
    monkeypatch.setattr(
        expansion.os,
        "sysconf",
        lambda name: {"SC_CLK_TCK": 100, "SC_PAGE_SIZE": 4096}.get(name, original_sysconf(name)),
    )
    return root


def record(
    root: Path, pid: int, parent: int, start: int, cpu: int, reaped: int, pages: int
) -> None:
    path = root / str(pid)
    path.mkdir()
    fields = ["0"] * 50
    fields[0], fields[1], fields[19] = "S", str(parent), str(start)
    fields[11], fields[13] = str(cpu), str(reaped)
    (path / "stat").write_text(f"{pid} (synthetic) " + " ".join(fields))
    (path / "statm").write_text(f"{pages} {pages}")
    (path / "task" / str(pid)).mkdir(parents=True)
    (path / "task" / str(pid) / "children").write_text("")


def test_all_thread_children_counted_once_with_reaped_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = fake_proc(tmp_path, monkeypatch)
    record(root, 100, 1, 10, 10, 20, 2)
    record(root, 200, 100, 20, 7, 3, 1)
    record(root, 999, 1, 99, 100, 0, 100)  # Not a descendant; never admitted.
    for tid in (101, 102):
        (root / "100/task" / str(tid)).mkdir()
        (root / "100/task" / str(tid) / "children").write_text("200 999")
    assert expansion.process_children(100) == {200: 20}
    assert expansion.process_tree(100) == (3 * 4096, 0.4)
    assert ram.descendants(100) == {100: 10, 200: 20}


def test_reused_child_start_identity_is_not_counted_or_tracked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = fake_proc(tmp_path, monkeypatch)
    record(root, 100, 1, 10, 10, 0, 2)
    record(root, 200, 100, 999, 100, 0, 100)

    def children(pid: int) -> dict[int, int]:
        return {200: 20} if pid == 100 else {}

    monkeypatch.setattr(expansion, "process_children", children)
    monkeypatch.setattr(ram, "process_children", children)
    assert expansion.process_tree(100) == (2 * 4096, 0.1)
    assert ram.descendants(100) == {100: 10}


def test_teacher_budget_includes_guardian_and_terminal_reaped_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = Mock(return_value=(600, 11.0))
    monkeypatch.setattr(teacher, "process_tree", usage)
    peaks = {"rss": 0.0, "cpu": 0.0}
    config = dict(caps=dict(rss_bytes=1000, cpu_seconds=10))
    with pytest.raises(RuntimeError, match="Owned teacher process-tree"):
        teacher.owned_resources(config, peaks)
    usage.assert_called_once_with(os.getpid())
    assert peaks == dict(rss=600, cpu=11.0)
    assert teacher.owned_resources(config, peaks, enforce=False) == dict(
        rss_bytes=600, cpu_seconds=11.0
    )


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="Linux process supervision")
def test_thread_spawned_separate_session_is_accounted_and_stopped(tmp_path: Path) -> None:
    script = """
import json,subprocess,threading,time
ready=threading.Event()
def child():
 p=subprocess.Popen([__import__('sys').executable,'-c','import time; x=bytearray(16*1024**2); time.sleep(30)'], start_new_session=True)
 print(json.dumps({'child':p.pid}),flush=True)
 ready.set()
 p.wait()
t=threading.Thread(target=child);t.start();ready.wait();time.sleep(30)
"""
    import json
    import time

    with (tmp_path / "worker.log").open("wb") as log:
        worker = subprocess.Popen(
            [sys.executable, "-u", "-c", script], stdout=log, start_new_session=True
        )
        child = None
        tracked: dict[int, int] = {}
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                text = (tmp_path / "worker.log").read_text()
                if text:
                    child = json.loads(text)["child"]
                    tracked = ram.descendants(worker.pid)
                    if child in tracked:
                        break
                time.sleep(0.01)
            assert child is not None and child in tracked
            own_rss = int(Path(f"/proc/{worker.pid}/statm").read_text().split()[1]) * os.sysconf(
                "SC_PAGE_SIZE"
            )
            assert expansion.process_tree(worker.pid)[0] > own_rss
            ram.stop_and_wait(worker, tracked)
            assert all(not ram.alive(pid, start) for pid, start in tracked.items())
        finally:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=3)
            if child is not None and child in tracked and ram.alive(child, tracked[child]):
                os.kill(child, 9)
