"""Run the guardian's real state machine using only synthetic files and mocked processes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tools import run_musicfm_teacher as teacher


@dataclass
class Clock:
    elapsed: float = 0.0

    def monotonic(self) -> float:
        return self.elapsed

    def time(self) -> float:
        return 100 + self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds
        assert self.elapsed < 1, (
            "Synthetic worker must retire before the normal disk-check interval"
        )


@dataclass
class Harness:
    config: Path
    approval: Path
    root: Path
    worker: Mock
    stop: Mock
    events: list[str]

    def run(self) -> None:
        teacher.supervise(self.config, self.approval, self.root)


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    config_path = tmp_path / "config.json"
    approval_path = tmp_path / "approval.json"
    root = tmp_path / "run"
    source_file = "tools/run_musicfm_teacher.py"
    config = {
        "mode": "fit",
        "eval_batch_size": 16,
        "not_before_epoch": 90,
        "deadline_epoch": 190,
        "source_sha256": {
            source_file: teacher.sha(Path(teacher.__file__).parents[1] / source_file)
        },
        "cpus": [2],
        "gpu_uuid": "synthetic-gpu",
        "caps": {"wall_seconds": 100, "rss_bytes": 1000, "cpu_seconds": 100, "output_bytes": 10000},
    }
    teacher.save(config_path, config)
    teacher.save(
        approval_path,
        {
            "authorized": True,
            "config_sha256": teacher.sha(config_path),
            "allow_training": True,
        },
    )
    clock = Clock()
    monkeypatch.setattr(teacher, "time", clock)
    monkeypatch.setattr(teacher.os, "sched_getaffinity", lambda _: {2}, raising=False)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "synthetic-gpu")
    monkeypatch.setattr(teacher, "proc", lambda _: (123, "S"))
    monkeypatch.setattr(teacher, "descendants", lambda _: {999: 456})
    monkeypatch.setattr(teacher, "capacity_guard", Mock())
    monkeypatch.setattr(teacher, "resources", Mock())
    monkeypatch.setattr(teacher, "process_tree", lambda _: (400, 1.0))
    events: list[str] = []
    worker = Mock(pid=999, returncode=0)
    worker.poll.side_effect = [None, 0]
    worker.wait.return_value = 0

    def launch(*_args: object, **_kwargs: object) -> Mock:
        (root / "worker").mkdir()
        teacher.save(
            root / "worker/result.json", {"status": "COMPLETE_FIXED_PILOT_NOT_RELEASE_EVIDENCE"}
        )
        return worker

    monkeypatch.setattr(teacher, "subprocess", SimpleNamespace(Popen=launch, STDOUT=-2))
    stop = Mock()
    monkeypatch.setattr(teacher, "stop_and_wait", stop)
    original_progress = teacher.save_or_replace

    def progress(path: Path, value: dict[str, Any]) -> None:
        events.append(path.name)
        original_progress(path, value)

    def disk(_root: Path) -> int:
        events.append("disk-check")
        return 100

    monkeypatch.setattr(teacher, "save_or_replace", progress)
    monkeypatch.setattr(teacher, "disk_bytes", disk)
    return Harness(config_path, approval_path, root, worker, stop, events)


def test_completion_forces_final_disk_progress_and_caps_before_success(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_save = teacher.save
    events = harness.events

    def process_tree(_pid: int) -> tuple[int, float]:
        events.append("owned-check")
        return 400, 1.0

    def save(path: Path, value: dict[str, Any]) -> None:
        if path.name == "held-success.json":
            assert events[-3:] == ["disk-check", "progress.json", "owned-check"]
            assert (
                teacher.load(harness.root / "progress.json")["state"]
                == "WORKER_COMPLETE_PENDING_CHECKS"
            )
        original_save(path, value)

    monkeypatch.setattr(teacher, "process_tree", process_tree)
    monkeypatch.setattr(teacher, "save", save)
    harness.run()
    controller = teacher.load(harness.root / "controller.json")
    assert controller["state"] == "COMPLETE_NO_RAM_OWNERSHIP"
    assert controller["error"] is controller["cleanup_error"] is None
    assert controller["worker_exit"] == 0
    assert controller["terminal_resources"] == {"rss_bytes": 400, "cpu_seconds": 1.0}
    assert teacher.load(harness.root / "held-success.json")["result_sha256"] == teacher.sha(
        harness.root / "worker/result.json"
    )
    assert not (harness.root / "failure.json").exists()
    harness.stop.assert_not_called()
    harness.worker.wait.assert_called_once_with()


@pytest.mark.parametrize(
    "failure_site", ["disk", "progress", "final-cap", "success-write", "validation"]
)
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_late_failure_never_reports_complete_and_preserves_error_during_cleanup(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
    cleanup_fails: bool,
) -> None:
    error = RuntimeError("synthetic late " + failure_site)
    original_progress = teacher.save_or_replace
    original_save = teacher.save

    def progress(path: Path, value: dict[str, Any]) -> None:
        if failure_site == "progress" and path.name == "progress.json":
            raise error
        original_progress(path, value)

    def save(path: Path, value: dict[str, Any]) -> None:
        if failure_site == "success-write" and path.name == "held-success.json":
            raise error
        if failure_site == "validation" and path.name == "result.json":
            value = {"status": "INCOMPLETE"}
        original_save(path, value)

    def process_tree(_pid: int) -> tuple[int, float]:
        return 400, 101.0 if "progress.json" in harness.events else 1.0

    monkeypatch.setattr(teacher, "save_or_replace", progress)
    monkeypatch.setattr(teacher, "save", save)
    if failure_site == "disk":
        monkeypatch.setattr(teacher, "disk_bytes", lambda _: 10001)
    if failure_site == "final-cap":
        monkeypatch.setattr(teacher, "process_tree", process_tree)
    if cleanup_fails:
        harness.stop.side_effect = RuntimeError("synthetic cleanup failure")
    with pytest.raises((RuntimeError, ValueError)) as raised:
        harness.run()
    expected = {
        "disk": "Output disk cap",
        "final-cap": "Owned teacher process-tree finite CPU/RSS cap",
        "validation": "Incomplete worker result",
    }.get(failure_site, str(error))
    assert str(raised.value) == expected
    failure = teacher.load(harness.root / "failure.json")
    controller = teacher.load(harness.root / "controller.json")
    assert failure["state"] == controller["state"] == "FAILED"
    assert failure["error"] == controller["error"] == repr(raised.value)
    assert not (harness.root / "held-success.json").exists()
    harness.stop.assert_called_once_with(harness.worker, {999: 456})
    if cleanup_fails:
        assert (
            failure["cleanup_error"]
            == controller["cleanup_error"]
            == "RuntimeError('synthetic cleanup failure')"
        )
    else:
        assert "cleanup_error" not in failure
        assert controller["cleanup_error"] is None
