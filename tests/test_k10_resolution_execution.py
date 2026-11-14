"""Metadata/control-flow only; no GPU/tensor/server/model execution."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import experiments.k10_resolution_execution as execution
import tools.run_k10_gpu1_resource as qualified
import tools.run_k10_resolution_fit as guardian
from experiments.k10_gpu1_resource import BACKEND_POLICY, GPU0, GPU1, POLICY
from tests.test_k10_resolution_fit import gpu_arms
from tools.musicfm_crop_io import load, save, sha


def config() -> dict[str, Any]:
    return {
        "caps": execution.CAPS.copy(),
        "policy": POLICY.copy(),
        "backend_policy": BACKEND_POLICY.copy(),
        "gpu_uuid": GPU1,
        "gpu0_uuid": GPU0,
        "model_order": ["baseline", "candidate"],
        "updates_per_model": 8280,
        "selection_events": 120,
        "crop_frames": 320,
        "qualification_sha256": execution.QUALIFIED.copy(),
        "not_before_epoch": 1.0,
        "deadline_epoch": 1801.0,
        "protected": {
            "protected_starts": [
                {"pid": pid, "start_ticks": 100} for pid in (1478814, 3448680, 1481756, 3451013)
            ],
            "active_job_starts": [
                {"pid": pid, "start_ticks": 100} for pid in (1294, 1307, 3814436, 3814440)
            ],
        },
    }


def test_fixed_contract_requires_all_caps_backend_and_protected_jobs() -> None:
    contract_config = config()
    execution.contract(contract_config)
    for key, invalid in (
        ("caps", None),
        ("backend_policy", {}),
        ("model_order", ["candidate", "baseline"]),
        ("updates_per_model", 8279),
    ):
        changed = copy.deepcopy(contract_config)
        changed[key] = invalid
        with pytest.raises(ValueError):
            execution.contract(changed)
    contract_config["protected"]["active_job_starts"].pop()
    with pytest.raises(ValueError, match="eight"):
        execution.contract(contract_config)


def test_qualified_pure_helpers_are_reused_without_policy_mutation() -> None:
    for name in ("environment", "sensor", "violation", "start_identity", "terminate_own"):
        assert getattr(guardian, name) is getattr(qualified, name)
    assert guardian.POLICY == POLICY
    assert execution.CAPS["wall_seconds"] == 1800 and execution.CAPS["cpu_seconds"] == 3600


@pytest.mark.parametrize("fail_second", [False, True])
def test_guardian_separate_workers_releases_and_completed_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_second: bool
) -> None:
    run_config = config()
    now = time.time()
    run_config.update(not_before_epoch=now - 1, deadline_epoch=now + 1799)
    # Time window is checked separately in contract; use a live fake wall here.
    monkeypatch.setattr(guardian, "contract", lambda _: None)
    monkeypatch.setattr(guardian, "bindings", lambda _: None)
    monkeypatch.setattr(guardian, "environment", lambda _: None)
    monkeypatch.setattr(guardian, "ADMISSION_SECONDS", 0)
    monkeypatch.setattr(guardian, "uncertainty", lambda *args: {"diagnostic": True})
    monkeypatch.setattr(guardian, "movements", lambda *args: {"paired": True})
    monkeypatch.setattr(guardian, "start_identity", lambda _: 100)
    monkeypatch.setattr(guardian, "process_tree", lambda _: (1024, 1.0))
    monkeypatch.setattr(guardian, "disk_bytes", lambda _: 100)
    monkeypatch.setattr(
        guardian.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3)
    )
    original_read = Path.read_text

    def read(path: Path, *args: Any, **kwargs: Any) -> str:  # noqa: ANN401
        if str(path) == "/proc/meminfo":
            return "MemAvailable: 104857600 kB\n"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(
        guardian,
        "sensor",
        lambda *_: {"owned_bytes": 0, "free_bytes": 10 * 1024**3, "gpu0_free_bytes": 10 * 1024**3},
    )
    baseline_result, candidate_result, historical_result = gpu_arms()
    historical = tmp_path / "old.json"
    save(historical, historical_result)
    import gzip

    groups = tmp_path / "groups.csv.gz"
    groups.write_bytes(gzip.compress(b"id,derived_uncertainty_group\nb0,g0\n", mtime=0))
    run_config.update(
        historical_gpu_control=str(historical),
        historical_gpu_control_sha256=sha(historical),
        groups=str(groups),
        rows=[],
    )
    config_path = tmp_path / "config.json"
    approval = tmp_path / "approval.json"
    save(config_path, run_config)
    save(
        approval,
        {
            "authorized": True,
            "allow_sequential_pair": True,
            "allow_training": True,
            "config_sha256": sha(config_path),
            "deadline_epoch": run_config["deadline_epoch"],
        },
    )
    pids = []

    class Child:
        def __init__(self, command: list[str], **_kwargs: Any) -> None:  # noqa: ANN401
            self.pid = 1000 + len(pids)
            self.returncode = 0
            pids.append(self.pid)
            name = command[command.index("--model") + 1]
            if fail_second and name == "candidate":
                self.returncode = 1
            output_dir = Path(command[command.index("--output") + 1])
            output_dir.mkdir()
            save(
                output_dir / "result.json",
                baseline_result if name == "baseline" else candidate_result,
            )

        def poll(self) -> int:
            return self.returncode

        def wait(self) -> int:
            return self.returncode

    monkeypatch.setattr(guardian.subprocess, "Popen", Child)
    output = tmp_path / "run"
    if fail_second:
        with pytest.raises(RuntimeError, match="no retry"):
            guardian.supervise(config_path, approval, output)
        assert len(set(pids)) == 2
        assert not load(output / "controller.json")["complete"]
        assert (output / "baseline/result.json").exists()
        assert not (output / "result.json").exists()
        return
    guardian.supervise(config_path, approval, output)
    assert len(set(pids)) == 2
    assert load(output / "controller.json")["complete"]
    assert load(output / "result.json")["continuation_gate"]["continuation_earned"]
    assert all((output / f"{name}-released.json").exists() for name in ("baseline", "candidate"))


def test_missing_approval_never_creates_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guardian, "contract", lambda _: None)
    monkeypatch.setattr(guardian, "bindings", lambda _: None)
    config_path = tmp_path / "c.json"
    approval_path = tmp_path / "p.json"
    save(config_path, config())
    save(approval_path, {"authorized": False})
    with pytest.raises(ValueError, match="approval"):
        guardian.supervise(config_path, approval_path, tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("change", ["forward", "policy", "protected", "historical", "groups"])
def test_bindings_reject_changed_qualified_source_and_evidence_before_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    import sys

    helper = tmp_path / "helper.py"
    history = tmp_path / "old.json"
    groups = tmp_path / "groups.gz"
    plan = tmp_path / "plan.json"
    sensor = tmp_path / "sensor"
    for path in (helper, history, groups, sensor):
        path.write_bytes(b"pinned")
    save(plan, {"uncertainty_groups_sha256": sha(groups)})
    interpreter = {"path": sys.executable, "sha256": sha(Path(sys.executable))}
    qualification = {
        "policy": POLICY.copy(),
        "protected": {"starts": [1]},
        "source_sha256": {"helper.py": sha(helper)},
        "interpreter": interpreter,
        "runtime": {
            "modules": {},
            "files": [],
            "nvidia_smi": {"path": str(sensor), "sha256": sha(sensor)},
        },
    }
    run_config = {
        **copy.deepcopy(qualification),
        "qualification_root": str(tmp_path),
        "plan": str(plan),
        "historical_gpu_control": str(history),
        "historical_gpu_control_sha256": sha(history),
        "groups": str(groups),
        "groups_sha256": sha(groups),
    }
    monkeypatch.setattr(execution, "contract", lambda _: None)
    monkeypatch.setattr(execution, "qualification", lambda _: qualification)
    monkeypatch.setattr(execution, "ROOT", tmp_path)
    monkeypatch.setattr(execution, "sources", lambda: qualification["source_sha256"])
    execution.bindings(run_config)
    if change == "forward":
        helper.write_bytes(b"different forward, same init")
    elif change in ("policy", "protected"):
        run_config[change] = {}
    else:
        (history if change == "historical" else groups).write_bytes(b"changed")
    with pytest.raises(ValueError):
        execution.bindings(run_config)
