"""Permit, source and owner-lifecycle tests; never launch a model or real child."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from experiments.musicfm_bn_execution import bound_json, score_admission, useful_signal
from tools import run_musicfm_bn as guard


def test_interpreter_guard_preserves_overlay_even_with_same_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    overlay = tmp_path / "overlay-python"
    shared = tmp_path / "shared-python"
    overlay.symlink_to(sys.executable)
    shared.symlink_to(sys.executable)
    assert overlay.resolve() == shared.resolve()
    monkeypatch.setattr(guard.sys, "executable", str(overlay))
    guard.verify_original_interpreter({"command": [str(overlay), "-m", "worker"]})
    with pytest.raises(ValueError, match="isolated Python"):
        guard.verify_original_interpreter({"command": [str(shared), "-m", "worker"]})


@pytest.mark.parametrize("command", [None, [], [None], "python", ["relative/python"]])
def test_interpreter_guard_rejects_missing_or_ambiguous_command(command: object) -> None:
    with pytest.raises(ValueError, match="isolated Python"):
        guard.verify_original_interpreter({"command": command})


@pytest.fixture
def permit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(guard.__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    monkeypatch.setattr(guard.time, "time", lambda: 1000.0)
    monkeypatch.setattr(guard.os, "sched_getaffinity", lambda _: {8, 9}, raising=False)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-00000000-0000-0000-0000-000000000000")
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(key, "1")
    config = dict(
        mode="single-bn-diagnostic",
        caps=dict(guard.CAPS),
        allocator_bytes=16 * guard.GIB,
        gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
        torch_threads=2,
        cpus=[8, 9],
        not_before_epoch=995,
        deadline_epoch=2795,
        protected_starts=[],
        authorized_stopped_pids=[],
        source_sha256={name: guard.sha(root / name) for name in guard.REQUIRED_SOURCE},
        output_root=str((tmp_path / "out").resolve()),
        selected_checkpoint_sha256="b" * 64,
    )
    approval = dict(
        authorized=True,
        allow_bn_diagnostic=True,
        allow_training=False,
        config_sha256="c" * 64,
        authorized_stopped_pids=[],
    )
    return config, approval


def test_exact_diagnostic_permit_is_accepted_without_any_execution(
    permit: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    config, approval = permit
    guard.validate_launch(config, approval, "c" * 64)


def test_guard_view_drops_only_authorized_stops() -> None:
    config = {
        "protected_starts": [
            {"pid": 1, "start_ticks": 2},
            {"pid": 3, "start_ticks": 4},
        ],
        "authorized_stopped_pids": [3],
    }
    view = guard.guard_view(config)
    assert view["protected_starts"] == [{"pid": 1, "start_ticks": 2}]
    assert config["protected_starts"] == [
        {"pid": 1, "start_ticks": 2},
        {"pid": 3, "start_ticks": 4},
    ]


def test_authorized_stop_must_be_protected_and_approved(
    permit: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    config, approval = permit
    config["protected_starts"] = [{"pid": 999999, "start_ticks": 111966827}]
    config["authorized_stopped_pids"] = [999999]
    approval["authorized_stopped_pids"] = [999999]
    guard.validate_launch(config, approval, "c" * 64)

    approval["authorized_stopped_pids"] = []
    with pytest.raises(ValueError, match="Authorized-stopped"):
        guard.validate_launch(config, approval, "c" * 64)

    config["authorized_stopped_pids"] = [999]
    approval["authorized_stopped_pids"] = [999]
    with pytest.raises(ValueError, match="Authorized-stopped"):
        guard.validate_launch(config, approval, "c" * 64)


@pytest.mark.parametrize(
    "defect",
    [
        "training",
        "hash",
        "cpu_cap",
        "allocator",
        "affinity",
        "missing_source",
        "source_bytes",
        "path_override",
        "outside_window",
    ],
)
def test_permit_cannot_weaken_resources_or_source_binding(
    permit: tuple[dict[str, Any], dict[str, Any]], defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, approval = permit
    if defect == "training":
        approval["allow_training"] = True
    elif defect == "hash":
        approval["config_sha256"] = "d" * 64
    elif defect == "cpu_cap":
        config["caps"]["cpu_seconds"] += 1
    elif defect == "allocator":
        config["allocator_bytes"] += 1
    elif defect == "affinity":
        config["cpus"] = [8]
    elif defect == "missing_source":
        config["source_sha256"].pop("experiments/musicfm_bn_execution.py")
    elif defect == "source_bytes":
        config["source_sha256"]["experiments/musicfm_bn_execution.py"] = "0" * 64
    elif defect == "path_override":
        monkeypatch.setenv("PYTHONPATH", "/unreviewed")
    else:
        config["deadline_epoch"] = 999
    with pytest.raises(ValueError):
        guard.validate_launch(config, approval, "c" * 64)


def test_import_identity_rejects_external_numeric_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Bind all actual local imports, then inject a competing editable-checkout module.
    source = Path(guard.__file__).resolve().parents[1]
    bindings = {
        str(path.relative_to(source)): guard.sha(path)
        for folder in ("drone", "tools", "experiments")
        for path in (source / folder).rglob("*.py")
    }
    guard.verify_import_paths(bindings)
    shadow = ModuleType("experiments.unreviewed_numeric")
    shadow.__file__ = str(tmp_path / "unreviewed_numeric.py")
    monkeypatch.setitem(sys.modules, shadow.__name__, shadow)
    with pytest.raises(ValueError, match="outside"):
        guard.verify_import_paths(bindings)


def test_late_timeout_and_cleanup_error_cannot_leave_complete_controller(
    permit: tuple[dict[str, Any], dict[str, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, approval = permit
    config_path = tmp_path / "config.json"
    approval_path = tmp_path / "approval.json"
    raw = json.dumps(config).encode()
    config_path.write_bytes(raw)
    approval["config_sha256"] = hashlib.sha256(raw).hexdigest()
    approval_path.write_text(json.dumps(approval))
    output = Path(config["output_root"])
    clock = [1000.0]
    monkeypatch.setattr(guard.time, "time", lambda: clock[0])
    monkeypatch.setattr(guard.time, "sleep", lambda _: None)
    monkeypatch.setattr(guard, "verify_import_paths", lambda _: None)
    monkeypatch.setattr(
        guard,
        "read_original",
        lambda _: {
            "config": b'{"protected_starts":[],"source_sha256":{}}',
            "launch": json.dumps({"command": [sys.executable]}).encode(),
        },
    )
    monkeypatch.setattr(guard, "original_failures_absent", lambda: True)
    monkeypatch.setattr(
        guard,
        "verify_terminal_teacher",
        lambda *_, **__: SimpleNamespace(
            result_sha256="a" * 64,
            checkpoint_sha256="b" * 64,
            original_gate_earned=False,
            best_update=828,
        ),
    )
    monkeypatch.setattr(guard, "proc", lambda _: (11, "R"))
    monkeypatch.setattr(guard, "descendants", lambda _: {900000: 12})
    monkeypatch.setattr(guard, "alive", lambda *_: False)
    monkeypatch.setattr(guard, "capacity_guard", lambda *_: None)
    monkeypatch.setattr(guard, "disk_bytes", lambda _: 0)
    monkeypatch.setattr(
        guard, "owned_resources", lambda *_, **__: {"rss_bytes": 1, "cpu_seconds": 1.0}
    )
    queries = 0

    def resources(*_: object) -> None:
        nonlocal queries
        queries += 1
        if queries == 2:  # Final query after worker exit, not the first admission.
            clock[0] = 3000

    monkeypatch.setattr(guard, "resources", resources)

    class Child:
        pid = 900000
        returncode: int | None = None

        def poll(self) -> int:
            self.returncode = 0
            return 0

        def wait(self) -> int:
            self.returncode = 0
            return 0

    def spawn(
        command: list[str],
        *,
        stdout: object,
        stderr: int,
        env: dict[str, str],
        cwd: Path,
        start_new_session: bool,
    ) -> Child:
        assert cwd == Path(guard.__file__).resolve().parents[1]
        assert start_new_session and env["PYTHONNOUSERSITE"] == "1"
        assert Path(command[command.index("--config") + 1]).is_absolute()
        (output / "worker").mkdir()
        (output / "worker/result.json").write_text(
            json.dumps(
                dict(
                    status="COMPLETE_SINGLE_BN_DIAGNOSTIC_NOT_PROMOTED", promotion_authorized=False
                )
            )
        )
        return Child()

    monkeypatch.setattr(guard.subprocess, "Popen", spawn)

    def cleanup(*_: object) -> None:
        raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(guard, "stop_and_wait", cleanup)
    with pytest.raises(TimeoutError):
        guard.supervise(config_path, approval_path, output)
    failure = json.loads((output / "failure.json").read_text())
    controller = json.loads((output / "controller.json").read_text())
    assert "cutoff" in failure["error"] and "cleanup failure" in failure["cleanup_error"]
    assert controller["complete"] is False and controller["cleanup_error"] is not None


def test_useful_signal_does_not_allow_bp_weighted_regression() -> None:
    baseline = {
        "beatport": {"exact_correct": 83, "symmetric_units10": 900, "asymmetric_units10": 890},
        "fmak": {"exact_correct": 166},
    }
    candidate = {"beatport": dict(baseline["beatport"]), "fmak": {"exact_correct": 171}}
    assert useful_signal(candidate, baseline)
    candidate["beatport"]["symmetric_units10"] -= 1
    assert not useful_signal(candidate, baseline)


def test_postseal_budget_requires_the_complete_fresh_comparison() -> None:
    assert score_admission(303, 81.2, 20)["fits_budget"] is True
    assert score_admission(302, 81.2, 20)["fits_budget"] is False
    with pytest.raises(ValueError):
        score_admission(float("nan"), 81.2, 20)


def test_metadata_parser_uses_only_the_pinned_bytes(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    raw = b'{"value":1}'
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert bound_json(path, digest, lambda: None) == {"value": 1}
    path.write_bytes(b'{"value":2}')
    with pytest.raises(ValueError, match="changed"):
        bound_json(path, digest, lambda: None)
