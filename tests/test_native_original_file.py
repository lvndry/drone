"""Admission and resource supervision tests; no audio, CQT, or model execution."""

from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    import original_file as measurement
else:
    measurement = importlib.import_module("research.drone-native-cqt.composed.original_file")


def test_aggregate_process_group_includes_decoder_and_ignores_other_jobs() -> None:
    sample = measurement.parse_group("10 10 100 01:02\n11 10 300 00:03.25\n12 12 999 99:59\n", 10)
    assert sample.pids == (10, 11)
    assert sample.rss_bytes == 400 * 1024
    assert sample.cpu_seconds == 65.25
    assert measurement.elapsed_seconds("1-02:03:04") == 93784
    with pytest.raises(ValueError):
        measurement.elapsed_seconds("nan:01")
    with pytest.raises(ValueError):
        measurement.parse_group("10 10 -1 00:00", 10)


def test_config_rejects_reordered_panel_and_changed_original_pcm_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    panel = tmp_path / "panel.json"
    rows = [dict(id=str(i), audio_sha256=str(i), mac_manifest_sha256=f"m{i}") for i in range(5)]
    panel.write_text(json.dumps(dict(cases=rows)))
    monkeypatch.setattr(measurement, "PANEL_PATH", panel)
    monkeypatch.setattr(measurement, "pinned", lambda _spec: tmp_path)
    monkeypatch.setattr(measurement.platform, "system", lambda: "Darwin")
    config: dict[str, object] = dict(
        format="drone-original-file-measurement-v1",
        warm_repetitions=3,
        candidate="fused-within-product-mac-magnitude-platform-log1p",
        linux_cpu=None,
        build_context=dict(
            profile="release",
            **dict.fromkeys(
                ["host_description", "rustc_version", "target_triple", "rustflags", "fma_lowering"],
                "test",
            ),
        ),
        worker={},
        ffmpeg={},
        ort={},
        soxr={},
        bundle_contract={},
    )
    cases = [
        dict(
            id=r["id"],
            audio=dict(sha256=r["audio_sha256"]),
            manifest=dict(sha256=r["mac_manifest_sha256"]),
        )
        for r in rows
    ]
    config["cases"] = cases
    assert len(measurement.validate_config(config)) == 5
    config["cases"] = list(reversed(cases))
    with pytest.raises(ValueError, match="membership/order"):
        measurement.validate_config(config)
    config["cases"] = cases
    cases[0]["audio"] = dict(sha256="different")
    with pytest.raises(ValueError, match="audio identity"):
        measurement.validate_config(config)


def test_worker_failure_stops_and_retires_owned_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = tmp_path / "synthetic-worker"
    worker.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    worker.chmod(0o755)
    monkeypatch.setattr(measurement, "MAX_WALL_SECONDS", 0.05)
    monkeypatch.setattr(measurement, "SAMPLE_SECONDS", 0.01)
    config: dict[str, object] = dict(
        worker=dict(path=str(worker), sha256=measurement.sha256(worker))
    )
    affinity = measurement.get_affinity()
    if affinity is not None:
        config["linux_cpu"] = min(affinity)
    snapshot = tmp_path / "config.json"
    snapshot.write_text("{}")
    with pytest.raises(RuntimeError, match="budget exceeded"):
        measurement.run_case(config, snapshot, tmp_path, 0, measurement.InvocationBudget.start())
    result = json.loads((tmp_path / "case-0-resources.json").read_text())
    assert result["complete"] is False
    assert result["remaining_pids"] == []
    assert "budget exceeded" in result["failure"]


def test_pinned_content_change_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.write_bytes(b"original")
    spec = dict(path=str(source), sha256=measurement.sha256(source))
    assert measurement.pinned(spec) == source.resolve()
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Pinned file changed"):
        measurement.pinned(spec)


def test_default_cli_validates_without_starting_a_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}")
    output = tmp_path / "not-created"
    monkeypatch.setattr(
        sys, "argv", ["measurement", "--config", str(config), "--output", str(output)]
    )
    monkeypatch.setattr(measurement, "validate_config", lambda _config: [])

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Default validation must not start a worker")

    monkeypatch.setattr(measurement, "run_case", forbidden)
    measurement.main()
    assert json.loads(capsys.readouterr().out)["executed"] is False
    assert not output.exists()


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
def test_output_size_scan_only_ignores_retired_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[OSError]
) -> None:
    wav = tmp_path / "retiring.wav"
    wav.write_bytes(b"wav")
    original = Path.stat
    calls = 0

    def racing_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        nonlocal calls
        if path == wav:
            calls += 1
            if calls >= 2:
                raise error("simulated retirement/error")
        return original(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", racing_stat)
    if error is FileNotFoundError:
        assert measurement.owned_bytes(tmp_path) == 0
    else:
        with pytest.raises(PermissionError):
            measurement.owned_bytes(tmp_path)


def test_cleanup_never_signals_reused_pid_or_its_new_descendants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(measurement, "process_table", lambda: {10: (1, 10), 11: (10, 10)})
    monkeypatch.setattr(measurement, "process_start", lambda pid: {10: 999, 11: 2}[pid])
    killed: list[int] = []
    monkeypatch.setattr(os, "kill", lambda pid, _sig: killed.append(pid))

    class Finished:
        pid = 10

        def wait(self, timeout: float) -> int:
            return 0

    with pytest.raises(RuntimeError, match="Untracked group survivors"):
        measurement.retire(cast(subprocess.Popen[bytes], Finished()), {10: 1})
    assert killed == []


def test_cleanup_refreshes_children_born_after_first_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = {10: 1, 12: 3}
    monkeypatch.setattr(
        measurement,
        "process_table",
        lambda: {pid: (10 if pid == 11 else 1, 10 if pid in (10, 11) else 12) for pid in live},
    )

    def start(pid: int) -> int:
        if pid not in live:
            raise ProcessLookupError(pid)
        return live[pid]

    monkeypatch.setattr(measurement, "process_start", start)
    killed: list[tuple[int, int]] = []

    def kill(pid: int, sig: int) -> None:
        killed.append((pid, sig))
        if sig == signal.SIGTERM:
            live[11] = 2
        else:
            del live[pid]

    monkeypatch.setattr(os, "kill", kill)

    class Child:
        pid = 10

        def wait(self, timeout: float) -> int:
            if 10 in live:
                raise subprocess.TimeoutExpired("owned child", timeout)
            return 0

    measurement.retire(cast(subprocess.Popen[bytes], Child()), {10: 1})
    assert (11, signal.SIGKILL) in killed
    assert live == {12: 3}


def test_budget_accumulates_controller_completed_and_live_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(measurement.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(measurement, "accounted_cpu", lambda: 350.0)
    budget = measurement.InvocationBudget(0.0, 0.0)
    budget.check(200.0)
    with pytest.raises(RuntimeError, match="Whole-invocation"):
        budget.check(251.0)
    # A later case retains already-consumed CPU rather than restarting its allowance.
    monkeypatch.setattr(measurement, "accounted_cpu", lambda: 601.0)
    with pytest.raises(RuntimeError, match="Whole-invocation"):
        budget.check()


def test_completed_numerical_disagreement_withholds_benchmark_qualification() -> None:
    good = dict(allclose=True, argmax_agrees=True)
    runs = [dict(comparison_mac=good.copy(), comparison_linux=good.copy()) for _ in range(4)]
    assert measurement.numerical_summary(dict(runs=runs))["benchmark_qualified"] is True
    runs[2]["comparison_linux"]["allclose"] = False
    summary = measurement.numerical_summary(dict(runs=runs))
    assert summary == dict(numerical_pass=False, unchanged_key=True, benchmark_qualified=False)
    runs[2]["comparison_linux"]["argmax_agrees"] = False
    assert measurement.numerical_summary(dict(runs=runs))["unchanged_key"] is False


def test_linux_start_identity_handles_parentheses_in_process_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(measurement.sys, "platform", "linux")
    line = "123 (name) with spaces) " + " ".join(["S"] + ["0"] * 18 + ["9876"])
    monkeypatch.setattr(Path, "read_text", lambda _path: line)
    assert measurement.process_start(123) == 9876


def test_execution_alarm_covers_admission_and_retains_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}")
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys, "argv", ["measurement", "--config", str(config), "--output", str(output), "--execute"]
    )
    previous = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM)
    }

    def interrupted_admission(_config: dict[str, object]) -> list[dict[str, object]]:
        assert signal.alarm(0) > 0
        signal.raise_signal(signal.SIGALRM)
        raise AssertionError("Admission should have been interrupted")

    monkeypatch.setattr(measurement, "validate_config", interrupted_admission)
    try:
        with pytest.raises(InterruptedError):
            measurement.main()
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    receipt = json.loads((output / "result.json").read_text())
    assert receipt["complete"] is False
    assert receipt["benchmark_qualified"] is False
    assert "InterruptedError" in receipt["failure"]


def test_unseen_child_after_leader_exit_is_not_certified_as_empty_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(measurement, "process_table", lambda: {11: (1, 10)})

    def start(pid: int) -> int:
        if pid == 10:
            raise ProcessLookupError(pid)
        return 2

    monkeypatch.setattr(measurement, "process_start", start)
    killed: list[int] = []
    monkeypatch.setattr(os, "kill", lambda pid, _sig: killed.append(pid))

    class Finished:
        pid = 10

        def wait(self, timeout: float) -> int:
            return 0

    with pytest.raises(RuntimeError, match="Untracked group survivors"):
        measurement.retire(cast(subprocess.Popen[bytes], Finished()), {10: 1})
    assert killed == []
