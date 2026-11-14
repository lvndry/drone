"""RAM-path safety/crop tests use synthetic bytes only, never decoders or audio."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tools.prepare_musicfm_ram_cache as ram
from tools.musicfm_crop_io import file_identity


def owned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ram.OwnedCache:
    parent = tmp_path / "shm"
    output = tmp_path / "receipts"
    parent.mkdir()
    output.mkdir()
    monkeypatch.setattr(ram, "RAM_PARENT", parent)
    return ram.OwnedCache(
        parent / ("drone-musicfm-fit-" + "a" * 32), output, ["wave-0001.f32le"], "a" * 32
    )


def test_page_forecast_is_per_file_not_rounded_total() -> None:
    rows = [{"role": "fit", "canonical_samples": 1} for _ in range(3)]
    rows += [{"role": "validation", "canonical_samples": 22050}]
    assert ram.forecast(rows, 4096) == {"fit": 12288, "validation": 98304}


def test_readonly_sample_crops_preserve_bits_and_close_mapping(tmp_path: Path) -> None:
    values = np.array(
        [0.0, -0.0, 1.0, np.nextafter(np.float32(1), np.float32(2)), 4.0], dtype="<f4"
    )
    path = tmp_path / "wave"
    values.tofile(path)
    path.chmod(0o444)
    row = {"cache": str(path), "samples": len(values), "cache_identity": file_identity(path)}
    actual = ram.read_crop(row, 1, 3, watch=lambda: None)
    assert actual.tobytes() == values[1:4].tobytes()
    path.unlink()  # Returned data owns bytes, no live mapping/file handle.
    assert actual.tobytes() == values[1:4].tobytes()


def test_crop_rejects_changed_cache_and_invalid_offsets(tmp_path: Path) -> None:
    path = tmp_path / "wave"
    np.arange(5, dtype="<f4").tofile(path)
    row = {"cache": str(path), "samples": 5, "cache_identity": file_identity(path)}
    for start, length in [(-1, 1), (0, 0), (4, 2)]:
        with pytest.raises(ValueError, match="outside"):
            ram.read_crop(row, start, length, watch=lambda: None)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity"):
        ram.read_crop(row, 0, 1, watch=lambda: None)


def test_owned_cleanup_keeps_other_actor_and_disk_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = owned(tmp_path, monkeypatch)
    other = cache.root.parent / "other-actor"
    other.write_text("preserve")
    (cache.root / "wave-0001.f32le").write_bytes(b"partial")
    ram.cleanup_from_config({"ram_root": str(cache.root), "cache_token": "a" * 32}, cache.output)
    assert not cache.root.exists()
    assert other.read_text() == "preserve"
    assert (cache.output / "cache-ownership.json").exists()


@pytest.mark.parametrize("danger", ["unknown", "symlink", "changed-witness"])
def test_cleanup_refuses_unowned_or_symlink_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    danger: str,
) -> None:
    cache = owned(tmp_path, monkeypatch)
    other = tmp_path / "protected"
    other.write_text("preserve")
    if danger == "unknown":
        (cache.root / "unknown").write_text("unowned")
    elif danger == "symlink":
        (cache.root / "wave-0001.f32le").symlink_to(other)
    else:
        (cache.root / "OWNER.json").write_text("{}")
    with pytest.raises(RuntimeError):
        cache.cleanup()
    assert other.read_text() == "preserve"
    assert cache.root.exists()


def test_capacity_guard_checks_forecast_and_every_pressure_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config: dict[str, Any] = {"protected_starts": [{"pid": 1, "start_ticks": 2}]}
    calls = []
    monkeypatch.setattr(ram, "protected_live", lambda value: calls.append(value))
    memory = [ram.MEMORY_RESERVE + 10]
    mount = [ram.MOUNT_RESERVE + 10]
    disk = [ram.DISK_RESERVE + 10]
    monkeypatch.setattr(ram, "mem_available", lambda: memory[0])
    monkeypatch.setattr(
        ram, "available", lambda path: mount[0] if path == ram.RAM_PARENT else disk[0]
    )
    ram.capacity_guard(config, tmp_path, 10)
    for bound in (memory, mount, disk):
        bound[0] -= 11
        with pytest.raises(RuntimeError):
            ram.capacity_guard(config, tmp_path)
        bound[0] += 11
    with pytest.raises(RuntimeError):
        ram.capacity_guard(config, tmp_path, 11)
    assert len(calls) == 5


def test_active_decode_pages_count_before_file_is_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = owned(tmp_path, monkeypatch)
    stage = ram.RAMPreparation({}, cache.output, cache)
    monkeypatch.setattr(ram.Preparation, "guard", lambda *_: None)
    monkeypatch.setattr(ram, "capacity_guard", lambda *_: None)
    monkeypatch.setattr(ram, "FIT_LIMIT", stage.completed_fit_bytes)
    stage.active_fit = cache.root / "wave-0001.f32le"
    stage.guard()  # No file exists yet; prelaunch guard is valid.
    stage.active_fit.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="page-rounded"):
        stage.guard()
    cache.cleanup()


def test_full_decode_profile_and_conflicting_identity_stop_without_seek(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    cache = owned(tmp_path, monkeypatch)
    for name in ("records", "logs", "validation"):
        (cache.output / name).mkdir()
    source = tmp_path / "source"
    source.write_bytes(b"synthetic source witness")
    values = np.arange(1025, dtype="<f4")
    config: dict[str, Any] = {
        "ffmpeg": "/frozen/ffmpeg",
        "caps": {"reference_bytes": 256 * 1024**2},
    }
    stage = ram.RAMPreparation(config, cache.output, cache)
    monkeypatch.setattr(stage, "guard", lambda *_: None)
    commands = []

    def fake_decode(command: list[str], target: Path, stderr: Path, maximum: int) -> dict[str, Any]:
        commands.append(command)
        values.tofile(target)
        stderr.write_text("retained synthetic warning")
        return {"command": command, "exit": 0}

    monkeypatch.setattr(stage, "decode", fake_decode)
    stage.families[hashlib.sha256(values.tobytes()).hexdigest()] = [{"role": "fit", "target": 2}]
    row = {
        "id": "synthetic:one",
        "role": "fit",
        "target": 1,
        "path": str(source),
        "sha256": ram.sha(source),
    }
    with pytest.raises(ValueError, match="conflicting-target"):
        stage.prepare(row, 1)
    assert len(commands) == 1
    assert "-ss" not in commands[0] and "-t" not in commands[0]
    assert commands[0] == ram.decode_command(Path("/frozen/ffmpeg"), source, None, 12 * 24000)
    assert (cache.output / "records/0001.json").exists()
    assert (cache.output / "logs/0001-full.stderr").read_text() == "retained synthetic warning"
    assert stage.rows == []
    cache.cleanup()


@pytest.mark.parametrize("samples, valid", [(0, False), (1024, False), (1025, True)])
def test_reflected_stft_sample_boundary(samples: int, valid: bool) -> None:
    if valid:
        ram.validate_samples(samples)
    else:
        with pytest.raises(ValueError, match="1024"):
            ram.validate_samples(samples)


def test_handoff_requires_live_fresh_successor_before_ack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.run_musicfm_ram_cache as guardian
    from tools.musicfm_crop_io import save

    source = tmp_path / "guardian.py"
    source.write_text("frozen future guardian")
    ready = tmp_path / "ready.json"
    approval = tmp_path / "approved.json"
    config = {"_executed_config_sha256": "config-sha", "cache_token": "a" * 32}
    save(
        approval,
        {
            "authorized": True,
            "config_sha256": "config-sha",
            "cache_token": "a" * 32,
            "guardian_source_path": str(source),
            "guardian_source_sha256": ram.sha(source),
            "guardian_pid": 123,
            "guardian_start_ticks": 456,
            "guardian_ready_path": str(ready),
        },
    )
    monkeypatch.setattr(guardian, "alive", lambda pid, start: (pid, start) == (123, 456))
    monkeypatch.setattr(guardian.time, "time", lambda: 100.0)

    def write_ready(epoch: float) -> None:
        if ready.exists():
            ready.unlink()
        save(
            ready,
            {
                "ready": True,
                "config_sha256": "config-sha",
                "cache_token": "a" * 32,
                "pid": 123,
                "start_ticks": 456,
                "last_guard_epoch": epoch,
            },
        )

    write_ready(98.9)
    with pytest.raises(RuntimeError, match="fresh"):
        guardian.handoff(config, approval)
    write_ready(99.5)
    assert guardian.handoff(config, approval)["guardian_pid"] == 123
    monkeypatch.setattr(guardian, "alive", lambda *_: False)
    with pytest.raises(RuntimeError, match="live"):
        guardian.handoff(config, approval)


def test_stopped_worker_pid_is_never_rediscovered_or_killed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess
    from typing import cast

    import tools.run_musicfm_ram_cache as guardian

    class Finished:
        pid = 123

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float) -> int:
            return 0

    monkeypatch.setattr(
        guardian, "descendants", lambda *_: pytest.fail("Do not inspect reused PID")
    )
    monkeypatch.setattr(guardian, "alive", lambda *_: False)
    guardian.stop_and_wait(cast(subprocess.Popen[bytes], Finished()), {123: 456})
