"""No weights, CUDA, package installation or scoring needed for preflight contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch

from tools.preflight_musicfm import (
    EVIDENCE,
    environment_fingerprint,
    inspection,
    pool_key_head,
    prospective_cap,
    sha,
    shape_contract,
    state_estimate,
    validate_config,
    verify_evidence,
)


def test_pinned_artifacts_are_small_metadata_not_weights() -> None:
    pins = verify_evidence()
    assert pins["source"]["commit"] == "b83ebedb401bcef639b26b05c0c8bee1dc2dfe71"
    assert pins["artifacts"]["weights_revision"] == "4513b38bc25ad1d227b1980819b9691ba97f4d87"
    assert (EVIDENCE / "weights-lfs-pointer.txt").stat().st_size < 200
    stats = json.loads((EVIDENCE / "msd-stats.json").read_bytes())
    assert stats["melspec_2048_mean"] == 6.768444971712967
    assert stats["melspec_2048_std"] == 18.417922652295623


def test_default_inspection_does_not_import_torch() -> None:
    script = "import json,sys; from tools.preflight_musicfm import inspection; d=inspection(); assert 'torch' not in sys.modules; print(json.dumps(d['estimated']))"
    output = subprocess.check_output([sys.executable, "-c", script], text=True)
    assert json.loads(output)["shape"]["encoder_frames"] == 300
    assert inspection()["measured_model_resources"] is None


@pytest.mark.parametrize(
    "samples,frames", [(288000, 300), (287999, 300), (288001, 300), (1025, 1), (288960, 301)]
)
def test_center_trim_and_two_strides_preserve_shape_contract(samples: int, frames: int) -> None:
    assert shape_contract(samples)["encoder_frames"] == frames
    with pytest.raises(ValueError):
        shape_contract(1024)


def test_nominal_state_floor_and_prospective_free_memory_reserve() -> None:
    scenario = state_estimate(330_000_000)
    assert scenario["total_bytes"] == 5_280_000_000
    assert scenario["total_bytes"] > 4 * 1024**3
    assert prospective_cap(22971, 98304) == 12288
    assert prospective_cap(18000, 98304) == 8827
    with pytest.raises(ValueError):
        prospective_cap(16000, 98304)


def test_resource_head_reduces_time_and_first_step_allocates_state() -> None:
    head = torch.nn.Linear(1024, 24, bias=False, device="cpu")
    hidden = torch.arange(3, dtype=torch.float32, device="cpu")[None, :, None].expand(1, 3, 1024)
    logits = pool_key_head(head, hidden)
    assert logits.shape == (1, 24)
    torch.testing.assert_close(logits, head(torch.ones(1, 1024, device="cpu")))
    optimizer = torch.optim.AdamW(head.parameters(), foreach=False, fused=False)
    assert not optimizer.state
    torch.nn.functional.cross_entropy(
        logits, torch.zeros(1, dtype=torch.long, device="cpu")
    ).backward()
    optimizer.step()
    assert len(optimizer.state) == 1
    moment = next(iter(optimizer.state.values()))
    assert moment["exp_avg"].shape == head.weight.shape
    assert moment["exp_avg_sq"].shape == head.weight.shape
    with pytest.raises(ValueError):
        pool_key_head(head, torch.zeros(1, 1024, 3, device="cpu"))


def test_window_source_and_protected_services_are_mandatory(tmp_path: Path) -> None:
    import tools.preflight_musicfm as module

    now = datetime.now(timezone.utc)
    from tools import run_cqt_expansion as guard

    config = {
        "tool_sha256": sha(Path(module.__file__)),
        "source_index_sha256": sha(EVIDENCE / "source-index.json"),
        "artifact_index_sha256": sha(EVIDENCE / "artifact-index.json"),
        "dependency_source_index_sha256": sha(EVIDENCE / "dependency-source-index.json"),
        "guard_sha256": sha(Path(guard.__file__)),
        "protocol_sha256": sha(EVIDENCE.parent / "PROTOCOL.md"),
        "environment_fingerprint_sha256": environment_fingerprint(),
        "start_utc": (now - timedelta(seconds=1)).isoformat(),
        "deadline_utc": (now + timedelta(seconds=600)).isoformat(),
        "protected_vllm_pids": [111, 222],
        "precision": "bf16-autocast-fp32-parameters",
        "attention": "eager",
        "steady_steps": 20,
        "requested_cap_mib": 12288,
        "reserve_mib": 8192,
        "gpu_uuid": "GPU-example",
        "root": str(tmp_path / "new-stage"),
    }
    validate_config(config, now.timestamp())
    with pytest.raises(ValueError):
        validate_config(dict(config, protected_vllm_pids=[111, 111]), now.timestamp())
    with pytest.raises(ValueError):
        validate_config(dict(config, requested_cap_mib=20000), now.timestamp())
    with pytest.raises(ValueError):
        validate_config(config, now.timestamp() + 1000)


def test_analytical_active_state_does_not_allocate_moments_for_unused_head() -> None:
    scenario = state_estimate(100, 80)
    assert scenario["fp32_parameter_bytes"] == 400
    assert scenario["total_bytes"] == 1360
    with pytest.raises(ValueError):
        state_estimate(100, 101)


def test_nvml_metadata_does_not_mix_device_or_process(monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.preflight_musicfm as module

    def device_output(*args: object, **kwargs: object) -> str:
        return "GPU-a, 0, 22971, 98304\n"

    monkeypatch.setattr(subprocess, "check_output", device_output)
    assert module.gpu_metadata("GPU-a")["free_mib"] == 22971
    with pytest.raises(ValueError):
        module.gpu_metadata("GPU-b")

    def processes(*args: object, **kwargs: object) -> str:
        return "GPU-a, 100, 512\nGPU-a, 200, 40000\nGPU-b, 100, 700\n"

    monkeypatch.setattr(subprocess, "check_output", processes)
    assert module.gpu_worker_mib("GPU-a", 100) == 512
    assert module.gpu_worker_mib("GPU-a", 999) == 0
