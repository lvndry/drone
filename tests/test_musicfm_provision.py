"""CPU-only regression checks for wheel isolation, safe layout and measured evidence."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tools import check_musicfm_cpu as helper
from tools.musicfm_resource_report import STUDY, load_raw, summarize
from tools.preflight_musicfm import sha
from tools.provision_musicfm import select_wheels


def lock(path: Path, name: str, version: str, origin: str) -> Path:
    filename = f"{name}-{version}-py3-none-any.whl"
    digest = "a" * 64
    path.write_text(
        f'[[packages]]\nname="{name}"\nversion="{version}"\n'
        f'wheels=[{{url="{origin}/{filename}",size=100,hashes={{sha256="{digest}"}}}}]\n'
    )
    return path


def test_wheel_selection_rejects_shadowing_and_unapproved_origins(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Torch/NumPy"):
        select_wheels(
            lock(tmp_path / "torch.toml", "torch", "2.15.0", "https://files.pythonhosted.org")
        )
    with pytest.raises(ValueError, match="origin"):
        select_wheels(lock(tmp_path / "evil.toml", "einops", "0.8.1", "https://unapproved.invalid"))
    chosen = select_wheels(
        lock(
            tmp_path / "audio.toml",
            "torchaudio",
            "2.11.0+cpu",
            "https://download.pytorch.org/whl/cpu",
        )
    )
    assert chosen[0]["version"] == "2.11.0+cpu"


def test_safe_cpu_layout_counts_state_not_model_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.pt"
    torch.save(
        {
            "state_dict": {
                "model.x": torch.zeros(2, 3, device="cpu"),
                "model.n": torch.zeros(1, dtype=torch.int64, device="cpu"),
            }
        },
        path,
    )
    monkeypatch.setattr(helper, "WEIGHT_BYTES", path.stat().st_size)
    monkeypatch.setattr(helper, "WEIGHT_SHA", sha(path))
    result = helper.checkpoint_layout(path)
    assert result["passed"] and result["weights_only"] and result["mmap"]
    assert result["state_tensor_count"] == 2
    assert result["counts_are_state_tensors_not_parameter_counts"]
    assert result["totals_by_dtype"]["torch.float32"]["elements"] == 6
    assert not result["cuda_initialized"]


def test_measured_resource_maxima_and_failed_candidate_are_preserved() -> None:
    result = summarize()
    assert result["model_parameters_including_new_head"] == 328957080
    assert result["checkpoint_state_elements"] == 329170226
    assert result["peak_allocated_bytes"] == 5493349376
    assert result["peak_reserved_bytes"] == 6142558208
    assert result["maximum_sampled_worker_nvml_mib"] == 6654
    assert result["minimum_sampled_global_free_mib"] == 16312
    assert result["failed_provision"]["wall_seconds"] > 0
    assert len(result["steps"]) == 21
    assert not result["accuracy_scored"]


def test_raw_evidence_tampering_is_rejected(tmp_path: Path) -> None:
    import json
    import shutil

    data = tmp_path / "data"
    data.mkdir()
    shutil.copyfile(STUDY / "data/runtime-index.json", data / "runtime-index.json")
    index = json.loads((data / "runtime-index.json").read_bytes())
    raw = (STUDY / "data" / index["path"]).read_bytes()
    (data / index["path"]).write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="compressed"):
        load_raw(tmp_path)
