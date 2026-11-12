"""Synthetic frontend/encoder only; complete coverage and conditional batching invariants."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from experiments.musicfm_eval import compare_panels
from experiments.musicfm_pilot import evaluate_panel
from experiments.musicfm_teacher import KeyTeacher
from tools.musicfm_crop_io import save


class ToyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1024)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
            self.linear.weight[0, 0] = 1.0
        self.front_batches = []
        self.encoder_batches = []

    def preprocessing(self, wave: torch.Tensor, names: list[str]) -> dict[str, torch.Tensor]:
        self.front_batches.append(wave.shape[0])
        assert wave.shape[0] == 1
        return {"melspec_2048": wave.mean(1, keepdim=True)}

    def normalize(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return features

    def encoder(self, feature: torch.Tensor) -> tuple[None, list[torch.Tensor]]:
        self.encoder_batches.append(feature.shape[0])
        hidden = self.linear(feature).unsqueeze(1)
        return None, [hidden] * 13


def test_batch16_uses_individual_frontends_and_retains_all_tail_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.musicfm_pilot as pilot

    backbone = ToyBackbone()
    teacher = KeyTeacher(backbone)
    ids = [str(i) for i in range(20)]
    rows = {
        identity: {"samples": 288000 + (1 if int(identity) % 2 else 1025), "value": int(identity)}
        for identity in ids
    }

    def wave(
        row: dict[str, Any], start: int, length: int, watch: Callable[[], None]
    ) -> torch.Tensor:
        return torch.full((1, length), row["value"] + start / 288000, dtype=torch.float32)

    monkeypatch.setattr(pilot, "wave", wave)
    one = evaluate_panel(
        teacher, ids, rows, lambda: None, collect_predictions=True, collect_logits=True
    )
    fast = evaluate_panel(
        teacher,
        ids,
        rows,
        lambda: None,
        collect_predictions=True,
        collect_logits=True,
        batch_size=16,
    )
    assert one["windows"] == fast["windows"] == 40
    np.testing.assert_allclose(one["logits"], fast["logits"], atol=1e-5, rtol=0)
    assert set(backbone.front_batches) == {1}
    assert 16 in backbone.encoder_batches


def test_composed_proof_rejects_panel_order_and_numerical_changes() -> None:
    metadata = [{"id": str(i), "samples": 1200, "windows": 1} for i in range(376)]
    ref = {
        "tracks": 376,
        "windows": 376,
        "logits": np.zeros((376, 24)).tolist(),
        "per_song": metadata,
    }
    assert compare_panels(ref, ref)["passed"]
    changed = {**ref, "logits": np.ones((376, 24)).tolist()}
    assert not compare_panels(ref, changed)["passed"]
    with pytest.raises(ValueError, match="order"):
        compare_panels(ref, {**ref, "per_song": list(reversed(metadata))})


def test_heartbeat_allows_slow_query_but_not_dead_or_stale_guardian(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tools.run_musicfm_teacher as guard

    ready = tmp_path / "ready.json"
    save(
        ready,
        {
            "pid": 123,
            "start_ticks": 456,
            "teacher_config_sha256": "config",
            "last_guard_epoch": 97.0,
        },
    )
    monkeypatch.setattr(guard, "alive", lambda pid, start: (pid, start) == (123, 456))
    monkeypatch.setattr(guard, "capacity_guard", lambda *_: None)
    current = [100.0]
    ticks = [1.0]
    monkeypatch.setattr(guard.time, "time", lambda: current[0])
    monkeypatch.setattr(guard.time, "monotonic", lambda: ticks[0])
    watch, stats = guard.worker_watch(
        {"_config_sha256": "config", "deadline_epoch": 200.0}, tmp_path, 123
    )
    watch()
    assert stats["max_observed_heartbeat_age_seconds"] == 3.0
    current[0] = 108.0
    ticks[0] = 2.0
    with pytest.raises(RuntimeError, match="guardian"):
        watch()
