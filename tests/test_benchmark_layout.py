from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import drone.benchmark as benchmark_module
from drone.checkpoints import load_checkpoint, save_checkpoint
from drone.features import N_CORE
from drone.model import KeyCNN


def test_audio_benchmark_uses_the_k10_feature_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, KeyCNN())
    model, payload = load_checkpoint(checkpoint)
    shapes = []
    model.register_forward_pre_hook(lambda _, args: shapes.append(tuple(args[0].shape)))
    features = np.ones((N_CORE, 56), dtype=np.float32)
    monkeypatch.setattr(benchmark_module, "load_checkpoint", lambda _: (model, payload))
    monkeypatch.setattr(benchmark_module, "prepare", lambda *a: (features, ("C", "major")))
    benchmark_module.benchmark(checkpoint, audio="unused.wav", iterations=1)
    assert all(shape[1] == N_CORE for shape in shapes)
    assert shapes[-2:] == [(1, N_CORE, 56), (1, N_CORE, 56)]
