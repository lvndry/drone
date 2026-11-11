from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import drone
from drone.features import N_CORE
from drone.keys import decode_key
from drone.model import KeyCNN


def test_predictor_matches_direct_k10_call_at_full_length(monkeypatch, tmp_path: Path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model = KeyCNN().eval()
    features = np.random.default_rng(13).random((N_CORE, 56), dtype=np.float32)
    monkeypatch.setattr(drone, "load_checkpoint", lambda *a: (model, {}))
    monkeypatch.setattr(drone, "prepare", lambda *a, **kw: (features, None))
    with torch.inference_mode():
        expected = model(torch.from_numpy(features[None])).softmax(1)[0]
    prediction = drone.Predictor(weights=tmp_path / "unused.pt").predict("track.wav")
    assert prediction[:2] == decode_key(int(expected.argmax()))
    assert prediction[2] == pytest.approx(float(expected.max()), abs=1e-7)
