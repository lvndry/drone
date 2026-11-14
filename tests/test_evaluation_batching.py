from __future__ import annotations

import numpy as np
import pytest
import torch

import train
from drone.datasets import Track
from drone.features import N_CORE


class MeanClassifier(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        signal = features.mean((1, 2)) - 0.75
        return torch.stack((signal, torch.zeros_like(signal)), dim=1)


def test_evaluation_predictions_are_independent_of_other_track_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = [
        np.ones((N_CORE, 8), dtype=np.float32),
        np.ones((N_CORE, 16), dtype=np.float32),
        np.full((N_CORE, 8), 0.1, dtype=np.float32),
    ]
    monkeypatch.setattr(train, "prepare", lambda track, _: (features[int(track.id)], None))
    tracks = [Track(str(index), "unused.wav", ("C", "major")) for index in range(3)]
    for batch in (1, 2, 3):
        prediction = train.evaluate_tracks(MeanClassifier(), tracks, "unused", "cpu", batch)
        np.testing.assert_array_equal(prediction, [0, 0, 1])


@pytest.mark.parametrize("empty,batch", [(True, 1), (False, 0), (False, -1)])
def test_evaluation_requires_tracks_and_positive_batch(empty: bool, batch: int) -> None:
    tracks = [] if empty else [Track("0", "unused.wav", ("C", "major"))]
    with pytest.raises(ValueError):
        train.evaluate_tracks(MeanClassifier(), tracks, "unused", "cpu", batch)


def test_k10_split_quarantines_training_copies_of_validation_audio() -> None:
    references = [0] * 20 + [1] * 20
    original_train, original_validation = train.validation_split(references)
    hashes = [str(index) for index in range(len(references))]
    duplicate = original_train[0]
    hashes[duplicate] = hashes[original_validation[0]]
    fit, validation = train.validation_split(references, audio_hashes=hashes)
    assert validation == original_validation
    assert duplicate not in fit
    assert not {hashes[index] for index in fit} & {hashes[index] for index in validation}
    hashes[original_validation[-1]] = hashes[original_validation[0]]
    with pytest.raises(ValueError, match="conflicting labels"):
        train.validation_split(references, audio_hashes=hashes)
