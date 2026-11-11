import numpy as np
import pytest
import torch

import train
from drone.datasets import Track
from drone.features import N_CORE


class MeanClassifier(torch.nn.Module):
    def forward(self, x):
        signal = x.mean((1, 2)) - 0.75
        return torch.stack((signal, torch.zeros_like(signal)), dim=1)


def test_evaluation_predictions_are_independent_of_other_track_lengths(monkeypatch):
    features = [
        np.ones((N_CORE, 8), dtype=np.float32),
        np.ones((N_CORE, 16), dtype=np.float32),
        np.full((N_CORE, 8), 0.1, dtype=np.float32),
    ]
    monkeypatch.setattr(train, "prepare", lambda track, _: (features[int(track.id)], None))
    tracks = [Track(str(i), "unused.wav", ("C", "major")) for i in range(3)]
    for batch in (1, 2, 3):
        prediction = train.evaluate_tracks(MeanClassifier(), tracks, "unused", "cpu", batch)
        np.testing.assert_array_equal(prediction, [0, 0, 1])


@pytest.mark.parametrize("tracks,batch", [([], 1), ([0], 0), ([0], -1)])
def test_evaluation_requires_tracks_and_positive_batch(tracks, batch):
    with pytest.raises(ValueError):
        train.evaluate_tracks(MeanClassifier(), tracks, "unused", "cpu", batch)


def test_k10_split_quarantines_training_copies_of_validation_audio():
    refs = [0] * 20 + [1] * 20
    original_train, original_val = train.validation_split(refs)
    hashes = [str(i) for i in range(len(refs))]
    duplicate = original_train[0]
    hashes[duplicate] = hashes[original_val[0]]
    fit, validation = train.validation_split(refs, audio_hashes=hashes)
    assert validation == original_val
    assert duplicate not in fit
    assert not {hashes[i] for i in fit} & {hashes[i] for i in validation}
    hashes[original_val[-1]] = hashes[original_val[0]]
    with pytest.raises(ValueError, match="conflicting labels"):
        train.validation_split(refs, audio_hashes=hashes)
