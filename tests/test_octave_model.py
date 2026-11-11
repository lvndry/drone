import pytest
import torch

from experiments.octave_model import OctaveNet, compress


def test_octave_head_maps_a_based_bins_to_c_based_interleaved_modes():
    for tonic in range(12):
        for mode in range(2):
            raw = torch.zeros(1, 2, 12, 1)
            raw[0, mode, (tonic - 9) % 12, 0] = 1
            assert OctaveNet.canonical_logits(raw).argmax().item() == tonic * 2 + mode


def test_cqt_compression_handles_silence_and_gain_consistently():
    assert not compress(torch.zeros(2, 99, 100)).any()
    magnitude = torch.rand(2, 99, 100) + 0.01
    torch.testing.assert_close(compress(magnitude), compress(magnitude * 0.25))
    assert compress(magnitude).min() >= 0 and compress(magnitude).max() <= 1


def test_octave_network_outputs_logits_and_propagates_supervised_gradients():
    torch.set_num_threads(1)
    model = OctaveNet()
    logits = model(torch.rand(2, 84, 256))
    assert logits.shape == (2, 24) and torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 19])).backward()
    convolution = model.down[0]
    assert isinstance(convolution, torch.nn.Conv2d)
    gradient = convolution.weight.grad
    assert gradient is not None and gradient.abs().sum() > 0
    with pytest.raises(ValueError):
        model(torch.rand(2, 33, 256))
