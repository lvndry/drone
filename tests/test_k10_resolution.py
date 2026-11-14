from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import nn

from drone.model import KeyCNN, params_count
from experiments.k10_resolution import SemitoneGridCNN


@pytest.fixture(autouse=True)
def bounded_cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("frames", [8, 65, 320])
def test_true_length_shapes_keep_frequency_and_floor_time(frames: int) -> None:
    model = SemitoneGridCNN().eval()
    shapes: list[tuple[int, ...]] = []

    def capture(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        shapes.append(tuple(output.shape))

    handles = [
        module.register_forward_hook(capture)
        for module in model.net
        if isinstance(module, nn.MaxPool2d)
    ]
    try:
        with torch.inference_mode():
            logits = model(torch.randn(2, 192, frames))
    finally:
        for handle in handles:
            handle.remove()
    assert shapes == [
        (2, 16, 96, frames // 2),
        (2, 32, 96, frames // 4),
        (2, 32, 96, frames // 8),
    ]
    assert logits.shape == (2, 24)
    assert torch.isfinite(logits).all()


def test_exact_parameter_allocation_and_unchanged_first_two_initializations() -> None:
    torch.manual_seed(13)
    baseline = KeyCNN()
    torch.manual_seed(13)
    candidate = SemitoneGridCNN()
    assert params_count(baseline) == 109128
    assert params_count(candidate) == 108936
    assert params_count(candidate.head) == 73752
    assert params_count(candidate.net) == 35184
    assert [params_count(candidate.net[start : start + 7]) for start in (0, 7, 14)] == [
        2544,
        14016,
        18624,
    ]
    expected = baseline.net[:14].state_dict()
    actual = candidate.net[:14].state_dict()
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[name], actual[name]) for name in expected)
    assert [layer.kernel_size for layer in candidate.net if isinstance(layer, nn.MaxPool2d)] == [
        (2, 2),
        (1, 2),
        (1, 2),
    ]


def test_synthetic_update_reaches_every_convolution_and_absolute_frequency_head() -> None:
    torch.manual_seed(13)
    model = SemitoneGridCNN().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    loss = nn.functional.cross_entropy(model(torch.randn(2, 192, 16)), torch.tensor([0, 23]))
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all()
        if name.endswith("weight"):
            assert not torch.equal(before[name], parameter), name
    # The full absolute-frequency readout is retained, with no tied tonic or
    # octave-folded output. These structural checks do not prove equivariance.
    assert model.head.weight.shape == (24, 3072)


@pytest.mark.parametrize("shape", [(1, 192, 7), (1, 191, 8), (192, 8), (0, 192, 8)])
def test_rejects_unusable_core_feature_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="time >= 8"):
        SemitoneGridCNN()(torch.zeros(shape))
