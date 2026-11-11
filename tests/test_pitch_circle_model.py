from __future__ import annotations

from collections.abc import Generator

import pytest
import torch
from torch import nn

from experiments.pitch_circle_model import (
    CircularFrequencyConv,
    PitchCircleNet,
    compute_estimate,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_setup() -> Generator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(previous)


def rotate_features(x: torch.Tensor, semitones: int) -> torch.Tensor:
    return torch.roll(x.reshape(len(x), 8, 24, x.shape[-1]), 2 * semitones, dims=2).reshape_as(x)


def rotate_logits(x: torch.Tensor, semitones: int) -> torch.Tensor:
    return torch.roll(x.reshape(len(x), 12, 2), semitones, dims=1).reshape_as(x)


@pytest.mark.parametrize(
    "dtype,rtol,atol", [(torch.float64, 1e-10, 1e-12), (torch.float32, 1e-5, 1e-6)]
)
def test_all12_folded_rotations_preserve_geometry_for_every_time_remainder(
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    torch.manual_seed(13)
    model = PitchCircleNet().to(dtype=dtype).eval()
    with torch.inference_mode():
        for frames in (*range(8, 16), *range(31, 39)):
            x = torch.randn(2, 192, frames, dtype=dtype)
            ordinary = model(x)
            assert ordinary.shape == (2, 24)
            for shift in range(12):
                torch.testing.assert_close(
                    model(rotate_features(x, shift)),
                    rotate_logits(ordinary, shift),
                    rtol=rtol,
                    atol=atol,
                )


def test_readout_maps_each_halfbin_to_canonical_tonic_mode_without_offset() -> None:
    model = PitchCircleNet().double()
    with torch.no_grad():
        model.readout.weight.zero_()
        assert model.readout.bias is not None
        model.readout.bias.zero_()
        model.readout.weight[0, 0, 0, 0] = 1
        model.readout.weight[1, 0, 1, 0] = 1
    feature = torch.zeros(1, 64, 24, 1, dtype=torch.float64)
    feature[0, 0, :, 0] = torch.arange(24, dtype=torch.float64)
    logits = model.canonical_logits(model.readout(feature))
    assert torch.equal(logits, torch.arange(24, dtype=torch.float64).reshape(1, 24))


@pytest.mark.parametrize("dilation", [1, 2, 4])
def test_only_frequency_wraps_and_temporal_edges_are_zero_padded(dilation: int) -> None:
    conv = CircularFrequencyConv(1, 1, dilation).double()
    with torch.no_grad():
        conv.conv.weight.fill_(1)
        assert conv.conv.bias is not None
        conv.conv.bias.zero_()
    x = torch.zeros(1, 1, 24, 5, dtype=torch.float64)
    x[0, 0, 0, 0] = 1
    output = conv(x)
    assert output.shape == x.shape
    assert output[0, 0, 24 - dilation, 0] > 0  # Frequency wraps around C.
    assert torch.count_nonzero(output[..., -1]) == 0  # Time never wraps.


@pytest.mark.parametrize(
    "shape", [(0, 192, 8), (1, 208, 8), (1, 191, 8), (1, 192, 7), (192, 8), (1, 1, 192, 8)]
)
def test_invalid_shape_and_too_short_time_are_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="batch,192,time"):
        PitchCircleNet()(torch.zeros(shape))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_features_fail_before_batchnorm_state_changes(value: float) -> None:
    model = PitchCircleNet().train()
    x = torch.zeros(1, 192, 8)
    x[0, 0, 0] = value
    with pytest.raises(ValueError, match="finite floating"):
        model(x)
    assert all(
        layer.num_batches_tracked == 0
        for layer in model.modules()
        if isinstance(layer, nn.BatchNorm2d)
    )


def test_integer_features_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite floating"):
        PitchCircleNet()(torch.zeros(1, 192, 8, dtype=torch.int64))


def test_network_is_trainable_without_mutating_input_and_uses_matched_regularization() -> None:
    torch.manual_seed(37)
    model = PitchCircleNet().train()
    assert sum(p.numel() for p in model.parameters()) == 73506
    assert sum(isinstance(layer, nn.BatchNorm2d) for layer in model.modules()) == 6
    assert sum(isinstance(layer, nn.ELU) for layer in model.modules()) == 6
    dropout = [layer for layer in model.modules() if isinstance(layer, nn.Dropout2d)]
    assert len(dropout) == 1 and dropout[0].p == 0.2
    assert not any(isinstance(layer, nn.Linear) for layer in model.modules())
    x = torch.randn(3, 192, 17, requires_grad=True)
    original = x.detach().clone()
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 11, 23]))
    loss.backward()
    assert torch.isfinite(loss) and x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0 and torch.equal(x.detach(), original)
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = model.readout.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(model.readout.weight, before)


@pytest.mark.parametrize("frames", [8, 15, 16, 319, 320, 321])
def test_analytic_cost_matches_actual_conv_shapes_including_padding(frames: int) -> None:
    model = PitchCircleNet().eval()
    measured_macs = []
    sizes = []
    time_shapes = []

    def observe(layer: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        assert isinstance(layer, nn.Conv2d)
        measured_macs.append(
            output.numel()
            * (layer.in_channels // layer.groups)
            * layer.kernel_size[0]
            * layer.kernel_size[1]
        )
        sizes.extend((inputs[0].numel(), output.numel()))
        time_shapes.append(output.shape[-1])

    handles = [
        layer.register_forward_hook(observe)
        for layer in model.modules()
        if isinstance(layer, nn.Conv2d)
    ]
    with torch.inference_mode():
        model(torch.zeros(1, 192, frames))
    for handle in handles:
        handle.remove()
    estimate = compute_estimate(frames)
    assert estimate["convolution_macs"] == sum(measured_macs)
    assert estimate["largest_fp32_tensor_bytes"] == max(sizes) * 4
    assert estimate["parameters"] == sum(p.numel() for p in model.parameters())
    assert time_shapes[-1] == estimate["output_frames"] == frames // 8


def test_noncontiguous_input_has_identical_pitch_and_time_interpretation() -> None:
    model = PitchCircleNet().eval()
    x = torch.randn(1, 192, 32)[:, :, ::2]
    assert not x.is_contiguous()
    with torch.inference_mode():
        assert torch.equal(model(x), model(x.contiguous()))


def test_folded_rotation_is_not_misrepresented_as_physical_cqt_reslicing() -> None:
    from drone.features import window

    # A C at the second octave's start crosses the octave boundary when resliced.
    padded = torch.zeros(208, 8)
    padded[8 + 24, :] = 1
    core = torch.from_numpy(window(padded.numpy(), 0)).unsqueeze(0)
    physical = torch.from_numpy(window(padded.numpy(), 1)).unsqueeze(0)
    abstract = rotate_features(core, -1)
    assert not torch.equal(physical, abstract)
    assert torch.equal(physical.reshape(1, 8, 24, 8).sum(1), abstract.reshape(1, 8, 24, 8).sum(1))
