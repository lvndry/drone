"""Research-only pitch-circle model; no training/checkpoint/production dispatch.

Equivariance concerns rotations of the folded feature representation only.
Physical finite-band CQT reslicing also moves energy between octave channels.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

CHANNELS = (16, 16, 32, 32, 64, 64)
FREQUENCY_DILATIONS = (1, 1, 2, 2, 4, 4)
PITCH_BINS = 24
OCTAVES = 8
MIN_FRAMES = 8
SPEC = {
    "architecture": "pitch-circle-v1",
    "input": "C-based192-bin K10 CQT, folded8octaves x24halfbins",
    "channels": CHANNELS,
    "frequency_dilations": FREQUENCY_DILATIONS,
    "activation": "ELU",
    "normalization": "BatchNorm2d after each convolution",
    "dropout": "single Dropout2d0.2 after third temporal pool",
    "pool": "MaxPool2d(1,2) after each pair",
    "readout": "Conv64to2 kernel(2,1) stride(2,1), time mean only",
    "class_order": "C-based-tonic-times-two-plus-minor",
    "frequency_receptive_field_before_readout": 29,
    "frequency_receptive_field_with_readout": 30,
    "time_receptive_field": 36,
    "time_stride": 8,
    "equivariance": "C12 rotations of folded features by2halfbins; not physical CQT reslices",
}


class CircularFrequencyConv(nn.Module):
    """Circular padding on pitch only; time retains ordinary zero padding."""

    def __init__(self, cin: int, cout: int, frequency_dilation: int) -> None:
        super().__init__()
        self.frequency_dilation = frequency_dilation
        self.conv = nn.Conv2d(cin, cout, 3, dilation=(frequency_dilation, 1), padding=(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.frequency_dilation
        return self.conv(F.pad(x, (0, 0, d, d), mode="circular"))


class PitchCircleNet(nn.Module):
    """Fixed shared-tonic/mode scorer preserving octave channels at its input."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = OCTAVES
        for index, (width, dilation) in enumerate(zip(CHANNELS, FREQUENCY_DILATIONS, strict=True)):
            layers.extend(
                (CircularFrequencyConv(previous, width, dilation), nn.BatchNorm2d(width), nn.ELU())
            )
            if index % 2 == 1:
                layers.append(nn.MaxPool2d((1, 2)))
            previous = width
        layers.append(nn.Dropout2d(0.2))
        self.net = nn.Sequential(*layers)
        self.readout = nn.Conv2d(64, 2, (2, 1), stride=(2, 1))

    @staticmethod
    def canonical_logits(scores: torch.Tensor) -> torch.Tensor:
        """B,mode,tonic,time -> C major,C minor,C# major,C# minor,..."""
        return scores.mean(-1).transpose(1, 2).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.ndim != 3
            or x.shape[0] < 1
            or x.shape[1] != OCTAVES * PITCH_BINS
            or x.shape[2] < MIN_FRAMES
        ):
            raise ValueError("Expected nonempty batch,192,time with at least8 frames")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("Expected finite floating-point CQT features")
        folded = x.reshape(x.shape[0], OCTAVES, PITCH_BINS, x.shape[2])
        return self.canonical_logits(self.readout(self.net(folded)))


def compute_estimate(frames: int, batch: int = 1) -> dict[str, Any]:
    """Analytic FP32 convolution MACs and largest tensor, not timing or peak RSS.

    Includes explicit circular-pad tensors. Excludes autograd saved tensors,
    convolution workspace, normalization/ELU/pooling work and input finite checks.
    """
    if frames < MIN_FRAMES or batch < 1:
        raise ValueError("Need positive batch and at least8 frames")
    time, previous = frames, OCTAVES
    rows = []
    largest = batch * OCTAVES * PITCH_BINS * frames
    parameters = 0
    for index, (width, dilation) in enumerate(zip(CHANNELS, FREQUENCY_DILATIONS, strict=True)):
        padded = batch * previous * (PITCH_BINS + 2 * dilation) * time
        activations = batch * width * PITCH_BINS * time
        macs = activations * previous * 9
        parameters += width * previous * 9 + width + 2 * width  # Conv bias and BN affine.
        largest = max(largest, padded, activations)
        rows.append(
            dict(
                layer=index + 1,
                channels=width,
                frames=time,
                macs=macs,
                output_elements=activations,
                padded_elements=padded,
            )
        )
        if index % 2 == 1:
            time //= 2
        previous = width
    readout_macs = batch * 12 * time * 2 * 64 * 2
    parameters += 2 * 64 * 2 + 2
    return dict(
        frames=frames,
        batch=batch,
        convolution_macs=sum(row["macs"] for row in rows) + readout_macs,
        parameters=parameters,
        largest_fp32_tensor_bytes=largest * 4,
        output_frames=time,
        layers=rows,
        readout_macs=readout_macs,
        caveat="Arithmetic/storage estimate only; circular/dilated kernels need real benchmarking",
    )
