"""Supervised octave-aware pilot inspired by ChromaNet/S-KEY (ICASSP 2025).

Independent implementation of the published temporal-only downsampling,
ConvNeXt blocks and octave-pooled tonic/mode head. No pretrained weights.
Reference: https://github.com/deezer/skey (MIT code; research-only weights).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

FEATURE_SPEC = {
    "profile": "octave-cqt-segments-v1",
    "sample_rate": 22050,
    "seconds": 32,
    "segments": "centers-at-quarter-and-three-quarter-duration-clamped",
    "transform": "nnAudio-0.3.4-VQT-gamma0",
    "fmin": 27.5,
    "bins": 99,
    "bins_per_octave": 12,
    "hop": 512,
    "compression": "relative-amplitude-db-80-to-unit-silence-zero",
    "training": "choose-one-segment-crop84-start0to15-label-minus-crop",
    "evaluation": "crop-start12-average-two-segment-probabilities",
}


def compress(magnitude: torch.Tensor) -> torch.Tensor:
    peak = magnitude.amax(dim=(-2, -1), keepdim=True)
    db = 20 * (magnitude.clamp_min(1e-9).log10() - peak.clamp_min(1e-9).log10())
    return torch.where(peak > 1e-8, db.clamp(-80, 0) / 80 + 1, 0)


class Block(nn.Module):
    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel,
            padding=kernel // 2,
            groups=channels,
            padding_mode="replicate",
        )
        self.expand = nn.Linear(channels, channels * 4)
        self.contract = nn.Linear(channels * 4, channels)
        self.scale = nn.Parameter(torch.full((channels,), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = F.layer_norm(y, y.shape[1:]).permute(0, 2, 3, 1)
        y = (self.contract(F.gelu(self.expand(y))) * self.scale).permute(0, 3, 1, 2)
        if self.training:
            mask = torch.empty((len(x), 1, 1, 1), device=x.device).bernoulli_(0.9) / 0.9
            y = y * mask
        return x + y


class OctaveNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels, kernels = (2, 3, 40, 40, 30, 10, 3), (7, 7, 7, 7, 7, 5, 5)
        self.down = nn.ModuleList()
        self.blocks = nn.ModuleList()
        previous = 1
        for width, kernel in zip(channels, kernels, strict=True):
            self.down.append(nn.Conv2d(previous, width, (1, 2), stride=(1, 2)))
            self.blocks.append(Block(width, kernel))
            previous = width
        self.classifier = nn.Conv2d(previous, 2, 1)
        self.mode_norm = nn.BatchNorm2d(2, affine=False)

    @staticmethod
    def canonical_logits(x: torch.Tensor) -> torch.Tensor:
        """A-based pitch bins -> C-based tonic*2+minor, retaining mode channels."""
        return torch.roll(x, 9, dims=2).squeeze(-1).transpose(1, 2).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 84 or x.shape[2] < 128:
            raise ValueError("Expected batch,84,time with at least 128 frames")
        x = x.unsqueeze(1)
        for down, block in zip(self.down, self.blocks, strict=True):
            x = block(F.gelu(down(F.layer_norm(x, x.shape[1:]))))
        x = x.reshape(len(x), x.shape[1], 7, 12, x.shape[-1]).mean(2).mean(-1, keepdim=True)
        return self.canonical_logits(self.mode_norm(self.classifier(x)))
