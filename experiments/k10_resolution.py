"""One fixed higher-resolution, narrower K10 research candidate.

The final frequency grid has 96 positions (two CQT bins per position). This is
not physical pitch equivariance: finite CQT boundaries, channel representations
and the unrestricted absolute-frequency head still matter. Final local features
have a frequency receptive field of 22 input bins, versus canonical K10's 36;
their temporal receptive field remains 36 frames. The final head and time mean
combine these features globally. Finer indexing is a hypothesis, not a proof.

Parameter allocation, including convolution biases and affine BN parameters:
block1 2,544 + block2 14,016 + block3 18,624 + head (32*96*24+24)
= 108,936. No accuracy, CPU speed or total-memory advantage is established.
"""

from __future__ import annotations

import torch
from torch import nn

from drone.features import N_CORE
from drone.model import _block


class SemitoneGridCNN(nn.Module):
    """Retain absolute frequency positions while reallocating late channels.

    Input is the unchanged core CQT, shape (batch,192,time), with time >= 8.
    The name describes grid spacing, not an equivariant output constraint.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *_block(1, 16, pool=False),
            nn.MaxPool2d((2, 2)),
            *_block(16, 32, pool=False),
            nn.MaxPool2d((1, 2)),
            *_block(32, 32, pool=False),
            nn.MaxPool2d((1, 2)),
            nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d((N_CORE // 2, 1)),
        )
        self.head = nn.Linear(32 * (N_CORE // 2), 24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != N_CORE or x.shape[2] < 8 or x.shape[0] < 1:
            raise ValueError("Expected nonempty batch,192,time with time >= 8")
        return self.head(self.net(x.unsqueeze(1)).flatten(1))
