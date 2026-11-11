"""Key detection CNN matching the K10 (MIREX 2026, MIT) architecture.

Input: (B, N_CORE, T) — 192 core CQT bins, T frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from drone.features import N_CORE

C1, C2, C3 = 16, 32, 64


def _block(cin: int, cout: int, pool: bool) -> list[nn.Module]:
    layers: list[nn.Module] = [
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ELU(),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return layers


class KeyCNN(nn.Module):
    def __init__(self, width: float = 1.0):
        super().__init__()
        c1, c2, c3 = int(C1 * width), int(C2 * width), int(C3 * width)
        # The head pools over TIME ONLY: key identity lives in absolute
        # frequency position, so frequency structure must reach the readout
        # (a full 2-D global pool made the net transposition-invariant and
        # pinned training at chance).
        self.net = nn.Sequential(
            *(_block(1, c1, True) + _block(c1, c2, True) + _block(c2, c3, True)),
            nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d((N_CORE // 8, 1)),  # 192 -> 24 after 3 halvings
        )
        self.head = nn.Linear(c3 * (N_CORE // 8), 24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_CORE, T) -> (B, 1, N_CORE, T)
        h = self.net(x.unsqueeze(1)).flatten(1)
        return self.head(h)


def params_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
