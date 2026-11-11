"""Small CNN for 24-class key detection. Input: (B, 33, T) CQT."""

import torch
import torch.nn as nn


class KeyCNN(nn.Module):
    def __init__(self, n_sem=33, t=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 8), stride=(1, 4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 8), stride=(2, 4)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(4, 8), stride=(2, 4)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 24),
        )

    def forward(self, x):
        # x: (B, T, n_sem) -> (B, 1, n_sem, T)
        return self.head(self.net(x.unsqueeze(1)))


def params_count(model):
    return sum(p.numel() for p in model.parameters())
