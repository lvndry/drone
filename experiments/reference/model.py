"""Small key CNN controls. Input: (batch, 33 bands, time).

The frequency-pooling experiment keeps six final frequency positions instead of
averaging them away. Its small profile remains below the 2 MiB FP32 weight budget.
Global pooling remains the default until controlled real-data experiments justify
promoting a different model.
"""

import torch.nn as nn

MODEL_SIZES = {"small": 1, "medium": 2, "large": 4}


class KeyCNN(nn.Module):
    def __init__(self, size="small", dropout=0.3, pooling="global"):
        super().__init__()
        if size not in MODEL_SIZES:
            raise ValueError(f"Unknown model size: {size}")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")
        if pooling not in ("global", "frequency"):
            raise ValueError(f"Unknown pooling: {pooling}")
        height = 6 if pooling == "frequency" else 1
        self.config = {"size": size, "dropout": dropout, "pooling": pooling}
        a, b, c = (n * MODEL_SIZES[size] for n in (32, 64, 128))
        self.net = nn.Sequential(
            nn.Conv2d(1, a, kernel_size=(1, 8), stride=(1, 4)),
            nn.BatchNorm2d(a),
            nn.ReLU(),
            nn.Conv2d(a, b, kernel_size=(4, 8), stride=(2, 4)),
            nn.BatchNorm2d(b),
            nn.ReLU(),
            nn.Conv2d(b, c, kernel_size=(4, 8), stride=(2, 4)),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((height, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c * height, 24),
        )

    def forward(self, x):
        # x: (B, n_sem, T) -> (B, 1, n_sem, T)
        return self.head(self.net(x.unsqueeze(1)))


def params_count(model):
    return sum(p.numel() for p in model.parameters())
