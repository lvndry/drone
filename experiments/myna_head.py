"""Shared prospective Myna head kernels for fitting and label-blind cost qualification.

Only NumPy, Torch and the standard library are imported. No pretrained encoder,
reference labels or training entry point is loaded by importing this module.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def make_head(device: str = "cpu") -> nn.Sequential:
    """One dropout after the first ReLU; never instantiate the pretrained encoder."""
    return nn.Sequential(
        nn.Linear(384, 4096, device=device, dtype=torch.float32),
        nn.ReLU(),
        nn.Dropout(0.99),
        nn.Linear(4096, 4096, device=device, dtype=torch.float32),
        nn.ReLU(),
        nn.Linear(4096, 24, device=device, dtype=torch.float32),
    )


def weighted_ce(logits: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
    """Unnormalized importance weights: a random batch must not redefine the objective."""
    _require(
        logits.ndim == 2
        and logits.shape[1] == 24
        and targets.shape == weights.shape == (len(logits),),
        "Invalid fitting batch",
    )
    _require(
        bool(torch.isfinite(logits).all())
        and bool(torch.isfinite(weights).all())
        and bool((weights > 0).all()),
        "Nonfinite/invalid objective input",
    )
    return (functional.cross_entropy(logits, targets, reduction="none") * weights).mean()


def mean_logits(
    chunks: Iterator[NDArray[np.float32]], expected_windows: int
) -> NDArray[np.float64]:
    """Average window logits, not probabilities or input embeddings, in FP64."""
    total = np.zeros(24, dtype=np.float64)
    seen = 0
    for chunk in chunks:
        _require(
            chunk.dtype == np.float32
            and chunk.ndim == 2
            and chunk.shape[1] == 24
            and bool(np.isfinite(chunk).all()),
            "Invalid window logits",
        )
        # Fixed window order makes aggregation independent of chunk/batch boundaries.
        for row in chunk:
            total += row.astype(np.float64)
        seen += len(chunk)
    _require(seen == expected_windows and seen > 0, "Missing/extra selection windows")
    return total / seen


def make_optimizer(head: nn.Module) -> torch.optim.Adam:
    return torch.optim.Adam(
        head.parameters(),
        lr=3e-4,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
        fused=False,
    )


def train_batch(
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    embeddings: NDArray[np.float32],
    targets: NDArray[np.int64],
    weights: NDArray[np.float32],
    watch: Callable[[], None],
) -> float:
    """The actual timed fitting path, including transfers/checks and scalar synchronization."""
    watch()
    _require(
        embeddings.dtype == np.float32
        and targets.dtype == np.int64
        and weights.dtype == np.float32,
        "Different training dtype",
    )
    _require(
        embeddings.ndim == 2 and embeddings.shape[1] == 384 and 0 < len(embeddings) <= 64,
        "Different fitting batch geometry",
    )
    device = next(head.parameters()).device
    head.train()
    optimizer.zero_grad(set_to_none=True)
    values = torch.from_numpy(embeddings).to(device)
    target_tensor = torch.from_numpy(targets).to(device)
    weight_tensor = torch.from_numpy(weights).to(device)
    loss = weighted_ce(head(values), target_tensor, weight_tensor)
    _require(bool(torch.isfinite(loss)), "Nonfinite loss; no optimizer step")
    loss.backward()
    _require(
        all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in head.parameters()
        ),
        "Missing/nonfinite gradients; no optimizer step",
    )
    optimizer.step()
    return float(loss.detach().cpu())


def window_logits(
    head: nn.Module,
    embeddings: NDArray[np.float32],
    batch_size: int,
    watch: Callable[[], None],
) -> Iterator[NDArray[np.float32]]:
    """The actual ordered evaluation path including host/device transfers and synchronization."""
    _require(
        embeddings.dtype == np.float32
        and embeddings.ndim == 2
        and embeddings.shape[1] == 384
        and len(embeddings) > 0,
        "Invalid selection embeddings",
    )
    _require(1 <= batch_size <= 64, "Unsupported head evaluation batch")
    device = next(head.parameters()).device
    head.eval()
    with torch.inference_mode():
        for offset in range(0, len(embeddings), batch_size):
            watch()
            values = np.array(embeddings[offset : offset + batch_size], copy=True)
            yield head(torch.from_numpy(values).to(device)).cpu().numpy()
