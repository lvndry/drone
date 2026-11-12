"""Fixed Euclidean SAM, coupled randomness and unperturbed-only BN buffers."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

RHO = 0.05


def rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    return torch.get_rng_state(), torch.cuda.get_rng_state(
        device
    ) if device.type == "cuda" else None


def restore_rng(state: tuple[torch.Tensor, torch.Tensor | None], device: torch.device) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    radius: float = RHO,
    loss_function: Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ] = nn.functional.cross_entropy,
) -> torch.Tensor:
    """Two sequential backwards, exact restore, one outer AdamW update.

    radius=0 exists for formula/parity tests only; study fixes0.05. Both passes
    use training BN batch statistics; only first-pass buffers are retained.
    """
    if radius < 0 or not model.training:
        raise ValueError("SAM requires training mode and nonnegative radius")
    device = inputs.device
    before = rng_state(device)
    optimizer.zero_grad()
    loss = loss_function(model(inputs), targets)
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite first SAM loss")
    loss.backward()
    after = rng_state(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise RuntimeError("Missing/nonfinite first SAM gradient")
    originals = [p.detach().clone() for p in parameters]
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(p.grad) for p in parameters])
    )
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite global SAM gradient norm")
    scale = radius / (float(norm) + 1e-12)
    try:
        with torch.no_grad():
            for parameter in parameters:
                gradient = parameter.grad
                assert gradient is not None
                parameter.add_(gradient, alpha=scale)
        optimizer.zero_grad()
        restore_rng(before, device)
        perturbed = loss_function(model(inputs), targets)
        if not torch.isfinite(perturbed):
            raise RuntimeError("Nonfinite perturbed SAM loss")
        perturbed.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise RuntimeError("Missing/nonfinite perturbed SAM gradient")
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals, strict=True):
                parameter.copy_(original)
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        restore_rng(after, device)
    optimizer.step()
    return loss.detach()
