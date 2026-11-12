"""Numerical and optimization contracts for the research-only fused graph."""

from __future__ import annotations

import io

import pytest
import torch
from torch import nn

from drone.model import KeyCNN
from experiments.k10_reparameterization import (
    FusedKeyCNN,
    ReparameterizedKeyCNN,
    TrainingBranches,
)


def test_initial_function_rng_and_source_are_preserved() -> None:
    torch.manual_seed(13)
    source = KeyCNN().eval()
    before = {k: v.clone() for k, v in source.state_dict().items()}
    rng = torch.get_rng_state().clone()
    candidate = ReparameterizedKeyCNN(source)
    assert torch.equal(rng, torch.get_rng_state())
    branches = [m for m in candidate.modules() if isinstance(m, TrainingBranches)]
    first = branches[0].side_conv.weight.flatten()[:16]
    second = branches[1].side_conv.weight.flatten()[:16] * 4
    assert not torch.allclose(first, second)
    x = torch.randn(2, 192, 65)
    with torch.inference_mode():
        torch.testing.assert_close(candidate(x), source(x), rtol=0, atol=0)
    for k, v in source.state_dict().items():
        assert torch.equal(v, before[k])


@pytest.mark.parametrize("frames", [8, 65, 320, 1615])
def test_nontrivial_branches_fuse_and_serialize(frames: int) -> None:
    torch.manual_seed(73)
    candidate = ReparameterizedKeyCNN(KeyCNN()).double().eval()
    # Exercise nonzero branches, nondefault statistics AND biases; checking only
    # the zero-initialized model would not test the fusion algebra.
    with torch.no_grad():
        for module in candidate.modules():
            if isinstance(module, nn.BatchNorm2d):
                assert module.running_mean is not None and module.running_var is not None
                module.weight.normal_()
                module.bias.normal_()
                module.running_mean.normal_()
                module.running_var.uniform_(0.3, 2)
    before = {k: v.clone() for k, v in candidate.state_dict().items()}
    deploy = candidate.to_deploy()
    assert sum(isinstance(m, nn.Conv2d) for m in deploy.modules()) == 6
    assert not any(isinstance(m, (TrainingBranches, nn.BatchNorm2d)) for m in deploy.modules())
    x = torch.randn(1, 192, frames, dtype=torch.float64)
    with torch.inference_mode():
        torch.testing.assert_close(deploy(x), candidate(x), rtol=1e-10, atol=1e-10)
    buf = io.BytesIO()
    torch.save(deploy.state_dict(), buf)
    buf.seek(0)
    # The loader has no reference to the trained/candidate model. Production
    # checkpoint factories expect BN keys and must not silently accept this graph.
    restored = FusedKeyCNN().double().eval()
    restored.load_state_dict(torch.load(buf, weights_only=True), strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(restored(x), deploy(x), rtol=0, atol=0)
    for k, v in candidate.state_dict().items():
        assert torch.equal(v, before[k])


def test_side_branch_learns_after_zero_scale_initialization() -> None:
    torch.manual_seed(13)
    branch = TrainingBranches(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16))
    optimizer = torch.optim.SGD(branch.parameters(), lr=0.1)
    x, target = torch.randn(2, 1, 9, 11), torch.randn(2, 16, 9, 11)
    for step in range(2):
        optimizer.zero_grad()
        (branch(x) - target).square().mean().backward()
        assert branch.side_bn.weight.grad is not None
        assert branch.side_bn.weight.grad.abs().sum() > 0
        if step == 1:
            assert branch.side_conv.weight.grad is not None
            assert branch.side_conv.weight.grad.abs().sum() > 0
        optimizer.step()


def test_conversion_rejects_training_and_mixed_modes() -> None:
    candidate = ReparameterizedKeyCNN(KeyCNN())
    with pytest.raises(ValueError, match="evaluation mode"):
        candidate.to_deploy()
    candidate.eval()
    candidate.model.net[0].train()
    with pytest.raises(ValueError, match="evaluation mode"):
        candidate.to_deploy()


def test_noncanonical_width_is_rejected() -> None:
    with pytest.raises(ValueError, match="canonical"):
        ReparameterizedKeyCNN(KeyCNN(width=2))


def test_initial_training_logits_and_main_gradients_match() -> None:
    torch.manual_seed(13)
    source = KeyCNN().train()
    candidate = ReparameterizedKeyCNN(source)
    x, target = torch.randn(2, 192, 65), torch.tensor([3, 18])
    rng = torch.get_rng_state().clone()
    expected = source(x)
    nn.functional.cross_entropy(expected, target).backward()
    torch.set_rng_state(rng)
    actual = candidate(x)
    nn.functional.cross_entropy(actual, target).backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for i, module in enumerate(source.net):
        replacement = candidate.model.net[i]
        if isinstance(module, nn.Conv2d):
            assert isinstance(replacement, TrainingBranches)
            torch.testing.assert_close(replacement.main_conv.weight.grad, module.weight.grad)
        elif isinstance(module, nn.BatchNorm2d):
            branch = candidate.model.net[i - 1]
            assert isinstance(branch, TrainingBranches)
            torch.testing.assert_close(branch.main_bn.weight.grad, module.weight.grad)
            torch.testing.assert_close(branch.main_bn.running_mean, module.running_mean)
            torch.testing.assert_close(branch.main_bn.running_var, module.running_var)
    torch.testing.assert_close(candidate.model.head.weight.grad, source.head.weight.grad)


def test_float32_fusion_after_actual_branch_updates() -> None:
    torch.manual_seed(13)
    candidate = ReparameterizedKeyCNN(KeyCNN())
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(
            candidate(torch.randn(2, 192, 65)), torch.tensor([3, 18])
        )
        loss.backward()
        optimizer.step()
    candidate.eval()
    deploy = candidate.to_deploy()
    with torch.inference_mode():
        for frames in (8, 65, 320, 1615):
            x = torch.randn(1, 192, frames)
            torch.testing.assert_close(deploy(x), candidate(x), rtol=2e-5, atol=2e-6)


def test_mixed_source_modes_are_rejected() -> None:
    source = KeyCNN().train()
    source.net[1].eval()
    with pytest.raises(ValueError, match="uniform"):
        ReparameterizedKeyCNN(source)


def test_cpu_construction_and_separate_checkpoint_factory() -> None:
    with pytest.raises(ValueError, match="CPU source"):
        ReparameterizedKeyCNN(KeyCNN().to("meta"))
    with torch.device("meta"):
        deploy = FusedKeyCNN()
    assert all(t.device.type == "cpu" for t in deploy.state_dict().values())
    with pytest.raises(RuntimeError, match="Missing key"):
        KeyCNN().load_state_dict(deploy.state_dict(), strict=True)
