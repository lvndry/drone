"""Training-boundary checks on a tiny stand-in, not MusicFM accuracy evidence."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.musicfm_teacher import KeyTeacher, accumulated_update, make_optimizer


class ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 1024)

    def preprocessing(self, wave: Tensor, names: list[str]) -> dict[str, Tensor]:
        assert wave.dtype == torch.float32 and names == ["melspec_2048"]
        return {names[0]: wave.mean(dim=1, keepdim=True)}

    def normalize(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        return features

    def encoder(self, features: Tensor) -> tuple[None, list[Tensor]]:
        hidden = self.projection(features).unsqueeze(1)
        return None, [hidden] * 13


def examples() -> list[tuple[Tensor, int]]:
    return [(torch.full((1, 1025), (i + 1) / 32), i % 24) for i in range(32)]


def test_head_is_reproducible_and_does_not_consume_global_rng() -> None:
    backbone = ToyBackbone()
    state = torch.get_rng_state().clone()
    first, second = KeyTeacher(backbone), KeyTeacher(backbone)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(first.head.weight, second.head.weight)


def test_accumulation_matches_mean_batch_objective_without_batch_dependent_layers() -> None:
    teacher = KeyTeacher(ToyBackbone())
    reference = copy.deepcopy(teacher)
    optimizer, other = make_optimizer(teacher), make_optimizer(reference)
    batch = examples()
    loss = accumulated_update(teacher, optimizer, batch, amp=False)
    expected = F.cross_entropy(
        reference(torch.cat([x for x, _ in batch])), torch.tensor([y for _, y in batch])
    )
    expected.backward()
    other.step()
    assert loss == pytest.approx(expected.item(), rel=1e-6)
    for actual, wanted in zip(teacher.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, wanted, atol=1e-7, rtol=1e-5)
        assert actual.grad is None
    for state in optimizer.state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
        assert state["step"] == 1
    assert [g["lr"] for g in optimizer.param_groups] == [1e-5, 1e-4]


@pytest.mark.parametrize("length", [0, 31, 33])
def test_wrong_accumulation_never_commits_an_update(length: int) -> None:
    teacher = KeyTeacher(ToyBackbone())
    optimizer = make_optimizer(teacher)
    before = copy.deepcopy(teacher.state_dict())
    batch = (examples() * 2)[:length]
    with pytest.raises(ValueError, match="Exactly 32"):
        accumulated_update(teacher, optimizer, batch, amp=False)
    assert not optimizer.state
    for key, value in teacher.state_dict().items():
        assert torch.equal(value, before[key])


def test_nonfinite_loss_aborts_before_optimizer_commit() -> None:
    teacher = KeyTeacher(ToyBackbone())
    optimizer = make_optimizer(teacher)
    before = copy.deepcopy(teacher.state_dict())
    batch = examples()
    batch[17] = (torch.full((1, 1025), float("nan")), 3)
    with pytest.raises(ValueError, match="Nonfinite microbatch loss"):
        accumulated_update(teacher, optimizer, batch, amp=False)
    assert not optimizer.state
    for key, value in teacher.state_dict().items():
        assert torch.equal(value, before[key])


def test_nonfinite_gradient_aborts_even_with_finite_losses() -> None:
    teacher = KeyTeacher(ToyBackbone())
    optimizer = make_optimizer(teacher)
    before = copy.deepcopy(teacher.state_dict())
    hook = teacher.head.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    with pytest.raises(ValueError, match="accumulated gradients"):
        accumulated_update(teacher, optimizer, examples(), amp=False)
    hook.remove()
    assert not optimizer.state
    for key, value in teacher.state_dict().items():
        assert torch.equal(value, before[key])
