"""Training-boundary checks on a tiny stand-in, not MusicFM accuracy evidence."""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.musicfm_key import song_windows
from experiments.musicfm_teacher import (
    KeyTeacher,
    accumulated_update,
    evaluate_song,
    make_optimizer,
    seed_training,
)


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
    return [(torch.full((1, 1200), (i + 1) / 32), i % 24) for i in range(32)]


def test_training_seed_is_reset_after_an_upstream_constructor_changes_it() -> None:
    def draw() -> tuple[float, float, Tensor]:
        return random.random(), float(np.random.random()), torch.rand(4)

    seed_training()
    expected = draw()
    random.seed(142)
    np.random.seed(142)
    torch.manual_seed(142)
    seed_training()
    actual = draw()
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


def test_single_microbatch_rejects_one_encoded_position_before_update() -> None:
    teacher = KeyTeacher(ToyBackbone())
    optimizer = make_optimizer(teacher)
    batch = examples()
    batch[0] = (torch.zeros(1, 1199), 0)
    with pytest.raises(ValueError, match="1200 samples"):
        accumulated_update(teacher, optimizer, batch, amp=False)
    assert not optimizer.state


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
    batch[17] = (torch.full((1, 1200), float("nan")), 3)
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


def test_streamed_song_evaluation_matches_explicit_weighted_reference() -> None:
    teacher = KeyTeacher(ToyBackbone())
    samples = 2 * 288000 + 1199
    windows = song_windows(samples)
    inputs = [torch.full((1, w.stop - w.start), float(i + 1)) for i, w in enumerate(windows)]
    with torch.inference_mode():
        explicit = (
            sum(
                teacher(x)[0].numpy().astype(np.float64) * w.weight
                for x, w in zip(inputs, windows, strict=True)
            )
            / samples
        )
    actual = evaluate_song(teacher, samples, zip(windows, inputs, strict=True))
    np.testing.assert_array_equal(actual, explicit)
    assert not teacher.training
    assert all(p.grad is None for p in teacher.parameters())
    with pytest.raises(ValueError, match="complete song"):
        evaluate_song(teacher, samples, zip(windows[:-1], inputs[:-1], strict=True))
    with pytest.raises(ValueError, match="window context"):
        evaluate_song(teacher, samples, [(windows[0], torch.zeros(1, 288001))])
