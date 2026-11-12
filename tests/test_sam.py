"""Synthetic CPU gradient/state proofs, without any study fitting data."""

from __future__ import annotations

import ast
import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn

from experiments.sam_step import step
from experiments.samstudy import AUDIT, ROOT, source_checks


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(4, 24)
        self.masks: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(2) if x.ndim == 3 else x
        x = self.bn(x)
        dropped = self.dropout(x)
        self.masks.append((dropped != 0).detach().clone())
        return self.head(dropped)


def assert_state_equal(a: nn.Module, b: nn.Module) -> None:
    for name, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[name]), name


def test_zero_radius_exact_adamw_parity_rng_bn_and_step_count() -> None:
    torch.manual_seed(13)
    x, y = torch.randn(8, 4), torch.arange(8)
    raw = Toy().train()
    sam = copy.deepcopy(raw)
    ordinary = torch.optim.AdamW(raw.parameters(), lr=0.001, weight_decay=0.0001)
    robust = torch.optim.AdamW(sam.parameters(), lr=0.001, weight_decay=0.0001)
    for _ in range(3):
        initial = torch.get_rng_state()
        ordinary.zero_grad()
        loss = nn.functional.cross_entropy(raw(x), y)
        loss.backward()
        ordinary.step()
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(initial)
        actual = step(sam, robust, x, y, radius=0)
        assert torch.equal(actual, loss.detach())
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert_state_equal(raw, sam)
        assert torch.equal(sam.masks[-1], sam.masks[-2])
    assert sam.bn.num_batches_tracked == 3
    assert all(state["step"] == 3 for state in robust.state.values())


def test_perturbation_formula_and_one_outer_step() -> None:
    torch.manual_seed(37)
    model = nn.Linear(4, 24).train()
    initial = copy.deepcopy(model)
    x, y = torch.randn(8, 4), torch.arange(8)
    initial.zero_grad()
    loss = nn.functional.cross_entropy(initial(x), y)
    loss.backward()
    gradients = [p.grad for p in initial.parameters()]
    assert all(g is not None for g in gradients)
    gradients = [g for g in gradients if g is not None]
    norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients]))
    expected = [
        p.detach() + 0.05 * g / (norm + 1e-12)
        for p, g in zip(initial.parameters(), gradients, strict=True)
    ]
    observed: list[list[torch.Tensor]] = []

    def capture(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        observed.append([p.detach().clone() for p in module.parameters()])

    model.register_forward_pre_hook(capture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    step(model, optimizer, x, y)
    for actual, want in zip(observed[1], expected, strict=True):
        torch.testing.assert_close(actual, want, rtol=1e-6, atol=1e-7)
    assert all(state["step"] == 1 for state in optimizer.state.values())


def test_second_pass_exception_restores_exact_weights_first_bn_and_rng() -> None:
    torch.manual_seed(73)
    model = Toy().train()
    original = copy.deepcopy(model)
    expected = copy.deepcopy(model)
    x, y = torch.randn(8, 4), torch.arange(8)
    initial_rng = torch.get_rng_state()
    expected(x)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(initial_rng)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    count = 0

    def fail_second(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("synthetic second-pass failure")
        return nn.functional.cross_entropy(logits, targets)

    with pytest.raises(RuntimeError, match="second-pass"):
        step(model, optimizer, x, y, loss_function=fail_second)
    for actual, before in zip(model.parameters(), original.parameters(), strict=True):
        assert torch.equal(actual, before)
    for name, value in expected.named_buffers():
        assert torch.equal(dict(model.named_buffers())[name], value)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert not optimizer.state
    assert torch.equal(model.masks[0], model.masks[1])


def test_disabled_branch_literal_and_executable_parity() -> None:
    source_checks()
    audit = json.loads(AUDIT.read_bytes())
    tree = ast.parse((ROOT / "experiments/k10_multidomain.py").read_text())
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    branch = next(
        n
        for n in ast.walk(train)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "training_step is None"
    )
    current = "\n".join(ast.unparse(n) for n in branch.body)

    def build(body: str) -> Callable[..., torch.Tensor]:
        namespace: dict[str, Any] = dict(torch=torch, np=np)
        code = 'def run(model,optimizer,xs,ys):\n    targets=None\n    args=type("Args",(),{"device":"cpu"})()\n'
        code += (
            "\n".join("    " + line for line in body.splitlines()) + "\n    return loss.detach()\n"
        )
        exec(compile(code, "<frozen-batch-proof>", "exec"), namespace)
        return cast(Callable[..., torch.Tensor], namespace["run"])

    raw_step, disabled_step = build(audit["ordinary_batch_source"]), build(current)
    torch.manual_seed(13)
    raw = Toy().train()
    disabled = copy.deepcopy(raw)
    a = torch.optim.AdamW(raw.parameters(), lr=0.001, weight_decay=0.0001)
    b = torch.optim.AdamW(disabled.parameters(), lr=0.001, weight_decay=0.0001)
    xs = list(np.random.default_rng(13).normal(size=(8, 4, 9)).astype(np.float32))
    for _ in range(3):
        rng = torch.get_rng_state()
        expected = raw_step(raw, a, xs, list(range(8)))
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        actual = disabled_step(disabled, b, xs, list(range(8)))
        assert torch.equal(actual, expected)
        assert torch.equal(after, torch.get_rng_state())
        assert_state_equal(raw, disabled)


def test_source_guard_rejects_default_math_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audit = json.loads(AUDIT.read_bytes())
    audit["ordinary_batch_ast"] = "changed"
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    monkeypatch.setattr("experiments.samstudy.AUDIT", path)
    with pytest.raises(ValueError, match="ordinary batch"):
        source_checks()


def test_real_k10_all_six_bn_buffers_advance_once() -> None:
    from drone.model import KeyCNN

    torch.set_num_threads(4)
    torch.manual_seed(13)
    model = KeyCNN().train()
    expected = copy.deepcopy(model)
    x, y = torch.randn(2, 192, 8), torch.tensor([0, 19])
    rng = torch.get_rng_state()
    expected(x)
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    step(model, optimizer, x, y)
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(bns) == 6
    assert all(m.num_batches_tracked == 1 for m in bns)
    for name, value in expected.named_buffers():
        assert torch.equal(dict(model.named_buffers())[name], value)
    assert torch.equal(torch.get_rng_state(), after)
