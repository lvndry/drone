"""Fixed full-token M/O/S residual heads over frozen 64×24 K10 activations."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import nn

VERSION = "frozen-k10-temporal-v1"


def permutation(seed: int, recording: str, tokens: int, epoch: int | None) -> np.ndarray:
    """Independent PCG64: SHA256 of compact UTF8 JSON; full digest big-endian seed."""
    encoded = json.dumps(
        [VERSION, seed, recording, "development" if epoch is None else "fitting", epoch],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(encoded).digest(), "big")
    ).permutation(tokens)


def masked(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return torch.where(valid, x, torch.zeros_like(x))


class Block(nn.Module):
    def __init__(self, kernel: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(64, eps=1e-5)
        self.depth = nn.Conv1d(
            64,
            64,
            kernel,
            groups=64,
            dilation=dilation,
            padding=(kernel - 1) * dilation // 2,
            bias=True,
        )
        self.point = nn.Linear(64, 64, bias=True)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        base = masked(x, valid)
        value = masked(self.norm(base), valid)
        value = self.depth(value.transpose(1, 2)).transpose(1, 2)
        value = masked(value, valid)
        value = masked(nn.functional.gelu(value, approximate="none"), valid)
        value = masked(self.point(masked(value, valid)), valid)
        return masked(base + value, valid)


class TemporalHead(nn.Module):
    def __init__(self, kernel: int) -> None:
        super().__init__()
        if kernel not in (1, 5):
            raise ValueError("Only frozen M/kernel1 or O,S/kernel5")
        self.kernel = kernel
        self.project = nn.Linear(1536, 64, bias=True)
        self.norm = nn.LayerNorm(64, eps=1e-5)
        self.blocks = nn.ModuleList([Block(kernel, 1), Block(kernel, 4)])
        self.readout = nn.Linear(64, 24, bias=True)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                fan_in = module.weight.shape[1] * math.prod(module.weight.shape[2:])
                nn.init.normal_(module.weight, mean=0, std=1 / math.sqrt(fan_in))
                assert module.bias is not None
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                assert module.bias is not None
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 4 or hidden.shape[1:3] != (64, 24) or lengths.shape != (hidden.shape[0],):
            raise ValueError("Expected B×64×24×N and B valid lengths")
        if not torch.compiler.is_compiling() and not torch.jit.is_tracing():
            if bool(((lengths < 1) | (lengths > hidden.shape[-1])).any()):
                raise ValueError("Invalid hidden token length")
        valid = (torch.arange(hidden.shape[-1], device=hidden.device)[None, :] < lengths[:, None])[
            :, :, None
        ]
        x = hidden.flatten(1, 2).transpose(1, 2)
        x = masked(x, valid)
        x = masked(self.project(x), valid)
        x = masked(self.norm(x), valid)
        x = masked(nn.functional.gelu(x, approximate="none"), valid)
        for block in self.blocks:
            x = block(x, valid)
        pooled = masked(x, valid).sum(1) / lengths[:, None]
        return self.readout(pooled)


def initial_heads(seed: int) -> dict[str, TemporalHead]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        ordered = TemporalHead(5)
        shuffled = copy.deepcopy(ordered)
        mean = TemporalHead(1)
        for name, value in mean.state_dict().items():
            source = ordered.state_dict()[name]
            if source.shape != value.shape:
                source = source[:, :, 2:3]
            value.copy_(source)
    return dict(M=mean, O=ordered, S=shuffled)


def microbatches(indices: Iterable[int], lengths: list[int], limit: int = 8192) -> list[list[int]]:
    result: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for index in indices:
        size = lengths[index]
        if size < 1:
            raise ValueError("Need positive token lengths")
        if current and max(longest, size) * (len(current) + 1) > limit:
            result.append(current)
            current = []
            longest = 0
        current.append(index)
        longest = max(longest, size)
        if size > limit:
            result.append(current)
            current = []
            longest = 0
    if current:
        result.append(current)
    return result


def better(candidate: dict[str, float], best: dict[str, float] | None) -> bool:
    """Exact first, equally weighted domain CE second, earlier epoch last."""
    return best is None or (-candidate["exact"], candidate["ce"], candidate["epoch"]) < (
        -best["exact"],
        best["ce"],
        best["epoch"],
    )


def gates(
    canonical: dict[str, dict[str, Any]], outputs: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, object]:
    ordered = outputs["O"]
    delta = {
        d: {m: ordered[d][m] - canonical[d][m] for m in ("exact", "mirex")}
        for d in ("beatport", "fmak")
    }

    def correct(value: dict[str, Any]) -> int:
        return round(value["exact"] * value["tracks"])

    def credit(value: dict[str, Any]) -> int:
        return round(value["mirex"] * value["tracks"] * 10)

    candidate = (
        correct(ordered["fmak"]) - correct(canonical["fmak"]) >= 5
        and correct(ordered["beatport"]) >= correct(canonical["beatport"])
        and all(credit(ordered[d]) >= credit(canonical[d]) for d in ("beatport", "fmak"))
    )
    mechanism_delta = {
        arm: ordered["fmak"]["exact"] - outputs[arm]["fmak"]["exact"] for arm in ("M", "S")
    }
    mechanism_counts = {
        arm: correct(ordered["fmak"]) - correct(outputs[arm]["fmak"]) for arm in ("M", "S")
    }
    return dict(
        candidate_pass=candidate,
        candidate_delta=delta,
        mechanism_pass=all(v >= 3 for v in mechanism_counts.values()),
        mechanism_net_exact_counts=mechanism_counts,
        mechanism_fmak_exact_delta=mechanism_delta,
        sole_candidate="O",
        promoted=False,
    )
