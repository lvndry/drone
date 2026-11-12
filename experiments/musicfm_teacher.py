"""Prospective key adapter and accumulated update; no execution/fit entrypoint."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Self

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.musicfm_key import SongLogits, SongWindow

if TYPE_CHECKING:
    from numpy.typing import NDArray

ACCUMULATION = 32
MIN_TRAIN_SAMPLES = 1200  # Five mel steps -> two positions for microbatch-one BatchNorm.


def seed_training(seed: int = 13) -> None:
    """Call AFTER upstream construction/strict load; its quantizer resets seed to 142."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


class KeyTeacher(nn.Module):
    """Use original MusicFM frontend/encoder and last-layer time-mean readout."""

    def __init__(self, backbone: nn.Module, *, seed: int = 13) -> None:
        super().__init__()
        self.backbone = backbone
        # Seed this new CPU head independently without disturbing dropout's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.head = nn.Linear(1024, 24, dtype=torch.float32, device="cpu")

    def _rotary_position(self) -> nn.Module | None:
        # Tiny synthetic backbones need no Conformer; the actual pinned model does.
        conformer = getattr(self.backbone, "conformer", None)
        if conformer is None:
            return None
        position = getattr(conformer, "embed_positions", None)
        if (
            not isinstance(position, nn.Module)
            or type(position).__name__ != "Wav2Vec2ConformerRotaryPositionalEmbedding"
            or not isinstance(getattr(position, "inv_freq", None), Tensor)
        ):
            raise ValueError("Expected the pinned Conformer rotary cache contract")
        length = getattr(position, "cached_sequence_length", "missing")
        cached = getattr(position, "cached_rotary_positional_embedding", "missing")
        if length is None and cached is None:
            return position
        if (
            type(length) is not int
            or length <= 0
            or not isinstance(cached, Tensor)
            or cached.dtype not in (torch.float32, torch.bfloat16)
            or cached.shape != (2, length, 1, 1, position.inv_freq.numel() * 2)
        ):
            raise ValueError("Malformed pinned Conformer rotary cache")
        return position

    def _clear_rotary_cache(self) -> None:
        position = self._rotary_position()
        if position is not None:
            setattr(position, "cached_sequence_length", None)
            setattr(position, "cached_rotary_positional_embedding", None)

    def train(self, mode: bool = True) -> Self:
        changed = self.training != mode
        super().train(mode)
        if changed:
            # Upstream keys this dtype-dependent cache only by sequence length.
            # eval() must also protect the direct B16 encoder path.
            self._clear_rotary_cache()
        return self

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        # These ordinary attributes are not buffers and nn.Module.to() cannot move them.
        self._clear_rotary_cache()
        super()._apply(fn, recurse=recurse)
        return self

    def forward(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 2 or waveform.shape[1] <= 1024 or waveform.dtype != torch.float32:
            raise ValueError("Expected actual FP32 waveforms with more than 1024 samples")
        with torch.autocast(waveform.device.type, enabled=False):
            features = getattr(self.backbone, "normalize")(
                getattr(self.backbone, "preprocessing")(waveform, ["melspec_2048"])
            )
        position = self._rotary_position()
        cached = getattr(position, "cached_rotary_positional_embedding", None)
        expected_dtype = (
            torch.get_autocast_dtype(waveform.device.type)
            if torch.is_autocast_enabled(waveform.device.type)
            else features["melspec_2048"].dtype
        )
        if isinstance(cached, Tensor) and (
            cached.dtype != expected_dtype or cached.device != waveform.device
        ):
            self._clear_rotary_cache()
        _, hidden = getattr(self.backbone, "encoder")(features["melspec_2048"])
        embedding = hidden[12]
        if (
            embedding.ndim != 3
            or embedding.shape[0] != waveform.shape[0]
            or embedding.shape[2] != 1024
        ):
            raise ValueError("Unexpected MusicFM last hidden state")
        return self.head(embedding.mean(dim=1))


def make_optimizer(teacher: KeyTeacher) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {"params": teacher.backbone.parameters(), "lr": 1e-5},
            {"params": teacher.head.parameters(), "lr": 1e-4},
        ],
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
        fused=False,
    )


def accumulated_update(
    teacher: KeyTeacher,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[tuple[Tensor, int]],
    *,
    amp: bool = True,
) -> float:
    """Commit exactly 32 hard-CE microbatches, abort before step on nonfinite data.

    Input tensors are actual-length CPU or device waveforms, shape (1, samples).
    This is not a true batch of 32 for BatchNorm. The caller owns source/cache
    validation, sampling, scheduling and the continuous resource supervisor.
    """
    if len(examples) != ACCUMULATION:
        raise ValueError("Exactly 32 examples are required for one update")
    if any(type(label) is not int or not 0 <= label < 24 for _, label in examples):
        raise ValueError("Targets must be integer Drone classes 0–23")
    if any(
        wave.ndim != 2
        or wave.shape[0] != 1
        or wave.dtype != torch.float32
        or wave.shape[1] < MIN_TRAIN_SAMPLES
        for wave, _ in examples
    ):
        raise ValueError(
            "Each microbatch needs one actual-length FP32 waveform of at least 1200 samples"
        )
    device = teacher.head.weight.device
    if amp and device.type != "cuda":
        raise ValueError("The prospective AMP recipe requires CUDA BF16")
    teacher.train()
    optimizer.zero_grad(set_to_none=True)
    labels = torch.tensor([label for _, label in examples], device=device, dtype=torch.long)
    total = 0.0
    for index, (waveform, _) in enumerate(examples):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = teacher(waveform.to(device=device))
            loss = F.cross_entropy(logits.float(), labels[index : index + 1])
        value = loss.detach().item()
        if not math.isfinite(value):
            raise ValueError("Nonfinite microbatch loss; no optimizer step")
        total += value
        (loss / ACCUMULATION).backward()
    gradients = [p.grad for p in teacher.parameters() if p.grad is not None]
    if not gradients or not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):
        raise ValueError("Nonfinite or missing accumulated gradients; no optimizer step")
    optimizer.step()
    # Also ensures evaluation does not retain the last update's gradient buffers.
    optimizer.zero_grad(set_to_none=True)
    return total / ACCUMULATION


def evaluate_song(
    teacher: KeyTeacher,
    samples: int,
    windows: Iterable[tuple[SongWindow, Tensor]],
) -> NDArray[np.float64]:
    """Batch-one reference: every actual window, FP32 eval, FP64 mean.

    Waveforms must come from the immutable prepared cache, not a new decoder.
    The iterable can stream one CPU waveform at a time. The caller maintains
    resource/service guards; this helper neither loads labels nor scores keys.
    """
    accumulator = SongLogits(samples)
    device = teacher.head.weight.device
    teacher.eval()
    with torch.inference_mode(), torch.autocast(device.type, enabled=False):
        for window, waveform in windows:
            if waveform.shape != (1, window.stop - window.start):
                raise ValueError("Actual waveform must match its complete-song window context")
            logits = teacher(waveform.to(device=device))
            if logits.shape != (1, 24) or logits.dtype != torch.float32:
                raise ValueError("Expected FP32 batch-one teacher logits")
            accumulator.add(window, logits[0].cpu().numpy())
    return accumulator.finish()
