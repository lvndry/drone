"""Deterministic content-deduplicated training/validation identities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from drone.keys import KeyIndices

SEED = 13


def validation_split(
    refs: KeyIndices, seed: int = SEED, audio_hashes: Sequence[str] | None = None
) -> tuple[list[int], list[int]]:
    """Fixed seeded, stratified split; training augmentation never touches validation."""
    rng = np.random.default_rng(seed)
    val: list[int] = []
    for key in sorted(set(refs)):
        indices = np.flatnonzero(np.asarray(refs) == key)
        rng.shuffle(indices)
        if len(indices) > 1:
            val.extend(indices[: max(1, len(indices) // 10)].tolist())
    val = sorted(val)
    chosen = set(val)
    train: list[int] = [i for i in range(len(refs)) if i not in chosen]
    if audio_hashes is not None:
        if len(audio_hashes) != len(refs):
            raise ValueError("Audio identities must align with key labels")
        labels_by_audio: dict[str, int] = {}
        for digest, ref in zip(audio_hashes, refs, strict=True):
            if digest in labels_by_audio and labels_by_audio[digest] != ref:
                raise ValueError("Identical audio has conflicting labels; resolve before splitting")
            labels_by_audio[digest] = ref
        # Preserve the original validation identities. Quarantine training copies
        # of validation audio, and count each remaining recording only once.
        seen: set[str] = set()
        unique_val: list[int] = []
        for i in val:
            if audio_hashes[i] not in seen:
                unique_val.append(i)
                seen.add(audio_hashes[i])
        unique_train: list[int] = []
        for i in train:
            if audio_hashes[i] not in seen:
                unique_train.append(i)
                seen.add(audio_hashes[i])
        train, val = unique_train, unique_val
    if not train or not val:
        raise ValueError(
            "Need at least two distinct examples in a class for a training/validation split"
        )
    return train, val
