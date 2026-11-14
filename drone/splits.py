"""Deterministic content-deduplicated training/validation identities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from drone.keys import KeyIndices

SEED = 13


def validation_split(
    references: KeyIndices, seed: int = SEED, audio_hashes: Sequence[str] | None = None
) -> tuple[list[int], list[int]]:
    """Fixed seeded, stratified split; training augmentation never touches validation."""
    rng = np.random.default_rng(seed)
    validation: list[int] = []
    for key in sorted(set(references)):
        indices = np.flatnonzero(np.asarray(references) == key)
        rng.shuffle(indices)
        if len(indices) > 1:
            validation.extend(indices[: max(1, len(indices) // 10)].tolist())
    validation = sorted(validation)
    chosen = set(validation)
    train: list[int] = [index for index in range(len(references)) if index not in chosen]
    if audio_hashes is not None:
        if len(audio_hashes) != len(references):
            raise ValueError("Audio identities must align with key labels")
        labels_by_audio: dict[str, int] = {}
        for digest, reference in zip(audio_hashes, references, strict=True):
            if digest in labels_by_audio and labels_by_audio[digest] != reference:
                raise ValueError("Identical audio has conflicting labels; resolve before splitting")
            labels_by_audio[digest] = reference
        # Preserve the original validation identities. Quarantine training copies
        # of validation audio, and count each remaining recording only once.
        seen: set[str] = set()
        unique_validation: list[int] = []
        for index in validation:
            if audio_hashes[index] not in seen:
                unique_validation.append(index)
                seen.add(audio_hashes[index])
        unique_train: list[int] = []
        for index in train:
            if audio_hashes[index] not in seen:
                unique_train.append(index)
                seen.add(audio_hashes[index])
        train = unique_train
        validation = unique_validation
    if not train or not validation:
        raise ValueError(
            "Need at least two distinct examples in a class for a training/validation split"
        )
    return train, validation
