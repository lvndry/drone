"""Pure schedule/moment primitives for studies/musicfm-key-pilot/BN_PROTOCOL.md.

No model, waveform reader, checkpoint, scoring or execution entry point is provided.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Domain = Literal["beatport", "fmak"]
FloatArray = NDArray[np.float32] | NDArray[np.float64]
SALT = "musicfm-bn-posthoc-v1"
PER_DOMAIN = 256
FIT_COUNTS: dict[Domain, int] = {"beatport": 1229, "fmak": 4033}
DOMAIN_MASS: dict[Domain, Fraction] = {
    "beatport": Fraction(1230, 2206),
    "fmak": Fraction(976, 2206),
}


def _domain(domain: Domain) -> None:
    if domain not in FIT_COUNTS:
        raise ValueError("Expected beatport or fmak domain")


def _identity(identity: str) -> None:
    if not isinstance(identity, str) or not identity:
        raise ValueError("Expected a nonempty canonical string ID")


def _digest(parts: Sequence[str | int]) -> bytes:
    return hashlib.sha256(
        json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    ).digest()


def membership(domain: Domain, identities: Sequence[str]) -> tuple[str, ...]:
    """Select 256 fitting IDs by the fixed hash rank, independent of input order."""
    _domain(domain)
    for identity in identities:
        _identity(identity)
    if len(identities) != FIT_COUNTS[domain] or len(set(identities)) != len(identities):
        raise ValueError("Expected the complete unique fitting-ID pool")
    return tuple(
        sorted(
            identities,
            key=lambda identity: (
                _digest([SALT, "membership", domain, identity]),
                json.dumps(identity, ensure_ascii=True).encode("ascii"),
            ),
        )[:PER_DOMAIN]
    )


def crop_bounds(domain: Domain, identity: str, samples: int) -> tuple[int, int]:
    """Return start/length using the protocol's bounded rejection mapping."""
    _domain(domain)
    _identity(identity)
    if type(samples) is not int or samples < 1200:
        raise ValueError("Expected an integer recording length of at least 1200 samples")
    length = min(samples, 288000)
    choices = samples - length + 1
    limit = (2**256 // choices) * choices
    for counter in range(1024):
        value = int.from_bytes(_digest([SALT, "crop", domain, identity, counter]), "big")
        if value < limit:
            return value % choices, length
    raise ValueError("Crop rejection limit exhausted; no replacement is permitted")


def recording_weight(domain: Domain) -> Fraction:
    """Exact rational mass; convert to FP64 only inside the numerical estimator."""
    _domain(domain)
    return DOMAIN_MASS[domain] / PER_DOMAIN


@dataclass(frozen=True)
class Clip:
    domain: Domain
    identity: str
    samples: int
    start: int
    length: int
    weight: Fraction


def schedule(beatport: Mapping[str, int], fmak: Mapping[str, int]) -> tuple[Clip, ...]:
    """Build the fixed BP-then-FMA schedule from fitting-only metadata supplied by caller.

    This does not establish source roles or cache identity: those require the sealed
    input manifest. A missing/invalid selected recording aborts, never backfills.
    """
    result: list[Clip] = []
    for domain, rows in (("beatport", beatport), ("fmak", fmak)):
        for identity in membership(domain, list(rows)):
            samples = rows[identity]
            start, length = crop_bounds(domain, identity, samples)
            result.append(Clip(domain, identity, samples, start, length, recording_weight(domain)))
    return tuple(result)


@dataclass(frozen=True)
class PopulationMoments:
    mean: NDArray[np.float64]
    variance: NDArray[np.float64]

    def __post_init__(self) -> None:
        for value in (self.mean, self.variance):
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.float64
                or value.ndim != 1
                or value.size == 0
                or not np.isfinite(value).all()
            ):
                raise ValueError("Expected nonempty finite FP64 channel vectors")
        if self.mean.shape != self.variance.shape or (self.variance < 0).any():
            raise ValueError("Moment shapes must agree and variance must be nonnegative")
        for name in ("mean", "variance"):
            value = getattr(self, name).copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)


def recording_moments(values: FloatArray) -> PopulationMoments:
    """Two-pass population moments over actual N1 BN input support, 64 channels at a time."""
    if (
        not isinstance(values, np.ndarray)
        or values.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or values.ndim not in (3, 4)
        or values.shape[0] != 1
        or any(size == 0 for size in values.shape)
    ):
        raise ValueError("Expected nonempty FP32/FP64 N1,C,T or N1,C,F,T input")
    channels = values.shape[1]
    mean = np.empty(channels, dtype=np.float64)
    variance = np.empty(channels, dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for begin in range(0, channels, 64):
                end = min(begin + 64, channels)
                block = np.asarray(values[0, begin:end], dtype=np.float64).reshape(end - begin, -1)
                if not np.isfinite(block).all():
                    raise ValueError("Nonfinite BN input")
                center = block.mean(axis=1, dtype=np.float64)
                delta = block - center[:, None]
                mean[begin:end] = center
                variance[begin:end] = np.mean(delta * delta, axis=1, dtype=np.float64)
    except FloatingPointError as error:
        raise ValueError("FP64 moment computation overflowed") from error
    return PopulationMoments(mean, variance)


class WeightedPopulation:
    """Streaming weighted Welford merge; each recording has its declared total mass."""

    def __init__(self) -> None:
        self._mean: NDArray[np.float64] | None = None
        self._central: NDArray[np.float64] | None = None
        self._mass = 0.0
        self._exact_mass = Fraction(0)

    @property
    def exact_mass(self) -> Fraction:
        return self._exact_mass

    def add(self, moments: PopulationMoments, weight: Fraction) -> None:
        if not isinstance(weight, Fraction) or weight <= 0:
            raise ValueError("Expected a positive exact rational recording weight")
        try:
            mass = float(weight)
        except OverflowError as error:
            raise ValueError("Weight must be representable in FP64") from error
        total = self._mass + mass
        if not np.isfinite(total) or mass <= 0:
            raise ValueError("Weight must be finite and positive in FP64")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if self._mean is None:
                    mean = moments.mean.copy()
                    central = mass * moments.variance
                else:
                    if moments.mean.shape != self._mean.shape or self._central is None:
                        raise ValueError("All recordings must have the same channel count")
                    delta = moments.mean - self._mean
                    mean = self._mean + delta * (mass / total)
                    central = (
                        self._central
                        + mass * moments.variance
                        + delta * delta * (self._mass * mass / total)
                    )
                if not np.isfinite(mean).all() or not np.isfinite(central).all():
                    raise ValueError("Nonfinite merged moments")
        except FloatingPointError as error:
            raise ValueError("FP64 merge overflowed") from error
        self._mean = mean
        self._central = central
        self._mass = total
        self._exact_mass += weight

    def finish(self) -> PopulationMoments:
        if self._mean is None or self._central is None:
            raise ValueError("Cannot finish an empty population")
        return PopulationMoments(self._mean, self._central / self._mass)
