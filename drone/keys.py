"""The shared key contract: class_id = tonic_index * 2 + mode (minor = 1)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from numbers import Integral

TONICS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
LABEL_SCHEMA = "tonic-times-two-plus-minor-v1"
WEIGHTS = {"exact": 1.0, "fifth": 0.5, "relative": 0.3, "parallel": 0.2, "other": 0.0}

Key = tuple[str, str]


def encode_key(tonic: str, mode: str) -> int:
    tonic = FLATS.get(tonic, tonic)
    if tonic not in TONICS or mode not in ("major", "minor"):
        raise ValueError(f"Invalid key: {tonic} {mode}")
    return TONICS.index(tonic) * 2 + int(mode == "minor")


def decode_key(class_id: int) -> Key:
    if not isinstance(class_id, Integral) or not 0 <= class_id < 24:
        raise ValueError(f"Invalid key class: {class_id}")
    return TONICS[class_id // 2], "minor" if class_id % 2 else "major"


def transpose_key(class_id: int, semitones: int) -> int:
    decode_key(class_id)
    return ((class_id // 2 + int(semitones)) % 12) * 2 + class_id % 2


def parse_key(text: str) -> Key | None:
    """Resolve ambiguous labels to the first option and normalize flats."""
    parts = text.split("|")[0].strip().split()
    if len(parts) != 2:
        return None
    try:
        return decode_key(encode_key(*parts))
    except ValueError:
        return None


def key_relation(pred: int, ref: int, *, symmetric: bool = True) -> str:
    """Use euphoria-bench's convention: both ascending and descending fifths count."""
    decode_key(pred)
    decode_key(ref)
    interval = (pred // 2 - ref // 2) % 12
    same_mode = pred % 2 == ref % 2
    if same_mode and interval == 0:
        return "exact"
    if same_mode and interval in ((5, 7) if symmetric else (7,)):
        return "fifth"
    if not same_mode and interval == (3 if ref % 2 else 9):
        return "relative"
    if not same_mode and interval == 0:
        return "parallel"
    return "other"


def score_breakdown(
    pred: Sequence[int], ref: Sequence[int], *, symmetric: bool = True
) -> Counter[str]:
    if len(pred) != len(ref) or not len(ref):
        raise ValueError("Scoring requires equally sized, nonempty predictions and references")
    return Counter(key_relation(p, r, symmetric=symmetric) for p, r in zip(pred, ref, strict=True))


def score_mirex(
    pred: Sequence[int], ref: Sequence[int], *, symmetric: bool = True
) -> tuple[float, float]:
    """Return (exact fraction, weighted MIREX fraction)."""
    counts = score_breakdown(pred, ref, symmetric=symmetric)
    return counts["exact"] / len(ref), sum(WEIGHTS[k] * v for k, v in counts.items()) / len(ref)
