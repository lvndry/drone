from __future__ import annotations

import numpy as np
import pytest

from drone.keys import (
    TONICS,
    decode_key,
    encode_key,
    key_relation,
    parse_key,
    score_breakdown,
    score_mirex,
    transpose_key,
)


def test_scoring_accepts_numpy_integer_arrays() -> None:
    pred = np.array([0, 1], dtype=np.int64)
    ref = np.array([0, 1], dtype=np.int64)
    assert score_mirex(pred, ref) == (1.0, 1.0)
    assert score_breakdown(pred, ref)["exact"] == 2


@pytest.mark.parametrize("class_id", range(24))
def test_all_keys_round_trip_and_transpose(class_id: int) -> None:
    assert encode_key(*decode_key(class_id)) == class_id
    assert transpose_key(class_id, 0) == class_id
    assert transpose_key(class_id, 12) == class_id
    for shift in range(-6, 6):
        shifted = transpose_key(class_id, shift)
        assert shifted % 2 == class_id % 2
        assert transpose_key(shifted, -shift) == class_id


@pytest.mark.parametrize("tonic", TONICS)
@pytest.mark.parametrize("mode", ["major", "minor"])
def test_scoring_relationships_for_every_key(tonic: str, mode: str) -> None:
    ref = encode_key(tonic, mode)
    opposite = ref ^ 1
    cases: list[tuple[int, str, float]] = [
        (ref, "exact", 1.0),
        (transpose_key(ref, 7), "fifth", 0.5),
        (transpose_key(ref, -7), "fifth", 0.5),
        (transpose_key(opposite, 9 if mode == "major" else 3), "relative", 0.3),
        (opposite, "parallel", 0.2),
        (transpose_key(ref, 1), "other", 0.0),
    ]
    for pred, relation, weight in cases:
        assert key_relation(pred, ref) == relation
        assert score_mirex([pred], [ref]) == (float(pred == ref), weight)


def test_invalid_inputs_and_flat_labels() -> None:
    assert parse_key("Bb minor | C major") == ("A#", "minor")
    assert parse_key("not a key") is None
    for bad in [-1, 24, 1.5]:
        with pytest.raises(ValueError):
            decode_key(bad)  # ty: ignore  # 1.5 proves the Integral guard rejects floats

    for pred, ref in [([], []), ([0], []), ([24], [0])]:
        with pytest.raises(ValueError):
            score_mirex(pred, ref)


def test_symmetric_and_asymmetric_fifth_conventions_are_explicit() -> None:
    c_major = encode_key("C", "major")
    f_major = encode_key("F", "major")
    assert score_mirex([f_major], [c_major]) == (0.0, 0.5)
    assert score_mirex([f_major], [c_major], symmetric=False) == (0.0, 0.0)
