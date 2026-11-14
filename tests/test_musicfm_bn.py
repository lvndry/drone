"""Synthetic-only qualification of BN_PROTOCOL schedule and population estimator."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction

import numpy as np
import pytest

from experiments import musicfm_bn as bn


def pools() -> tuple[dict[str, int], dict[str, int]]:
    return (
        {f"bp-{index}": 14_400_000 + index for index in range(1229)},
        {f"fma-{index}": 1200 + index for index in range(4033)},
    )


def test_schedule_is_512_deterministic_and_has_exact_domain_masses() -> None:
    beatport, fma = pools()
    result = bn.schedule(beatport, fma)
    assert result == bn.schedule(dict(reversed(list(beatport.items()))), fma)
    assert len(result) == 512
    assert len({(clip.domain, clip.identity) for clip in result}) == 512
    for domain, clips in (("beatport", result[:256]), ("fmak", result[256:])):
        assert all(clip.domain == domain for clip in clips)
        assert sum((clip.weight for clip in clips), Fraction()) == bn.DOMAIN_MASS[domain]
        assert all(0 <= clip.start <= clip.samples - clip.length for clip in clips)
    assert sum((clip.weight for clip in result), Fraction()) == 1
    assert all(clip.length == 288000 for clip in result[:256])
    assert all(clip.start == 0 and clip.length == clip.samples for clip in result[256:])
    assert any(clip.start > 480 * 24000 for clip in result[:256])


def test_membership_fails_closed_on_pool_identity_errors() -> None:
    beatport, fma = pools()
    with pytest.raises(ValueError, match="complete unique"):
        bn.membership("beatport", list(beatport)[:-1])
    ids = list(beatport)
    ids[-1] = ids[0]
    with pytest.raises(ValueError, match="complete unique"):
        bn.membership("beatport", ids)
    selected = bn.membership("fmak", list(fma))[0]
    fma[selected] = 1199
    with pytest.raises(ValueError, match="1200"):
        bn.schedule(beatport, fma)


def test_schedule_version_golden_pins_encoding_rank_and_domain_crop() -> None:
    beatport, _ = pools()
    ranked = bn.membership("beatport", list(beatport))
    assert ranked[:5] == ("bp-1112", "bp-190", "bp-133", "bp-607", "bp-142")
    encoded = json.dumps(ranked, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    assert hashlib.sha256(encoded).hexdigest() == (
        "b6d9437834727d203a74856fb763fad12ca1161f89d8d90062add76aa444c212"
    )
    assert bn.crop_bounds("beatport", "canonical-é", 14400000) == (7633240, 288000)
    assert bn.crop_bounds("fmak", "canonical-é", 14400000) == (6564983, 288000)


def test_membership_digest_ties_use_canonical_json_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    beatport, _ = pools()
    ids = list(beatport)
    ids[0:3] = ["é", '"quoted"', "\\backslash"]
    monkeypatch.setattr(bn, "_digest", lambda _: b"\x00" * 32)
    expected = sorted(ids, key=lambda value: json.dumps(value, ensure_ascii=True).encode("ascii"))
    assert bn.membership("beatport", ids) == tuple(expected[:256])


@pytest.mark.parametrize("samples", [1200, 1201, 288000])
def test_short_and_exact_crops_are_not_padded(samples: int) -> None:
    assert bn.crop_bounds("beatport", "canonical-é", samples) == (0, samples)


@pytest.mark.parametrize("samples", [True, 1199, -1, 1200.0])
def test_invalid_sample_counts_are_rejected(samples: int) -> None:
    with pytest.raises(ValueError, match="1200"):
        bn.crop_bounds("fmak", "id", samples)


def test_hash_mapping_handles_rejection_and_has_hard_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def digest(parts: list[str | int]) -> bytes:
        counter = int(parts[-1])
        calls.append(counter)
        return ((2**256 - 1) if counter == 0 else 8).to_bytes(32, "big")

    monkeypatch.setattr(bn, "_digest", digest)
    assert bn.crop_bounds("fmak", "id", 288002) == (2, 288000)
    assert calls == [0, 1]
    monkeypatch.setattr(bn, "_digest", lambda _: b"\xff" * 32)
    with pytest.raises(ValueError, match="rejection limit"):
        bn.crop_bounds("fmak", "id", 288002)


def fixtures() -> tuple[bn.PopulationMoments, bn.PopulationMoments]:
    return (
        bn.recording_moments(np.array([[[0, 2], [1, 1]]], dtype=np.float32)),
        bn.recording_moments(np.array([[[6], [5]]], dtype=np.float32)),
    )


def test_between_recording_variance_and_equal_recording_mass() -> None:
    first, second = fixtures()
    result = bn.WeightedPopulation()
    result.add(first, Fraction(1, 4))
    result.add(second, Fraction(3, 4))
    np.testing.assert_allclose(result.finish().mean, [4.75, 4], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.finish().variance, [4.9375, 3], rtol=1e-12, atol=1e-12)
    assert result.exact_mass == 1
    assert not result.finish().mean.flags.writeable


def test_actual_domain_weights_across_512_recordings() -> None:
    beatport_moments, fmak_moments = fixtures()
    result = bn.WeightedPopulation()
    for domain, moments in (("beatport", beatport_moments), ("fmak", fmak_moments)):
        for _ in range(256):
            result.add(moments, bn.recording_weight(domain))
    beatport_share = float(Fraction(1230, 2206))
    fmak_share = float(Fraction(976, 2206))
    np.testing.assert_allclose(
        result.finish().mean,
        [beatport_share + 6 * fmak_share, beatport_share + 5 * fmak_share],
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        result.finish().variance,
        [beatport_share + 25 * beatport_share * fmak_share, 16 * beatport_share * fmak_share],
        atol=1e-12,
        rtol=1e-12,
    )
    assert result.exact_mass == 1


def test_merge_order_is_stable_and_matches_explicit_weighted_positions() -> None:
    rng = np.random.default_rng(29)
    arrays = [rng.normal(index * 0.7, 2, size=(1, 4, index + 1)) for index in range(40)]
    weights = [Fraction(index + 1, 820) for index in range(40)]
    forward = bn.WeightedPopulation()
    reverse = bn.WeightedPopulation()
    for indices, accumulator in ((range(40), forward), (reversed(range(40)), reverse)):
        for index in indices:
            accumulator.add(bn.recording_moments(arrays[index]), weights[index])
    # Independent reference: each position gets its recording's mass / support.
    values = np.concatenate([array[0] for array in arrays], axis=1)
    position_weights = np.concatenate(
        [
            np.full(array.shape[-1], float(weight) / array.shape[-1])
            for array, weight in zip(arrays, weights, strict=True)
        ]
    )
    mean = np.sum(values * position_weights, axis=1)
    variance = np.sum((values - mean[:, None]) ** 2 * position_weights, axis=1)
    for accumulator in (forward, reverse):
        np.testing.assert_allclose(accumulator.finish().mean, mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(accumulator.finish().variance, variance, rtol=1e-12, atol=1e-12)


def test_bn2d_axes_channel_chunks_and_noncontiguous_inputs() -> None:
    values = np.arange(1 * 129 * 3 * 5, dtype=np.float32).reshape(1, 129, 3, 5)[..., ::-1]
    result = bn.recording_moments(values)
    np.testing.assert_array_equal(result.mean, values.astype(np.float64).mean(axis=(0, 2, 3)))
    np.testing.assert_allclose(result.variance, values.astype(np.float64).var(axis=(0, 2, 3)))


def test_identical_constant_clips_have_zero_population_variance() -> None:
    result = bn.WeightedPopulation()
    for length in (1, 300, 1000):
        result.add(bn.recording_moments(np.full((1, 3, length), 7.0)), Fraction(1, 3))
    np.testing.assert_array_equal(result.finish().mean, [7, 7, 7])
    np.testing.assert_array_equal(result.finish().variance, [0, 0, 0])


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((2, 3, 5)),
        np.zeros((1, 0, 5)),
        np.zeros((1, 3, 0)),
        np.zeros((3, 5)),
        np.zeros((1, 2, 3), dtype=np.int64),
        np.zeros((1, 2, 3), dtype=np.float16),
        np.array([[[np.nan]]]),
        np.array([[[np.inf]]]),
        np.array([[[1e308, -1e308]]]),
    ],
)
def test_invalid_inputs_never_produce_moments(values: bn.FloatArray) -> None:
    with pytest.raises(ValueError):
        bn.recording_moments(values)


def test_moment_and_merge_validation_preserves_previous_state() -> None:
    for mean, variance in (([1], [-1]), ([np.inf], [0]), ([1, 2], [0])):
        with pytest.raises(ValueError):
            bn.PopulationMoments(
                np.array(mean, dtype=np.float64), np.array(variance, dtype=np.float64)
            )
    accumulator = bn.WeightedPopulation()
    with pytest.raises(ValueError, match="empty"):
        accumulator.finish()
    first, _ = fixtures()
    accumulator.add(first, Fraction(1, 2))
    before = accumulator.finish()
    for weight in (Fraction(0), Fraction(-1), Fraction(10**400), Fraction(1, 10**400)):
        with pytest.raises(ValueError):
            accumulator.add(first, weight)
    with pytest.raises(ValueError, match="channel count"):
        accumulator.add(bn.recording_moments(np.ones((1, 3, 4))), Fraction(1, 2))
    np.testing.assert_array_equal(accumulator.finish().mean, before.mean)
    np.testing.assert_array_equal(accumulator.finish().variance, before.variance)
    assert accumulator.exact_mass == Fraction(1, 2)


def test_overflowing_merge_is_atomic() -> None:
    accumulator = bn.WeightedPopulation()
    first = bn.PopulationMoments(np.array([1e308]), np.array([0.0]))
    second = bn.PopulationMoments(np.array([-1e308]), np.array([0.0]))
    accumulator.add(first, Fraction(1, 2))
    with pytest.raises(ValueError, match="overflowed"):
        accumulator.add(second, Fraction(1, 2))
    np.testing.assert_array_equal(accumulator.finish().mean, first.mean)
    np.testing.assert_array_equal(accumulator.finish().variance, first.variance)
    assert accumulator.exact_mass == Fraction(1, 2)
