from __future__ import annotations

import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from experiments.reference.features import (
    FFT_SIZE,
    N_SEM,
    SR,
    build_cqt,
    extract,
    normalize,
    resample_time,
    shift_features,
)
from experiments.reference.train import augment_batch


def write_tone(path: Path, frequency: float, seconds: float = 0.25) -> None:
    times = np.arange(int(SR * seconds)) / SR
    samples = (np.sin(2 * np.pi * frequency * times) * 16000).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SR)
        stream.writeframes(samples.tobytes())


@pytest.mark.parametrize("band", [0, 3, 12, 24, 31, 32])
def test_tones_land_in_semitone_bands(band: int) -> None:
    times = np.arange(SR) / SR
    features = build_cqt(np.sin(2 * np.pi * 55 * 2 ** (band / 12) * times))
    assert features.shape[1] == N_SEM
    assert features.mean(axis=0).argmax(axis=0) == band


def test_time_resampling_preserves_channels_and_endpoints() -> None:
    features = np.array([[0, 100, 1000], [10, 110, 1010]], dtype=np.float32)
    actual = resample_time(features, 5)
    np.testing.assert_allclose(actual[0], features[0])
    np.testing.assert_allclose(actual[-1], features[-1])
    np.testing.assert_allclose(actual[:, 1] - actual[:, 0], 100)
    np.testing.assert_allclose(actual[:, 2] - actual[:, 0], 1000)
    np.testing.assert_allclose(resample_time(features[:1], 3), np.repeat(features[:1], 3, axis=0))


def test_zero_shift_identity_and_no_frequency_wrap() -> None:
    features = np.zeros((300, N_SEM), dtype=np.float32)
    features[:, -1] = 10
    features = normalize(features)
    np.testing.assert_array_equal(shift_features(features, 0), features)
    np.testing.assert_allclose(shift_features(features, 1), 0)
    features = np.zeros((300, N_SEM), dtype=np.float32)
    features[:, 0] = 10
    np.testing.assert_allclose(shift_features(normalize(features), -1), 0)


def test_augmentation_applies_same_shift_to_audio_and_label() -> None:
    features = np.zeros((24, 300, N_SEM), dtype=np.float32)
    features[:, :, 12] = 1
    shifts = np.tile(np.arange(-6, 6), 2)
    shifted, labels = augment_batch(features, list(range(24)), shifts)
    for class_id, shift in enumerate(shifts):
        assert shifted[class_id].mean(axis=0).argmax(axis=0) == 12 + shift
        assert labels[class_id] // 2 == (class_id // 2 + shift) % 12
        assert labels[class_id] % 2 == class_id % 2
    np.testing.assert_array_equal(features[:, :, 12], 1)


def test_short_and_silent_audio_are_finite() -> None:
    features = build_cqt(np.zeros(10, dtype=np.float32))
    assert features.shape == (1, N_SEM)
    np.testing.assert_array_equal(normalize(features), 0)
    with pytest.raises(ValueError, match="nonempty"):
        build_cqt(np.array([]))
    assert FFT_SIZE > 10


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required for audio integration")
def test_cache_distinguishes_files_and_changed_content(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    write_tone(first, 110)
    write_tone(second, 220)
    low_tone = extract("track", first, tmp_path)
    high_tone = extract("track", second, tmp_path)
    assert low_tone.mean(axis=0).argmax(axis=0) == 12
    assert high_tone.mean(axis=0).argmax(axis=0) == 24
    np.testing.assert_array_equal(extract("different-id", first, tmp_path), low_tone)
    write_tone(first, 220)
    np.testing.assert_array_equal(extract("track", first, tmp_path), high_tone)
    assert len(list((tmp_path / "cqt").rglob("*.npy"))) == 2


def test_blockwise_bands_are_bitwise_identical_to_full_stft_reference() -> None:
    from experiments.reference.features import SEMIBASE, stft_magnitude

    rng = np.random.default_rng(13)
    # Spans several FFT blocks, including a final partial block.
    waveform = rng.normal(size=360_000).astype(np.float32)
    magnitude = stft_magnitude(waveform)
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1 / SR)
    indices = np.rint(12 * np.log2(np.maximum(freqs, 1.0) / SEMIBASE)).astype(int)
    bands = np.zeros((len(magnitude), N_SEM), dtype=np.float32)
    for band in range(N_SEM):
        bands[:, band] = magnitude[:, indices == band].sum(axis=1)
    expected = np.log1p(10 * bands / max(float(np.percentile(bands, 99)), 1e-9))
    np.testing.assert_array_equal(build_cqt(waveform), expected)


def test_band_extraction_does_not_retain_a_full_track_fft_matrix() -> None:
    import tracemalloc

    waveform = np.zeros(FFT_SIZE + 12000 * 512, dtype=np.float32)
    full_magnitude_bytes = 12001 * (FFT_SIZE // 2 + 1) * 4
    tracemalloc.start()
    try:
        bands = build_cqt(waveform)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert bands.shape == (12001, N_SEM)
    assert peak < full_magnitude_bytes / 2
