import json
import subprocess
import sys

import numpy as np
import pytest

import tools.evaluate_octave as evaluation
from tools.evaluate_octave import calibrated, evaluation_decode, main, uncertainty


def test_temperature_preserves_class_decisions_and_probability_mass():
    p = np.random.default_rng(13).random((20, 24))
    p /= p.sum(1, keepdims=True)
    for temperature in (0.25, 1, 4):
        result = calibrated(p, temperature)
        np.testing.assert_allclose(result.sum(1), 1)
        np.testing.assert_array_equal(result.argmax(1), p.argmax(1))
    perfect = np.eye(24)[[0, 19]]
    result = uncertainty(perfect, np.array([0, 19]))
    assert result == {"nll": 0.0, "brier": 0.0, "ece_15_equal_width": 0.0}


def test_final_evaluation_rejects_ineligible_candidate_before_opening_corpus(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"eligible_for_final_evaluation": False}))
    args = ["evaluate", "--mode", "fmak-final", "--selection", str(selection)]
    for name in ("fmak", "beatport", "partition", "groups", "output"):
        args += ["--" + name, str(tmp_path / name)]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match="development gate"):
        main()


def test_broken_seek_falls_back_to_actual_decoded_duration(monkeypatch):
    def broken_seek(row):
        raise subprocess.CalledProcessError(69, ["ffmpeg"])

    full = np.linspace(-1, 1, 64 * 22050, dtype=np.float32)
    monkeypatch.setattr(evaluation, "decode", broken_seek)
    monkeypatch.setattr(
        evaluation.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, stdout=full.tobytes(), stderr=b""
        ),
    )
    segments, diagnostic = evaluation_decode({"id": "broken-mp3", "path": "audio.mp3"})
    np.testing.assert_array_equal(segments[0], full[: 32 * 22050])
    np.testing.assert_array_equal(segments[1], full[32 * 22050 :])
    assert diagnostic["actual_duration_seconds"] == 64
    assert diagnostic["route"] == "full-decode-actual-duration"


def test_valid_fast_decode_is_preserved(monkeypatch):
    segments = np.ones((2, 705600), dtype=np.float32)
    monkeypatch.setattr(evaluation, "decode", lambda row: segments)
    result, diagnostic = evaluation_decode({"id": "valid"})
    assert result is segments and diagnostic is None
