import json
import sys

import numpy as np
import pytest

from tools.evaluate_octave import calibrated, main, uncertainty


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
