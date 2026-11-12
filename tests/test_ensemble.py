"""Synthetic ensemble math and immutable-identity rejection tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from drone.features import CROP_FRAMES, FEATURE_SPEC
from drone.keys import LABEL_SCHEMA
from experiments.ensemble import (
    SEEDS,
    combine,
    extract,
    probabilities,
    teacher_report,
    terminal_sam,
    validate_cohort,
    validate_payload,
    verify_archive,
)


def metrics(value: float) -> dict[str, dict[str, float]]:
    return {d: dict(mirex=value, exact=value, mirex_asymmetric=value) for d in ("beatport", "fmak")}


def test_probability_mixture_not_logit_average_and_ties() -> None:
    logits = [np.full((1, 24), -100, dtype=np.float32) for _ in SEEDS]
    logits[0][0, :2] = [10, 0]
    logits[1][0, :2] = [0, 2]
    logits[2][0, :2] = [0, 2]
    result = combine(logits)
    assert result.dtype == np.float64
    assert result.argmax(1).item() == 1
    assert probabilities(np.mean(logits, axis=0)).argmax(1).item() == 0
    np.testing.assert_allclose(result.sum(1), 1, atol=1e-15)
    assert combine([np.zeros((1, 24))] * 3).argmax(1).item() == 0
    np.testing.assert_allclose(
        probabilities(logits[0]), probabilities(logits[0] + 1000), atol=1e-15
    )


def test_member_order_shape_and_nonfinite_reject() -> None:
    good = [np.zeros((2, 24))] * 3
    with pytest.raises(ValueError):
        combine(good, (37, 13, 73))
    for wrong in (good[:2], [*good[:2], np.zeros((3, 24))], [*good[:2], np.full((2, 24), np.nan)]):
        with pytest.raises(ValueError):
            combine(wrong)
    with pytest.raises(ValueError):
        probabilities(np.zeros((2, 23)))


def test_teacher_gate_original_hurdle_and_one_cohort() -> None:
    members = [metrics(0.75), metrics(0.77), metrics(0.74)]
    bp = [metrics(0.72)] * 3
    candidate = metrics(0.765)
    result = teacher_report(members, candidate, bp)
    assert result["teacher_gate"]["qualifies_as_teacher"]
    assert result["bp_reference_hurdle_pass"]
    assert not result["uniformly_nonregressing_vs_all_members"]
    assert result["ensemble_candidates"] == result["ensemble_replicates"] == 1
    assert set(result["comparisons"]) == {"13", "37", "73"}
    assert "standard error" in result["uncertainty"]
    assert not result["distillation_authorized"]
    candidate["beatport"]["exact"] = 0.749
    assert not teacher_report(members, candidate, bp)["teacher_gate"]["qualifies_as_teacher"]
    candidate = metrics(0.755)
    result = teacher_report(members, candidate, bp)
    assert result["bp_reference_hurdle_pass"]
    assert not result["teacher_gate"]["qualifies_as_teacher"]


def test_checkpoint_schema_metadata_corruption() -> None:
    run = dict(arm="joint", seed=13, source_sha256={"model": "source"}, best_epoch=20)
    payload: dict[str, Any] = dict(
        format_version=2,
        feature_spec=FEATURE_SPEC,
        label_schema=LABEL_SCHEMA,
        crop=CROP_FRAMES,
        model_config={"width": 1.0},
        metadata=dict(run, epoch=20, experiment="k10-multidomain-development-v1"),
    )
    validate_payload(payload, run)
    for key, value in (
        ("format_version", 1),
        ("label_schema", "wrong"),
        ("feature_spec", "wrong"),
        ("model_config", {"width": 2.0}),
        ("crop", 32),
    ):
        bad = copy.deepcopy(payload)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_payload(bad, run)
    for key, value in (("seed", 37), ("source_sha256", {}), ("epoch", 21)):
        bad = copy.deepcopy(payload)
        bad["metadata"][key] = value
        with pytest.raises(ValueError):
            validate_payload(bad, run)


def test_archive_source_hash_and_boundary(tmp_path: Path) -> None:
    from drone.features import audio_digest

    file = tmp_path / "source.py"
    file.write_text("original")
    expected = {"source.py": audio_digest(file)}
    verify_archive(tmp_path, expected, lambda: None)
    file.write_text("changed")
    with pytest.raises(ValueError, match="Archived source"):
        verify_archive(tmp_path, expected, lambda: None)
    with pytest.raises(ValueError):
        verify_archive(tmp_path, {"../outside": "anything"}, lambda: None)


def test_cohort_fixed_order_labels_and_source() -> None:
    rows = [
        dict(id="beatport:fit", role="train", domain="beatport"),
        dict(id="fmak:fit", role="train", domain="fmak"),
        dict(id="beatport:val", role="validation", domain="beatport"),
    ]
    base = dict(
        config={"recipe": "fixed"},
        feature_manifest_sha256="manifest",
        fit_ids=["beatport:fit", "fmak:fit"],
        validation_ids=["beatport:val"],
        source_sha256={"model": "source"},
    )
    runs = [dict(base, seed=seed) for seed in SEEDS]
    bp = [dict(r, fit_ids=["beatport:fit"]) for r in runs]
    validate_cohort(runs, bp, rows, "manifest", {"model": "source"})
    for key, value in (
        ("source_sha256", {}),
        ("feature_manifest_sha256", "changed"),
        ("fit_ids", ["fmak:fit"]),
    ):
        bad = copy.deepcopy(runs)
        bad[1][key] = value
        with pytest.raises(ValueError):
            validate_cohort(bad, bp, rows, "manifest", {"model": "source"})
    with pytest.raises(ValueError):
        validate_cohort(list(reversed(runs)), bp, rows, "manifest", {"model": "source"})


class LengthModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == 192
        return x.mean((1, 2))[:, None].repeat(1, 24)


def test_stream_true_lengths_order_no_padding(tmp_path: Path) -> None:
    rows = []
    for index, length in enumerate((8, 11, 8, 9, 13)):
        value = np.full((208, length), index + 1, dtype=np.float16)
        path = tmp_path / f"{index}.npy"
        np.save(path, value)
        rows.append(dict(cache=path.name, frames=length))
    output = extract(LengthModel(), rows, tmp_path, "cpu", lambda: None)
    np.testing.assert_array_equal(output, np.arange(1, 6)[:, None] * np.ones((5, 24)))


def test_sam_terminal_receipt_required(tmp_path: Path) -> None:
    result = tmp_path / "sam-result.json"
    exit_path = tmp_path / "candidate.exit"
    result.write_text(
        json.dumps(
            dict(
                artifact_version="k10-vanilla-sam-adamw-v1",
                seed=13,
                status="complete",
                kind="candidate",
            )
        )
    )
    with pytest.raises(FileNotFoundError):
        terminal_sam(result, exit_path)
    exit_path.write_text("1\n")
    with pytest.raises(ValueError):
        terminal_sam(result, exit_path)
    exit_path.write_text("0\n")
    assert terminal_sam(result, exit_path)["status"] == "complete"
    result.write_text(
        json.dumps(dict(artifact_version="k10-vanilla-sam-adamw-v1", seed=13, status="running"))
    )
    with pytest.raises(ValueError):
        terminal_sam(result, exit_path)
