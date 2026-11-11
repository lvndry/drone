"""Synthetic-only geometry, leakage, grouping and pinned API compatibility."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.key_ranker import Budget, configuration, load_artifact, save_artifact
from experiments.tonal_features import (
    FEATURE_NAMES,
    candidate_features,
    grouped_targets,
    quarter_bounds,
    relative_key,
)


def test_order_modes_and_relative() -> None:
    core = np.zeros((192, 11), dtype=np.float32)
    core[0] = np.log1p(10)
    values = candidate_features(core)
    assert values.shape == (24, 24)
    assert len(FEATURE_NAMES) == 24
    np.testing.assert_array_equal(values[:, 20], np.tile([0, 1], 12))
    assert values[0, 0] > values[2, 0]
    assert values[0, 12] > values[2, 12]
    assert relative_key(0) == 19
    assert relative_key(19) == 0
    for candidate in range(24):
        assert relative_key(relative_key(candidate)) == candidate
        assert relative_key(candidate) % 2 != candidate % 2


def test_silence_and_finite_contract() -> None:
    values = candidate_features(np.zeros((192, 8), dtype=np.float32))
    np.testing.assert_array_equal(values[:, 20], np.tile([0, 1], 12))
    np.testing.assert_array_equal(np.delete(values, 20, axis=1), 0)
    for bad in (np.nan, np.inf, -1, 1000):
        core = np.zeros((192, 8), dtype=np.float32)
        core[0, 0] = bad
        with pytest.raises(ValueError):
            candidate_features(core)
    for shape in ((208, 8), (192, 7)):
        with pytest.raises(ValueError):
            candidate_features(np.zeros(shape))


def test_quarters_and_chunk_boundaries() -> None:
    assert quarter_bounds(11) == [(0, 3), (3, 6), (6, 9), (9, 11)]
    core = np.zeros((192, 11), dtype=np.float32)
    for quarter, (a, b) in enumerate(quarter_bounds(11)):
        core[2 * quarter, a:b] = np.log1p(4)
    one = candidate_features(core, chunk_frames=99)
    many = candidate_features(core, chunk_frames=2)
    np.testing.assert_allclose(one, many, rtol=1e-6, atol=1e-7)
    # C major: C is triad, C# off scale, D in scale but not triad, D# off scale.
    np.testing.assert_allclose(one[0, 15:19], [1 / 3, -1 / 5, 0, -1 / 5], atol=1e-7)
    assert one[6, 19] > one[0, 19]  # last eighth is D#


def test_audio_only_and_group_integrity() -> None:
    assert set(inspect.signature(candidate_features).parameters) == {
        "core",
        "check",
        "chunk_frames",
    }
    targets, groups = grouped_targets(np.array([0, 19, 23]))
    np.testing.assert_array_equal(groups.reshape(3, 24), np.arange(3)[:, None] + np.zeros((3, 24)))
    np.testing.assert_array_equal(targets.reshape(3, 24).sum(1), 1)
    np.testing.assert_array_equal(targets.reshape(3, 24).argmax(1), [0, 19, 23])
    for invalid in (np.array([]), np.array([24]), np.array([1.0]), np.array([[0]])):
        with pytest.raises(ValueError):
            grouped_targets(invalid)


def test_feature_space_candidate_rotation() -> None:
    rng = np.random.default_rng(1)
    core = np.log1p(rng.uniform(0, 1, (8, 24, 13))).reshape(192, 13).astype(np.float32)
    initial = candidate_features(core)
    shifted = candidate_features(np.roll(core.reshape(8, 24, 13), 6, axis=1).reshape(192, 13))
    np.testing.assert_allclose(shifted, np.roll(initial, 6, axis=0), atol=2e-7, rtol=1e-5)


def test_pinned_synthetic_api_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The full suite has a historical multi-GiB Torch peak; this API unit test
    # must not inherit that unrelated process high-water mark. Guard tested below.
    monkeypatch.setattr(
        "experiments.key_ranker.resource.getrusage", lambda _: SimpleNamespace(ru_maxrss=0)
    )
    catboost = pytest.importorskip("catboost")
    rng = np.random.default_rng(13)
    # Only artificial arrays: no audio, labels or synthetic accuracy result.
    x = rng.normal(size=(72, 24)).astype(np.float32)
    targets, groups = grouped_targets(np.array([0, 9, 22]))
    pool = catboost.Pool(x, label=targets, group_id=groups, feature_names=list(FEATURE_NAMES))
    model = catboost.CatBoostRanker(**configuration(13))
    budget = Budget(60, tmp_path)
    model.fit(pool, verbose=False, callbacks=[budget])
    assert model.tree_count_ == 128
    save_artifact(model, tmp_path, {"synthetic_only": True}, 13)
    restored = load_artifact(tmp_path)
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))
    assert restored.get_all_params()["loss_function"] == "QuerySoftMax:beta=1"
    info_path = tmp_path / "artifact.json"
    original = info_path.read_text()
    for name, wrong in (
        ("feature_names", list(reversed(FEATURE_NAMES))),
        ("catboost_version", "0"),
        ("label_schema", "wrong"),
        ("model_sha256", "0"),
    ):
        info = json.loads(original)
        info[name] = wrong
        info_path.write_text(json.dumps(info))
        with pytest.raises(ValueError):
            load_artifact(tmp_path)
    info_path.write_text(original)
    blob = tmp_path / "model.cbm"
    blob.write_bytes(blob.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        load_artifact(tmp_path)


def test_budget_callback_returns_false_instead_of_wrapped_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "experiments.key_ranker.resource.getrusage", lambda _: SimpleNamespace(ru_maxrss=0)
    )
    budget = Budget(-1, tmp_path)
    assert not budget.after_iteration(object())
    assert isinstance(budget.failure, TimeoutError)


def test_final_method_gate_cannot_be_replaced_by_original_hurdle(tmp_path: Path) -> None:
    from experiments.key_ranker import VERSION, summarize

    paths = []
    for seed in (13, 37, 73):
        result = dict(
            seed=seed,
            artifact_version=VERSION,
            status="complete",
            control_reproduced=True,
            identity={},
            feature_array_sha256="same",
            method_gate={"qualifies_for_confirmation": True},
            metrics={
                d: dict(mirex=0.755, mirex_asymmetric=0.75, exact=0.7) for d in ("beatport", "fmak")
            },
            control_metrics={
                d: dict(mirex=0.75, mirex_asymmetric=0.75, exact=0.7) for d in ("beatport", "fmak")
            },
            bp_metrics={
                d: dict(mirex=0.72, mirex_asymmetric=0.72, exact=0.65) for d in ("beatport", "fmak")
            },
        )
        path = tmp_path / f"seed{seed}.json"
        path.write_text(json.dumps(result))
        paths.append(path)
    result = summarize(paths)
    assert result["original_hurdle_pass"]
    assert result["raw_joint_safeguard"]
    assert not result["method_gate"]["qualifies_for_confirmation"]
    assert not result["qualifies_for_external_freeze"]


def test_memory_budget_rejects_real_peak_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    peak = 3 * 1024**3 // (1 if sys.platform == "darwin" else 1024)
    monkeypatch.setattr(
        "experiments.key_ranker.resource.getrusage", lambda _: SimpleNamespace(ru_maxrss=peak)
    )
    budget = Budget(60, tmp_path)
    assert not budget.after_iteration(object())
    assert isinstance(budget.failure, MemoryError)
