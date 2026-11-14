from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from experiments.reference.checkpoints import load_checkpoint, save_checkpoint
from experiments.reference.model import KeyCNN, params_count
from experiments.small import RECIPES, Recipe, confirmation_recipes, load_development, run_recipe


def test_research_and_production_checkpoints_reject_each_other(tmp_path: Path) -> None:
    from drone.checkpoints import load_checkpoint as load_production
    from drone.checkpoints import save_checkpoint as save_production
    from drone.model import KeyCNN as ProductionModel

    research = tmp_path / "research.pt"
    production = tmp_path / "production.pt"
    save_checkpoint(research, KeyCNN())
    save_production(production, ProductionModel())
    with pytest.raises(ValueError, match="Incompatible"):
        load_production(research)
    with pytest.raises(ValueError, match="Incompatible"):
        load_checkpoint(production)


def test_frequency_position_is_preserved_in_small_model_and_checkpoint(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    model = KeyCNN(pooling="frequency").eval()
    assert params_count(model) * 4 < 2 * 1024 * 1024
    features = torch.randn(2, 33, 300)
    with torch.no_grad():
        assert model.net(features.unsqueeze(1)).shape == (2, 128, 6, 1)
        expected = model(features)
    checkpoint = tmp_path / "frequency.pt"
    save_checkpoint(checkpoint, model)
    restored, _ = load_checkpoint(checkpoint)
    with torch.no_grad():
        torch.testing.assert_close(expected, restored(features))
    # Existing artifacts without a pooling field still reconstruct the global control.
    payload = torch.load(checkpoint, weights_only=True)
    global_model = KeyCNN()
    payload["model_config"] = {"size": "small", "dropout": 0.3}
    payload["state_dict"] = global_model.state_dict()
    torch.save(payload, checkpoint)
    assert load_checkpoint(checkpoint)[0].config["pooling"] == "global"


def test_experiment_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    random_generator = np.random.default_rng(1)
    features = random_generator.normal(size=(8, 300, 33)).astype(np.float32)
    meta = {
        "refs": [0, 0, 0, 0, 1, 1, 1, 1],
        "ids": [str(index) for index in range(8)],
        "tracks": [{"sha256": str(index)} for index in range(8)],
        "frames": 300,
    }
    training = [0, 1, 2, 4, 5, 6]
    validation = [3, 7]
    recipe = Recipe("resume-test", pooling="frequency", dropout=0.3, smoothing=0)
    full = run_recipe(
        features,
        meta,
        training,
        validation,
        recipe,
        seed=13,
        output=tmp_path / "full",
        epochs=3,
        batch=3,
        threads=1,
    )
    paused = run_recipe(
        features,
        meta,
        training,
        validation,
        recipe,
        seed=13,
        output=tmp_path / "resumed",
        epochs=3,
        batch=3,
        threads=1,
        stop_after=1,
    )
    assert paused["status"] == "paused"
    resumed = run_recipe(
        features,
        meta,
        training,
        validation,
        recipe,
        seed=13,
        output=tmp_path / "resumed",
        epochs=3,
        batch=3,
        threads=1,
    )
    assert full["history"] == resumed["history"]
    assert full["validation"] == resumed["validation"]
    full_state = torch.load(tmp_path / "full/last.pt", weights_only=True)["model"]
    resumed_state = torch.load(tmp_path / "resumed/last.pt", weights_only=True)["model"]
    for name in full_state:
        torch.testing.assert_close(full_state[name], resumed_state[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="identity differs"):
        run_recipe(
            features,
            meta,
            training,
            validation,
            recipe,
            seed=37,
            output=tmp_path / "resumed",
            epochs=3,
            batch=3,
            threads=1,
        )


def test_development_loader_never_opens_test_data_and_quarantines_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta: dict[str, Any] = {
        "refs": [0] * 10,
        "ids": [str(index) for index in range(10)],
        "tracks": [{"sha256": str(index)} for index in range(10)],
    }
    opened = []

    def fake_load(data: str | Path, tag: str) -> tuple[np.ndarray, dict[str, Any]]:
        opened.append(tag)
        assert tag == "train"
        return np.zeros((10, 300, 33), np.float32), meta

    monkeypatch.setattr("experiments.small.load_prepared", fake_load)
    _, _, training, validation = load_development("unused")
    assert opened == ["train"]
    meta["tracks"][validation[0]]["sha256"] = meta["tracks"][training[0]]["sha256"]
    _, _, cleaned_train, cleaned_validation = load_development("unused")
    assert cleaned_validation == validation
    assert training[0] not in cleaned_train
    assert not {meta["tracks"][index]["sha256"] for index in cleaned_train} & {
        meta["tracks"][index]["sha256"] for index in cleaned_validation
    }
    meta["refs"][training[0]] = 1
    with pytest.raises(ValueError, match="conflicting labels"):
        load_development("unused")


def test_confirmation_requires_same_protocol_and_material_screen_gain() -> None:
    results: list[dict[str, Any]] = [
        {
            "status": "complete",
            "identity": {"recipe": asdict(recipe)},
            "validation": {"mirex": 0.4, "exact": 0.3},
        }
        for recipe in RECIPES
    ]
    results[2]["validation"]["mirex"] = 0.6
    results[3]["validation"]["mirex"] = 0.41
    summary = {
        "status": "complete",
        "manifest_sha256": "data",
        "split_sha256": "split",
        "source_sha256": {},
        "results": results,
    }
    selected = confirmation_recipes(summary, "data", "split", {})
    assert [recipe.name for recipe in selected] == ["control", "frequency"]
    with pytest.raises(ValueError, match="matching data"):
        confirmation_recipes(summary, "different", "split", {})
