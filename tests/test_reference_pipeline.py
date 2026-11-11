from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_reference_features import write_tone

from experiments.reference import load_model, predict
from experiments.reference.checkpoints import load_checkpoint, save_checkpoint
from experiments.reference.datasets import Track, load_beatport, load_giantsteps
from experiments.reference.features import FEATURE_SPEC
from experiments.reference.model import KeyCNN
from experiments.reference.train import (
    assert_disjoint,
    evaluate_checkpoint,
    evaluate_set,
    load_prepared,
    prepare_dir,
    prepared_path,
    train_model,
    validation_split,
)


@pytest.fixture(autouse=True)
def one_cpu_thread() -> Iterator[None]:
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_checkpoint_round_trip_and_version_rejection(tmp_path: Path) -> None:
    torch.manual_seed(13)
    model = KeyCNN().eval()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model)
    restored = load_model(path)
    x = torch.randn(2, 33, 300)
    with torch.no_grad():
        torch.testing.assert_close(model(x), restored(x))
    torch.save(model.state_dict(), path)
    with pytest.raises(ValueError, match="Incompatible"):
        load_model(path)
    save_checkpoint(path, model)
    payload: dict[str, Any] = torch.load(path, weights_only=True)
    payload["feature_spec"] = {**FEATURE_SPEC, "version": 1}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Incompatible"):
        load_model(path)
    with pytest.raises(FileNotFoundError, match="No pretrained weights ship"):
        load_model(tmp_path / "absent.pt")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required for audio integration")
def test_inference_decodes_interleaved_classes(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    write_tone(audio, 110)
    model = KeyCNN()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        head = model.head[-1]
        assert isinstance(head, torch.nn.Linear) and head.bias is not None
        head.bias[1] = 10
    weights = tmp_path / "weights.pt"
    save_checkpoint(weights, model)
    tonic, mode, confidence = predict(str(audio), str(tmp_path / "cache"), weights=weights)
    assert (tonic, mode) == ("C", "minor")
    assert confidence > 0.99


def test_split_and_overlap_guards() -> None:
    refs = [key for key in range(24) for _ in range(10)]
    train, val = validation_split(refs)
    assert validation_split(refs) == (train, val)
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(range(len(refs)))
    assert sorted(refs[i] for i in val) == list(range(24))
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(["1", "2"], ["2", "3"])
    with pytest.raises(ValueError, match="at least two"):
        validation_split([0, 1])


def test_loaders_fail_for_missing_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="labels"):
        load_beatport(tmp_path, tmp_path / "labels")
    with pytest.raises(ValueError, match="Missing GiantSteps"):
        load_giantsteps(tmp_path)
    with pytest.raises(FileNotFoundError, match="prepare"):
        load_prepared(tmp_path, "train")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required for audio integration")
def test_prepare_train_evaluate_and_infer(tmp_path: Path) -> None:
    tracks = []
    for i in range(8):
        audio = tmp_path / f"{i}.wav"
        write_tone(audio, (110 if i < 4 else 220) + i * 0.2)
        tracks.append(Track(str(i), audio, ("A", "major" if i < 4 else "minor")))
    train_features, meta = prepare_dir("train", tracks, tmp_path, data=tmp_path)
    assert train_features.shape == (8, 300, 33)
    cached, cached_meta = prepare_dir("train", tracks, tmp_path, data=tmp_path)
    np.testing.assert_array_equal(cached, train_features)
    assert cached_meta == meta
    # A changed label must invalidate the prepared bundle, even with identical audio.
    tracks[0].key = ("C", "major")
    _, changed = prepare_dir("train", tracks, tmp_path, data=tmp_path)
    assert changed["refs"][0] == 0
    tracks[0].key = ("A", "major")
    prepare_dir("train", tracks, tmp_path, data=tmp_path)
    held_out = []
    for i in range(2):
        audio = tmp_path / f"test-{i}.wav"
        write_tone(audio, 165 + i * 110)
        held_out.append(Track(f"test-{i}", audio, ("E", "major" if i == 0 else "minor")))
    prepare_dir("test", held_out, tmp_path, data=tmp_path)
    checkpoint = train_model(tmp_path, "cpu", epochs=1, batch=4)
    record = json.loads((checkpoint.parent / "run.json").read_text())
    assert record["checkpoint_sha256"]
    assert record["history"][0]["loss"] > 0
    # Evaluation must work without any training feature bundle present.
    prepared_path(tmp_path, "train").unlink()
    result = evaluate_checkpoint(tmp_path, checkpoint, "cpu")
    assert result["tracks"] == 2
    assert sum(result["counts"].values()) == 2
    assert 0 <= result["mirex"] <= 1
    model, _ = load_checkpoint(checkpoint)
    prediction = predict(str(held_out[0].mp3), str(tmp_path / "infer"), weights=checkpoint)
    assert len(prediction) == 3
    with pytest.raises(ValueError, match="nonempty"):
        evaluate_set(model, np.empty((0, 300, 33), dtype=np.float32), [], "cpu")
    # Checkpoint time resolution is part of the feature contract.
    prepare_dir("test", held_out, tmp_path, data=tmp_path, frames=200)
    with pytest.raises(ValueError, match="frame counts"):
        evaluate_checkpoint(tmp_path, checkpoint, "cpu")


def test_small_batch_can_overfit() -> None:
    torch.manual_seed(13)
    model = KeyCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.zeros(4, 33, 300)
    x[:2, :, :150] = 1
    x[2:, :, 150:] = 2
    target = torch.tensor([0, 0, 1, 1])
    for _ in range(40):
        model.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), target)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        assert model(x).argmax(1).tolist() == target.tolist()


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_model_size_profiles_and_checkpoint_config(tmp_path: Path, size: str) -> None:
    from experiments.reference.model import params_count

    counts = {"small": 331704, "medium": 1318744, "large": 5258904}
    model = KeyCNN(size=size, dropout=0.15).eval()
    assert params_count(model) == counts[size]
    with torch.no_grad():
        assert model(torch.zeros(2, 33, 300)).shape == (2, 24)
    path = tmp_path / f"{size}.pt"
    save_checkpoint(path, model)
    restored, metadata = load_checkpoint(path)
    assert restored.config == {"size": size, "dropout": 0.15, "pooling": "global"}
    assert metadata["model_config"] == restored.config


def test_training_cli_forwards_model_and_recipe_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    from experiments.reference import train

    received = {}
    monkeypatch.setattr(train, "train_model", lambda *a, **kw: received.update(kw))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reference-train",
            "train",
            "--data",
            str(tmp_path),
            "--model-size",
            "large",
            "--lr",
            "0.002",
            "--dropout",
            "0.1",
            "--label-smoothing",
            "0.05",
            "--weight-decay",
            "0.001",
        ],
    )
    train.main()
    assert received["model_size"] == "large"
    assert received["lr"] == 0.002
    assert received["dropout"] == 0.1
    assert received["label_smoothing"] == 0.05
    assert received["weight_decay"] == 0.001


def test_benchmark_reports_explicit_cpu_budget(tmp_path: Path) -> None:
    from experiments.reference.benchmark import benchmark

    path = tmp_path / "model.pt"
    save_checkpoint(path, KeyCNN())
    result = benchmark(path, iterations=3, threads=1)
    assert result["parameters"] == 331704
    assert result["threads"] == 1
    assert result["pytorch_forward"]["iterations"] == 3
    assert result["pytorch_forward"]["p95_ms"] > 0
