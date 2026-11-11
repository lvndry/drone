import json
import shutil
import wave

import numpy as np
import pytest
import torch

from drone import load_model, predict
from drone.checkpoints import load_checkpoint, save_checkpoint
from drone.datasets import Track, load_beatport, load_giantsteps, parse_key
from drone.features import (
    CROP_FRAMES,
    FEATURE_SPEC,
    N_CORE,
    SR,
    window,
)
from drone.keys import decode_key, encode_key
from drone.model import KeyCNN, params_count
from train import (
    assert_disjoint,
    evaluate_checkpoint,
    load_manifest,
    prepare_dir,
    train_model,
    validation_split,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def write_tone(path, hz, seconds=3):
    t = np.arange(int(SR * seconds)) / SR
    samples = (np.sin(2 * np.pi * hz * t) * 0.5).astype(np.float32)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SR)
        out.writeframes((samples * 32767).astype(np.int16).tobytes())


def test_checkpoint_round_trip_and_version_rejection(tmp_path):
    torch.manual_seed(13)
    model = KeyCNN().eval()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model)
    restored = load_model(path)
    x = torch.randn(2, N_CORE, 128)
    with torch.no_grad():
        torch.testing.assert_close(model(x), restored(x))
    torch.save(model.state_dict(), path)
    with pytest.raises(ValueError, match="Incompatible"):
        load_model(path)
    torch.load(path, weights_only=True)  # raw state dict loads fine
    save_checkpoint(tmp_path / "good.pt", model)
    good = torch.load(tmp_path / "good.pt", weights_only=True)
    good["feature_spec"] = "stft-33bands-a1-f4-zscore-v1"
    torch.save(good, path)
    with pytest.raises(ValueError, match="Incompatible"):
        load_model(path)
    with pytest.raises(FileNotFoundError, match="No trained checkpoint"):
        load_model(tmp_path / "absent.pt")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required for audio integration")
def test_inference_decodes_interleaved_classes(tmp_path):
    audio = tmp_path / "audio.wav"
    write_tone(audio, 110)
    model = KeyCNN()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias[1] = 10
    weights = tmp_path / "weights.pt"
    save_checkpoint(weights, model)
    tonic, mode, confidence = predict(str(audio), str(tmp_path / "cache"), weights=weights)
    assert (tonic, mode) == decode_key(1) == ("C", "minor")
    assert confidence > 0.99


def test_split_and_overlap_guards():
    refs = [key for key in range(24) for _ in range(10)]
    train, val = validation_split(refs)
    assert validation_split(refs) == (train, val)
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(range(len(refs)))
    # stratified: every class appears in validation
    assert sorted(refs[i] for i in val) == list(range(24))
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(["1", "2"], ["2", "3"])
    with pytest.raises(ValueError, match="at least two"):
        validation_split([0, 1])


def test_parse_key_normalizes_flats_and_rejects_garbage():
    assert parse_key("Ab major") == ("G#", "major")
    assert parse_key("D# minor | E major") == ("D#", "minor")
    assert parse_key("B minor") == ("B", "minor")
    assert parse_key("C#") is None
    assert parse_key("Dorian minor") is None
    assert encode_key("C", "major") == 0
    assert encode_key("C", "minor") == 1
    assert encode_key("B", "minor") == 23


def test_loaders_fail_for_missing_data(tmp_path):
    (tmp_path / "labels").mkdir()
    with pytest.raises(ValueError, match="No tracks"):
        load_beatport(tmp_path, tmp_path / "labels")
    with pytest.raises(ValueError, match="Missing GiantSteps"):
        load_giantsteps(tmp_path)
    with pytest.raises(FileNotFoundError, match="prepare"):
        load_manifest(tmp_path, "train")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required for audio integration")
def test_prepare_train_evaluate_and_infer(tmp_path):
    tracks = []
    for i in range(8):
        audio = tmp_path / f"{i}.wav"
        write_tone(audio, 110 if i < 4 else 220)
        tracks.append(Track(str(i), str(audio), ("A", "major" if i < 4 else "minor")))
    (tmp_path / "labels").mkdir()
    # write label files for the loader used by prepare_dir
    meta = prepare_dir("train", tracks, tmp_path, data=tmp_path)
    assert len(meta["refs"]) == 8
    assert meta["refs"] == [encode_key("A", "major")] * 4 + [encode_key("A", "minor")] * 4
    assert meta["feature_spec"] == FEATURE_SPEC
    cached = prepare_dir("train", tracks, tmp_path, data=tmp_path)
    assert cached == meta
    # A changed label must invalidate the prepared bundle, even with identical audio.
    tracks[0].key = ("C", "major")
    changed = prepare_dir("train", tracks, tmp_path, data=tmp_path)
    assert changed["refs"][0] == 0
    tracks[0].key = ("A", "major")
    prepare_dir("train", tracks, tmp_path, data=tmp_path)

    held_out = []
    for i in range(2):
        audio = tmp_path / f"test-{i}.wav"
        write_tone(audio, 165 + i * 110)
        held_out.append(Track(f"test-{i}", str(audio), ("E", "major" if i == 0 else "minor")))
    prepare_dir("test", held_out, tmp_path, data=tmp_path)
    checkpoint = train_model(tmp_path, "cpu", epochs=1, batch=4)
    record = json.loads((checkpoint.parent / "run.json").read_text())
    assert record["checkpoint_sha256"]
    assert record["history"][0]["loss"] > 0
    # Evaluation must work without any training manifest present: refs/paths are
    # self-contained in the test manifest.
    from train import prepared_path

    prepared_path(tmp_path, "train").unlink()
    result = evaluate_checkpoint(tmp_path, checkpoint, "cpu")
    assert result["tracks"] == 2
    assert sum(result["counts"].values()) == 2
    assert 0 <= result["mirex"] <= 1
    model, _ = load_checkpoint(checkpoint)
    prediction = predict(str(held_out[0].audio), str(tmp_path / "infer"), weights=checkpoint)
    assert len(prediction) == 3


def test_extract_cqt_shapes_and_finiteness():
    import drone.features as feats

    samples = np.random.default_rng(0).standard_normal(SR * 4).astype(np.float32)
    features = feats.cqt(samples)
    n_frames = 1 + (int(SR * 4) - feats.FFT) // feats.HOP + feats.FFT // feats.HOP
    assert features.shape == (feats.N_TOTAL, n_frames)
    assert features.shape[0] == 208
    assert np.isfinite(features).all()
    core = window(features, 0)
    assert core.shape[0] == N_CORE == 192


def test_small_crop_can_overfit():
    torch.manual_seed(13)
    model = KeyCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.zeros(4, N_CORE, 128)
    x[:2, :48, :] = 1
    x[2:, 48:, :] = 2
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


def test_width_scales_parameters_and_checkpoint_config(tmp_path):
    base = params_count(KeyCNN())
    narrow = params_count(KeyCNN(width=0.5))
    assert narrow < base
    assert base < 200_000  # K10-class size, not the old 331k profile
    model = KeyCNN(width=0.7).eval()
    with torch.no_grad():
        assert model(torch.zeros(2, N_CORE, 128)).shape == (2, 24)
    path = tmp_path / "w.pt"
    save_checkpoint(path, model, width=0.7)
    restored, metadata = load_checkpoint(path)
    assert metadata["model_config"] == {"width": 0.7}
    assert params_count(restored) == params_count(KeyCNN(width=0.7))


def test_benchmark_reports_explicit_cpu_budget(tmp_path):
    from drone.benchmark import benchmark

    path = tmp_path / "model.pt"
    save_checkpoint(path, KeyCNN())
    result = benchmark(path, iterations=3, threads=1)
    assert result["parameters"] == params_count(KeyCNN())
    assert result["threads"] == 1
    assert result["pytorch_forward"]["iterations"] == 3
    assert result["pytorch_forward"]["p95_ms"] > 0
    assert result["feature_crop"] == CROP_FRAMES
