"""Corrections exported by Euphoria: hashing, reading, matching, mixing, fine-tuning."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import blake3
import numpy as np
import pytest
import torch

from drone.checkpoints import load_checkpoint, save_checkpoint
from drone.corrections import (
    euphoria_content_hash,
    is_held_out,
    match_corrections,
    read_corrections_export,
    split_corrections,
)
from drone.datasets import Track
from drone.features import CROP_FRAMES, FEATURE_SPEC
from drone.keys import LABEL_SCHEMA
from drone.model import KeyCNN
from tests.test_pipeline import write_tone
from tools.synthetic_corrections import camelot, write_synthetic_corrections
from train import (
    CORRECTIONS_FIT,
    CORRECTIONS_HELD_OUT,
    PROVENANCE_FIELDS,
    correction_batches,
    evaluate_checkpoint,
    finetune_on_corrections,
    load_manifest,
    prepare_corrections,
    prepare_dir,
    prepared_path,
)

FIXTURES = Path(__file__).parent / "fixtures"
EUPHORIA_EXPORT = FIXTURES / "euphoria-corrections.jsonl"
EUPHORIA_HASHED_AUDIO = FIXTURES / "c-major-triad.wav"


@pytest.fixture(autouse=True)
def one_cpu_thread() -> Iterator[None]:
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_content_hash_matches_the_one_euphoria_exported() -> None:
    """The export was written by `euphoria corrections export` after Euphoria
    analyzed the fixture, so its hash is Euphoria's own."""
    first = json.loads(EUPHORIA_EXPORT.read_text().splitlines()[0])
    assert euphoria_content_hash(EUPHORIA_HASHED_AUDIO) == first["content_hash"]


def test_reading_keeps_key_corrections_and_counts_tempo_only_ones() -> None:
    parsed = read_corrections_export(EUPHORIA_EXPORT)
    assert parsed.tempo_only_count == 1
    assert len(parsed.key_corrections) == 1
    correction = parsed.key_corrections[0]
    assert correction.key == ("A", "minor")
    assert correction.detected_key == ("C", "major")
    assert correction.corrected_at == 1_794_409_200


def test_unknown_schema_versions_and_malformed_hashes_are_refused(tmp_path: Path) -> None:
    line = json.loads(EUPHORIA_EXPORT.read_text().splitlines()[0])
    newer = tmp_path / "newer.jsonl"
    newer.write_text(json.dumps({**line, "schema_version": 2}) + "\n")
    with pytest.raises(ValueError, match="schema_version 2"):
        read_corrections_export(newer)
    short = tmp_path / "short.jsonl"
    short.write_text(json.dumps({**line, "content_hash": "abc"}) + "\n")
    with pytest.raises(ValueError, match="64 hex"):
        read_corrections_export(short)
    bad_key = tmp_path / "bad-key.jsonl"
    bad_key.write_text(json.dumps({**line, "corrected_key": {"tonic": 12, "mode": "minor"}}) + "\n")
    with pytest.raises(ValueError, match="pitch class"):
        read_corrections_export(bad_key)


def test_files_are_matched_by_content_wherever_they_are_and_whatever_they_are_called(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "library" / "deeper"
    nested.mkdir(parents=True)
    shutil.copy(EUPHORIA_HASHED_AUDIO, nested / "renamed.WAV")
    write_tone(tmp_path / "library" / "other.wav", 330)
    (tmp_path / "library" / "notes.txt").write_text("not audio")
    parsed = read_corrections_export(EUPHORIA_EXPORT)
    matched = match_corrections(parsed.key_corrections, tmp_path / "library")
    assert [track.audio for track in matched.tracks] == [str(nested / "renamed.WAV")]
    assert matched.tracks[0].id == parsed.key_corrections[0].content_hash
    assert matched.tracks[0].key == ("A", "minor")
    assert matched.unmatched_hashes == []
    empty = tmp_path / "empty"
    empty.mkdir()
    unmatched = match_corrections(parsed.key_corrections, empty)
    assert unmatched.tracks == []
    assert unmatched.unmatched_hashes == [parsed.key_corrections[0].content_hash]


def test_a_track_keeps_its_side_of_the_split_as_corrections_grow() -> None:
    def track(index: int) -> Track:
        content_hash = blake3.blake3(str(index).encode()).hexdigest()
        return Track(content_hash, "unused.wav", ("C", "major"))

    tracks = [track(index) for index in range(40)]
    fit, held_out = split_corrections(tracks)
    grown_fit, grown_held_out = split_corrections(
        tracks + [track(index) for index in range(40, 80)]
    )
    assert {track.id for track in held_out} <= {track.id for track in grown_held_out}
    assert {track.id for track in fit} <= {track.id for track in grown_fit}
    assert all(is_held_out(track.id) for track in held_out)
    with pytest.raises(ValueError, match="Correct more keys"):
        split_corrections([track for track in tracks if not is_held_out(track.id)])


def test_every_batch_holds_the_same_share_of_corrections() -> None:
    rng = np.random.default_rng(0)
    batches = correction_batches(100, 7, 32, 0.25, rng)
    assert all(len(corrections) == 8 for _, corrections in batches)
    base_seen = np.concatenate([base for base, _ in batches])
    assert sorted(base_seen.tolist()) == list(range(100))
    corrections_seen = np.concatenate([corrections for _, corrections in batches])
    counts = np.bincount(corrections_seen, minlength=7).tolist()
    assert max(counts) - min(counts) <= 1
    with pytest.raises(ValueError, match="room for base"):
        correction_batches(10, 5, 2, 0.75, rng)


def test_synthetic_camelot_codes_match_euphoria() -> None:
    assert camelot(9, "minor") == "8A"
    assert camelot(0, "major") == "8B"
    assert camelot(4, "minor") == "9A"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required for audio integration")
def test_prepare_and_finetune_report_giantsteps_and_held_out_corrections(tmp_path: Path) -> None:
    data = tmp_path / "data"
    base = []
    for index in range(6):
        audio = tmp_path / f"base-{index}.wav"
        write_tone(audio, 110 * (1 + index % 3) + index)
        base.append(Track(f"base-{index}", str(audio), ("A", "major" if index < 3 else "minor")))
    prepare_dir("train", base, data, data=data)
    test = []
    for index in range(2):
        audio = tmp_path / f"test-{index}.wav"
        write_tone(audio, 165 + index * 110)
        test.append(Track(f"test-{index}", str(audio), ("E", "major" if index else "minor")))
    prepare_dir("test", test, data, data=data)

    export = write_synthetic_corrections(tmp_path / "synthetic", 24, seconds=3.0)
    summary = prepare_corrections(data, export, tmp_path / "synthetic" / "audio")
    assert summary["matched"] == 24
    assert summary["tempo_only_skipped"] == 1
    assert summary["fit"] + summary["held_out"] == 24
    assert len(load_manifest(data, CORRECTIONS_HELD_OUT)["refs"]) == summary["held_out"]

    torch.manual_seed(13)
    start = tmp_path / "start.pt"
    save_checkpoint(start, KeyCNN(), train_ids=[track.id for track in base])
    tuned = finetune_on_corrections(data, start, "cpu", epochs=1, batch=4)
    report = json.loads((tuned.parent / "finetune.json").read_text())
    for name, tracks in (("giantsteps", 2), ("corrections_held_out", summary["held_out"])):
        assert report[name]["before"]["tracks"] == tracks
        assert report[name]["after"]["tracks"] == tracks
        assert sum(report[name]["paired_change"].values()) == tracks
    assert report["config"]["corrections_per_batch"] == 1
    assert report["config"]["corrections_fit"] == summary["fit"]
    assert evaluate_checkpoint(data, tuned, "cpu")["tracks"] == 2


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required for audio integration")
def test_corrected_giantsteps_audio_is_refused(tmp_path: Path) -> None:
    data = tmp_path / "data"
    export = write_synthetic_corrections(tmp_path / "synthetic", 4, seconds=2.0)
    audio = sorted((tmp_path / "synthetic" / "audio").glob("*.wav"))
    prepare_dir("test", [Track("test-0", str(audio[0]), ("C", "major"))], data, data=data)
    with pytest.raises(ValueError, match="GiantSteps test audio"):
        prepare_corrections(data, export, tmp_path / "synthetic" / "audio")
    assert not (data / CORRECTIONS_FIT).exists()


def synthetic_manifest(ids: list[str], hashes: list[str]) -> dict[str, Any]:
    """Declarative manifests plus synthetic arrays; never read an audio library."""
    return {
        "feature_spec": FEATURE_SPEC,
        "label_schema": LABEL_SCHEMA,
        "crop": CROP_FRAMES,
        "ids": ids,
        "refs": [0] * len(ids),
        "tracks": [
            {
                "id": track_id,
                "path": "synthetic-unread.wav",
                "sha256": digest,
                "key": ["C", "major"],
            }
            for track_id, digest in zip(ids, hashes, strict=True)
        ],
    }


@pytest.fixture
def synthetic_finetune_data(tmp_path: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    manifests = {
        "train": synthetic_manifest(["base"], ["a" * 64]),
        "test": synthetic_manifest(["gs"], ["b" * 64]),
        CORRECTIONS_FIT: synthetic_manifest(["fit"], ["c" * 64]),
        CORRECTIONS_HELD_OUT: synthetic_manifest(["held"], ["d" * 64]),
    }
    return tmp_path, manifests


def write_synthetic_manifests(data: Path, manifests: dict[str, dict[str, Any]]) -> None:
    for tag, manifest in manifests.items():
        path = prepared_path(data, tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    ("field", "value", "evaluation"),
    [
        ("train_ids", ["gs"], "GiantSteps"),
        ("validation_ids", ["gs"], "GiantSteps"),
        ("training_audio_sha256", ["b" * 64], "GiantSteps"),
        ("training_audio_sha256", ["d" * 64], "corrections-held-out"),
        ("validation_ids", ["held"], "corrections-held-out"),
    ],
)
def test_finetune_rejects_inherited_exposure_before_scoring_or_gradients(
    synthetic_finetune_data: tuple[Path, dict[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: list[str],
    evaluation: str,
) -> None:
    data, manifests = synthetic_finetune_data
    write_synthetic_manifests(data, manifests)
    checkpoint = data / "start.pt"
    inherited: dict[str, Any] = {field: value}
    save_checkpoint(checkpoint, KeyCNN(), **inherited)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Exposure must be rejected before scoring, sampling or optimizer creation")

    monkeypatch.setattr("train.score_tracks", forbidden)
    monkeypatch.setattr("train.sample_features", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    with pytest.raises(ValueError, match=evaluation + " overlaps starting-checkpoint"):
        finetune_on_corrections(data, checkpoint, "cpu", epochs=1, batch=2)
    assert not (data / "runs").exists()


def test_finetune_rejects_actual_base_test_overlap_before_scoring(
    synthetic_finetune_data: tuple[Path, dict[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, manifests = synthetic_finetune_data
    # Different IDs, identical bytes: the ordinary trainer already rejects this.
    manifests["train"] = synthetic_manifest(["base-alias"], ["b" * 64])
    write_synthetic_manifests(data, manifests)
    checkpoint = data / "start.pt"
    save_checkpoint(checkpoint, KeyCNN())

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("New fitting overlap must be rejected before evaluation or training")

    monkeypatch.setattr("train.score_tracks", forbidden)
    monkeypatch.setattr("train.sample_features", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    with pytest.raises(ValueError, match="GiantSteps overlaps new fine-tuning data"):
        finetune_on_corrections(data, checkpoint, "cpu", epochs=1, batch=2)


def test_filtered_held_audio_is_not_added_to_training_provenance(
    synthetic_finetune_data: tuple[Path, dict[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, manifests = synthetic_finetune_data
    manifests["train"] = synthetic_manifest(["base", "held-as-base"], ["a" * 64, "d" * 64])
    write_synthetic_manifests(data, manifests)
    checkpoint = data / "start.pt"
    inherited: dict[str, Any] = {
        "train_ids": ["prior-fit"],
        "validation_ids": ["prior-validation"],
        "training_audio_sha256": ["e" * 64],
    }
    save_checkpoint(checkpoint, KeyCNN(), **inherited)
    seen: list[str] = []

    def sample(
        track: Track, data: Path, random_generator: np.random.Generator, T: int
    ) -> tuple[np.ndarray, int]:
        seen.append(track.id)
        return np.zeros((192, 8), dtype=np.float32), 0

    def score(
        model: torch.nn.Module, meta: dict[str, Any], data: Path, device: str
    ) -> dict[str, Any]:
        return {
            "tracks": len(meta["refs"]),
            "exact": 1.0,
            "mirex": 1.0,
            "counts": {"exact": len(meta["refs"])},
            "predictions": list(meta["refs"]),
        }

    monkeypatch.setattr("train.sample_features", sample)
    monkeypatch.setattr("train.score_tracks", score)
    tuned = finetune_on_corrections(data, checkpoint, "cpu", epochs=1, batch=2)
    _, payload = load_checkpoint(tuned, "cpu")
    assert sorted(seen) == ["base", "fit"]
    metadata = payload["metadata"]
    assert metadata["train_ids"] == ["prior-fit", "base", "fit"]
    assert metadata["validation_ids"] == ["prior-validation"]
    assert metadata["training_audio_sha256"] == ["e" * 64, "a" * 64, "c" * 64]
    report = json.loads((tuned.parent / "finetune.json").read_text())
    assert report["inherited_provenance"]["present_fields"] == list(PROVENANCE_FIELDS)
    assert report["inherited_provenance"]["missing_fields"] == []
    # An ordinary evaluation on the truly unseen held-out correction stays valid.
    prepared_path(data, "test").write_text(json.dumps(manifests[CORRECTIONS_HELD_OUT]))
    monkeypatch.setattr("train.evaluate_tracks", lambda *args: np.array([0]))
    assert evaluate_checkpoint(data, tuned, "cpu")["tracks"] == 1


def test_legacy_missing_provenance_is_reported_as_unknown(
    synthetic_finetune_data: tuple[Path, dict[str, dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, manifests = synthetic_finetune_data
    write_synthetic_manifests(data, manifests)
    checkpoint = data / "start.pt"
    save_checkpoint(checkpoint, KeyCNN(), train_ids=["known-prior"])
    monkeypatch.setattr(
        "train.sample_features", lambda *args, **kwargs: (np.zeros((192, 8), dtype=np.float32), 0)
    )
    monkeypatch.setattr(
        "train.score_tracks",
        lambda model, meta, *args: dict(tracks=1, exact=1.0, mirex=1.0, predictions=[0]),
    )
    tuned = finetune_on_corrections(data, checkpoint, "cpu", epochs=1, batch=2)
    report = json.loads((tuned.parent / "finetune.json").read_text())
    assert report["inherited_provenance"]["present_fields"] == ["train_ids"]
    assert report["inherited_provenance"]["missing_fields"] == [
        "validation_ids",
        "training_audio_sha256",
    ]
    assert "missing witnesses" in report["inherited_provenance"]["policy"]
