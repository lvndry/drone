"""Corrections exported by Euphoria: hashing, reading, matching, mixing, fine-tuning."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import blake3
import numpy as np
import pytest
import torch

from drone.checkpoints import save_checkpoint
from drone.corrections import (
    euphoria_content_hash,
    is_held_out,
    match_corrections,
    read_corrections_export,
    split_corrections,
)
from drone.datasets import Track
from drone.model import KeyCNN
from tests.test_pipeline import write_tone
from tools.synthetic_corrections import camelot, write_synthetic_corrections
from train import (
    CORRECTIONS_FIT,
    CORRECTIONS_HELD_OUT,
    correction_batches,
    evaluate_checkpoint,
    finetune_on_corrections,
    load_manifest,
    prepare_corrections,
    prepare_dir,
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
    assert {t.id for t in held_out} <= {t.id for t in grown_held_out}
    assert {t.id for t in fit} <= {t.id for t in grown_fit}
    assert all(is_held_out(t.id) for t in held_out)
    with pytest.raises(ValueError, match="Correct more keys"):
        split_corrections([track for track in tracks if not is_held_out(track.id)])


def test_every_batch_holds_the_same_share_of_corrections() -> None:
    rng = np.random.default_rng(0)
    batches = correction_batches(100, 7, 32, 0.25, rng)
    assert all(len(corrections) == 8 for _, corrections in batches)
    base_seen = np.concatenate([base for base, _ in batches])
    assert sorted(base_seen.tolist()) == list(range(100))
    corrections_seen = np.concatenate([corrections for _, corrections in batches])
    counts = np.bincount(corrections_seen, minlength=7)
    assert counts.max() - counts.min() <= 1
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
    save_checkpoint(start, KeyCNN(), train_ids=[t.id for t in base])
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
