"""Write synthetic chord progressions and an Euphoria corrections export for them.

A stand-in for a real `euphoria corrections export` when nobody has corrected
any keys yet: it exercises matching, feature caching, fine-tuning and scoring
end to end. The audio is synthesized, so results on it say nothing about the
gain on real music.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

from drone.corrections import SUPPORTED_SCHEMA_VERSION, euphoria_content_hash
from drone.features import SR
from drone.keys import TONICS, decode_key, transpose_key

MAJOR_PROGRESSION = ((0, 4, 7), (5, 9, 12), (7, 11, 14), (0, 4, 7))
MINOR_PROGRESSION = ((0, 3, 7), (5, 8, 12), (7, 11, 14), (0, 3, 7))
CORRECTED_AT = 1_794_409_200


def tone(frequency: float, seconds: float, harmonics: int, brightness: float) -> np.ndarray:
    time = np.arange(int(SR * seconds)) / SR
    signal = np.zeros_like(time)
    for harmonic in range(1, harmonics + 1):
        partial = frequency * harmonic
        if partial >= SR / 2:
            break
        signal += np.sin(2 * np.pi * partial * time) * brightness ** (harmonic - 1)
    envelope = np.minimum(1.0, time / 0.02) * np.exp(-time * 1.5)
    return signal * envelope


def progression(class_id: int, seconds: float, rng: np.random.Generator) -> np.ndarray:
    """I-IV-V-I in major or i-iv-V-i in minor, with a bass root and noise."""
    tonic_pitch_class = class_id // 2
    chords = MINOR_PROGRESSION if class_id % 2 else MAJOR_PROGRESSION
    octave = int(rng.integers(3, 5))
    harmonics = int(rng.integers(3, 9))
    brightness = float(rng.uniform(0.3, 0.8))
    chord_seconds = seconds / len(chords)
    pieces = []
    for chord in chords:
        segment = np.zeros(int(SR * chord_seconds))
        for interval in chord:
            midi = 12 * (octave + 1) + tonic_pitch_class + interval
            frequency = 440.0 * 2 ** ((midi - 69) / 12)
            segment += tone(frequency, chord_seconds, harmonics, brightness)[: len(segment)]
        bass_frequency = 440.0 * 2 ** ((12 * octave + tonic_pitch_class + chord[0] - 69) / 12)
        segment += 0.8 * tone(bass_frequency, chord_seconds, 2, 0.5)[: len(segment)]
        pieces.append(segment)
    signal = np.concatenate(pieces)
    signal += rng.normal(0, 0.02, len(signal))
    return 0.8 * signal / np.max(np.abs(signal))


def write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SR)
        wav_file.writeframes(np.round(samples * 32767).astype("<i2").tobytes())


def camelot(pitch_class: int, mode: str) -> str:
    major_pitch_class = pitch_class if mode == "major" else (pitch_class + 3) % 12
    return f"{(major_pitch_class * 7 + 7) % 12 + 1}{'B' if mode == 'major' else 'A'}"


def key_fields(class_id: int) -> dict[str, object]:
    tonic, mode = decode_key(class_id)
    pitch_class = TONICS.index(tonic)
    return {
        "name": f"{tonic} {mode}",
        "camelot": camelot(pitch_class, mode),
        "tonic": pitch_class,
        "mode": mode,
    }


def write_synthetic_corrections(
    output: str | Path, count: int, *, seconds: float = 8.0, seed: int = 13
) -> Path:
    """Write `count` progressions under `output`/audio and their export.

    Keys cycle through all 24 classes. Each detected key is a fifth away, a
    common detector error, and one tempo-only correction is added so the
    reader's skipping is exercised too.
    """
    output = Path(output)
    audio = output / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lines = []
    for index in range(count):
        class_id = index % 24
        path = audio / f"synthetic-{index:04d}.wav"
        write_wav(path, progression(class_id, seconds, rng))
        lines.append(
            {
                "schema_version": SUPPORTED_SCHEMA_VERSION,
                "content_hash": euphoria_content_hash(path),
                "corrected_key": key_fields(class_id),
                "corrected_bpm": None,
                "corrected_at": CORRECTED_AT + index,
                "detected_key": key_fields(transpose_key(class_id, 7)),
                "detected_bpm": 124.0,
                "analyzer_version": "synthetic",
                "analyzed_at": CORRECTED_AT,
            }
        )
    lines.append(
        {
            "schema_version": SUPPORTED_SCHEMA_VERSION,
            "content_hash": "0" * 64,
            "corrected_key": None,
            "corrected_bpm": 126.0,
            "corrected_at": CORRECTED_AT + count,
            "detected_key": None,
            "detected_bpm": None,
            "analyzer_version": None,
            "analyzed_at": None,
        }
    )
    export = output / "corrections.jsonl"
    export.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=96)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    export = write_synthetic_corrections(
        args.output, args.count, seconds=args.seconds, seed=args.seed
    )
    print(f"Wrote {args.count} tracks under {args.output / 'audio'} and {export}")


if __name__ == "__main__":
    main()
