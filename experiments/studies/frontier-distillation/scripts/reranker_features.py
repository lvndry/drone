"""Candidate and acoustic features for the top-2 key reranker.

Class index = tonic * 2 + minor, tonics C..B as 0..11. Every acoustic feature is
expressed as candidate-1 minus candidate-2 so swapping the candidates negates it.
"""

from __future__ import annotations

import numpy as np

CORE_LOW_BIN = 8
CORE_HIGH_BIN = 200
BINS_PER_SEMITONE = 2
SEMITONES_PER_OCTAVE = 12
CORE_OCTAVES = 8
BASS_OCTAVES = 3
FINAL_FRACTION = 0.2

KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
TEMPERLEY_MAJOR = np.array(
    [0.748, 0.060, 0.488, 0.082, 0.670, 0.460, 0.096, 0.715, 0.104, 0.366, 0.057, 0.400]
)
TEMPERLEY_MINOR = np.array(
    [0.712, 0.084, 0.474, 0.618, 0.049, 0.460, 0.105, 0.747, 0.404, 0.067, 0.133, 0.330]
)
PROFILES = {
    "krumhansl": (KRUMHANSL_MAJOR, KRUMHANSL_MINOR),
    "temperley": (TEMPERLEY_MAJOR, TEMPERLEY_MINOR),
}

RELATIONS = ("parallel", "relative", "fifth", "fourth", "semitone", "other")
CHROMA_SOURCES = ("mean", "final", "bass")
ACOUSTIC_NAMES = ("tonic", "third_in_mode", "fifth", "krumhansl", "temperley")


def octave_chroma(cqt: np.ndarray) -> np.ndarray:
    """Fold the 192 core bins of a (208, frames) CQT into (octaves, 12, frames)."""
    core = np.asarray(cqt, dtype=np.float32)[CORE_LOW_BIN:CORE_HIGH_BIN]
    frames = core.shape[1]
    semitones = (
        core.astype(np.float64)
        .reshape(CORE_OCTAVES * SEMITONES_PER_OCTAVE, BINS_PER_SEMITONE, frames)
        .sum(axis=1)
    )
    return semitones.reshape(CORE_OCTAVES, SEMITONES_PER_OCTAVE, frames)


def normalized(vector: np.ndarray) -> np.ndarray:
    total = float(vector.sum())
    if total <= 0:
        return np.full_like(vector, 1.0 / vector.size)
    return vector / total


def chroma_views(cqt: np.ndarray) -> dict[str, np.ndarray]:
    """Return normalized 12-bin chroma for the whole track, its final 20% of frames and octaves 1-3."""
    per_octave = octave_chroma(cqt)
    full = per_octave.sum(axis=0)
    frames = full.shape[1]
    final_frames = max(1, int(np.ceil(FINAL_FRACTION * frames)))
    bass = per_octave[:BASS_OCTAVES].sum(axis=0)
    return {
        "mean": normalized(full.mean(axis=1)),
        "final": normalized(full[:, -final_frames:].mean(axis=1)),
        "bass": normalized(bass.mean(axis=1)),
    }


def rotated_profile(profile_name: str, key_class: int) -> np.ndarray:
    tonic = key_class // 2
    minor = key_class % 2
    profile = PROFILES[profile_name][minor]
    return np.roll(profile, tonic)


def profile_correlation(chroma: np.ndarray, profile_name: str, key_class: int) -> float:
    profile = rotated_profile(profile_name, key_class)
    if np.std(chroma) == 0:
        return 0.0
    return float(np.corrcoef(chroma, profile)[0, 1])


def candidate_evidence(chroma: np.ndarray, key_class: int) -> np.ndarray:
    tonic = key_class // 2
    minor = key_class % 2
    major_third = chroma[(tonic + 4) % 12]
    minor_third = chroma[(tonic + 3) % 12]
    third_in_mode = (minor_third - major_third) if minor else (major_third - minor_third)
    return np.array(
        [
            chroma[tonic],
            third_in_mode,
            chroma[(tonic + 7) % 12],
            profile_correlation(chroma, "krumhansl", key_class),
            profile_correlation(chroma, "temperley", key_class),
        ]
    )


def relation(first_class: int, second_class: int) -> str:
    """Relation of candidate 2 to candidate 1."""
    first_tonic = first_class // 2
    first_minor = first_class % 2
    second_tonic = second_class // 2
    second_minor = second_class % 2
    interval = (second_tonic - first_tonic) % 12
    if first_minor != second_minor:
        if interval == 0:
            return "parallel"
        major_tonic = first_tonic if not first_minor else second_tonic
        minor_tonic = first_tonic if first_minor else second_tonic
        if (minor_tonic - major_tonic) % 12 == 9:
            return "relative"
    else:
        if interval == 7:
            return "fifth"
        if interval == 5:
            return "fourth"
    if interval in (1, 11):
        return "semitone"
    return "other"


def acoustic_feature_names() -> list[str]:
    return [f"{source}_{name}_diff" for source in CHROMA_SOURCES for name in ACOUSTIC_NAMES]


def cnn_feature_names() -> list[str]:
    return [
        "probability_first",
        "probability_second",
        "log_margin",
        "stability_first",
        "stability_second",
        "minor_diff",
    ] + [f"relation_{name}" for name in RELATIONS]


def feature_names() -> list[str]:
    return cnn_feature_names() + acoustic_feature_names()


def log_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def candidate_summary(log_probabilities: np.ndarray) -> dict[str, np.ndarray]:
    """From (tracks, models, shifts, 24) log-probabilities, the top-2 candidates and CNN features."""
    normalized_log_probabilities = log_softmax(log_probabilities.astype(np.float64))
    per_shift = normalized_log_probabilities.mean(axis=1)
    averaged = log_softmax(per_shift.mean(axis=1))
    probabilities = np.exp(averaged)
    order = np.argsort(-averaged, axis=1)
    first = order[:, 0]
    second = order[:, 1]
    rows = np.arange(len(first))
    shift_argmax = per_shift.argmax(axis=2)
    stability_first = (shift_argmax == first[:, None]).mean(axis=1)
    stability_second = (shift_argmax == second[:, None]).mean(axis=1)
    relation_onehot = np.zeros((len(first), len(RELATIONS)))
    for row_index, (first_class, second_class) in enumerate(zip(first, second)):
        relation_onehot[
            row_index, RELATIONS.index(relation(int(first_class), int(second_class)))
        ] = 1.0
    cnn_features = np.column_stack(
        [
            probabilities[rows, first],
            probabilities[rows, second],
            averaged[rows, first] - averaged[rows, second],
            stability_first,
            stability_second,
            (first % 2) - (second % 2),
            relation_onehot,
        ]
    )
    return {"first": first, "second": second, "cnn_features": cnn_features}


def acoustic_features(
    views: dict[str, np.ndarray], first_class: int, second_class: int
) -> np.ndarray:
    differences = [
        candidate_evidence(views[source], first_class)
        - candidate_evidence(views[source], second_class)
        for source in CHROMA_SOURCES
    ]
    return np.concatenate(differences)
