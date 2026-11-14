"""Where do two model families fail on clean GiantSteps+ tracks, and do they fail the same way?"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from frontier_train import WORK_DIRECTORY as WORK
from rescore_gsplus import load_gsplus
from score_madmom import panel_references


def relation(prediction: int, reference: int) -> str:
    if prediction == reference:
        return "exact"
    predicted_tonic, predicted_minor = divmod(prediction, 2)
    reference_tonic, reference_minor = divmod(reference, 2)
    interval = (predicted_tonic - reference_tonic) % 12
    if predicted_minor == reference_minor and interval in (5, 7):
        return "fifth"
    if predicted_minor != reference_minor and interval == (3 if reference_minor else 9):
        return "relative"
    if predicted_minor != reference_minor and interval == 0:
        return "parallel"
    return "other"


def family(path: str, group: str, panel: str) -> np.ndarray:
    return np.exp(np.load(path)[f"{group}/{panel}"]).mean(axis=(0, 2)).argmax(-1)


def main() -> None:
    gsplus, _ = load_gsplus(WORK / "gsplus-download" / "annotations" / "new_annotations")
    track_ids, references = panel_references()["gsplus_clean"]
    references = np.array(references)
    k10 = family(str(WORK / "analysis-r6.npz"), "shift12", "gsplus_clean")
    myna = family(str(WORK / "analysis-myna.npz"), "myna", "gsplus_clean")
    k10_wrong = k10 != references
    myna_wrong = myna != references
    both_wrong = k10_wrong & myna_wrong
    same_wrong_answer = both_wrong & (k10 == myna)
    print(
        f"tracks {len(references)}: K10 wrong {k10_wrong.sum()}, Myna wrong {myna_wrong.sum()}, both wrong {both_wrong.sum()}, both wrong with the same key {same_wrong_answer.sum()}"
    )
    print(
        f"only K10 wrong {(k10_wrong & ~myna_wrong).sum()}, only Myna wrong {(myna_wrong & ~k10_wrong).sum()}; an oracle picking the right family would score {100 * (~both_wrong).mean():.2f}%"
    )
    print(
        "K10 error relations:",
        dict(
            Counter(
                relation(int(prediction), int(reference))
                for prediction, reference in zip(k10[k10_wrong], references[k10_wrong])
            )
        ),
    )
    print(
        "shared same-key error relations:",
        dict(
            Counter(
                relation(int(prediction), int(reference))
                for prediction, reference in zip(
                    k10[same_wrong_answer], references[same_wrong_answer]
                )
            )
        ),
    )
    confidence = np.array([gsplus[track]["confidence"] for track in track_ids], dtype=object)
    for label in sorted({str(value) for value in confidence}):
        mask = np.array([str(value) == label for value in confidence])
        print(
            f"annotator confidence {label}: tracks {mask.sum()}, K10 exact {100 * (~k10_wrong[mask]).mean():.1f}%, shared same-key errors {same_wrong_answer[mask].sum()}"
        )
    shared = [
        {
            "track": track,
            "reference": int(reference),
            "both_predict": int(prediction),
            "relation": relation(int(prediction), int(reference)),
            "confidence": gsplus[track]["confidence"],
        }
        for track, reference, prediction, flag in zip(track_ids, references, k10, same_wrong_answer)
        if flag
    ]
    (WORK / "ceiling-shared-errors.json").write_text(json.dumps(shared, indent=1))


if __name__ == "__main__":
    main()
