"""For tracks both model families get 'wrong' the same way, what do the other human sources say?"""

from __future__ import annotations

import json
from collections import Counter

from frontier_train import GIANTSTEPS_DIRECTORY
from frontier_train import WORK_DIRECTORY as WORK
from rescore_gsplus import load_original

ORIGINAL_ANNOTATIONS = GIANTSTEPS_DIRECTORY / "annotations" / "key"
MODE_WORDS = ("dorian", "phrygian", "mixolydian", "lydian", "locrian")


def main() -> None:
    shared = json.loads((WORK / "ceiling-shared-errors.json").read_text())
    original = load_original(ORIGINAL_ANNOTATIONS)
    annotation_directory = WORK / "gsplus-download" / "annotations" / "new_annotations"
    agree_with_original = 0
    modal_tracks = 0
    mode_counts: Counter = Counter()
    comments = []
    fields: set[str] = set()
    explained = 0
    for item in shared:
        record = json.loads((annotation_directory / f"{item['track']}.json").read_text())
        fields.update(record)
        matches_original = original.get(item["track"]) == item["both_predict"]
        agree_with_original += matches_original
        key_text = " ".join(record.get("key") or []).lower()
        found = [word for word in MODE_WORDS if word in key_text]
        mode_counts.update(found)
        modal_tracks += bool(found)
        if record.get("comments"):
            comments.append(record["comments"])
        low_confidence = record.get("confidence") is not None and record["confidence"] < 2
        explained += bool(matches_original or found or record.get("comments") or low_confidence)
    print(f"{explained} of {len(shared)} have at least one independent sign of ambiguity")
    print(f"shared same-key errors: {len(shared)}")
    print(f"the models' key equals the ORIGINAL GiantSteps label for {agree_with_original} of them")
    print(f"expert label names a church mode for {modal_tracks} of them: {dict(mode_counts)}")
    print(f"annotation fields: {sorted(fields)}")
    print(f"tracks with an expert comment: {len(comments)}")
    for comment in comments[:8]:
        print("  -", str(comment)[:160])


if __name__ == "__main__":
    main()
