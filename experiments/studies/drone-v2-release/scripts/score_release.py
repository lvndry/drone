"""Score packaged-predictor outputs on GiantSteps and FMA panels, overall and by DJ genre."""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

STUDY_SCRIPTS = Path(__file__).resolve().parents[2] / "frontier-distillation" / "scripts"
sys.path.insert(0, str(STUDY_SCRIPTS))
import numpy as np  # noqa: E402
from compare import interval, is_test_half  # noqa: E402
from rescore_gsplus import load_gsplus, load_original  # noqa: E402

from drone.keys import encode_key, score_mirex  # noqa: E402

GIANTSTEPS = Path(os.environ["DRONE_GIANTSTEPS"])
WORK = Path(os.environ["DRONE_WORK"])
MANIFEST = Path(os.environ["DRONE_PREP"]) / "manifest.json"


def predicted_classes(path: str) -> dict[str, int]:
    return {
        row["track_id"]: encode_key(row["tonic"], row["mode"])
        for row in json.loads(Path(path).read_text())
    }


def main() -> None:
    versions = {
        version: predicted_classes(f"predictions-{version}.json") for version in ("v1", "v2")
    }
    keyfinder = {
        f"giantsteps:{track}": record["drone_class"]
        for track, record in json.loads((WORK / "keyfinder-giantsteps.json").read_text()).items()
    }
    rows = json.loads(MANIFEST.read_text())["tracks"]
    original = load_original(GIANTSTEPS / "annotations" / "key")
    gsplus, _ = load_gsplus(WORK / "gsplus-download" / "annotations" / "new_annotations")
    panels: dict[str, list[tuple[str, int]]] = defaultdict(list)
    genre_of: dict[str, str] = {}
    for row in rows:
        if row["panel"] in ("valid", "holdout"):
            track = f"fmak:{row['track_id']}:{row['panel']}"
            panels[f"fma_{row['panel']}"].append((track, int(row["key_label"])))
            genre_of[track] = row["genre"]
    for track_id, label in sorted(original.items()):
        panels["giantsteps_original"].append((f"giantsteps:{track_id}", label))
    for track_id, record in sorted(gsplus.items()):
        if f"giantsteps:{track_id}" not in versions["v2"]:
            continue
        panels["giantsteps_expert_primary"].append((f"giantsteps:{track_id}", record["primary"]))
        if record["clean"]:
            panels["giantsteps_expert_clean"].append((f"giantsteps:{track_id}", record["primary"]))
            if is_test_half(track_id):
                panels["giantsteps_expert_clean_test_half"].append(
                    (f"giantsteps:{track_id}", record["primary"])
                )
    generator = np.random.default_rng(0)
    report: dict = {"panels": {}, "fma_genres": {}}
    for panel, items in panels.items():
        references = [label for _, label in items]
        entry = {"tracks": len(items)}
        correct = {}
        systems = dict(versions)
        if panel.startswith("giantsteps"):
            systems = {**versions, "libkeyfinder": keyfinder}
        for name, predictions in systems.items():
            classes = [predictions[track] for track, _ in items]
            exact, weighted = score_mirex(classes, references)
            entry[name] = {"exact": round(100 * exact, 2), "weighted": round(100 * weighted, 2)}
            correct[name] = np.array(
                [value == reference for value, reference in zip(classes, references)], dtype=float
            )
        for baseline in [name for name in systems if name != "v2"]:
            low, high = interval(correct["v2"] - correct[baseline], generator)
            entry[f"v2_minus_{baseline}"] = {
                "difference": round(100 * float((correct["v2"] - correct[baseline]).mean()), 2),
                "interval_95": [round(100 * low, 2), round(100 * high, 2)],
            }
        report["panels"][panel] = entry
    by_genre: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for panel in ("fma_valid", "fma_holdout"):
        for track, label in panels[panel]:
            by_genre[genre_of[track]].append((track, label))
    for genre, items in sorted(by_genre.items(), key=lambda pair: -len(pair[1])):
        references = [label for _, label in items]
        report["fma_genres"][genre] = {
            "tracks": len(items),
            **{
                name: round(
                    100 * score_mirex([versions[name][track] for track, _ in items], references)[0],
                    1,
                )
                for name in versions
            },
        }
    Path("release-scores.json").write_text(json.dumps(report, indent=1))
    for panel, entry in report["panels"].items():
        print(panel, json.dumps(entry))
    print("FMA genres (exact v1 / v2):")
    for genre, entry in report["fma_genres"].items():
        print(f"  {genre:28s} {entry['tracks']:4d}  {entry['v1']:5.1f}  {entry['v2']:5.1f}")


if __name__ == "__main__":
    main()
