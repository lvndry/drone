"""Top-2 reranker for the K10 key classifier.

fit:   learn when to replace the CNN's first guess with its second, from out-of-fold logits.
apply: score a panel before and after reranking.
dev:   nested cross-validation on one panel, for development only.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from environment import required_path

sys.path.insert(0, str(required_path("DRONE_SOURCE")))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from reranker_features import (  # noqa: E402
    CHROMA_SOURCES,
    acoustic_features,
    candidate_summary,
    chroma_views,
    feature_names,
)
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

WORK_DIRECTORY = required_path("DRONE_WORK")
PREPARED_DIRECTORY = required_path("DRONE_PREP")
CHROMA_CACHE = WORK_DIRECTORY / "reranker-dev" / "chroma-cache.npz"
REGULARIZATION_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
THRESHOLD_GRID = np.round(np.arange(0.5, 1.0, 0.01), 2)
NEVER_OVERRIDE = 1.01
CROSS_VALIDATION_SPLITS = 5
SEED = 13


class ProbabilityModel(Protocol):
    def probability(self, features: np.ndarray) -> np.ndarray: ...


@dataclass
class TrackSet:
    log_probabilities: np.ndarray
    references: np.ndarray
    cache_keys: list[str]
    loaders: list[Callable[[], np.ndarray]]
    groups: np.ndarray | None = None


def manifest_index() -> dict[str, dict]:
    rows = json.loads((PREPARED_DIRECTORY / "manifest.json").read_text())["tracks"]
    return {row["track_id"]: row for row in rows}


def path_loader(path: Path) -> Callable[[], np.ndarray]:
    return lambda: np.load(path)


def array_loader(array: np.ndarray) -> Callable[[], np.ndarray]:
    return lambda: array


def load_fold_files(pattern: str) -> TrackSet:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no fold files match {pattern}")
    index = manifest_index()
    log_probabilities = []
    references = []
    track_ids: list[str] = []
    groups = []
    for fold_position, path in enumerate(paths):
        stored = np.load(path)
        logits = stored["logits"]
        log_probabilities.append(logits[:, None])
        references.append(stored["references"].astype(int))
        track_ids.extend(str(track_id) for track_id in stored["track_ids"])
        match = re.search(r"fold(\d+)", path)
        groups.append(np.full(len(logits), int(match.group(1)) if match else fold_position))
    if len(set(track_ids)) != len(track_ids):
        raise SystemExit("fold files overlap: duplicated track ids")
    missing = [track_id for track_id in track_ids if track_id not in index]
    if missing:
        raise SystemExit(
            f"{len(missing)} fold track ids are not in the manifest, e.g. {missing[:5]}"
        )
    references_array = np.concatenate(references)
    manifest_labels = np.array([int(index[track_id]["key_label"]) for track_id in track_ids])
    mismatched = int((manifest_labels != references_array).sum())
    if mismatched:
        raise SystemExit(f"{mismatched} fold references disagree with manifest key_label")
    cqt_paths = [PREPARED_DIRECTORY / index[track_id]["cqt_path"] for track_id in track_ids]
    print(f"loaded {len(paths)} fold files, {len(track_ids)} tracks", flush=True)
    return TrackSet(
        log_probabilities=np.concatenate(log_probabilities),
        references=references_array,
        cache_keys=[str(path) for path in cqt_paths],
        loaders=[path_loader(path) for path in cqt_paths],
        groups=np.concatenate(groups),
    )


def panel_tracks(panel: str) -> tuple[list[str], list[Callable[[], np.ndarray]], list[int]]:
    """Track order, CQT loaders and references for one panel, in the order of analyze.load_panels()."""
    if panel in ("valid", "holdout"):
        from frontier_train import load_manifest

        rows = load_manifest()[panel]
        cqt_paths = [PREPARED_DIRECTORY / row["cqt_path"] for row in rows]
        return (
            [str(path) for path in cqt_paths],
            [path_loader(path) for path in cqt_paths],
            [int(row["key_label"]) for row in rows],
        )
    from analyze import load_panels

    arrays, references, track_ids = load_panels()[panel]
    return (
        [f"giantsteps:{track_id}" for track_id in track_ids],
        [array_loader(array) for array in arrays],
        [int(reference) for reference in references],
    )


def load_logits_file(logits_path: str, array_name: str, panel: str | None) -> TrackSet:
    stored = np.load(logits_path)
    if "track_ids" in stored.files and array_name == "logits":
        index = manifest_index()
        track_ids = [str(track_id) for track_id in stored["track_ids"]]
        cqt_paths = [PREPARED_DIRECTORY / index[track_id]["cqt_path"] for track_id in track_ids]
        return TrackSet(
            log_probabilities=stored["logits"][:, None],
            references=stored["references"].astype(int),
            cache_keys=[str(path) for path in cqt_paths],
            loaders=[path_loader(path) for path in cqt_paths],
        )
    if panel is None:
        raise SystemExit("--panel is required for arrays without track ids")
    array = stored[array_name]
    if array.ndim == 3:
        array = array[None]
    log_probabilities = np.transpose(array, (1, 0, 2, 3))
    cache_keys, loaders, references = panel_tracks(panel)
    if len(cache_keys) != len(log_probabilities):
        raise SystemExit(
            f"{array_name} has {len(log_probabilities)} tracks but panel {panel} has {len(cache_keys)}"
        )
    return TrackSet(
        log_probabilities=log_probabilities,
        references=np.array(references),
        cache_keys=cache_keys,
        loaders=loaders,
    )


def load_chroma_cache() -> dict[str, np.ndarray]:
    if not CHROMA_CACHE.is_file():
        return {}
    stored = np.load(CHROMA_CACHE)
    return {key: stored[key] for key in stored.files}


def save_chroma_cache(cache: dict[str, np.ndarray]) -> None:
    CHROMA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    pending = CHROMA_CACHE.with_suffix(".tmp.npz")
    np.savez(pending, **cache)  # ty: ignore[invalid-argument-type]
    pending.replace(CHROMA_CACHE)


def stacked_views(cqt: np.ndarray) -> np.ndarray:
    views = chroma_views(cqt)
    return np.stack([views[source] for source in CHROMA_SOURCES])


def track_chroma(tracks: TrackSet) -> list[dict[str, np.ndarray]]:
    cache = load_chroma_cache()
    todo = [position for position, key in enumerate(tracks.cache_keys) if key not in cache]
    if todo:
        print(
            f"computing chroma for {len(todo)} tracks ({len(tracks.cache_keys) - len(todo)} cached)",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=8) as pool:
            computed = list(
                pool.map(lambda position: stacked_views(tracks.loaders[position]()), todo)
            )
        for position, views in zip(todo, computed):
            cache[tracks.cache_keys[position]] = views
        save_chroma_cache(cache)
    return [dict(zip(CHROMA_SOURCES, cache[key])) for key in tracks.cache_keys]


def build_features(tracks: TrackSet) -> dict[str, np.ndarray]:
    summary = candidate_summary(tracks.log_probabilities)
    views = track_chroma(tracks)
    acoustic = np.stack(
        [
            acoustic_features(track_views, int(first_class), int(second_class))
            for track_views, first_class, second_class in zip(
                views, summary["first"], summary["second"]
            )
        ]
    )
    features = np.column_stack([summary["cnn_features"], acoustic])
    return {
        "features": features,
        "first": summary["first"],
        "second": summary["second"],
        "references": tracks.references,
    }


class LogisticModel:
    kind = "logistic"

    def __init__(self, regularization: float) -> None:
        self.regularization = regularization

    def fit(self, features: np.ndarray, targets: np.ndarray) -> LogisticModel:
        self.mean = features.mean(axis=0)
        self.scale = features.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        self.estimator = LogisticRegression(C=self.regularization, max_iter=5000)
        self.estimator.fit((features - self.mean) / self.scale, targets)
        return self

    def probability(self, features: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba((features - self.mean) / self.scale)[:, 1]

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "regularization": self.regularization,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.estimator.coef_[0].tolist(),
            "intercept": float(self.estimator.intercept_[0]),
        }


class SavedLogistic:
    def __init__(self, stored: dict) -> None:
        self.mean = np.array(stored["mean"])
        self.scale = np.array(stored["scale"])
        self.coefficients = np.array(stored["coefficients"])
        self.intercept = stored["intercept"]

    def probability(self, features: np.ndarray) -> np.ndarray:
        scores = ((features - self.mean) / self.scale) @ self.coefficients + self.intercept
        return 1.0 / (1.0 + np.exp(-scores))


class BoostedModel:
    kind = "gbt"

    def __init__(self, regularization: float) -> None:
        self.regularization = regularization

    def fit(self, features: np.ndarray, targets: np.ndarray) -> BoostedModel:
        self.estimator = HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=150,
            learning_rate=0.05,
            min_samples_leaf=20,
            l2_regularization=self.regularization,
            random_state=SEED,
        )
        self.estimator.fit(features, targets)
        return self

    def probability(self, features: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(features)[:, 1]


MODEL_CLASSES = {"logistic": LogisticModel, "gbt": BoostedModel}
GBT_REGULARIZATION_GRID = (0.0, 1.0, 10.0)


def eligible_rows(data: dict[str, np.ndarray]) -> np.ndarray:
    return (data["references"] == data["first"]) | (data["references"] == data["second"])


def fit_model(
    kind: str, regularization: float, data: dict[str, np.ndarray], rows: np.ndarray
) -> LogisticModel | BoostedModel:
    selected = rows & eligible_rows(data)
    targets = (data["references"][selected] == data["second"][selected]).astype(int)
    return MODEL_CLASSES[kind](regularization).fit(data["features"][selected], targets)


def cross_validation_groups(data: dict[str, np.ndarray], groups: np.ndarray | None) -> np.ndarray:
    if groups is not None:
        return groups
    targets = np.where(data["references"] == data["second"], 1, 0)
    splitter = StratifiedKFold(n_splits=CROSS_VALIDATION_SPLITS, shuffle=True, random_state=SEED)
    assigned = np.zeros(len(targets), dtype=int)
    for fold_index, (_, test_rows) in enumerate(splitter.split(np.zeros(len(targets)), targets)):
        assigned[test_rows] = fold_index
    return assigned


def out_of_fold_probability(
    kind: str, regularization: float, data: dict[str, np.ndarray], groups: np.ndarray
) -> np.ndarray:
    probability = np.zeros(len(groups))
    for group in np.unique(groups):
        held = groups == group
        model = fit_model(kind, regularization, data, ~held)
        probability[held] = model.probability(data["features"][held])
    return probability


def log_loss(probability: np.ndarray, targets: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return float(-np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped)))


def override_outcome(
    data: dict[str, np.ndarray], probability: np.ndarray, threshold: float
) -> dict:
    overrides = probability > threshold
    right = int((overrides & (data["references"] == data["second"])).sum())
    wrong = int((overrides & (data["references"] == data["first"])).sum())
    total = len(probability)
    correct_before = int((data["references"] == data["first"]).sum())
    correct_after = correct_before + right - wrong
    return {
        "tracks": total,
        "exact_before": round(100 * correct_before / total, 2),
        "exact_after": round(100 * correct_after / total, 2),
        "overrides": int(overrides.sum()),
        "overrides_right": right,
        "overrides_wrong": wrong,
        "overrides_neither": int(overrides.sum()) - right - wrong,
    }


def choose_threshold(data: dict[str, np.ndarray], probability: np.ndarray) -> tuple[float, int]:
    """Highest-net-gain threshold; ties go to the higher (more conservative) threshold."""
    best_threshold = NEVER_OVERRIDE
    best_gain = 0
    for threshold in THRESHOLD_GRID:
        outcome = override_outcome(data, probability, float(threshold))
        gain = outcome["overrides_right"] - outcome["overrides_wrong"]
        if gain > best_gain or (gain == best_gain and gain > 0 and threshold > best_threshold):
            best_gain = gain
            best_threshold = float(threshold)
    return best_threshold, best_gain


def select_and_fit(
    kind: str, data: dict[str, np.ndarray], groups: np.ndarray, verbose: bool
) -> dict:
    """Choose regularization by out-of-fold log loss and the threshold by out-of-fold net gain, then refit on all rows."""
    eligible = eligible_rows(data)
    targets = (data["references"] == data["second"]).astype(int)
    grid = REGULARIZATION_GRID if kind == "logistic" else GBT_REGULARIZATION_GRID
    candidates = []
    for regularization in grid:
        probability = out_of_fold_probability(kind, regularization, data, groups)
        loss = log_loss(probability[eligible], targets[eligible])
        candidates.append((loss, regularization, probability))
        if verbose:
            print(
                f"  {kind} regularization={regularization}: out-of-fold log loss {loss:.4f}",
                flush=True,
            )
    loss, regularization, probability = min(candidates, key=lambda candidate: candidate[0])
    threshold, gain = choose_threshold(data, probability)
    model = fit_model(kind, regularization, data, np.ones(len(groups), dtype=bool))
    return {
        "model": model,
        "regularization": regularization,
        "threshold": threshold,
        "cv_log_loss": round(loss, 4),
        "cv_outcome": override_outcome(data, probability, threshold),
        "cv_net_gain": gain,
        "cv_probability": probability,
    }


def candidate_statistics(data: dict[str, np.ndarray]) -> dict:
    total = len(data["references"])
    in_top_two = eligible_rows(data)
    second_right = data["references"] == data["second"]
    return {
        "tracks": total,
        "top1_exact": round(100 * float((data["references"] == data["first"]).mean()), 2),
        "top2_coverage": round(100 * float(in_top_two.mean()), 2),
        "second_is_right": int(second_right.sum()),
        "eligible_rows": int(in_top_two.sum()),
    }


def command_fit(arguments: argparse.Namespace) -> None:
    tracks = load_fold_files(arguments.folds_glob)
    data = build_features(tracks)
    groups = cross_validation_groups(data, tracks.groups)
    statistics = candidate_statistics(data)
    print(json.dumps(statistics), flush=True)
    results = {kind: select_and_fit(kind, data, groups, True) for kind in arguments.compare}
    for kind, result in results.items():
        print(
            f"{kind}: threshold {result['threshold']} cv {json.dumps(result['cv_outcome'])}",
            flush=True,
        )
    chosen = results[arguments.model_type]
    output = Path(arguments.out)
    stored = {
        "feature_names": feature_names(),
        "model_type": arguments.model_type,
        "threshold": chosen["threshold"],
        "regularization": chosen["regularization"],
        "fitting_data": {"folds_glob": arguments.folds_glob, **statistics},
        "cross_validation": {
            kind: {
                "regularization": result["regularization"],
                "threshold": result["threshold"],
                "log_loss": result["cv_log_loss"],
                "outcome": result["cv_outcome"],
            }
            for kind, result in results.items()
        },
    }
    if arguments.model_type == "logistic":
        stored["logistic"] = chosen["model"].to_json()
    else:
        pickle_path = output.with_suffix(".gbt.pkl")
        pickle_path.write_bytes(pickle.dumps(chosen["model"].estimator))
        stored["gbt_pickle"] = str(pickle_path.resolve())
    if "logistic" in results and arguments.model_type != "logistic":
        stored["logistic"] = results["logistic"]["model"].to_json()
    output.write_text(json.dumps(stored, indent=2))
    print(f"wrote {output}", flush=True)


def load_saved_model(path: str) -> tuple[dict, ProbabilityModel]:
    stored = json.loads(Path(path).read_text())
    if stored["feature_names"] != feature_names():
        raise SystemExit("feature names in the model file do not match this code")
    if stored["model_type"] == "logistic":
        return stored, SavedLogistic(stored["logistic"])
    estimator = pickle.loads(Path(stored["gbt_pickle"]).read_bytes())

    class SavedBoosted:
        def probability(self, features: np.ndarray) -> np.ndarray:
            return estimator.predict_proba(features)[:, 1]

    return stored, SavedBoosted()


def mirex_scores(predictions: np.ndarray, references: np.ndarray) -> float:
    from drone.keys import score_mirex

    _, mirex = score_mirex(
        [int(value) for value in predictions], [int(value) for value in references]
    )
    return round(100 * mirex, 2)


def command_apply(arguments: argparse.Namespace) -> None:
    stored, model = load_saved_model(arguments.model)
    tracks = load_logits_file(arguments.logits, arguments.key, arguments.panel)
    data = build_features(tracks)
    probability = model.probability(data["features"])
    threshold = stored["threshold"]
    outcome = override_outcome(data, probability, threshold)
    reranked = np.where(probability > threshold, data["second"], data["first"])
    report = {
        "panel": arguments.panel,
        "array": arguments.key,
        "threshold": threshold,
        **candidate_statistics(data),
        **outcome,
        "mirex_before": mirex_scores(data["first"], data["references"]),
        "mirex_after": mirex_scores(reranked, data["references"]),
    }
    print(json.dumps(report, indent=2))


def command_dev(arguments: argparse.Namespace) -> None:
    """Nested cross-validation on one panel: every reported number comes from tracks unseen by its model and threshold."""
    tracks = load_logits_file(arguments.logits, arguments.key, arguments.panel)
    data = build_features(tracks)
    print("DEVELOPMENT ONLY", json.dumps(candidate_statistics(data)), flush=True)
    outer_groups = cross_validation_groups(data, None)
    report = {}
    for kind in arguments.compare:
        probability = np.zeros(len(outer_groups))
        decision_threshold = np.zeros(len(outer_groups))
        thresholds = []
        for group in np.unique(outer_groups):
            held = outer_groups == group
            inner_data = {name: values[~held] for name, values in data.items()}
            inner = select_and_fit(
                kind, inner_data, cross_validation_groups(inner_data, None), False
            )
            probability[held] = inner["model"].probability(data["features"][held])
            decision_threshold[held] = inner["threshold"]
            thresholds.append(inner["threshold"])
        overrides = probability > decision_threshold
        outcome = override_outcome(data, np.where(overrides, 1.0, 0.0), 0.5)
        report[kind] = {"outer_thresholds": thresholds, **outcome}
        print(
            f"{kind} (development only, nested CV on {arguments.panel}): {json.dumps(report[kind])}",
            flush=True,
        )
        whole = select_and_fit(kind, data, outer_groups, False)
        if kind == "logistic":
            weights = sorted(
                zip(feature_names(), whole["model"].estimator.coef_[0]),
                key=lambda pair: -abs(pair[1]),
            )
            print(
                "  largest standardized weights:",
                [(name, round(float(weight), 3)) for name, weight in weights[:8]],
            )
        report[kind]["single_level_cv"] = whole["cv_outcome"]
        report[kind]["single_level_threshold"] = whole["threshold"]
    if arguments.out:
        Path(arguments.out).write_text(
            json.dumps({"development_only": True, "panel": arguments.panel, **report}, indent=2)
        )
        print(f"wrote {arguments.out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fit_parser = commands.add_parser("fit")
    fit_parser.add_argument("--folds-glob", required=True)
    fit_parser.add_argument("--out", required=True)
    fit_parser.add_argument("--model-type", choices=sorted(MODEL_CLASSES), default="logistic")
    fit_parser.add_argument(
        "--compare", nargs="+", choices=sorted(MODEL_CLASSES), default=["logistic", "gbt"]
    )
    apply_parser = commands.add_parser("apply")
    apply_parser.add_argument("--model", required=True)
    apply_parser.add_argument("--logits", required=True)
    apply_parser.add_argument("--key", default="logits")
    apply_parser.add_argument("--panel")
    dev_parser = commands.add_parser("dev")
    dev_parser.add_argument("--logits", required=True)
    dev_parser.add_argument("--key", required=True)
    dev_parser.add_argument("--panel", required=True)
    dev_parser.add_argument(
        "--compare", nargs="+", choices=sorted(MODEL_CLASSES), default=["logistic", "gbt"]
    )
    dev_parser.add_argument("--out")
    arguments = parser.parse_args()
    if arguments.command == "fit" and arguments.model_type not in arguments.compare:
        arguments.compare.append(arguments.model_type)
    if arguments.command == "dev" and arguments.panel != "valid":
        raise SystemExit("dev is restricted to the valid panel")
    {"fit": command_fit, "apply": command_apply, "dev": command_dev}[arguments.command](arguments)


if __name__ == "__main__":
    main()
