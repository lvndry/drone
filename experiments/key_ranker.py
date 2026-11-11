"""One frozen CPU grouped-ranking falsification study, separate from production."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from catboost import CatBoostRanker

import numpy as np

from drone.features import FEATURE_SPEC as CQT_SPEC
from drone.features import N_TOTAL, audio_digest, window
from drone.keys import LABEL_SCHEMA
from experiments.ema import read_reference
from experiments.k10_multidomain import FROZEN_SHA256, domain_metrics, validate_rows
from experiments.paired import compare, movements, verify_control
from experiments.small import atomic_json
from experiments.tonal_features import (
    FEATURE_NAMES,
    FEATURE_SPEC,
    candidate_features,
    grouped_targets,
)

VERSION = "tonal-key-ranker-v1"
CATBOOST_VERSION = "1.2.10"
ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "experiments/studies/tonal-key-ranker"
PROTOCOL = STUDY / "PROTOCOL.md"
REVIEWED = ROOT / "experiments/studies/k10-capacity/data/reviewed-sources.json"
SOURCE_FILES = (
    "experiments/key_ranker.py",
    "experiments/tonal_features.py",
    "experiments/paired.py",
    "experiments/ema.py",
    "experiments/k10_multidomain.py",
    "experiments/small.py",
    "drone/checkpoints.py",
    "drone/features.py",
    "drone/keys.py",
    "pyproject.toml",
    "uv.lock",
)


def configuration(seed: int) -> dict[str, Any]:
    if seed not in (13, 37, 73):
        raise ValueError("Only canonical pilot13 and conditional37/73")
    return dict(
        loss_function="QuerySoftMax:beta=1",
        iterations=128,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=3,
        grow_policy="SymmetricTree",
        boosting_type="Plain",
        bootstrap_type="No",
        random_strength=1,
        border_count=64,
        task_type="CPU",
        thread_count=4,
        random_seed=seed,
        use_best_model=False,
        allow_writing_files=False,
    )


def source_identity() -> dict[str, Any]:
    return dict(
        source_sha256={name: audio_digest(ROOT / name) for name in SOURCE_FILES},
        protocol_sha256=audio_digest(PROTOCOL),
        reviewed_sources_sha256=audio_digest(REVIEWED),
        versions={
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "catboost",
                "scipy",
                "pandas",
                "plotly",
                "graphviz",
                "six",
                "matplotlib",
            )
        },
    )


class Budget:
    """Sample RSS/time/output; an exceeded budget makes the study inconclusive."""

    def __init__(self, seconds: int, output: Path) -> None:
        self.start, self.seconds, self.output = time.monotonic(), seconds, output
        self.peak_bytes = 0
        self.failure: BaseException | None = None

    def check(self) -> None:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.peak_bytes = int(rss * (1 if sys.platform == "darwin" else 1024))
        if self.peak_bytes > 2 * 1024**3:
            raise MemoryError("Study exceeded2GiB process peak RSS")
        if time.monotonic() - self.start > self.seconds:
            raise TimeoutError(f"Study exceeded fixed {self.seconds}s budget")
        if sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file()) > 1024**3:
            raise OSError("Study exceeded1GiB output storage")

    def after_iteration(self, info: object) -> bool:
        try:
            self.check()
        except (TimeoutError, MemoryError, OSError) as error:
            self.failure = error
            return False
        return True

    def report(self) -> dict[str, Any]:
        self.check()
        return dict(
            elapsed_seconds=time.monotonic() - self.start,
            peak_rss_bytes=self.peak_bytes,
            budget_seconds=self.seconds,
        )


def execution_guard() -> dict[str, Any]:
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    env = {name: os.environ.get(name) for name in names}
    if any(value != "4" for value in env.values()):
        raise ValueError("All four numerical thread environment variables must equal4")
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if sys.platform == "linux" and (not affinity or not set(affinity) <= {6, 7, 8, 9}):
        raise ValueError("Linux study must be pinned to CPU6–9")
    if importlib.metadata.version("catboost") != CATBOOST_VERSION:
        raise ValueError("CatBoost package must match pinned1.2.10")
    return dict(environment=env, affinity=affinity, task_type="CPU")


def references(
    args: argparse.Namespace, rows: list[dict[str, Any]], manifest_sha: str
) -> dict[str, Any]:
    joint = read_reference(args.reference_run, "joint", args.seed)
    bp = read_reference(args.reference_bp, "control", args.seed)
    reviewed = json.loads(REVIEWED.read_bytes())
    if joint["source_sha256"] != reviewed["archived"]["sources"]:
        raise ValueError("Archived joint source must be frozen673")
    for key in ("config", "feature_manifest_sha256", "validation_ids", "source_sha256"):
        if joint[key] != bp[key]:
            raise ValueError(f"Archived BP/joint mismatch: {key}")
    if joint["feature_manifest_sha256"] != manifest_sha:
        raise ValueError("Prepared feature manifest differs from archived controls")
    for role, field in (("train", "fit_ids"), ("validation", "validation_ids")):
        if [r["id"] for r in rows if r["role"] == role] != joint[field]:
            raise ValueError(f"Archived {field} differs")
    if [r["id"] for r in rows if r["role"] == "train" and r["domain"] == "beatport"] != bp[
        "fit_ids"
    ]:
        raise ValueError("BP fitting identities differ")
    control, proof = verify_control(
        args.paired_control,
        args.reference_run,
        args.reference_checkpoint or args.reference_run.with_name("best.pt"),
        [reviewed["reusable_control"]["sources"]],
    )
    result_path = args.paired_control.with_name("structured-result.json")
    replay = json.loads(result_path.read_bytes())
    artifact = args.paired_control.with_name(replay["validation_artifact"])
    if (
        replay.get("artifact_version") != "k10-structured-targets-v1"
        or replay.get("status") != "complete"
        or replay.get("kind") != "control"
        or replay.get("seed") != args.seed
        or not replay.get("control_reproduced")
        or replay.get("run_sha256") != audio_digest(args.paired_control)
        or replay.get("checkpoint_sha256") != proof["checkpoint_sha256"]
        or replay.get("validation_sha256") != audio_digest(artifact)
    ):
        raise ValueError("Need matching completed structured exact-replay proof")
    validation = json.loads(artifact.read_bytes())
    val_rows = [r for r in rows if r["role"] == "validation"]
    y = np.array([r["ref"] for r in val_rows])
    logits = np.asarray(validation["logits"])
    if (
        validation["ids"] != [r["id"] for r in val_rows]
        or validation["audio_sha256"] != [r["sha256"] for r in val_rows]
        or validation["refs"] != y.tolist()
        or logits.shape != (376, 24)
        or not np.isfinite(logits).all()
        or domain_metrics(logits.argmax(1), y, val_rows) != control["domain_metrics"]
    ):
        raise ValueError("Control validation identities/logits differ")
    if audio_digest(args.reference_bp.with_name("best.pt")) != bp["checkpoint_sha256"]:
        raise ValueError("Archived BP checkpoint hash differs")
    return dict(
        control_metrics=control["domain_metrics"],
        bp_metrics=bp["domain_metrics"],
        control_proof=dict(
            proof, replay_sha256=audio_digest(result_path), validation_sha256=audio_digest(artifact)
        ),
        bp_run_sha256=audio_digest(args.reference_bp),
        bp_checkpoint_sha256=bp["checkpoint_sha256"],
    )


def prepare(args: argparse.Namespace) -> None:
    if args.seed != 13:
        raise ValueError("Prepare once with canonical seed13")
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    args.output.mkdir(parents=True)
    budget = Budget(600, args.output)
    execution = execution_guard()
    identity = source_identity()
    manifest_path = args.data / "manifest.json"
    manifest_sha = audio_digest(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    if (
        not manifest["complete"]
        or manifest["feature_spec"] != CQT_SPEC
        or manifest["label_schema"] != LABEL_SCHEMA
        or manifest["frozen_manifest_sha256"] != FROZEN_SHA256
    ):
        raise ValueError("Need compatible complete quarantined development cache")
    rows = manifest["rows"]
    validate_rows(rows)
    counts = {
        (d, role): sum(r["domain"] == d and r["role"] == role for r in rows)
        for d in ("beatport", "fmak")
        for role in ("train", "validation")
    }
    if list(counts.values()) != [1230, 126, 976, 250]:
        raise ValueError("Need exact2206 fitting/376 validation identities")
    refs = references(args, rows, manifest_sha)
    identity.update(
        rows_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        feature_manifest_sha256=manifest_sha,
        feature_spec=FEATURE_SPEC,
        feature_names=list(FEATURE_NAMES),
        execution=execution,
    )
    atomic_json(args.output / "preparation-identity.json", identity)
    (args.output / "PROTOCOL.md").write_bytes(PROTOCOL.read_bytes())
    (args.output / "cache-manifest.json").write_bytes(manifest_path.read_bytes())
    values = np.empty((len(rows), 24, 24), dtype=np.float32)
    for i, row in enumerate(rows):
        budget.check()
        path = (args.data / row["cache"]).resolve()
        if (
            not path.is_relative_to(args.data.resolve())
            or audio_digest(path) != row["cache_sha256"]
        ):
            raise ValueError("Cache outside data root or content changed")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float16 or array.shape != (N_TOTAL, row["frames"]):
            raise ValueError("Cache shape/dtype differs")
        values[i] = candidate_features(window(array), check=budget.check)
    np.save(args.output / "features.npy", values, allow_pickle=False)
    safe_rows = [
        {k: r[k] for k in ("id", "sha256", "ref", "role", "domain", "cache_sha256", "frames")}
        for r in rows
    ]
    atomic_json(
        args.output / "prepared.json",
        dict(
            identity=identity,
            rows=safe_rows,
            feature_array_sha256=audio_digest(args.output / "features.npy"),
            pilot_references=refs,
            resources=budget.report(),
            complete=True,
        ),
    )


def save_artifact(model: CatBoostRanker, output: Path, identity: dict[str, Any], seed: int) -> None:
    model.save_model(str(output / "model.cbm"), format="cbm")
    atomic_json(
        output / "artifact.json",
        dict(
            artifact_version=VERSION,
            feature_spec=FEATURE_SPEC,
            feature_names=list(FEATURE_NAMES),
            label_schema=LABEL_SCHEMA,
            catboost_version=CATBOOST_VERSION,
            config=configuration(seed),
            tree_count=128,
            model_sha256=audio_digest(output / "model.cbm"),
            identity=identity,
        ),
    )


def load_artifact(output: Path) -> CatBoostRanker:
    from catboost import CatBoostRanker

    info = json.loads((output / "artifact.json").read_bytes())
    if (
        info["artifact_version"] != VERSION
        or info["feature_spec"] != FEATURE_SPEC
        or info["feature_names"] != list(FEATURE_NAMES)
        or info["label_schema"] != LABEL_SCHEMA
        or info["catboost_version"] != importlib.metadata.version("catboost")
        or info["catboost_version"] != CATBOOST_VERSION
        or info["config"] != configuration(info["config"]["random_seed"])
        or info["tree_count"] != 128
        or audio_digest(output / "model.cbm") != info["model_sha256"]
    ):
        raise ValueError("Incompatible or changed ranker artifact")
    model = CatBoostRanker()
    model.load_model(str(output / "model.cbm"), format="cbm")
    resolved = model.get_all_params()
    expected = configuration(info["config"]["random_seed"])
    for name in (
        "loss_function",
        "depth",
        "l2_leaf_reg",
        "grow_policy",
        "boosting_type",
        "bootstrap_type",
        "random_strength",
        "border_count",
        "task_type",
        "random_seed",
    ):
        if resolved.get(name) != expected[name]:
            raise ValueError(f"Model resolved parameter differs: {name}")
    if not np.isclose(resolved["learning_rate"], expected["learning_rate"], rtol=1e-7, atol=0):
        raise ValueError("Model learning rate differs")
    if model.tree_count_ != 128 or model.feature_names_ != list(FEATURE_NAMES):
        raise ValueError("Ranker model shape/features differ")
    return model


def fit(args: argparse.Namespace) -> None:
    from catboost import CatBoostRanker, Pool

    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    args.output.mkdir(parents=True)
    budget = Budget(300, args.output)
    execution = execution_guard()
    prepared = json.loads((args.prepared / "prepared.json").read_bytes())
    identity = source_identity()
    if not prepared["complete"] or any(identity[k] != prepared["identity"][k] for k in identity):
        raise ValueError("Prepared features source/protocol/package identities changed")
    if audio_digest(args.prepared / "features.npy") != prepared["feature_array_sha256"]:
        raise ValueError("Prepared features changed")
    rows = prepared["rows"]
    original = args.prepared / "cache-manifest.json"
    if audio_digest(original) != prepared["identity"]["feature_manifest_sha256"]:
        raise ValueError("Copied immutable cache manifest changed")
    original_rows = json.loads(original.read_bytes())["rows"]
    if [{k: r[k] for k in rows[0]} for r in original_rows] != rows:
        raise ValueError("Prepared rows differ from immutable cache manifest")
    validate_rows(rows)
    refs = references(args, rows, prepared["identity"]["feature_manifest_sha256"])
    if args.seed != 13:
        if args.pilot is None:
            raise ValueError("Confirmation requires passing seed13 pilot")
        pilot = json.loads(args.pilot.read_bytes())
        if (
            pilot["seed"] != 13
            or pilot["artifact_version"] != VERSION
            or pilot["status"] != "complete"
            or not pilot["method_gate"]["qualifies_for_confirmation"]
            or pilot["identity"] != prepared["identity"]
            or pilot["feature_array_sha256"] != prepared["feature_array_sha256"]
        ):
            raise ValueError("Pilot did not qualify or study identities changed")
    features = np.load(args.prepared / "features.npy", allow_pickle=False)
    if (
        features.shape != (2582, 24, 24)
        or features.dtype != np.float32
        or not np.isfinite(features).all()
    ):
        raise ValueError("Invalid prepared feature array")
    train = np.array([r["role"] == "train" for r in rows])
    val_rows = [r for r in rows if r["role"] == "validation"]
    y = np.array([r["ref"] for r in rows], dtype=np.int64)
    targets, groups = grouped_targets(y[train])
    pool = Pool(
        features[train].reshape(-1, 24),
        label=targets,
        group_id=groups,
        feature_names=list(FEATURE_NAMES),
    )
    atomic_json(
        args.output / "fit-identity.json",
        dict(
            identity=prepared["identity"],
            config=configuration(args.seed),
            references=refs,
            execution=execution,
            prepared_sha256=audio_digest(args.prepared / "prepared.json"),
        ),
    )
    model = CatBoostRanker(**configuration(args.seed))
    model.fit(pool, verbose=False, callbacks=[budget])
    if budget.failure is not None:
        raise budget.failure
    budget.check()
    save_artifact(model, args.output, prepared["identity"], args.seed)
    loaded = load_artifact(args.output)
    x = features[~train].reshape(-1, 24)
    logits = np.asarray(loaded.predict(x, thread_count=4)).reshape(376, 24)
    if not np.array_equal(logits, np.asarray(model.predict(x, thread_count=4)).reshape(376, 24)):
        raise ValueError("Ranker artifact roundtrip changed scores")
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite validation scores")
    metrics = domain_metrics(logits.argmax(1), y[~train], val_rows)
    atomic_json(
        args.output / "validation.json",
        dict(
            ids=[r["id"] for r in val_rows],
            audio_sha256=[r["sha256"] for r in val_rows],
            refs=y[~train].tolist(),
            logits=logits.tolist(),
            predictions=logits.argmax(1).tolist(),
            metrics=metrics,
        ),
    )
    result = dict(
        artifact_version=VERSION,
        seed=args.seed,
        status="complete",
        kind="candidate",
        identity=prepared["identity"],
        config=configuration(args.seed),
        control_reproduced=True,
        feature_array_sha256=prepared["feature_array_sha256"],
        **refs,
        metrics=metrics,
        method_gate=compare(refs["control_metrics"], metrics),
        confusion_movements=movements(refs["control_metrics"], metrics, y[~train], val_rows),
        resolved_params=model.get_all_params(),
        resources=budget.report(),
        artifact_sha256=audio_digest(args.output / "artifact.json"),
        validation_sha256=audio_digest(args.output / "validation.json"),
        promoted=False,
    )
    atomic_json(args.output / "result.json", result)


def summarize(paths: list[Path]) -> dict[str, Any]:
    results = [json.loads(p.read_bytes()) for p in paths]
    if (
        len(results) != 3
        or sorted(r["seed"] for r in results) != [13, 37, 73]
        or not next(r for r in results if r["seed"] == 13)["method_gate"][
            "qualifies_for_confirmation"
        ]
    ):
        raise ValueError("Need passing pilot and exactly three declared seeds")
    for r in results:
        if (
            r["status"] != "complete"
            or r["artifact_version"] != VERSION
            or not r["control_reproduced"]
            or r["identity"] != results[0]["identity"]
            or r["feature_array_sha256"] != results[0]["feature_array_sha256"]
        ):
            raise ValueError("Unmatched/incomplete study identities")
    means = {
        name: {
            d: {
                m: statistics.mean(r[name][d][m] for r in results)
                for m in ("mirex", "mirex_asymmetric", "exact")
            }
            for d in ("beatport", "fmak")
        }
        for name in ("metrics", "control_metrics", "bp_metrics")
    }
    delta = {
        d: means["metrics"][d]["mirex"] - means["bp_metrics"][d]["mirex"]
        for d in ("beatport", "fmak")
    }
    safeguard = all(
        means["metrics"][d][m] >= means["control_metrics"][d][m]
        for d in ("beatport", "fmak")
        for m in ("mirex", "exact")
    )
    original = delta["fmak"] >= 0.02 and delta["beatport"] >= 0
    method_gate = compare(means["control_metrics"], means["metrics"])
    return dict(
        artifact_version=VERSION,
        means=means,
        original_hurdle_delta=delta,
        original_hurdle_pass=original,
        raw_joint_safeguard=safeguard,
        method_gate=method_gate,
        qualifies_for_external_freeze=original
        and safeguard
        and method_gate["qualifies_for_confirmation"],
        canonical_seed=13,
        promoted=False,
        uncertainty="Reused development comparisons, not independent generalization evidence",
    )


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "fit"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--seed", type=int, choices=(13, 37, 73), default=13)
        p.add_argument("--reference-run", type=Path, required=True)
        p.add_argument("--reference-bp", type=Path, required=True)
        p.add_argument("--reference-checkpoint", type=Path)
        p.add_argument("--paired-control", type=Path, required=True)
        if name == "prepare":
            p.add_argument("--data", type=Path, required=True)
        else:
            p.add_argument("--prepared", type=Path, required=True)
            p.add_argument("--pilot", type=Path)
    p = sub.add_parser("summarize")
    p.add_argument("results", type=Path, nargs=3)
    args = parser.parse_args()
    if args.command == "summarize":
        print(json.dumps(summarize(args.results), indent=2))
        return
    try:
        (prepare if args.command == "prepare" else fit)(args)
    except (TimeoutError, MemoryError, OSError) as error:
        atomic_json(
            args.output / "inconclusive.json",
            dict(
                artifact_version=VERSION,
                status="resource_inconclusive",
                error=str(error),
                promoted=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
