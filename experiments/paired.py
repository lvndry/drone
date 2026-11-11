"""Shared evidence, scoring and gates for bounded paired classifier studies."""

from __future__ import annotations

import ast
import copy
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drone.checkpoints import load_checkpoint
from drone.features import audio_digest
from drone.keys import key_relation
from experiments.ema import assert_reproduced, evaluate_logits, preserve_rng, read_reference
from experiments.research_models import CheckpointLoader
from experiments.small import atomic_json, signature


def training_math(source: str) -> dict[str, str]:
    """Audit explicit math/selection components, excluding metadata and optional probes."""
    train = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "train"
    )
    epoch = next(
        n
        for n in train.body
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "epoch"
    )
    batch = next(n for n in epoch.body if isinstance(n, ast.For))
    statements = []
    for statement in batch.body:
        if isinstance(statement, ast.If):
            condition = ast.unparse(statement.test)
            if condition == "targets is None":
                statements.extend(statement.body)
                continue
            if condition == "probe is not None" or "time.monotonic()" in condition:
                continue
        statements.append(statement)
    components: dict[str, object] = dict(
        horizon=epoch.iter,
        batch=batch.iter,
        batch_math=statements,
    )
    selected = {
        "fitting",
        "validation",
        "val_features",
        "val_refs",
        "rng",
        "optimizer",
        "scheduler",
    }
    for node in train.body + epoch.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            if name in selected or name in {"order", "pred"}:
                components[name] = node.value
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "score_mirex"
        ):
            components["validation_score"] = node.value
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            name = ast.unparse(node.value.func)
            if name in {"torch.manual_seed", "scheduler.step", "record['history'].append"}:
                components[name] = node.value
        if isinstance(node, ast.If):
            condition = ast.unparse(node.test)
            if condition in {"score > best", "epoch - best_epoch >= 25"}:
                components[condition] = node.test

    def encode(value: object) -> str:
        if isinstance(value, ast.AST):
            return ast.dump(value, include_attributes=False)
        if isinstance(value, list):
            return "|".join(ast.dump(n, include_attributes=False) for n in value)
        raise TypeError("Unexpected math audit component")

    return {name: signature(encode(value)) for name, value in components.items()}


def ema_evidence_math(source: str) -> dict[str, str]:
    """Permit path-only loader changes and explicit budget injection, not scoring math."""
    tree = ast.parse(source)
    hashes = {}
    for original in tree.body:
        if isinstance(original, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "BUDGET_SECONDS" for t in original.targets
        ):
            hashes["default_budget"] = signature(ast.dump(original.value))
        if not isinstance(original, ast.FunctionDef) or original.name not in {
            "assert_reproduced",
            "read_reference",
            "preserve_rng",
            "evaluate_logits",
        }:
            continue
        node = copy.deepcopy(original)
        if node.name == "evaluate_logits":
            node.args.kwonlyargs = []
            node.args.kw_defaults = []
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "check_budget"
                ):
                    call.args = call.args[:1]
        hashes[node.name] = signature(ast.dump(node, include_attributes=False))
    return hashes


def verify_control(
    run_path: Path,
    archived_path: Path,
    checkpoint: Path,
    allowed_sources: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-verify immutable history, predictions and actual tensors; do not trust a pass flag."""
    archived = json.loads(archived_path.read_bytes())
    control = read_reference(run_path, "joint", archived["seed"])
    if control["source_sha256"] not in allowed_sources:
        raise ValueError("Control source is not an approved frozen source")
    if control.get("loss_spec", {"name": "hard-cross-entropy", "smoothing": 0.0}) != {
        "name": "hard-cross-entropy",
        "smoothing": 0.0,
    }:
        raise ValueError("Control must use hard CE")
    actual = run_path.with_name("best.pt")
    if audio_digest(actual) != control["checkpoint_sha256"]:
        raise ValueError("Control checkpoint hash differs from run")
    if audio_digest(checkpoint) != archived["checkpoint_sha256"]:
        raise ValueError("Archived checkpoint hash differs from run")
    archived_state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    payload = torch.load(actual, map_location="cpu", weights_only=True)
    if payload.get("model_config", {}).get("width", 1.0) != 1.0:
        raise ValueError("Control checkpoint width must be1")
    assert_reproduced(
        control, archived, current_state=payload["state_dict"], reference_state=archived_state
    )
    return control, dict(
        run_sha256=audio_digest(run_path),
        checkpoint_sha256=audio_digest(actual),
        archived_run_sha256=audio_digest(archived_path),
        archived_checkpoint_sha256=audio_digest(checkpoint),
        exact_history_predictions_state=True,
    )


def score_selected(
    output: Path,
    record: dict[str, Any],
    features: list[np.ndarray],
    refs: np.ndarray,
    rows: list[dict[str, Any]],
    device: str,
    start: float,
    budget_seconds: int,
    width: float = 1.0,
    *,
    loader: CheckpointLoader | None = None,
) -> dict[str, Any]:
    from experiments.k10_multidomain import domain_metrics

    if "validation_ids" in record and record["validation_ids"] != [r["id"] for r in rows]:
        raise ValueError("Final scoring validation identities differ from the selected run")
    with preserve_rng():
        if loader is None:
            model, payload = load_checkpoint(output / "best.pt", device)
            if payload["model_config"]["width"] != width:
                raise ValueError("Selected checkpoint width metadata differs")
        else:
            model, payload = loader(output / "best.pt", device)
        logits = evaluate_logits(model, features, device, start, budget_seconds=budget_seconds)
        metrics = domain_metrics(logits.argmax(1), refs, rows)
        if metrics != record["domain_metrics"]:
            raise ValueError("Selected logits differ from recorded validation predictions")
        path = output / "validation.json"
        atomic_json(
            path,
            dict(
                ids=[r["id"] for r in rows],
                audio_sha256=[r["sha256"] for r in rows],
                refs=refs.tolist(),
                logits=logits.tolist(),
                predictions=logits.argmax(1).tolist(),
                metrics=metrics,
            ),
        )
        return dict(
            metrics=metrics,
            validation_artifact=path.name,
            validation_sha256=audio_digest(path),
            checkpoint_sha256=audio_digest(output / "best.pt"),
        )


def compare(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    delta = {
        d: {m: candidate[d][m] - control[d][m] for m in ("mirex", "exact")}
        for d in ("beatport", "fmak")
    }
    passes = (
        delta["fmak"]["mirex"] >= 0.01
        and delta["beatport"]["mirex"] >= 0
        and all(delta[d]["exact"] >= 0 for d in delta)
    )
    return dict(delta=delta, qualifies_for_confirmation=passes)


def movements(
    control: dict[str, Any], candidate: dict[str, Any], refs: np.ndarray, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    categories = ["exact", "fifth", "relative", "parallel", "other"]
    result = {}
    for domain in ("beatport", "fmak"):
        indices = [i for i, r in enumerate(rows) if r["domain"] == domain]
        ids = [rows[i]["id"] for i in indices]
        if control[domain]["ids"] != ids or candidate[domain]["ids"] != ids:
            raise ValueError("Paired confusion identities differ")
        result[domain] = {}
        for name, symmetric in (
            ("repo_symmetric_fifths", True),
            ("existing_asymmetric_ascending_fifth", False),
        ):
            counts = np.zeros((5, 5), dtype=np.int64)
            for old, new, i in zip(
                control[domain]["predictions"],
                candidate[domain]["predictions"],
                indices,
                strict=True,
            ):
                a = categories.index(key_relation(old, int(refs[i]), symmetric=symmetric))
                b = categories.index(key_relation(new, int(refs[i]), symmetric=symmetric))
                counts[a, b] += 1
            result[domain][name] = dict(
                categories=categories,
                row_control_column_candidate=counts.tolist(),
                exact_promotions=int(counts[1:, 0].sum()),
                exact_demotions=int(counts[0, 1:].sum()),
            )
    return result


def summarize(results: list[dict[str, Any]], version: str) -> dict[str, Any]:
    seeds = {r["seed"]: r for r in results}
    if (
        len(results) != 3
        or sorted(seeds) != [13, 37, 73]
        or not seeds[13]["method_gate"]["qualifies_for_confirmation"]
    ):
        raise ValueError("Need passing pilot and exactly declared candidate seeds13/37/73")
    for r in results:
        if (
            r["status"] != "complete"
            or r["artifact_version"] != version
            or r["kind"] != "candidate"
            or not r["control_reproduced"]
        ):
            raise ValueError("Need completed candidates with verified controls")
        for key in (
            "source_sha256",
            "config",
            "feature_manifest_sha256",
            "protocol_sha256",
            "architecture",
            "width",
            "budget_seconds",
        ):
            current = (
                r["identity"].get(key) if key in {"architecture", "width"} else r["identity"][key]
            )
            reference = (
                seeds[13]["identity"].get(key)
                if key in {"architecture", "width"}
                else seeds[13]["identity"][key]
            )
            if current != reference:
                raise ValueError(f"Confirmation changed {key}")
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
    gate = compare(means["control_metrics"], means["metrics"])
    delta = {
        d: means["metrics"][d]["mirex"] - means["bp_metrics"][d]["mirex"]
        for d in ("beatport", "fmak")
    }
    original = delta["fmak"] >= 0.02 and delta["beatport"] >= 0
    safeguard = all(
        means["metrics"][d][m] >= means["control_metrics"][d][m]
        for d in ("beatport", "fmak")
        for m in ("mirex", "exact")
    )
    return dict(
        artifact_version=version,
        means=means,
        method_gate=gate,
        original_hurdle_delta=delta,
        original_hurdle_pass=original,
        raw_joint_safeguard=safeguard,
        qualifies_for_external_freeze=gate["qualifies_for_confirmation"] and original and safeguard,
        confusion_movements_by_seed={str(r["seed"]): r.get("confusion_movements") for r in results},
        canonical_seed=13,
        promoted=False,
        uncertainty="Seed means on reused development data, not independent generalization evidence",
    )
