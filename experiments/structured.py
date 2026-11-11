"""One fixed loss-only K10 experiment with replayed hard-CE controls."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drone.checkpoints import load_checkpoint
from drone.features import audio_digest
from drone.keys import LABEL_SCHEMA, WEIGHTS, key_relation
from experiments.ema import (
    assert_reproduced,
    evaluate_logits,
    preserve_rng,
    read_reference,
)
from experiments.small import atomic_json

VERSION = "k10-structured-targets-v1"
STUDY = Path(__file__).parent / "studies" / "k10-structured-targets"
PROTOCOL = STUDY / "PROTOCOL.md"
REVIEWED = STUDY / "data" / "reviewed-sources.json"
HARD_LOSS = {"name": "hard-cross-entropy", "smoothing": 0.0}
SOFT_LOSS = {
    "name": "mirex-structured-cross-entropy",
    "true_mass": 0.8,
    "neighbour_mass": 0.2,
    "fifths": "both",
    "label_schema": LABEL_SCHEMA,
}


def check_budget(start: float) -> None:
    if time.monotonic() - start > 1200:
        raise TimeoutError("Structured-targets fit exceeded20min including final scoring")


def target_matrix() -> torch.Tensor:
    """Derive normalized neighbours from the shared scoring contract, not label arithmetic."""
    matrix = torch.zeros(24, 24, dtype=torch.float32)
    for ref in range(24):
        weights = [0.0 if pred == ref else WEIGHTS[key_relation(pred, ref)] for pred in range(24)]
        total = sum(weights)
        if total <= 0:
            raise ValueError("Missing harmonic neighbours")
        matrix[ref] = torch.tensor([0.2 * weight / total for weight in weights])
        matrix[ref, ref] = 0.8
    return matrix


def compare_metrics(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    delta = {
        domain: {
            metric: candidate[domain][metric] - control[domain][metric]
            for metric in ("mirex", "exact")
        }
        for domain in ("beatport", "fmak")
    }
    passes = (
        delta["fmak"]["mirex"] >= 0.01
        and delta["beatport"]["mirex"] >= 0
        and all(delta[d]["exact"] >= 0 for d in delta)
    )
    return dict(delta=delta, qualifies_for_confirmation=passes)


def confusion_movements(
    control: dict[str, Any],
    pred: np.ndarray,
    refs: np.ndarray,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Paired category transitions expose exact demotions behind weighted gains."""
    categories = ["exact", "fifth", "relative", "parallel", "other"]
    output = {}
    for domain in ("beatport", "fmak"):
        indices = [i for i, row in enumerate(rows) if row["domain"] == domain]
        if control[domain]["ids"] != [rows[i]["id"] for i in indices]:
            raise ValueError("Paired confusion identities differ")
        previous = control[domain]["predictions"]
        if len(previous) != len(indices):
            raise ValueError("Paired confusion prediction count differs")
        output[domain] = {}
        for name, symmetric in (
            ("repo_symmetric_fifths", True),
            ("existing_asymmetric_ascending_fifth", False),
        ):
            counts = np.zeros((5, 5), dtype=np.int64)
            for old, i in zip(previous, indices, strict=True):
                before = categories.index(key_relation(old, int(refs[i]), symmetric=symmetric))
                after = categories.index(
                    key_relation(int(pred[i]), int(refs[i]), symmetric=symmetric)
                )
                counts[before, after] += 1
            output[domain][name] = dict(
                row_control_column_candidate=counts.tolist(),
                categories=categories,
                control_counts=counts.sum(1).tolist(),
                candidate_counts=counts.sum(0).tolist(),
                exact_promotions=int(counts[1:, 0].sum()),
                exact_demotions=int(counts[0, 1:].sum()),
            )
    return output


def require_pair(control: dict[str, Any], candidate: dict[str, Any]) -> None:
    """Only loss may differ between fresh paired runs; both must complete."""
    for run in (control, candidate):
        if run["status"] != "complete" or run["calibration_used"] or run["final_holdout_used"]:
            raise ValueError("Pair must be complete and development-only")
    for key in (
        "arm",
        "seed",
        "config",
        "parameters",
        "source_sha256",
        "feature_manifest_sha256",
        "fit_ids",
        "validation_ids",
    ):
        if control[key] != candidate[key]:
            raise ValueError(f"Paired runs differ beyond loss: {key}")
    if control["loss_spec"] != HARD_LOSS or candidate["loss_spec"] != SOFT_LOSS:
        raise ValueError("Need declared hard-CE control and structured-target candidate")


def source_checks(sources: dict[str, str], archived: dict[str, str]) -> None:
    reviewed = json.loads(REVIEWED.read_bytes())
    if archived != reviewed["archived"]:
        raise ValueError("Reference source is not archived commit673")
    for name, digest in reviewed["reviewed_base"].items():
        if name != "experiments/k10_multidomain.py" and sources.get(name) != digest:
            raise ValueError(f"Source differs from reviewed base: {name}")


def check_pilot(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        raise ValueError("Seeds37/73 require the passing canonical seed13 candidate pilot")
    pilot = json.loads(path.read_bytes())
    if (
        pilot.get("artifact_version") != VERSION
        or pilot.get("seed") != 13
        or pilot.get("kind") != "candidate"
        or pilot.get("status") != "complete"
        or not pilot.get("control_reproduced")
        or not pilot.get("method_gate", {}).get("qualifies_for_confirmation")
    ):
        raise ValueError("Need completed passing seed13 candidate pilot")
    for key in ("source_sha256", "config", "feature_manifest_sha256"):
        if pilot["identity"][key] != record[key]:
            raise ValueError(f"Confirmation changed {key}")
    if pilot["identity"]["protocol_sha256"] != audio_digest(PROTOCOL):
        raise ValueError("Confirmation changed protocol")


class StructuredProbe:
    """Preflight and final artifacts only; shared trainer retains the entire training loop."""

    def __init__(self, args: argparse.Namespace, record: dict[str, Any]) -> None:
        self.args = args
        self.kind = "candidate" if args.structured_targets else "control"
        self.reference = read_reference(args.reference_run, "joint", args.seed)
        self.bp = read_reference(args.reference_bp, "control", args.seed)
        for key in ("config", "feature_manifest_sha256", "fit_ids", "validation_ids"):
            if record[key] != self.reference[key]:
                raise ValueError(f"Archived joint reference mismatch: {key}")
        for key in ("config", "source_sha256", "feature_manifest_sha256", "validation_ids"):
            if self.reference[key] != self.bp[key]:
                raise ValueError(f"Archived BP and joint differ: {key}")
        if (
            record["arm"] != "joint"
            or len(record["fit_ids"]) != 2206
            or len(record["validation_ids"]) != 376
        ):
            raise ValueError("Need frozen joint2206 fitting/376 validation identities")
        if [i for i in record["fit_ids"] if i.startswith("beatport:")] != self.bp["fit_ids"]:
            raise ValueError("Archived BP fitting identities differ")
        source_checks(record["source_sha256"], self.reference["source_sha256"])
        if args.seed != 13:
            check_pilot(args.pilot_run, record)
        self.reference_checkpoint = args.reference_checkpoint or args.reference_run.with_name(
            "best.pt"
        )
        if not self.reference_checkpoint.is_file():
            raise ValueError(
                "Exact replay requires archived joint best.pt (or --reference-checkpoint)"
            )
        self.control: dict[str, Any] | None = None
        self.control_result: dict[str, Any] | None = None
        if self.kind == "candidate":
            self.control = read_reference(args.paired_control, "joint", args.seed)
            path = args.paired_control.with_name("structured-result.json")
            self.control_result = json.loads(path.read_bytes())
            if (
                self.control_result.get("artifact_version") != VERSION
                or self.control_result.get("kind") != "control"
                or self.control_result.get("seed") != args.seed
                or self.control_result.get("status") != "complete"
                or not self.control_result.get("control_reproduced")
                or self.control_result.get("run_sha256") != audio_digest(args.paired_control)
            ):
                raise ValueError("Candidate requires exactly reproduced fresh paired control first")
            for key in (
                "config",
                "source_sha256",
                "feature_manifest_sha256",
                "fit_ids",
                "validation_ids",
            ):
                if self.control[key] != record[key]:
                    raise ValueError(f"Fresh control differs: {key}")
            if self.control.get("loss_spec") != HARD_LOSS:
                raise ValueError("Fresh control loss differs")
            if self.control_result["identity"]["protocol_sha256"] != audio_digest(PROTOCOL):
                raise ValueError("Fresh control protocol differs")
        record["loss_spec"] = SOFT_LOSS if self.kind == "candidate" else HARD_LOSS
        self.identity = dict(
            artifact_version=VERSION,
            kind=self.kind,
            seed=args.seed,
            loss_spec=record["loss_spec"],
            config=record["config"],
            source_sha256=record["source_sha256"],
            feature_manifest_sha256=record["feature_manifest_sha256"],
            protocol_sha256=audio_digest(PROTOCOL),
            reviewed_sources_sha256=audio_digest(REVIEWED),
            reference_run_sha256=audio_digest(args.reference_run),
            reference_bp_sha256=audio_digest(args.reference_bp),
            reference_checkpoint_sha256=audio_digest(self.reference_checkpoint),
            source_diff={
                name: dict(
                    archived=self.reference["source_sha256"].get(name),
                    current=record["source_sha256"].get(name),
                )
                for name in sorted(
                    set(self.reference["source_sha256"]) | set(record["source_sha256"])
                )
                if self.reference["source_sha256"].get(name) != record["source_sha256"].get(name)
            },
            torch=str(torch.__version__),
            numpy=np.__version__,
            targets=target_matrix().tolist(),
            score_conventions=dict(
                primary="repo_symmetric_fifths",
                secondary="existing_score_mirex_symmetricFalse_ascending_fifth",
            ),
        )
        if args.paired_control:
            self.identity["paired_control_run_sha256"] = audio_digest(args.paired_control)
            self.identity["paired_control_result_sha256"] = audio_digest(
                args.paired_control.with_name("structured-result.json")
            )
        if args.pilot_run:
            self.identity["pilot_result_sha256"] = audio_digest(args.pilot_run)
        for path in (PROTOCOL, REVIEWED):
            (args.output / path.name).write_bytes(path.read_bytes())
        atomic_json(args.output / "structured-identity.json", self.identity)
        record["structured_identity"] = self.identity

    def budget_exhausted(self, start: float) -> None:
        atomic_json(
            self.args.output / "structured-result.json",
            dict(
                artifact_version=VERSION,
                kind=self.kind,
                seed=self.args.seed,
                identity=self.identity,
                status="budget_exhausted",
                control_reproduced=False,
                promoted=False,
                elapsed_seconds=time.monotonic() - start,
                method_gate={"qualifies_for_confirmation": False},
            ),
        )

    def finish(
        self,
        record: dict[str, Any],
        features: list[np.ndarray],
        refs: np.ndarray,
        rows: list[dict[str, Any]],
        start: float,
    ) -> dict[str, Any]:
        from experiments.k10_multidomain import domain_metrics

        result: dict[str, Any] = dict(
            artifact_version=VERSION,
            kind=self.kind,
            seed=self.args.seed,
            identity=self.identity,
            status="checking",
            control_reproduced=False,
            promoted=False,
            selected_epoch=record["best_epoch"],
            stop_epoch=record["history"][-1]["epoch"],
            training_elapsed_seconds=record["elapsed_seconds"],
            run_sha256=audio_digest(self.args.output / "run.json"),
            calibration_used=False,
            final_holdout_used=False,
        )
        try:
            check_budget(start)
            with preserve_rng():
                model, payload = load_checkpoint(self.args.output / "best.pt", self.args.device)
                if self.kind == "control":
                    archived_state = torch.load(
                        self.reference_checkpoint, map_location="cpu", weights_only=True
                    )["state_dict"]
                    assert_reproduced(
                        record,
                        self.reference,
                        current_state=payload["state_dict"],
                        reference_state=archived_state,
                    )
                else:
                    if self.control is None or self.control_result is None:
                        raise ValueError("Missing reproduced paired control")
                    require_pair(self.control, record)
                logits = evaluate_logits(model, features, self.args.device, start)
                metrics = domain_metrics(logits.argmax(1), refs, rows)
                if metrics != record["domain_metrics"]:
                    raise ValueError("Final logits differ from selected validation predictions")
                artifact_path = self.args.output / "structured-validation.json"
                atomic_json(
                    artifact_path,
                    dict(
                        artifact_version=VERSION,
                        ids=[r["id"] for r in rows],
                        audio_sha256=[r["sha256"] for r in rows],
                        refs=refs.tolist(),
                        logits=logits.tolist(),
                        predictions=logits.argmax(1).tolist(),
                        metrics=metrics,
                    ),
                )
                result.update(
                    status="complete",
                    control_reproduced=True,
                    metrics=metrics,
                    checkpoint_state_checked=self.kind == "control",
                    validation_artifact=artifact_path.name,
                    validation_sha256=audio_digest(artifact_path),
                    checkpoint_sha256=audio_digest(self.args.output / "best.pt"),
                    bp_metrics=self.bp["domain_metrics"],
                    archived_joint_metrics=self.reference["domain_metrics"],
                )
                if self.kind == "candidate":
                    assert self.control is not None
                    result.update(
                        confusion_movements=confusion_movements(
                            self.control["domain_metrics"], logits.argmax(1), refs, rows
                        ),
                        control_metrics=self.control["domain_metrics"],
                        method_gate=compare_metrics(self.control["domain_metrics"], metrics),
                    )
                check_budget(start)
        except (ValueError, TimeoutError, RuntimeError, OSError) as exc:
            result.update(
                status="budget_exhausted"
                if isinstance(exc, TimeoutError)
                else "reproduction_or_protocol_failed",
                error=str(exc),
                control_reproduced=False,
                method_gate={"qualifies_for_confirmation": False},
            )
        result["elapsed_seconds"] = time.monotonic() - start
        atomic_json(self.args.output / "structured-result.json", result)
        return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = {r["seed"]: r for r in results}
    if len(results) != 3 or sorted(seeds) != [13, 37, 73]:
        raise ValueError("Need exactly candidate seeds13/37/73")
    pilot = seeds[13]
    if not pilot["method_gate"]["qualifies_for_confirmation"]:
        raise ValueError("Canonical pilot did not qualify")
    for result in results:
        if (
            result["artifact_version"] != VERSION
            or result["status"] != "complete"
            or result["kind"] != "candidate"
            or not result["control_reproduced"]
        ):
            raise ValueError("Need completed candidates with reproduced paired controls")
        for key in (
            "source_sha256",
            "config",
            "feature_manifest_sha256",
            "protocol_sha256",
            "loss_spec",
            "targets",
        ):
            if result["identity"][key] != pilot["identity"][key]:
                raise ValueError(f"Confirmation changed {key}")
    means = {}
    for name in ("metrics", "control_metrics", "bp_metrics", "archived_joint_metrics"):
        means[name] = {
            domain: {
                metric: statistics.mean(r[name][domain][metric] for r in results)
                for metric in ("mirex", "mirex_asymmetric", "exact")
            }
            for domain in ("beatport", "fmak")
        }
    method = compare_metrics(means["control_metrics"], means["metrics"])
    original_delta = {
        d: means["metrics"][d]["mirex"] - means["bp_metrics"][d]["mirex"]
        for d in ("beatport", "fmak")
    }
    original_pass = original_delta["fmak"] >= 0.02 and original_delta["beatport"] >= 0
    safeguard = {
        d: {
            m: means["metrics"][d][m] - means["archived_joint_metrics"][d][m]
            for m in ("mirex", "exact")
        }
        for d in ("beatport", "fmak")
    }
    safeguard_pass = all(value >= 0 for metrics in safeguard.values() for value in metrics.values())
    return dict(
        artifact_version=VERSION,
        means=means,
        method_gate=method,
        original_hurdle_delta=original_delta,
        original_hurdle_pass=original_pass,
        raw_joint_safeguard=dict(delta=safeguard, passes=safeguard_pass),
        qualifies_for_external_freeze=method["qualifies_for_confirmation"]
        and original_pass
        and safeguard_pass,
        confusion_movements_by_seed={str(r["seed"]): r.get("confusion_movements") for r in results},
        canonical_seed=13,
        promoted=False,
        uncertainty="Training-seed means on reused development data; not independent generalization evidence",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize([json.loads(p.read_bytes()) for p in args.results])
    result["result_sha256"] = {str(p): audio_digest(p) for p in args.results}
    result["summary_source_sha256"] = audio_digest(Path(__file__))
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
