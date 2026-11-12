"""Fresh-control-first, one fixed SAM/AdamW accuracy study over the shared loop."""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drone.features import audio_digest
from experiments.ema import read_reference
from experiments.paired import compare, movements, score_selected, summarize, verify_control
from experiments.small import atomic_json

VERSION = "k10-vanilla-sam-adamw-v1"
ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "experiments/studies/k10-sam"
PROTOCOL = STUDY / "PROTOCOL.md"
AUDIT = STUDY / "data/base-audit.json"
REVIEWED = ROOT / "experiments/studies/k10-capacity/data/reviewed-sources.json"


def source_checks() -> dict[str, Any]:
    """Bind unchanged model/math helpers and the literal disabled ordinary batch."""
    audit = json.loads(AUDIT.read_bytes())
    for name, digest in audit["sources"].items():
        if name != "experiments/k10_multidomain.py" and audio_digest(ROOT / name) != digest:
            raise ValueError(f"Source differs from fixed reviewed base: {name}")
    tree = ast.parse((ROOT / "experiments/k10_multidomain.py").read_text())
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    branch = next(
        n
        for n in ast.walk(train)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "training_step is None"
    )
    actual = ast.dump(ast.Module(body=branch.body, type_ignores=[]), include_attributes=False)
    if actual != audit["ordinary_batch_ast"]:
        raise ValueError("Literal disabled-hook ordinary batch math changed")
    return dict(
        base_commit=audit["base_commit"],
        base_source_sha256=audit["sources"],
        ordinary_batch_ast_exact=True,
        audit_sha256=audio_digest(AUDIT),
    )


class SamStudy:
    """Lifecycle/provenance only: the shared fitter owns all data and selection."""

    def __init__(
        self,
        args: argparse.Namespace,
        record: dict[str, Any],
        hook_enabled: bool,
        val_rows: list[dict[str, Any]],
        val_refs: np.ndarray,
    ) -> None:
        self.args = args
        self.candidate = args.sam_candidate
        self.seconds = 2400 if self.candidate else 1200
        if (
            hook_enabled != self.candidate
            or record["arm"] != "joint"
            or record["parameters"] != 109128
        ):
            raise ValueError("Need the fixed K10 joint model with declared hook mode")
        self.joint = read_reference(args.reference_run, "joint", args.seed)
        self.bp = read_reference(args.reference_bp, "control", args.seed)
        reviewed = json.loads(REVIEWED.read_bytes())
        if self.joint["source_sha256"] != reviewed["archived"]["sources"]:
            raise ValueError("Archived joint must match approved frozen673 source")
        for key in ("config", "feature_manifest_sha256", "fit_ids", "validation_ids"):
            if record[key] != self.joint[key]:
                raise ValueError(f"Joint fitting recipe/data changed: {key}")
        if len(record["fit_ids"]) != 2206 or len(record["validation_ids"]) != 376:
            raise ValueError("Need fixed2206 fitting/376 validation identities")
        for key in ("config", "feature_manifest_sha256", "validation_ids", "source_sha256"):
            if self.joint[key] != self.bp[key]:
                raise ValueError(f"BP reference differs: {key}")
        if [i for i in record["fit_ids"] if i.startswith("beatport:")] != self.bp["fit_ids"]:
            raise ValueError("BP fitting identities changed")
        if audio_digest(args.reference_bp.with_name("best.pt")) != self.bp["checkpoint_sha256"]:
            raise ValueError("Archived BP checkpoint changed")
        self.reference_checkpoint = args.reference_checkpoint or args.reference_run.with_name(
            "best.pt"
        )
        if audio_digest(self.reference_checkpoint) != self.joint["checkpoint_sha256"]:
            raise ValueError("Archived joint checkpoint changed")
        checks = source_checks()
        self.control: dict[str, Any] | None = None
        self.proof: dict[str, Any] = {}
        if self.candidate:
            self.control, self.proof = verify_control(
                args.paired_control,
                args.reference_run,
                self.reference_checkpoint,
                [record["source_sha256"]],
            )
            proof_path = args.paired_control.with_name("sam-result.json")
            result = json.loads(proof_path.read_bytes())
            artifact = args.paired_control.with_name(result["validation_artifact"])
            if (
                result.get("artifact_version") != VERSION
                or result.get("kind") != "control"
                or result.get("status") != "complete"
                or not result.get("control_reproduced")
                or result.get("seed") != args.seed
                or result.get("run_sha256") != audio_digest(args.paired_control)
                or result.get("checkpoint_sha256") != self.proof["checkpoint_sha256"]
                or result.get("validation_sha256") != audio_digest(artifact)
                or result["identity"]["source_sha256"] != record["source_sha256"]
                or result["identity"]["protocol_sha256"] != audio_digest(PROTOCOL)
                or result["identity"]["sam_spec"]["enabled"]
            ):
                raise ValueError("Need fresh matching disabled-hook exact control proof")
            values = json.loads(artifact.read_bytes())
            from experiments.k10_multidomain import domain_metrics

            logits = np.asarray(values["logits"])
            if (
                values["ids"] != record["validation_ids"]
                or values["audio_sha256"] != [r["sha256"] for r in val_rows]
                or values["refs"] != val_refs.tolist()
                or logits.shape != (376, 24)
                or not np.isfinite(logits).all()
                or values["metrics"] != self.control["domain_metrics"]
                or domain_metrics(logits.argmax(1), val_refs, val_rows)
                != self.control["domain_metrics"]
            ):
                raise ValueError("Fresh control logits identities or metrics differ")
            self.proof.update(
                result_sha256=audio_digest(proof_path), validation_sha256=audio_digest(artifact)
            )
        if args.seed != 13:
            if args.pilot_run is None:
                raise ValueError("Only passing canonical13 earns37/73")
            pilot = json.loads(args.pilot_run.read_bytes())
            if (
                pilot.get("artifact_version") != VERSION
                or pilot.get("seed") != 13
                or pilot.get("kind") != "candidate"
                or pilot.get("status") != "complete"
                or not pilot.get("control_reproduced")
                or not pilot["method_gate"]["qualifies_for_confirmation"]
            ):
                raise ValueError("SAM pilot did not qualify")
            for key in ("source_sha256", "config", "feature_manifest_sha256"):
                if pilot["identity"][key] != record[key]:
                    raise ValueError(f"Confirmation changed {key}")
            if pilot["identity"]["protocol_sha256"] != audio_digest(PROTOCOL):
                raise ValueError("Confirmation changed SAM protocol")
        memory: dict[str, Any] = dict(allocator_fraction=0.04, required_free_bytes=8 * 1024**3)
        if args.device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            memory.update(free_bytes=free, total_bytes=total, allocator_cap_bytes=int(total * 0.04))
        self.identity = dict(
            artifact_version=VERSION,
            source_sha256=record["source_sha256"],
            config=record["config"],
            feature_manifest_sha256=record["feature_manifest_sha256"],
            protocol_sha256=audio_digest(PROTOCOL),
            reviewed_sources_sha256=audio_digest(REVIEWED),
            budget_seconds=self.seconds,
            width=1.0,
            source_audit=checks,
            sam_spec=dict(
                enabled=self.candidate,
                rho=0.05,
                norm="global-euclidean-all-trainable",
                dropout="coupled",
                bn="batch-statistics-both-passes-first-pass-buffers-only",
                outer="one-existing-AdamW-step",
                weight_decay="decoupled-outer-only",
            ),
            reference_run_sha256=audio_digest(args.reference_run),
            reference_bp_sha256=audio_digest(args.reference_bp),
            reference_checkpoint_sha256=audio_digest(self.reference_checkpoint),
            memory_guard=memory,
            torch=str(torch.__version__),
            numpy=np.__version__,
            source_diff={
                name: dict(
                    archived=self.joint["source_sha256"].get(name),
                    current=record["source_sha256"].get(name),
                )
                for name in sorted(set(record["source_sha256"]) | set(self.joint["source_sha256"]))
                if self.joint["source_sha256"].get(name) != record["source_sha256"].get(name)
            },
        )
        record["sam_identity"] = self.identity
        atomic_json(args.output / "sam-identity.json", self.identity)
        for path in (PROTOCOL, AUDIT, REVIEWED):
            (args.output / path.name).write_bytes(path.read_bytes())

    def failure(
        self,
        record: dict[str, Any],
        start: float,
        error: str,
        *,
        budget: bool = False,
        resource: bool = False,
    ) -> None:
        atomic_json(
            self.args.output / "sam-result.json",
            dict(
                artifact_version=VERSION,
                kind="candidate" if self.candidate else "control",
                seed=self.args.seed,
                status="budget_inconclusive"
                if budget
                else ("resource_inconclusive" if resource else "reproduction_or_protocol_failed"),
                error=error,
                identity=self.identity,
                control_reproduced=False,
                elapsed_seconds=time.monotonic() - start,
                history_epochs=len(record["history"]),
                method_gate={"qualifies_for_confirmation": False},
                promoted=False,
            ),
        )

    def finish(
        self,
        record: dict[str, Any],
        features: list[np.ndarray],
        refs: np.ndarray,
        rows: list[dict[str, Any]],
        start: float,
    ) -> None:
        try:
            if not self.candidate:
                self.control, self.proof = verify_control(
                    self.args.output / "run.json",
                    self.args.reference_run,
                    self.reference_checkpoint,
                    [record["source_sha256"]],
                )
            scored = score_selected(
                self.args.output,
                record,
                features,
                refs,
                rows,
                self.args.device,
                start,
                self.seconds,
                1.0,
            )
            result = dict(
                artifact_version=VERSION,
                kind="candidate" if self.candidate else "control",
                seed=self.args.seed,
                status="complete",
                identity=self.identity,
                control_reproduced=True,
                control_proof=self.proof,
                promoted=False,
                selected_epoch=record["best_epoch"],
                stop_epoch=record["history"][-1]["epoch"],
                history_epochs=len(record["history"]),
                training_elapsed_seconds=record["elapsed_seconds"],
                elapsed_seconds=time.monotonic() - start,
                run_sha256=audio_digest(self.args.output / "run.json"),
                **scored,
            )
            if self.candidate:
                assert self.control is not None
                result.update(
                    control_metrics=self.control["domain_metrics"],
                    bp_metrics=self.bp["domain_metrics"],
                    method_gate=compare(self.control["domain_metrics"], scored["metrics"]),
                    confusion_movements=movements(
                        self.control["domain_metrics"], scored["metrics"], refs, rows
                    ),
                )
            if self.args.device.startswith("cuda"):
                free, total = torch.cuda.mem_get_info()
                result["memory"] = dict(
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    final_free_bytes=free,
                    total_bytes=total,
                )
            if time.monotonic() - start > self.seconds:
                raise TimeoutError("Fixed total SAM study budget exceeded including scoring")
            atomic_json(self.args.output / "sam-result.json", result)
        except (ValueError, RuntimeError, TimeoutError, OSError) as error:
            self.failure(
                record,
                start,
                str(error),
                budget=isinstance(error, TimeoutError),
                resource=isinstance(error, torch.OutOfMemoryError),
            )
            raise


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summary_parser = argparse.ArgumentParser("Fixed SAM confirmation summary")
        summary_parser.add_argument("results", type=Path, nargs=3)
        summary_parser.add_argument("--output", type=Path, required=True)
        args = summary_parser.parse_args(sys.argv[2:])
        result = summarize([json.loads(p.read_bytes()) for p in args.results], VERSION)
        result["result_sha256"] = {str(p): audio_digest(p) for p in args.results}
        atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, choices=(13, 37, 73), default=13)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--reference-bp", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--paired-control", type=Path)
    parser.add_argument("--pilot-run", type=Path)
    args = parser.parse_args()
    if args.candidate != (args.paired_control is not None):
        parser.error(
            "Only candidate requires --paired-control; it must be fresh SAM disabled-hook control"
        )
    args.arm, args.sam_study, args.sam_candidate = "joint", True, args.candidate
    from experiments.k10_multidomain import train
    from experiments.sam_step import step

    try:
        train(args, training_step=step if args.candidate else None)
    except (ValueError, RuntimeError, TimeoutError, OSError) as error:
        if (args.output / "sam-result.json").exists():
            raise
        identity_path = args.output / "sam-identity.json"
        atomic_json(
            args.output / "sam-result.json",
            dict(
                artifact_version=VERSION,
                status="resource_inconclusive"
                if isinstance(error, (torch.OutOfMemoryError, TimeoutError))
                else "reproduction_or_protocol_failed",
                seed=args.seed,
                error=str(error),
                identity=json.loads(identity_path.read_bytes()) if identity_path.exists() else {},
                control_reproduced=False,
                method_gate={"qualifies_for_confirmation": False},
                promoted=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
