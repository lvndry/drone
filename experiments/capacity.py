"""Single fixed width-2 K10 capacity hypothesis; never a product tier or sweep."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drone.features import audio_digest
from experiments.ema import read_reference
from experiments.paired import (
    compare,
    ema_evidence_math,
    movements,
    score_selected,
    summarize,
    training_math,
    verify_control,
)
from experiments.small import atomic_json

VERSION = "k10-width2-capacity-v1"
STUDY = Path(__file__).parent / "studies" / "k10-capacity"
PROTOCOL, REVIEWED = STUDY / "PROTOCOL.md", STUDY / "data" / "reviewed-sources.json"


def memory_guard(device: str) -> dict[str, Any]:
    """Only the fixed wide candidate may raise the existing allocator cap."""
    if not device.startswith("cuda"):
        return dict(
            device=device, allocator_fraction=0.08, required_free_bytes=12 * 1024**3, cuda=False
        )
    free, total = torch.cuda.mem_get_info()
    if free < 12 * 1024**3:
        raise RuntimeError("Width2 needs12GiB currently free; preserve vLLM")
    torch.cuda.set_per_process_memory_fraction(0.08)
    return dict(
        device=device,
        allocator_fraction=0.08,
        required_free_bytes=12 * 1024**3,
        free_bytes=free,
        total_bytes=total,
        allocator_cap_bytes=int(total * 0.08),
        cuda=True,
    )


def record_oom(args: argparse.Namespace, error: BaseException) -> None:
    identity_path = args.output / "capacity-identity.json"
    identity = json.loads(identity_path.read_bytes()) if identity_path.exists() else {}
    result = dict(
        artifact_version=VERSION,
        kind="candidate" if args.capacity_probe else "control",
        seed=args.seed,
        identity=identity,
        status="resource_inconclusive",
        error=str(error),
        control_reproduced=False,
        promoted=False,
        method_gate={"qualifies_for_confirmation": False},
    )
    atomic_json(args.output / "capacity-result.json", result)


class CapacityProbe:
    """Thin capacity policy over shared replay, scoring and comparison helpers."""

    def __init__(
        self, args: argparse.Namespace, record: dict[str, Any], memory: dict[str, Any]
    ) -> None:
        self.args = args
        self.candidate = args.capacity_probe
        self.width, self.seconds = (2.0, 2700) if self.candidate else (1.0, 1200)
        self.reference = read_reference(args.reference_run, "joint", args.seed)
        self.bp = read_reference(args.reference_bp, "control", args.seed)
        for key in ("config", "feature_manifest_sha256", "fit_ids", "validation_ids"):
            if record[key] != self.reference[key]:
                raise ValueError(f"Archived joint mismatch: {key}")
        for key in ("config", "feature_manifest_sha256", "validation_ids", "source_sha256"):
            if self.reference[key] != self.bp[key]:
                raise ValueError(f"Archived references differ: {key}")
        if (
            record["arm"] != "joint"
            or len(record["fit_ids"]) != 2206
            or len(record["validation_ids"]) != 376
            or record["parameters"] != (361080 if self.candidate else 109128)
        ):
            raise ValueError("Need fixed joint identities and declared K10 width/parameter count")
        if [i for i in record["fit_ids"] if i.startswith("beatport:")] != self.bp["fit_ids"]:
            raise ValueError("Archived BP fitting identities differ")
        reviewed = json.loads(REVIEWED.read_bytes())
        if self.reference["source_sha256"] != reviewed["archived"]["sources"]:
            raise ValueError("Archived reference source differs from approved source673")
        for name, digest in reviewed["base"]["sources"].items():
            if (
                name not in {"experiments/k10_multidomain.py", "experiments/ema.py"}
                and record["source_sha256"].get(name) != digest
            ):
                raise ValueError(f"Source differs from reviewed base: {name}")
        evidence_math = ema_evidence_math(Path(__file__).with_name("ema.py").read_text())
        if evidence_math != reviewed["base"]["ema_evidence_math"]:
            raise ValueError("EMA evidence helpers changed beyond paths/default-budget injection")
        math = training_math(Path(__file__).with_name("k10_multidomain.py").read_text())
        if any(
            math != reviewed[label]["training_math"]
            for label in ("archived", "reusable_control", "base")
        ):
            raise ValueError("Training math differs from audited hard-CE controls")
        self.reference_checkpoint = args.reference_checkpoint or args.reference_run.with_name(
            "best.pt"
        )
        if not self.reference_checkpoint.is_file():
            raise ValueError("Need archived joint best.pt (or --reference-checkpoint)")
        self.control: dict[str, Any] | None = None
        self.control_proof: dict[str, Any] = {}
        if self.candidate:
            self.control, self.control_proof = verify_control(
                args.paired_control,
                args.reference_run,
                self.reference_checkpoint,
                [reviewed["reusable_control"]["sources"], record["source_sha256"]],
            )
            reused = self.control["source_sha256"] == reviewed["reusable_control"]["sources"]
            proof_path = args.paired_control.with_name(
                "structured-result.json" if reused else "capacity-result.json"
            )
            proof = json.loads(proof_path.read_bytes())
            expected_version = "k10-structured-targets-v1" if reused else VERSION
            if (
                proof.get("artifact_version") != expected_version
                or proof.get("seed") != args.seed
                or proof.get("status") != "complete"
                or proof.get("kind") != "control"
                or not proof.get("control_reproduced")
                or proof.get("run_sha256") != audio_digest(args.paired_control)
                or proof.get("checkpoint_sha256") != self.control_proof["checkpoint_sha256"]
            ):
                raise ValueError("Paired control lacks its matching completed exact-replay proof")
            artifact_path = args.paired_control.with_name(proof["validation_artifact"])
            if audio_digest(artifact_path) != proof["validation_sha256"]:
                raise ValueError("Paired control validation artifact hash changed")
            self.control_proof["original_result_sha256"] = audio_digest(proof_path)
            self.control_proof["validation_sha256"] = proof["validation_sha256"]
            for key in ("config", "feature_manifest_sha256", "fit_ids", "validation_ids"):
                if self.control[key] != record[key]:
                    raise ValueError(f"Paired control differs beyond capacity: {key}")
        if args.seed != 13:
            if args.pilot_run is None:
                raise ValueError("Confirmation needs passing canonical capacity seed13 pilot")
            pilot = json.loads(args.pilot_run.read_bytes())
            if (
                pilot.get("artifact_version") != VERSION
                or pilot.get("seed") != 13
                or pilot.get("status") != "complete"
                or pilot.get("kind") != "candidate"
                or not pilot.get("control_reproduced")
                or not pilot.get("method_gate", {}).get("qualifies_for_confirmation")
            ):
                raise ValueError("Capacity pilot did not qualify")
            for key in ("source_sha256", "config", "feature_manifest_sha256"):
                if pilot["identity"][key] != record[key]:
                    raise ValueError(f"Confirmation changed {key}")
            if pilot["identity"]["protocol_sha256"] != audio_digest(PROTOCOL):
                raise ValueError("Confirmation changed capacity protocol")
        self.identity = dict(
            artifact_version=VERSION,
            width=self.width,
            budget_seconds=self.seconds,
            source_sha256=record["source_sha256"],
            config=record["config"],
            feature_manifest_sha256=record["feature_manifest_sha256"],
            protocol_sha256=audio_digest(PROTOCOL),
            reviewed_sources_sha256=audio_digest(REVIEWED),
            training_math=math,
            evidence_helper_math=evidence_math,
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
            archived_source_sha256=self.reference["source_sha256"],
            reference_run_sha256=audio_digest(args.reference_run),
            reference_bp_sha256=audio_digest(args.reference_bp),
            control_proof=self.control_proof,
            memory_guard=memory,
            torch=str(torch.__version__),
            numpy=np.__version__,
            score_conventions=dict(
                primary="repo_symmetric_fifths", secondary="existing_asymmetric_ascending_fifth"
            ),
        )
        for path in (PROTOCOL, REVIEWED):
            (args.output / path.name).write_bytes(path.read_bytes())
        atomic_json(args.output / "capacity-identity.json", self.identity)
        record["capacity_identity"] = self.identity

    def failure(
        self,
        record: dict[str, Any],
        start: float,
        error: str,
        *,
        budget: bool = False,
        resource: bool = False,
    ) -> dict[str, Any]:
        result = dict(
            artifact_version=VERSION,
            kind="candidate" if self.candidate else "control",
            seed=self.args.seed,
            identity=self.identity,
            status="budget_inconclusive"
            if budget
            else ("resource_inconclusive" if resource else "reproduction_or_protocol_failed"),
            error=error,
            control_reproduced=False,
            promoted=False,
            elapsed_seconds=time.monotonic() - start,
            history_epochs=len(record["history"]),
            method_gate={"qualifies_for_confirmation": False},
        )
        atomic_json(self.args.output / "capacity-result.json", result)
        return result

    def finish(
        self,
        record: dict[str, Any],
        features: list[np.ndarray],
        refs: np.ndarray,
        rows: list[dict[str, Any]],
        start: float,
    ) -> dict[str, Any]:
        try:
            if not self.candidate:
                self.control, self.control_proof = verify_control(
                    self.args.output / "run.json",
                    self.args.reference_run,
                    self.reference_checkpoint,
                    [record["source_sha256"]],
                )
            scores = score_selected(
                self.args.output,
                record,
                features,
                refs,
                rows,
                self.args.device,
                start,
                self.seconds,
                self.width,
            )
            result = dict(
                artifact_version=VERSION,
                kind="candidate" if self.candidate else "control",
                seed=self.args.seed,
                identity=self.identity,
                status="complete",
                promoted=False,
                control_reproduced=True,
                control_proof=self.control_proof,
                selected_epoch=record["best_epoch"],
                stop_epoch=record["history"][-1]["epoch"],
                training_elapsed_seconds=record["elapsed_seconds"],
                elapsed_seconds=time.monotonic() - start,
                run_sha256=audio_digest(self.args.output / "run.json"),
                **scores,
            )
            if self.candidate:
                assert self.control is not None
                result.update(
                    control_metrics=self.control["domain_metrics"],
                    bp_metrics=self.bp["domain_metrics"],
                    method_gate=compare(self.control["domain_metrics"], scores["metrics"]),
                    confusion_movements=movements(
                        self.control["domain_metrics"], scores["metrics"], refs, rows
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
                raise TimeoutError("Fixed fit budget exhausted including scoring/artifacts")
            atomic_json(self.args.output / "capacity-result.json", result)
            return result
        except (ValueError, TimeoutError, RuntimeError, OSError) as exc:
            return self.failure(
                record,
                start,
                str(exc),
                budget=isinstance(exc, TimeoutError),
                resource=isinstance(exc, torch.OutOfMemoryError),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize([json.loads(p.read_bytes()) for p in args.results], VERSION)
    result["result_sha256"] = {str(p): audio_digest(p) for p in args.results}
    result["summary_source_sha256"] = audio_digest(Path(__file__))
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
