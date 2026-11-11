"""Fixed PitchCircle fit policy on the shared K10 loop and paired evidence helpers."""

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
from experiments.pitch_circle_model import SPEC
from experiments.research_models import ARCHITECTURE, ResearchModelHooks
from experiments.small import atomic_json

VERSION = "pitch-circle-joint-fit-v1"
STUDY = Path(__file__).parent / "studies" / "pitch-circle"
PROTOCOL, REVIEWED = STUDY / "PROTOCOL.md", STUDY / "data" / "fit-reviewed-sources.json"


class PitchCircleFit:
    """Evidence/preflight/reporting only; never owns a second training loop."""

    def __init__(
        self, args: argparse.Namespace, record: dict[str, Any], hooks: ResearchModelHooks
    ) -> None:
        self.args, self.hooks = args, hooks
        self.reference = read_reference(args.reference_run, "joint", args.seed)
        self.bp = read_reference(args.reference_bp, "control", args.seed)
        for key in ("config", "feature_manifest_sha256", "fit_ids", "validation_ids"):
            if record[key] != self.reference[key]:
                raise ValueError(f"Archived joint mismatch: {key}")
        for key in ("config", "feature_manifest_sha256", "validation_ids", "source_sha256"):
            if self.reference[key] != self.bp[key]:
                raise ValueError(f"Archived reference pair differs: {key}")
        if (
            record["arm"] != "joint"
            or record["parameters"] != 73506
            or len(record["fit_ids"]) != 2206
            or len(record["validation_ids"]) != 376
            or hooks.architecture != ARCHITECTURE
        ):
            raise ValueError("Need fixed PitchCircle architecture and joint2206/376 identities")
        if [i for i in record["fit_ids"] if i.startswith("beatport:")] != self.bp["fit_ids"]:
            raise ValueError("Archived BP fitting identities differ")
        reviewed = json.loads(REVIEWED.read_bytes())
        if self.reference["source_sha256"] != reviewed["archived"]["sources"]:
            raise ValueError("Archived source differs from frozen source673")
        for name, digest in reviewed["base"]["sources"].items():
            if (
                name not in {"experiments/k10_multidomain.py", "experiments/ema.py"}
                and record["source_sha256"].get(name) != digest
            ):
                raise ValueError(f"Current source differs from reviewed base: {name}")
        core = training_math(Path(__file__).with_name("k10_multidomain.py").read_text())
        if any(
            core != reviewed[name]["training_math"]
            for name in ("archived", "reusable_control", "base")
        ):
            raise ValueError("Fitting math differs from audited hard-CE control recipe")
        evidence = ema_evidence_math(Path(__file__).with_name("ema.py").read_text())
        if evidence != reviewed["base"]["ema_evidence_math"]:
            raise ValueError("Replay/scoring helpers changed beyond audited path/budget injection")
        archived_checkpoint = args.reference_checkpoint or args.reference_run.with_name("best.pt")
        self.control, self.proof = verify_control(
            args.paired_control,
            args.reference_run,
            archived_checkpoint,
            [reviewed["reusable_control"]["sources"], record["source_sha256"]],
        )
        if self.control["source_sha256"] == reviewed["reusable_control"]["sources"]:
            proof_path = args.paired_control.with_name("structured-result.json")
            proof = json.loads(proof_path.read_bytes())
            if (
                proof.get("artifact_version") != "k10-structured-targets-v1"
                or proof.get("seed") != args.seed
                or proof.get("kind") != "control"
                or proof.get("status") != "complete"
                or not proof.get("control_reproduced")
                or proof.get("run_sha256") != audio_digest(args.paired_control)
                or proof.get("checkpoint_sha256") != self.proof["checkpoint_sha256"]
            ):
                raise ValueError("Reused control lacks its matching completed replay result")
            validation = args.paired_control.with_name(proof["validation_artifact"])
            if audio_digest(validation) != proof["validation_sha256"]:
                raise ValueError("Reused control validation artifact changed")
            self.proof.update(
                original_result_sha256=audio_digest(proof_path),
                validation_sha256=proof["validation_sha256"],
            )
        for key in ("config", "feature_manifest_sha256", "fit_ids", "validation_ids"):
            if self.control[key] != record[key]:
                raise ValueError(f"Paired control differs beyond architecture: {key}")
        if args.seed != 13:
            if args.pilot_run is None:
                raise ValueError(
                    "Confirmation requires the passing canonical PitchCircle seed13 pilot"
                )
            pilot = json.loads(args.pilot_run.read_bytes())
            if (
                pilot.get("artifact_version") != VERSION
                or pilot.get("seed") != 13
                or pilot.get("status") != "complete"
                or not pilot.get("control_reproduced")
                or not pilot.get("method_gate", {}).get("qualifies_for_confirmation")
            ):
                raise ValueError("PitchCircle pilot did not qualify")
            for key in ("source_sha256", "config", "feature_manifest_sha256"):
                if pilot["identity"][key] != record[key]:
                    raise ValueError(f"Confirmation changed {key}")
            if pilot["identity"]["protocol_sha256"] != audio_digest(PROTOCOL):
                raise ValueError("Confirmation changed the fixed fit protocol")
        memory: dict[str, Any] = dict(allocator_fraction=0.04, required_free_bytes=8 * 1024**3)
        if args.device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            memory.update(free_bytes=free, total_bytes=total, allocator_cap_bytes=int(total * 0.04))
        self.identity = dict(
            artifact_version=VERSION,
            architecture=ARCHITECTURE,
            model_spec=SPEC,
            budget_seconds=1200,
            source_sha256=record["source_sha256"],
            config=record["config"],
            feature_manifest_sha256=record["feature_manifest_sha256"],
            protocol_sha256=audio_digest(PROTOCOL),
            reviewed_sources_sha256=audio_digest(REVIEWED),
            reference_run_sha256=audio_digest(args.reference_run),
            reference_bp_sha256=audio_digest(args.reference_bp),
            control_proof=self.proof,
            training_math=core,
            evidence_helper_math=evidence,
            paired_helpers_origin="679b7fdb021af64cb2816c8b61d571a5d3ac773c; optional research-loader extension",
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
            memory_guard=memory,
            torch=str(torch.__version__),
            numpy=np.__version__,
            score_conventions=dict(
                primary="repo_symmetric_fifths", secondary="existing_asymmetric_ascending_fifth"
            ),
            symmetry_scope="folded-feature C12 rotations only; not physical finite-band CQT reslices",
        )
        if args.pilot_run is not None:
            self.identity["pilot_result_sha256"] = audio_digest(args.pilot_run)
        for path in (PROTOCOL, REVIEWED):
            (args.output / path.name).write_bytes(path.read_bytes())
        atomic_json(args.output / "pitch-circle-identity.json", self.identity)
        record["research_identity"] = self.identity

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
            kind="candidate",
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
        atomic_json(self.args.output / "pitch-circle-result.json", result)
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
            scores = score_selected(
                self.args.output,
                record,
                features,
                refs,
                rows,
                self.args.device,
                start,
                1200,
                loader=self.hooks.loader,
            )
            result = dict(
                artifact_version=VERSION,
                kind="candidate",
                seed=self.args.seed,
                identity=self.identity,
                status="complete",
                promoted=False,
                control_reproduced=True,
                control_proof=self.proof,
                selected_epoch=record["best_epoch"],
                stop_epoch=record["history"][-1]["epoch"],
                training_elapsed_seconds=record["elapsed_seconds"],
                elapsed_seconds=time.monotonic() - start,
                run_sha256=audio_digest(self.args.output / "run.json"),
                **scores,
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
            if time.monotonic() - start > 1200:
                raise TimeoutError("PitchCircle exceeded its fixed1200-second fit/scoring budget")
            atomic_json(self.args.output / "pitch-circle-result.json", result)
            return result
        except (ValueError, TimeoutError, RuntimeError, OSError) as exc:
            return self.failure(
                record,
                start,
                str(exc),
                budget=isinstance(exc, TimeoutError),
                resource=isinstance(exc, torch.OutOfMemoryError),
            )


def record_oom(args: argparse.Namespace, error: BaseException) -> None:
    path = args.output / "pitch-circle-identity.json"
    identity = json.loads(path.read_bytes()) if path.exists() else {}
    atomic_json(
        args.output / "pitch-circle-result.json",
        dict(
            artifact_version=VERSION,
            kind="candidate",
            seed=args.seed,
            identity=identity,
            status="resource_inconclusive",
            error=str(error),
            control_reproduced=False,
            promoted=False,
            method_gate={"qualifies_for_confirmation": False},
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize([json.loads(p.read_bytes()) for p in args.results], VERSION)
    result["result_sha256"] = {str(p): audio_digest(p) for p in args.results}
    result["summary_source_sha256"] = audio_digest(Path(__file__))
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
