"""Fixed K10 EMA probe; training and checkpoint selection remain ordinary RAW."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
from torch import nn

from drone.features import CROP_FRAMES, audio_digest, window
from experiments.small import atomic_json

ModelT = TypeVar("ModelT", bound=nn.Module)

VERSION = "k10-ema-final-bn-v1"
DECAY = 0.99
SCHEDULE_SEED = 20261111
BUDGET_SECONDS = 1200
STUDY = Path(__file__).parent / "studies" / "k10-ema"
PROTOCOL = STUDY / "PROTOCOL.md"
REVIEWED = STUDY / "data" / "reviewed-sources.json"


@contextmanager
def preserve_rng() -> Generator[None]:
    """Do not let diagnostic work change the ordinary training trajectory."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def make_shadow(model: ModelT) -> ModelT:
    """Initialize explicitly from theta_0, without random model construction."""
    return copy.deepcopy(model).requires_grad_(False)


def update_shadow(shadow: nn.Module, model: nn.Module) -> None:
    """Average parameters only, including the first update from theta_0."""
    with torch.no_grad():
        for (name, target), (other, source) in zip(
            shadow.named_parameters(), model.named_parameters(), strict=True
        ):
            if name != other:
                raise ValueError("EMA parameter names differ")
            target.mul_(DECAY).add_(source.detach(), alpha=1 - DECAY)


def best_shadow(shadow: nn.Module, model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot EMA parameters with contemporaneous RAW buffers, never averaged BN."""
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for name, value in shadow.named_parameters():
        state[name] = value.detach().clone()
    return state


def bn_schedule(
    rows: list[dict[str, Any]], arrays: list[np.ndarray], fitting: list[int]
) -> list[dict[str, Any]]:
    """Freeze fitting-only center crops and batch membership without training RNG."""
    if len(fitting) != len(set(fitting)) or not fitting:
        raise ValueError("BN fitting indices must be nonempty and unique")
    result = []
    for position, index in enumerate(np.random.default_rng(SCHEDULE_SEED).permutation(fitting)):
        row_index = int(index)
        row = rows[row_index]
        array = arrays[row_index]
        if row["role"] != "train" or row["domain"] not in ("beatport", "fmak"):
            raise ValueError("BN can use fitting rows only")
        frames = array.shape[1]
        result.append(
            dict(
                index=row_index,
                id=row["id"],
                audio_sha256=row["sha256"],
                start=max(0, (frames - CROP_FRAMES) // 2),
                length=min(frames, CROP_FRAMES),
                padded_frames=max(0, CROP_FRAMES - frames),
                batch=position // 32,
                position=position,
                shift=0,
            )
        )
    return result


def schedule_batches(
    arrays: list[np.ndarray], schedule: list[dict[str, Any]]
) -> Iterator[torch.Tensor]:
    for offset in range(0, len(schedule), 32):
        features = []
        for item in schedule[offset : offset + 32]:
            feature = window(arrays[item["index"]], 0).astype(np.float32)
            if feature.shape[1] < CROP_FRAMES:
                feature = np.pad(feature, ((0, 0), (0, CROP_FRAMES - feature.shape[1])))
            else:
                feature = feature[:, item["start"] : item["start"] + CROP_FRAMES]
            features.append(feature)
        yield torch.from_numpy(np.stack(features))


def check_budget(start: float, budget_seconds: int = BUDGET_SECONDS) -> None:
    if time.monotonic() - start > budget_seconds:
        raise TimeoutError(f"Fit exceeded {budget_seconds}-second total training/scoring budget")


def recalibrate_bn(
    source: ModelT,
    arrays: list[np.ndarray],
    schedule: list[dict[str, Any]],
    device: str,
    start: float,
    *,
    expected_bn: int = 6,
) -> ModelT:
    """Reset BN only; dropout stays disabled; source and RNG remain unchanged."""
    with preserve_rng():
        model = copy.deepcopy(source)
        model.eval()
        layers = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        if len(layers) != expected_bn:
            raise ValueError(f"Expected {expected_bn} BatchNorm2d layers, found {len(layers)}")
        momenta = [layer.momentum for layer in layers]
        for layer in layers:
            layer.reset_running_stats()
            layer.momentum = None
            layer.train()
        with torch.inference_mode():
            for batch in schedule_batches(arrays, schedule):
                check_budget(start)
                model(batch.to(device))
        for layer, momentum in zip(layers, momenta, strict=True):
            layer.momentum = momentum
        model.eval()
        check_budget(start)
        return model


def evaluate_logits(
    model: nn.Module,
    features: list[np.ndarray],
    device: str,
    start: float,
    *,
    budget_seconds: int = BUDGET_SECONDS,
) -> np.ndarray:
    """Mirror evaluate_features grouping and true-length inference exactly."""
    model.eval()
    output = np.empty((len(features), 24), dtype=np.float32)
    for offset in range(0, len(features), 4):
        groups: dict[int, list[tuple[int, np.ndarray]]] = {}
        for index, feature in enumerate(features[offset : offset + 4], start=offset):
            groups.setdefault(feature.shape[1], []).append((index, feature))
        for entries in groups.values():
            check_budget(start, budget_seconds)
            indices, arrays = zip(*entries, strict=True)
            with torch.inference_mode():
                output[list(indices)] = (
                    model(torch.from_numpy(np.stack(arrays)).to(device)).cpu().numpy()
                )
    check_budget(start, budget_seconds)
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite EMA diagnostic logits")
    return output


def read_reference(path: Path, arm: str, seed: int) -> dict[str, Any]:
    run = json.loads(path.read_bytes())
    if run["status"] != "complete" or run["arm"] != arm or run["seed"] != seed:
        raise ValueError("Reference must be completed matching arm and seed")
    if run["calibration_used"] or run["final_holdout_used"]:
        raise ValueError("Reference crossed development boundary")
    return run


def assert_reproduced(
    current: dict[str, Any],
    reference: dict[str, Any],
    *,
    current_state: dict[str, torch.Tensor] | None = None,
    reference_state: dict[str, torch.Tensor] | None = None,
) -> None:
    """Exact comparisons: no tuned tolerances, seed selection or metric-only match."""
    for key in (
        "arm",
        "seed",
        "config",
        "feature_manifest_sha256",
        "fit_ids",
        "validation_ids",
        "history",
        "best_epoch",
        "early_stopped",
        "domain_metrics",
    ):
        if current[key] != reference[key]:
            raise ValueError(f"RAW reproduction failed: {key}")
    if reference_state is not None:
        if current_state is None or current_state.keys() != reference_state.keys():
            raise ValueError("RAW reproduction failed: checkpoint keys")
        if any(
            current_state[k].dtype != v.dtype or not torch.equal(current_state[k].cpu(), v.cpu())
            for k, v in reference_state.items()
        ):
            raise ValueError("RAW reproduction failed: checkpoint state")


def method_gate(outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = outputs["raw_bn"]["metrics"]
    candidate = outputs["ema_bn"]["metrics"]
    deltas = {
        domain: {
            metric: candidate[domain][metric] - baseline[domain][metric]
            for metric in ("mirex", "exact")
        }
        for domain in ("beatport", "fmak")
    }
    qualifies = (
        deltas["fmak"]["mirex"] >= 0.01
        and deltas["beatport"]["mirex"] >= 0
        and all(deltas[domain]["exact"] >= 0 for domain in deltas)
    )
    return dict(delta=deltas, qualifies_for_confirmation=qualifies)


def raw_accuracy_safeguard(outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Separate safeguard: BN effects cannot conceal a RAW accuracy regression."""
    raw = outputs["raw"]["metrics"]
    candidate = outputs["ema_bn"]["metrics"]
    deltas = {
        domain: {
            metric: candidate[domain][metric] - raw[domain][metric] for metric in ("mirex", "exact")
        }
        for domain in ("beatport", "fmak")
    }
    return dict(
        delta=deltas,
        passes=all(value >= 0 for metrics in deltas.values() for value in metrics.values()),
    )


class EmaProbe:
    """Optional hooks only; never owns the training loop or model selection."""

    def __init__(
        self,
        args: argparse.Namespace,
        record: dict[str, Any],
        model: nn.Module,
        rows: list[dict[str, Any]],
        arrays: list[np.ndarray],
        fitting: list[int],
    ) -> None:
        self.args = args
        self.rows = rows
        self.arrays = arrays
        self.reference = read_reference(args.reference_run, "joint", args.seed)
        self.bp = read_reference(args.reference_bp, "control", args.seed)
        self.startup_checks(record, fitting)
        self.shadow = make_shadow(model)
        self.schedule = bn_schedule(rows, arrays, fitting)
        schedule_path = args.output / "bn-schedule.json"
        atomic_json(
            schedule_path,
            dict(
                artifact_version=VERSION,
                seed=SCHEDULE_SEED,
                crop=CROP_FRAMES,
                batch=32,
                fitting_only=True,
                rows=self.schedule,
            ),
        )
        protocol_path = args.output / "EMA_PROTOCOL.md"
        protocol_path.write_bytes(PROTOCOL.read_bytes())
        (args.output / "ema-reviewed-sources.json").write_bytes(REVIEWED.read_bytes())
        self.identity = dict(
            artifact_version=VERSION,
            decay=DECAY,
            initial_shadow="theta_0",
            parameters_only=True,
            epoch_selection="ordinary_raw_earliest_best",
            bn_schedule_sha256=audio_digest(schedule_path),
            protocol_sha256=audio_digest(protocol_path),
            reference_run_sha256=audio_digest(args.reference_run),
            reference_bp_sha256=audio_digest(args.reference_bp),
            reference_source_sha256=self.reference["source_sha256"],
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
            reviewed_sources_sha256=audio_digest(REVIEWED),
            source_sha256=record["source_sha256"],
            seed=args.seed,
            feature_manifest_sha256=record["feature_manifest_sha256"],
            config=record["config"],
            torch=str(torch.__version__),
            numpy=np.__version__,
            device_guard="device_setup:8GiB_free,fraction0.04,TF32_off",
        )
        if args.pilot_run:
            self.identity["pilot_run_sha256"] = audio_digest(args.pilot_run)
        atomic_json(args.output / "ema-identity.json", self.identity)
        self.updates = 0

    def startup_checks(self, record: dict[str, Any], fitting: list[int]) -> None:
        if self.args.arm != "joint" or len(fitting) != 2206:
            raise ValueError("EMA protocol requires the fixed 2206-row joint fitting arm")
        for key in ("config", "feature_manifest_sha256", "fit_ids", "validation_ids"):
            if record[key] != self.reference[key]:
                raise ValueError(f"Reference preflight mismatch: {key}")
        for key in ("config", "feature_manifest_sha256", "validation_ids", "source_sha256"):
            if self.reference[key] != self.bp[key]:
                raise ValueError(f"Archived BP and joint reference differ: {key}")
        beatport_ids = [
            self.rows[index]["id"] for index in fitting if self.rows[index]["domain"] == "beatport"
        ]
        if beatport_ids != self.bp["fit_ids"]:
            raise ValueError("Archived BP fitting identities differ")
        reviewed = json.loads(REVIEWED.read_bytes())
        if self.reference["source_sha256"] != reviewed["archived"]:
            raise ValueError("Reference source is not the frozen archived commit673")
        for name, digest in reviewed["reviewed_base"].items():
            if (
                name != "experiments/k10_multidomain.py"
                and record["source_sha256"].get(name) != digest
            ):
                raise ValueError(f"Source differs from reviewed base: {name}")
        if self.args.seed != 13:
            if self.args.pilot_run is None:
                raise ValueError("Seeds 37/73 require the passing canonical seed13 pilot")
            pilot = json.loads(self.args.pilot_run.read_bytes())
            if (
                pilot.get("status") != "complete"
                or pilot.get("seed") != 13
                or pilot.get("artifact_version") != VERSION
                or not pilot.get("raw_reproduced")
                or not pilot.get("method_gate", {}).get("qualifies_for_confirmation")
            ):
                raise ValueError("Confirmation requires a completed passing seed13 pilot")
            identity = pilot["identity"]
            for key in ("source_sha256", "config", "feature_manifest_sha256"):
                if identity[key] != record[key]:
                    raise ValueError(f"Confirmation changed {key}")
            if identity["protocol_sha256"] != audio_digest(PROTOCOL):
                raise ValueError("Confirmation changed protocol")

    def budget_exhausted(self, start: float) -> None:
        atomic_json(
            self.args.output / "ema-result.json",
            dict(
                artifact_version=VERSION,
                seed=self.args.seed,
                identity=self.identity,
                status="budget_exhausted",
                raw_reproduced=False,
                updates=self.updates,
                elapsed_seconds=time.monotonic() - start,
                promoted=False,
                method_gate={"qualifies_for_confirmation": False},
            ),
        )

    def step(self, model: nn.Module) -> None:
        update_shadow(self.shadow, model)
        self.updates += 1

    def save_best(self, model: nn.Module) -> None:
        torch.save(best_shadow(self.shadow, model), self.args.output / "ema-selected.pt")

    def finish(
        self,
        record: dict[str, Any],
        features: list[np.ndarray],
        references: np.ndarray,
        validation_rows: list[dict[str, Any]],
        start: float,
    ) -> dict[str, Any]:
        from drone.checkpoints import load_checkpoint
        from experiments.k10_multidomain import domain_metrics

        result: dict[str, Any] = dict(
            artifact_version=VERSION,
            seed=self.args.seed,
            identity=self.identity,
            status="checking_raw_reproduction",
            raw_reproduced=False,
            updates=self.updates,
            selected_epoch=record["best_epoch"],
            stop_epoch=record["history"][-1]["epoch"],
            training_elapsed_seconds=record.get("elapsed_seconds"),
            calibration_used=False,
            final_holdout_used=False,
            promoted=False,
        )
        output_path = self.args.output / "ema-result.json"
        try:
            check_budget(start)
            with preserve_rng():
                raw, payload = load_checkpoint(self.args.output / "best.pt", self.args.device)
                reference_path = (
                    self.args.reference_checkpoint or self.args.reference_run.with_name("best.pt")
                )
                reference_state = None
                if reference_path.is_file():
                    reference_state = torch.load(
                        reference_path, map_location="cpu", weights_only=True
                    )["state_dict"]
                elif self.args.reference_checkpoint is not None:
                    raise ValueError("Explicit reference checkpoint is missing")
                assert_reproduced(
                    record,
                    self.reference,
                    current_state=payload["state_dict"],
                    reference_state=reference_state,
                )
                result.update(
                    raw_reproduced=True,
                    checkpoint_state_checked=reference_state is not None,
                    reference_checkpoint_sha256=audio_digest(reference_path)
                    if reference_state is not None
                    else None,
                )
                raw_logits = evaluate_logits(raw, features, self.args.device, start)
                if (
                    domain_metrics(raw_logits.argmax(1), references, validation_rows)
                    != record["domain_metrics"]
                ):
                    raise ValueError("RAW reproduction failed: final scoring predictions")
                ema = copy.deepcopy(raw)
                ema.load_state_dict(
                    torch.load(
                        self.args.output / "ema-selected.pt",
                        map_location=self.args.device,
                        weights_only=True,
                    )
                )
                models = {"raw": raw}
                timings = {}
                for name, model in (("raw_bn", raw), ("ema_bn", ema)):
                    began = time.monotonic()
                    models[name] = recalibrate_bn(
                        model, self.arrays, self.schedule, self.args.device, start
                    )
                    timings[name] = time.monotonic() - began
                outputs = {}
                for name, model in models.items():
                    logits = (
                        raw_logits
                        if name == "raw"
                        else evaluate_logits(model, features, self.args.device, start)
                    )
                    metrics = domain_metrics(logits.argmax(1), references, validation_rows)
                    artifact = dict(
                        ids=[row["id"] for row in validation_rows],
                        refs=references.tolist(),
                        logits=logits.tolist(),
                        predictions=logits.argmax(1).tolist(),
                        metrics=metrics,
                    )
                    path = self.args.output / f"ema-{name}-validation.json"
                    atomic_json(path, artifact)
                    state_path = self.args.output / f"ema-{name}-state.pt"
                    torch.save(model.state_dict(), state_path)
                    outputs[name] = dict(
                        metrics=metrics,
                        artifact=path.name,
                        sha256=audio_digest(path),
                        state_artifact=state_path.name,
                        state_sha256=audio_digest(state_path),
                    )
                result.update(
                    status="complete",
                    outputs=outputs,
                    method_gate=method_gate(outputs),
                    raw_accuracy_safeguard=raw_accuracy_safeguard(outputs),
                    bn_seconds=timings,
                    bp_metrics=self.bp["domain_metrics"],
                    raw_checkpoint_sha256=audio_digest(self.args.output / "best.pt"),
                    selected_ema_sha256=audio_digest(self.args.output / "ema-selected.pt"),
                )
                check_budget(start)
        except (ValueError, TimeoutError, RuntimeError, OSError) as error:
            result.update(
                status="budget_exhausted"
                if isinstance(error, TimeoutError)
                else "reproduction_or_protocol_failed",
                error=str(error),
                method_gate={"qualifies_for_confirmation": False},
            )
        result["elapsed_seconds"] = time.monotonic() - start
        atomic_json(output_path, result)
        return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed = {result["seed"]: result for result in results}
    if len(results) != 3 or sorted(by_seed) != [13, 37, 73]:
        raise ValueError("Need exactly declared seeds 13,37,73")
    pilot = by_seed[13]
    if not pilot["method_gate"]["qualifies_for_confirmation"]:
        raise ValueError("Canonical pilot did not qualify")
    for result in results:
        if result["status"] != "complete" or not result["raw_reproduced"]:
            raise ValueError("All RAW reproductions must pass")
        if result["artifact_version"] != VERSION:
            raise ValueError("Artifact version differs")
        for key in (
            "source_sha256",
            "config",
            "feature_manifest_sha256",
            "protocol_sha256",
            "bn_schedule_sha256",
        ):
            if result["identity"][key] != pilot["identity"][key]:
                raise ValueError(f"Confirmation changed {key}")
    means = {}
    for name in ("raw", "raw_bn", "ema_bn", "bp_control"):
        means[name] = {}
        for domain in ("beatport", "fmak"):
            means[name][domain] = {}
            for metric in ("mirex", "exact"):
                values = [
                    result["bp_metrics"][domain][metric]
                    if name == "bp_control"
                    else result["outputs"][name]["metrics"][domain][metric]
                    for result in results
                ]
                means[name][domain][metric] = statistics.mean(values)
    gate = method_gate({name: {"metrics": means[name]} for name in ("raw_bn", "ema_bn")})
    original_delta = {
        domain: means["ema_bn"][domain]["mirex"] - means["bp_control"][domain]["mirex"]
        for domain in ("beatport", "fmak")
    }
    original_pass = original_delta["fmak"] >= 0.02 and original_delta["beatport"] >= 0
    raw_guard = raw_accuracy_safeguard(
        {name: {"metrics": means[name]} for name in ("raw", "ema_bn")}
    )
    return dict(
        artifact_version=VERSION,
        means=means,
        method_gate=gate,
        original_hurdle_delta=original_delta,
        original_hurdle_pass=original_pass,
        raw_accuracy_safeguard=raw_guard,
        qualifies_for_external_freeze=(
            gate["qualifies_for_confirmation"] and original_pass and raw_guard["passes"]
        ),
        canonical_seed=13,
        promoted=False,
        uncertainty="Training-seed means; not independent generalization evidence",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize([json.loads(path.read_bytes()) for path in args.results])
    result["result_sha256"] = {str(path): audio_digest(path) for path in args.results}
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
