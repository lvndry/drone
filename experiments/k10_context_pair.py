"""One prospective matched CPU crop-length/audio-exposure pair; no release promotion."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import io
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from experiments.k10_context_cost import FEATURES, INITIAL, LABELS, PLAN, SOURCE_NAMES, make_crop
from experiments.k10_v2_pilot import paired_batches, selection_key, validate_feature_manifest
from tools.musicfm_crop_io import file_identity, load, save, sha
from tools.prepare_musicfm_ram_cache import save_or_replace

if TYPE_CHECKING:
    import torch

COST = "6c04aa1b138f5ebf4483e0fd58a064cc13685f3507d72b738837e7d45a27dc4b"
COST_CONFIG = "f1cc26a5211be8f9ce149572e45e5be39b2a5a5b575633539df213ab8aa101a8"
COST_CONTROLLER = "418e6919ceb0a1464d26b41aa4df4c3005454db9d6fe6af8736d30f3a7195c35"
UPDATES, EVERY, EVENTS = 8280, 69, 120
CAPS = {
    "wall_seconds": 21600,
    "cpu_seconds": 40000,
    "rss_bytes": 8 * 1024**3,
    "output_bytes": 512 * 1024**2,
}


def make_plan(cost_config: Path, cost: Path, controller: Path, groups: Path) -> dict[str, Any]:
    if (sha(cost_config), sha(cost), sha(controller)) != (COST_CONFIG, COST, COST_CONTROLLER):
        raise ValueError("Require the exact reviewed CPU cost and resource evidence")
    value, result, receipt = load(cost_config), load(cost), load(controller)
    if not receipt["complete"] or receipt["worker_exit"] != 0 or result["task_scores_computed"]:
        raise ValueError("Incomplete or task-scored qualification")
    root = Path(__file__).resolve().parents[1]
    if any(sha(root / n) != h for n, h in value["source_sha256"].items()):
        raise ValueError("Qualified shared numerical/resource source changed")
    p = load(Path(value["plan"]))
    if sha(groups) != p["uncertainty_groups_sha256"]:
        raise ValueError("Uncertainty groups differ from admitted data contract")
    old = root / "experiments/studies/k10-v2-pilot/data/results/expanded.json.gz"
    teacher_starts = [x for x in value["protected_starts"] if x["pid"] in (1294, 1307)]
    value.update(
        status="PROPOSED_CONTEXT_PAIR_NOT_AUTHORIZED",
        crop_frames=[320, 65],
        updates_per_shape=UPDATES,
        events=120,
        updates_per_event=EVERY,
        caps=CAPS.copy(),
        full_task_fit_allowed=True,
        not_before_epoch=None,
        deadline_epoch=None,
        protected_starts=[x for x in value["protected_starts"] if x["pid"] not in (1294, 1307)],
        teacher={
            "root": "/home/lysk-admin/drone-experiments/musicfm-key-pilot-20261112-seed13-051e4ad",
            "starts": teacher_starts,
            "config_sha256": "069d1afc396c65fcaa1d7cb993db7d4c930aec93eae377a0a611842dfcbbac28",
        },
        groups=str(groups),
        groups_sha256=sha(groups),
        cost=str(cost),
        cost_config=str(cost_config),
        cost_controller=str(controller),
        cost_sha256=COST,
        cost_config_sha256=COST_CONFIG,
        cost_controller_sha256=COST_CONTROLLER,
        historical_gpu_control=str(old),
        historical_gpu_control_sha256=sha(old),
        selection="equal-domain exact; symmetric MIREX tie-break; earliest tie",
        candidate_crop_frames=65,
        control_crop_frames=320,
        continuation="65 gains>=5 FMA exact and BP exact/symmetric/asymmetric nondecline vs320; historical control guard must pass",
        loss_target_policy="pinned reference transposed by negative pitch-bin shift",
        uncertainty={
            "bootstrap_replicates": 10000,
            "seed": 20261112,
            "scope": "descriptive adaptive development; one seed; metadata clusters only",
        },
    )
    value["source_sha256"] = {
        n: sha(root / n)
        for n in (*SOURCE_NAMES, "experiments/k10_context_pair.py", "tools/run_k10_context_pair.py")
    }
    return value


def contract(v: dict[str, Any]) -> None:
    if (
        v["crop_frames"] != [320, 65]
        or v["updates_per_shape"] != 8280
        or v["events"] != 120
        or v["updates_per_event"] != 69
        or v["batch"] != 32
        or v["seed"] != 13
        or v["cpus"] != [14, 15]
        or v["torch_threads"] != 2
        or v["interop_threads"] != 1
        or v["nice"] != 19
        or v["caps"] != CAPS
        or v["cost_sha256"] != COST
        or v["candidate_crop_frames"] != 65
        or v["control_crop_frames"] != 320
        or v["full_task_fit_allowed"] is not True
        or v["plan_sha256"] != PLAN
        or v["labels_sha256"] != LABELS
        or v["features_sha256"] != FEATURES
        or len(v["rows"]) != 5638
        or len({r["id"] for r in v["rows"]}) != 5638
        or v["runtime"] != {"python": "3.12.3", "torch": "2.15.0+cu132", "numpy": "2.5.4"}
        or {r["pid"] for r in v["protected_starts"]} != {1478814, 3448680, 1481756, 3451013}
        or len(v["protected_starts"]) != 4
        or {r["pid"] for r in v["teacher"]["starts"]} != {1294, 1307}
        or v["teacher"]["config_sha256"]
        != "069d1afc396c65fcaa1d7cb993db7d4c930aec93eae377a0a611842dfcbbac28"
    ):
        raise ValueError("Frozen matched CPU pair contract differs")


def metrics(ids: list[str], predictions: list[int], refs: dict[str, int]) -> dict[str, Any]:
    from drone.keys import score_mirex

    y = [refs[i] for i in ids]
    exact, sym = score_mirex(predictions, y)
    asym = score_mirex(predictions, y, symmetric=False)[1]
    return {
        "ids": ids,
        "predictions": predictions,
        "tracks": len(ids),
        "exact_correct": sum(a == b for a, b in zip(predictions, y, strict=True)),
        "symmetric_units10": round(sym * len(ids) * 10),
        "asymmetric_units10": round(asym * len(ids) * 10),
        "exact": exact,
        "mirex": sym,
        "mirex_asymmetric": asym,
    }


def checkpoint(
    path: Path,
    model: torch.nn.Module,
    provenance: dict[str, Any],
    watch: Callable[[], None],
    *,
    artifact_format: str = "EXPERIMENTAL_K10_CONTEXT_STATE_DICT_V1",
) -> str:
    import torch

    watch()
    temporary = path.with_suffix(".pending")
    torch.save(
        {
            "format": artifact_format,
            "state_dict": model.state_dict(),
            "provenance": provenance,
        },
        temporary,
    )
    watch()
    temporary.replace(path)
    return sha(path)


def evaluate_complete(
    model: torch.nn.Module,
    full: list[np.ndarray],
    watch: Callable[[], None],
    *,
    device: str = "cpu",
) -> np.ndarray:
    import torch

    from train import evaluate_features

    counts = {"tracks": 0, "frames": 0}

    class Finite(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            watch()
            y = self.base(x)
            if (
                y.shape != (x.shape[0], 24)
                or y.dtype != torch.float32
                or not bool(torch.isfinite(y).all())
            ):
                raise ValueError("Invalid full-song FP32 logits")
            counts["tracks"] += x.shape[0]
            counts["frames"] += x.shape[0] * x.shape[2]
            return y

    predictions = evaluate_features(Finite(), full, device, batch=4)
    if (
        predictions.shape != (len(full),)
        or not np.issubdtype(predictions.dtype, np.integer)
        or np.any((predictions < 0) | (predictions >= 24))
        or counts["tracks"] != len(full)
        or counts["frames"] != sum(x.shape[1] for x in full)
    ):
        raise ValueError("Incomplete full-song prediction coverage/shape/range")
    return predictions


def train_arm(
    length: int,
    p: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    array: Callable[[str], np.ndarray],
    output: Path,
    provenance: dict[str, Any],
    watch: Callable[[], None],
    *,
    model_factory: Callable[[], torch.nn.Module] | None = None,
    expected_init: str | None = None,
    device: str = "cpu",
    artifact_format: str = "EXPERIMENTAL_K10_CONTEXT_STATE_DICT_V1",
    result_status: str = "COMPLETE_CONTEXT_ARM_NOT_PROMOTED",
) -> dict[str, Any]:
    import torch

    from drone.features import window
    from drone.keys import transpose_key
    from drone.model import KeyCNN

    torch.manual_seed(13)
    factory = KeyCNN if model_factory is None else model_factory
    model = factory().to(device)
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    if digest.hexdigest() != (INITIAL if expected_init is None else expected_init):
        raise ValueError("Fresh initialization differs from qualified seed13")
    checkpoint_options = (
        {}
        if artifact_format == "EXPERIMENTAL_K10_CONTEXT_STATE_DICT_V1"
        else {"artifact_format": artifact_format}
    )
    evaluation_options = {} if device == "cpu" else {"device": device}
    output.mkdir(exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EVENTS)
    panels = {"beatport": p["bp_selection_guard"], "fmak": p["fma_inner_selection"]}
    ids = panels["beatport"] + panels["fmak"]
    refs = {i: r["ref"] for i, r in rows.items()}
    full = [window(array(i)).astype(np.float32) for i in ids]
    common, temporal, bp = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    exposure = Counter()
    coverage = Counter()
    history = []
    loss_sum = 0.0
    best = None
    best_metrics = None
    best_update = None
    start = time.monotonic()
    for update, batch in enumerate(
        paired_batches(p["bp_fit"], p["expanded_fma_fit"], updates=UPDATES), 1
    ):
        watch()
        xs = []
        targets = []
        for draw in batch:
            x, w = make_crop(array(draw.identity), draw, length)
            xs.append(x)
            targets.append(transpose_key(refs[draw.identity], -w["shift"]))
            common.update(
                f"{draw.domain}:{draw.identity}:{draw.augmentation_seed}:{w['shift']}\n".encode()
            )
            temporal.update(
                f"{draw.identity}:{draw.augmentation_seed}:{w['shift']}:{w['first']}\n".encode()
            )
            if draw.domain == "beatport":
                bp.update(
                    f"{draw.identity}:{draw.augmentation_seed}:{w['shift']}:{w['first']}\n".encode()
                )
            coverage[draw.identity] += 1
            exposure[draw.domain] += w["actual_frames"]
        model.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            model(torch.from_numpy(np.stack(xs)).to(device)), torch.tensor(targets, device=device)
        )
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        if any(
            t.grad is not None and not bool(torch.isfinite(t.grad).all())
            for t in model.parameters()
        ):
            raise ValueError("Nonfinite gradients; no optimizer commit")
        optimizer.step()
        loss_sum += float(loss.detach())
        if any(not bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
            raise ValueError("Nonfinite updated state")
        if update % EVERY:
            continue
        scheduler.step()  # Same timing as original K10 v2 driver.
        predictions = evaluate_complete(model, full, watch, **evaluation_options)
        measured = {
            d: metrics(
                domain_ids,
                predictions[
                    (0 if d == "beatport" else 126) : (126 if d == "beatport" else len(ids))
                ].tolist(),
                refs,
            )
            for d, domain_ids in panels.items()
        }
        key = selection_key(measured)
        if best is None or key > best:
            best, best_update, best_metrics = key, update, measured
            best_sha = checkpoint(
                output / "best.pt",
                model,
                {**provenance, "crop_frames": length, "selected_update": update},
                watch,
                **checkpoint_options,
            )
        history.append({"update": update, "train_ce": loss_sum / EVERY, "metrics": measured})
        loss_sum = 0.0
        save_or_replace(
            output / "progress.json",
            {
                "complete": False,
                "updates": update,
                "events": len(history),
                "best_update": best_update,
                "history": history,
            },
        )
        watch()
    if update != UPDATES or len(history) != EVENTS:
        raise ValueError("Incomplete fixed horizon")
    if best_metrics is None or best_update is None:
        raise ValueError("No selected checkpoint")
    final_sha = checkpoint(
        output / "final.pt",
        model,
        {**provenance, "crop_frames": length, "final_update": update},
        watch,
        **checkpoint_options,
    )
    if sha(output / "best.pt") != best_sha:
        raise ValueError("Selected checkpoint changed before reload")
    payload = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    expected_metadata = {**provenance, "crop_frames": length, "selected_update": best_update}
    if payload.get("format") != artifact_format or payload["provenance"] != expected_metadata:
        raise ValueError("Selected checkpoint metadata differs")
    restored = factory().to(device)
    restored.load_state_dict(payload["state_dict"], strict=True)
    reloaded = evaluate_complete(restored, full, watch, **evaluation_options)
    expected_predictions = (
        best_metrics["beatport"]["predictions"] + best_metrics["fmak"]["predictions"]
    )
    if not np.array_equal(reloaded, np.asarray(expected_predictions)):
        raise ValueError("Selected checkpoint reload prediction identity failed")
    result = {
        "status": result_status,
        "crop_frames": length,
        "updates": update,
        "events": EVENTS,
        "initial_state_sha256": digest.hexdigest(),
        "best_update": best_update,
        "best_metrics": best_metrics,
        "best_checkpoint_sha256": best_sha,
        "final_checkpoint_sha256": final_sha,
        "history": history,
        "common_draw_pitch_sha256": common.hexdigest(),
        "temporal_sha256": temporal.hexdigest(),
        "bp_augmented_sha256": bp.hexdigest(),
        "coverage": dict(coverage),
        "actual_frames": dict(exposure),
        "elapsed_seconds": time.monotonic() - start,
        "provenance": provenance,
        "confirmation_used": False,
        "calibration_used": False,
        "selected_reload_proof": {
            "strict_state_load": True,
            "tracks": len(ids),
            "frames": sum(x.shape[1] for x in full),
            "argmax_disagreements": 0,
            "scores_identical": True,
            "checkpoint_sha256": best_sha,
        },
    }
    watch()
    save(output / "result.json", result)
    return result


def compare(
    control: dict[str, Any], candidate: dict[str, Any], historical: dict[str, Any]
) -> dict[str, Any]:
    fields = (
        "initial_state_sha256",
        "common_draw_pitch_sha256",
        "coverage",
        "updates",
        "events",
        "provenance",
    )
    valid = (
        control["crop_frames"] == 320
        and candidate["crop_frames"] == 65
        and all(control[k] == candidate[k] for k in fields)
        and control["updates"] == 8280
        and control["events"] == 120
    )
    for arm in (control, candidate):
        proof = arm.get("selected_reload_proof", {})
        valid = valid and (
            arm.get("status") == "COMPLETE_CONTEXT_ARM_NOT_PROMOTED"
            and arm.get("confirmation_used") is False
            and arm.get("calibration_used") is False
            and arm["initial_state_sha256"] == INITIAL
            and proof.get("strict_state_load") is True
            and proof.get("scores_identical") is True
            and proof.get("tracks") == 376
            and proof.get("argmax_disagreements") == 0
        )
    for d in ("beatport", "fmak"):
        valid = (
            valid
            and control["best_metrics"][d]["ids"]
            == candidate["best_metrics"][d]["ids"]
            == historical["best_metrics"][d]["ids"]
        )
    if not valid:
        raise ValueError("Failed paired source/runtime/init/draw/horizon control")
    a, b = control["best_metrics"], candidate["best_metrics"]
    keys = ("exact_correct", "symmetric_units10", "asymmetric_units10")
    old = historical["best_metrics"]
    historical_guard = all(a[d][k] >= old[d][k] for d in ("beatport", "fmak") for k in keys)
    gate = b["fmak"]["exact_correct"] >= a["fmak"]["exact_correct"] + 5 and all(
        b["beatport"][k] >= a["beatport"][k] for k in keys
    )
    return {
        "technical_pair_valid": True,
        "canonical_candidate_crop": 65,
        "FMA_required_gain": 5,
        "FMA_exact_gain": b["fmak"]["exact_correct"] - a["fmak"]["exact_correct"],
        "context_exposure_gate": gate,
        "historical_CPU320_noninferiority_guard": historical_guard,
        "historical_gaps": {
            d: {k: a[d][k] - old[d][k] for k in keys} for d in ("beatport", "fmak")
        },
        "continuation_earned": gate and historical_guard,
        "promotion_authorized": False,
        "historical_comparison_scope": "CPU versus CUDA and two versus four Torch threads; score guard is not trajectory reproduction",
    }


def movements(
    control: dict[str, Any],
    candidate: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    groups: dict[str, str],
) -> dict[str, Any]:
    all_records = []
    domains = {}
    for domain in ("beatport", "fmak"):
        a, b = control["best_metrics"][domain], candidate["best_metrics"][domain]
        counts = Counter()
        strata = {name: Counter() for name in ("<2min", "2-<5min", "5-10min", ">10min")}
        stratum_groups = {name: set() for name in strata}
        for identity, x, y in zip(a["ids"], a["predictions"], b["predictions"], strict=True):
            ref = rows[identity]["ref"]
            seconds = rows[identity]["canonical_samples"] / 22050
            label = (
                "<2min"
                if seconds < 120
                else "2-<5min"
                if seconds < 300
                else "5-10min"
                if seconds <= 600
                else ">10min"
            )
            category = (
                "both_correct"
                if x == ref and y == ref
                else "repaired"
                if y == ref
                else "new_error"
                if x == ref
                else "both_wrong"
            )
            counts[category] += 1
            strata[label][category] += 1
            stratum_groups[label].add(groups[identity])
            all_records.append(
                {
                    "id": identity,
                    "domain": domain,
                    "ref": ref,
                    "control_prediction": x,
                    "candidate_prediction": y,
                    "movement": category,
                    "duration_seconds": seconds,
                    "duration_stratum": label,
                    "uncertainty_group": groups[identity],
                }
            )
        domains[domain] = {
            "paired_counts": dict(counts),
            "duration_strata": {
                label: {
                    "tracks": sum(c.values()),
                    "clusters": len(stratum_groups[label]),
                    "paired_counts": dict(c),
                }
                for label, c in strata.items()
            },
        }
    return {
        "domains": domains,
        "selected_per_ID": all_records,
        "scope": "development; BP previews do not establish full-length house accuracy",
    }


def uncertainty(
    control: dict[str, Any], candidate: dict[str, Any], refs: dict[str, int], groups: dict[str, str]
) -> dict[str, Any]:
    from drone.keys import score_mirex

    result = {}
    for d in ("beatport", "fmak"):
        a, b = control["best_metrics"][d], candidate["best_metrics"][d]
        ids = a["ids"]
        values = []
        for i, x, y in zip(ids, a["predictions"], b["predictions"], strict=True):

            def scores(v: int) -> np.ndarray:
                return np.array(
                    [
                        v == refs[i],
                        score_mirex([v], [refs[i]])[1],
                        score_mirex([v], [refs[i]], symmetric=False)[1],
                    ],
                    dtype=float,
                )

            values.append(scores(y) - scores(x))
        names = sorted({groups[i] for i in ids})
        sums = []
        sizes = []
        for name in names:
            indices = [n for n, i in enumerate(ids) if groups[i] == name]
            sums.append(np.sum([values[n] for n in indices], axis=0))
            sizes.append(len(indices))
        sums = np.asarray(sums)
        sizes = np.asarray(sizes)
        rng = np.random.default_rng(20261112)
        intervals = []
        for _ in range(10000):
            take = rng.integers(0, len(names), len(names))
            intervals.append(sums[take].sum(0) / sizes[take].sum())
        lo, hi = np.quantile(intervals, [0.025, 0.975], axis=0)
        result[d] = {
            "tracks": len(ids),
            "clusters": len(names),
            "largest_cluster": max(sizes).item(),
            "metrics": {
                k: {
                    "delta": float(np.mean(values, axis=0)[n]),
                    "ci95": [float(lo[n]), float(hi[n])],
                }
                for n, k in enumerate(("exact", "symmetric", "asymmetric"))
            },
        }
    return {
        "scope": "descriptive adaptive development; metadata clustering only; one paired seed; not fresh test",
        "replicates": 10000,
        "seed": 20261112,
        "domains": result,
    }


def worker(config: Path, approval: Path, output: Path, guardian: int) -> None:
    import torch

    from tools.run_k10_context_pair import worker_watch

    v, permit = load(config), load(approval)
    contract(v)
    if (
        os.getppid() != guardian
        or permit.get("authorized") is not True
        or permit.get("allow_context_pair") is not True
        or permit.get("allow_training") is not True
        or permit.get("config_sha256") != sha(config)
    ):
        raise ValueError("Exact private matched-pair approval required")
    watch = worker_watch(v, output.parent, guardian, sha(config))
    watch()
    root = Path(__file__).resolve().parents[1]
    if any(sha(root / n) != h for n, h in v["source_sha256"].items()):
        raise ValueError("Source changed")
    for k in (
        "plan",
        "labels",
        "features",
        "groups",
        "cost",
        "cost_config",
        "cost_controller",
        "historical_gpu_control",
    ):
        if sha(Path(v[k])) != v[k + "_sha256"]:
            raise ValueError("Bound data/evidence changed")
    base = load(Path(v["cost_config"]))
    for field in (
        "rows",
        "plan",
        "plan_sha256",
        "labels_sha256",
        "features_sha256",
        "module_receipts",
        "native_cpu_sha256",
        "interpreter",
        "runtime",
        "cpus",
        "torch_threads",
        "interop_threads",
        "nice",
    ):
        if v[field] != base[field]:
            raise ValueError("Qualified data/runtime invariant differs: " + field)
    validate_feature_manifest(load(Path(v["plan"])), load(Path(v["features"])))
    for name, receipt in v["module_receipts"].items():
        module = importlib.import_module(name)
        if module.__file__ is None or sha(Path(module.__file__)) != receipt["sha256"]:
            raise ValueError("Runtime module changed")
    if (
        sha(Path(sys.executable)) != v["interpreter"]["sha256"]
        or any(sha(Path(p)) != h for p, h in v["native_cpu_sha256"].items())
        or str(torch.__version__) != v["runtime"]["torch"]
        or np.__version__ != v["runtime"]["numpy"]
        or sys.version.split()[0] != v["runtime"]["python"]
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or torch.cuda.is_initialized()
    ):
        raise ValueError("CPU runtime differs")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output.mkdir(exist_ok=False)
    rows = {r["id"]: r for r in v["rows"]}
    for r in rows.values():
        watch()
        path = Path(r["cache_path"])
        before = file_identity(path)
        if (
            list(before) != list(r["cache_identity"])
            or sha(path) != r["cache_sha256"]
            or file_identity(path) != before
        ):
            raise ValueError("CQT identity changed")

    @lru_cache(maxsize=64)
    def array(identity: str) -> np.ndarray:
        watch()
        r = rows[identity]
        path = Path(r["cache_path"])
        if list(file_identity(path)) != list(r["cache_identity"]):
            raise ValueError("CQT stat changed")
        a = np.load(path, allow_pickle=False, mmap_mode="r")
        if (
            a.shape != (208, r["frames"])
            or a.dtype != np.float16
            or not np.isfinite(a).all()
            or list(file_identity(path)) != list(r["cache_identity"])
        ):
            raise ValueError("Invalid CQT")
        return a

    p = load(Path(v["plan"]))
    provenance = {
        "config_sha256": sha(config),
        "source_sha256": v["source_sha256"],
        "plan_sha256": v["plan_sha256"],
        "labels_sha256": v["labels_sha256"],
        "features_sha256": v["features_sha256"],
        "runtime": v["runtime"],
        "modules": v["module_receipts"],
        "threads": 2,
        "seed": 13,
    }
    arms = []
    for length in (320, 65):
        array.cache_clear()
        arms.append(train_arm(length, p, rows, array, output / f"crop-{length}", provenance, watch))
    old = load(Path(v["historical_gpu_control"]))
    if arms[0]["bp_augmented_sha256"] != old["bp_trace_sha256"]:
        raise ValueError("CPU320 crop/pitch replay differs from historical source")
    gate = compare(*arms, old)
    group_rows = list(
        csv.DictReader(io.StringIO(gzip.decompress(Path(v["groups"]).read_bytes()).decode()))
    )
    groups = {r["id"]: r["derived_uncertainty_group"] for r in group_rows}
    ci = uncertainty(*arms, {i: r["ref"] for i, r in rows.items()}, groups)
    errors = movements(*arms, rows, groups)
    watch()
    save(
        output / "result.json",
        {
            "status": "COMPLETE_MATCHED_CONTEXT_PAIR_NOT_PROMOTED",
            "arms": [
                {"path": f"crop-{n}/result.json", "sha256": sha(output / f"crop-{n}/result.json")}
                for n in (320, 65)
            ],
            "gate": gate,
            "uncertainty": ci,
            "paired_errors_and_duration": errors,
            "provenance": provenance,
            "confirmation_used": False,
            "calibration_used": False,
            "promotion_authorized": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    planning = sub.add_parser("plan")
    for n in ("cost-config", "cost", "controller", "groups", "output"):
        planning.add_argument("--" + n, type=Path, required=True)
    sealing = sub.add_parser("seal")
    for n in ("config", "output"):
        sealing.add_argument("--" + n, type=Path, required=True)
    for n in ("begin", "end"):
        sealing.add_argument("--" + n, required=True)
    run = sub.add_parser("worker")
    for n in ("config", "approval", "output"):
        run.add_argument("--" + n, type=Path, required=True)
    run.add_argument("--guardian", type=int, required=True)
    args = parser.parse_args()
    if args.mode == "plan":
        save(args.output, make_plan(args.cost_config, args.cost, args.controller, args.groups))
    elif args.mode == "seal":
        from tools.prepare_musicfm_waveforms import seal

        contract(load(args.config))
        save(args.output, seal(args.config, args.begin, args.end))
    else:
        worker(args.config, args.approval, args.output, args.guardian)


if __name__ == "__main__":
    main()
