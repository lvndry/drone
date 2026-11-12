"""Prospective paired K10 data pilot. Planning is metadata-only; training needs a permit."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

VERSION = "k10-v2-paired-updates-v1"
SEED = 13
STEPS_PER_EVENT = 69
EVENTS = 120
TOTAL_UPDATES = STEPS_PER_EVENT * EVENTS
BATCH = 32
BP_NUMERATOR = 1230
REFERENCE_ROWS = 2206


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def keyed(purpose: str) -> int:
    return int(digest(f"{VERSION}:{SEED}:{purpose}".encode())[:16], 16)


@dataclass(frozen=True)
class Draw:
    identity: str
    domain: str
    augmentation_seed: int


class Cycle:
    def __init__(self, identities: list[str], domain: str) -> None:
        if not identities or len(identities) != len(set(identities)):
            raise ValueError("Sampling pools must be nonempty and ID-unique")
        self.pool = sorted(identities)
        self.order: list[str] = []
        self.rng = random.Random(keyed("order:" + domain))
        self.domain = domain
        self.position = 0
        self.ordinal = 0

    def take(self) -> Draw:
        if self.position == len(self.order):
            self.order = self.pool.copy()
            self.rng.shuffle(self.order)
            self.position = 0
        identity = self.order[self.position]
        self.position += 1
        result = Draw(identity, self.domain, keyed(f"augmentation:{self.domain}:{self.ordinal}"))
        self.ordinal += 1
        return result


def paired_batches(
    bp: list[str], fma: list[str], updates: int = TOTAL_UPDATES
) -> Iterator[list[Draw]]:
    bp_cycle, fm_cycle = Cycle(bp, "beatport"), Cycle(fma, "fmak")
    for step in range(updates):
        previous = step * BATCH * BP_NUMERATOR // REFERENCE_ROWS
        current = (step + 1) * BATCH * BP_NUMERATOR // REFERENCE_ROWS
        yield (
            [bp_cycle.take() for _ in range(current - previous)]
            + [fm_cycle.take() for _ in range(BATCH - current + previous)]
        )


def selection_key(metrics: dict[str, dict[str, Any]]) -> tuple[Fraction, Fraction]:
    """Equal-domain exact, then equal-domain symmetric MIREX; equal key keeps earlier."""
    exact, symmetric = Fraction(), Fraction()
    for domain in ("beatport", "fmak"):
        row = metrics[domain]
        exact += Fraction(row["exact_correct"], row["tracks"])
        symmetric += Fraction(row["symmetric_units10"], 10 * row["tracks"])
    return exact / 2, symmetric / 2


def make_plan(study: Path, bp_manifest: Path, label_export: Path) -> dict[str, Any]:
    from tools.check_v2_contract import check
    from tools.propose_v2_contract import load_input

    check(study)
    data = study / "data"
    tracks, _ = load_input(data / "input.json.gz")
    roles = {
        r["id"]: r
        for r in csv.DictReader(
            io.StringIO(gzip.decompress((data / "exposure-ledger.csv.gz").read_bytes()).decode())
        )
    }
    overlay = {
        r["id"]: r
        for r in csv.DictReader(
            io.StringIO(
                gzip.decompress((data / "eligibility-overlay.csv.gz").read_bytes()).decode()
            )
        )
    }
    unique: dict[tuple[str, str], str] = {}
    by_id = {t.identity: t for t in tracks}
    for track in sorted(tracks, key=lambda t: (t.old_role != "fit", t.identity)):
        if overlay[track.identity]["source_integrity_status"] == "PROPOSED_QUARANTINE":
            continue
        role = roles[track.identity]["proposed_role"]
        unique.setdefault((role, track.audio_sha256), track.identity)
    pools = {
        name: sorted(v for (r, _), v in unique.items() if r == "PROPOSED_" + name)
        for name in ("expanded_fit", "inner_selection", "development_confirmation")
    }
    restricted = sorted(t.identity for t in tracks if t.old_role == "fit")
    if not set(restricted) <= set(pools["expanded_fit"]):
        raise ValueError("Old fitting cohort is not nested")
    raw = bp_manifest.read_bytes()
    if digest(raw) != "39fe1842a544bd818fe48c38d9b0916d7f6c04f5531894b0396d6e6d24d1f5f5":
        raise ValueError("Different current BP identity/role source")
    bp = [r for r in json.loads(raw)["rows"] if r["domain"] == "beatport"]
    bp_fit = sorted(r["id"] for r in bp if r["role"] == "train")
    bp_guard = sorted(r["id"] for r in bp if r["role"] == "validation")
    if (
        len(bp_fit),
        len(bp_guard),
        len(restricted),
        len(pools["expanded_fit"]),
        len(pools["inner_selection"]),
        len(pools["development_confirmation"]),
    ) != (1230, 126, 976, 4035, 250, 997):
        raise ValueError("Reviewed cohort counts differ")
    bp_hashes = {r["id"]: r["sha256"] for r in bp}
    bindings = {
        identity: by_id[identity].audio_sha256
        for identity in pools["expanded_fit"] + pools["inner_selection"]
    }
    bindings.update(bp_hashes)
    fit_hashes = {bindings[i] for i in bp_fit + pools["expanded_fit"]}
    val_hashes = {bindings[i] for i in bp_guard + pools["inner_selection"]}
    if fit_hashes & val_hashes or len(fit_hashes) != 1230 + 4035:
        raise ValueError("Exact-container uniqueness/separation failure")
    label_bytes = label_export.read_bytes()
    labels = json.loads(gzip.decompress(label_bytes))["labels"]
    if set(labels) != set(bindings) or any(
        type(v) is not int or not 0 <= v < 24 for v in labels.values()
    ):
        raise ValueError("Numeric label export membership/encoding differs")
    root = Path(__file__).resolve().parents[1]
    bound_sources = {
        name: digest((root / name).read_bytes())
        for name in (
            "drone/features.py",
            "drone/model.py",
            "drone/keys.py",
            "drone/checkpoints.py",
            "train.py",
            "experiments/octave.py",
            "tools/prepare_k10_v2_pilot.py",
        )
    }
    return {
        "bound_source_sha256": bound_sources,
        "expected_refs": labels,
        "label_export_sha256": digest(label_bytes),
        "status": "PROPOSED_PILOT_NOT_AUTHORIZED",
        "version": VERSION,
        "seed": SEED,
        "driver_sha256": digest(Path(__file__).read_bytes()),
        "python_version": sys.version,
        "sources": {
            "bp_feature_manifest_sha256": digest(raw),
            "assignment_ledger_sha256": digest((data / "exposure-ledger.csv.gz").read_bytes()),
            "eligibility_overlay_sha256": digest(
                (data / "eligibility-overlay.csv.gz").read_bytes()
            ),
            "proposal_sha256": digest((data / "proposal.json").read_bytes()),
        },
        "bp_fit": bp_fit,
        "bp_selection_guard": bp_guard,
        "restricted_fma_fit": restricted,
        "expanded_fma_fit": pools["expanded_fit"],
        "fma_inner_selection": pools["inner_selection"],
        "confirmation_reserved_container_unique": pools["development_confirmation"],
        "confirmation_raw_rows": 998,
        "train_feature_bindings": bindings,
        "schedule": {
            "updates": TOTAL_UPDATES,
            "batch": BATCH,
            "events": EVENTS,
            "updates_per_event": STEPS_PER_EVENT,
            "bp_mass": [1230, 2206],
            "independent_early_stopping": False,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "cosine_T_max_events": 120,
            "wall_cap_seconds_per_arm": 1200,
        },
        "selection": "equal-domain exact;then equal-domain symmetric MIREX;then earliest event",
        "confirmation_policy": "never opened by train;passed fixed pilot gate earns separate root review",
    }


def validate_feature_manifest(plan: dict[str, Any], manifest: dict[str, Any]) -> None:
    from drone.features import FEATURE_SPEC
    from drone.keys import LABEL_SCHEMA

    if (
        not manifest.get("complete")
        or manifest["feature_spec"] != FEATURE_SPEC
        or manifest["label_schema"] != LABEL_SCHEMA
    ):
        raise ValueError("Incomplete/incompatible physical feature manifest")
    if manifest.get("label_export_sha256") != plan["label_export_sha256"]:
        raise ValueError("Pinned numeric-label export provenance differs")
    rows = manifest["rows"]
    expected = plan["train_feature_bindings"]
    if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
        raise ValueError(
            "Training manifest must contain exactly fit+selection;no confirmation/calibration"
        )
    for row in rows:
        if row["sha256"] != expected[row["id"]] or row["ref"] != plan["expected_refs"][row["id"]]:
            raise ValueError("Feature identity/label binding differs")
        if not row.get("cache_sha256") or row["frames"] < 8:
            raise ValueError("Missing feature content hash/frames")
        if not row.get("canonical_model_input_sha256"):
            raise ValueError("Canonical22050 model-input identity verification required")
    if (
        manifest.get("canonical_identity_contract")
        != "drone.features.decode_mp3/read_wav_mono/resample_linear:22050:float32le-v1"
    ):
        raise ValueError("Canonical decoder contract differs")
    canonical_roles: dict[str, set[str]] = {}
    canonical_labels: dict[str, set[int]] = {}
    fitting = set(plan["bp_fit"] + plan["expanded_fma_fit"])
    for row in rows:
        key = row["canonical_model_input_sha256"]
        canonical_roles.setdefault(key, set()).add("fit" if row["id"] in fitting else "selection")
        canonical_labels.setdefault(key, set()).add(row["ref"])
    if any(len(v) > 1 for v in canonical_roles.values()):
        raise ValueError("Canonical model input crosses fit/selection;root review required")
    if any(len(v) > 1 for v in canonical_labels.values()):
        raise ValueError("Canonical identical input has conflicting labels;no automatic repair")


def approve(plan_bytes: bytes, permit: dict[str, Any]) -> None:
    if permit.get("allow_training") is not True or permit.get("plan_sha256") != digest(plan_bytes):
        raise ValueError("Separate root-reviewed, hash-bound training permit required")


def train(
    plan_path: Path, feature_path: Path, arm: str, output: Path, permit_path: Path, device: str
) -> None:
    import numpy as np
    import torch

    from drone.checkpoints import save_checkpoint
    from drone.features import CROP_FRAMES, N_TOTAL, audio_digest, window
    from drone.keys import score_mirex, transpose_key
    from drone.model import KeyCNN, params_count
    from experiments.octave import device_setup
    from experiments.small import atomic_json, snapshot_sources
    from train import evaluate_features

    plan_bytes = plan_path.read_bytes()
    approve(plan_bytes, json.loads(permit_path.read_bytes()))
    plan = json.loads(
        gzip.decompress(plan_bytes) if plan_bytes.startswith(b"\x1f\x8b") else plan_bytes
    )
    if plan["driver_sha256"] != digest(Path(__file__).read_bytes()):
        raise ValueError("Driver differs from approved plan")
    root = Path(__file__).resolve().parents[1]
    if any(
        digest((root / name).read_bytes()) != expected
        for name, expected in plan["bound_source_sha256"].items()
    ):
        raise ValueError("Reviewed physical/model/evaluator sources changed")
    if output.exists():
        raise ValueError("Fresh output required;no silent restart")
    start = time.monotonic()
    manifest_raw = feature_path.read_bytes()
    manifest = json.loads(manifest_raw)
    validate_feature_manifest(plan, manifest)
    rows = {r["id"]: r for r in manifest["rows"]}
    paths = {}
    for identity, row in rows.items():
        path = (feature_path.parent / row["cache"]).resolve()
        if not path.is_relative_to(feature_path.parent.resolve()):
            raise ValueError("Cache escapes declared feature root")
        if audio_digest(path) != row["cache_sha256"]:
            raise ValueError("Feature content changed")
        paths[identity] = path
    device_setup(device)
    torch.manual_seed(SEED)
    model = KeyCNN().to(device)
    if params_count(model) != 109128:
        raise ValueError("Reviewed K10 architecture differs")
    state = hashlib.sha256()
    for name, value in model.state_dict().items():
        state.update(name.encode())
        state.update(value.detach().cpu().numpy().tobytes())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 120)
    output.mkdir(parents=True)
    record: dict[str, Any] = {
        "status": "running",
        "arm": arm,
        "device": device,
        "plan_sha256": digest(plan_bytes),
        "feature_manifest_sha256": digest(manifest_raw),
        "initial_state_sha256": state.hexdigest(),
        "source_sha256": snapshot_sources(output),
        "runtime_versions": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
        "history": [],
        "updates": 0,
        "confirmation_used": False,
        "calibration_used": False,
        "config": plan["schedule"],
        "bp_trace_sha256": None,
    }

    @lru_cache(maxsize=64)
    def array(identity: str) -> np.ndarray:
        data = np.load(paths[identity], allow_pickle=False, mmap_mode="r")
        if (
            data.shape != (N_TOTAL, rows[identity]["frames"])
            or data.dtype != np.float16
            or not np.isfinite(data).all()
        ):
            raise ValueError("Invalid physical CQT cache")
        return data

    val_ids = plan["bp_selection_guard"] + plan["fma_inner_selection"]
    refs = np.array([rows[i]["ref"] for i in val_ids])
    val_features = [window(array(i)).astype(np.float32) for i in val_ids]
    fma = plan[arm + "_fma_fit"]
    trace = hashlib.sha256()
    coverage: dict[str, Counter[str]] = {d: Counter() for d in ("beatport", "fmak")}
    used_frames: Counter[str] = Counter()
    best: tuple[Fraction, Fraction] | None = None
    loss_sum = 0.0
    for update, batch in enumerate(paired_batches(plan["bp_fit"], fma), 1):
        model.train()
        xs = []
        ys = []
        for draw in batch:
            rng = np.random.default_rng(draw.augmentation_seed)
            shift = int(rng.integers(-4, 5))
            feature = window(array(draw.identity), shift)
            first = (
                int(rng.integers(0, feature.shape[1] - CROP_FRAMES + 1))
                if feature.shape[1] >= CROP_FRAMES
                else 0
            )
            crop = feature[:, first : first + CROP_FRAMES].astype(np.float32)
            xs.append(np.pad(crop, ((0, 0), (0, max(0, CROP_FRAMES - crop.shape[1])))))
            ys.append(transpose_key(rows[draw.identity]["ref"], -shift))
            coverage[draw.domain][draw.identity] += 1
            used_frames[draw.domain] += min(CROP_FRAMES, feature.shape[1])
            if draw.domain == "beatport":
                trace.update(f"{draw.identity}:{draw.augmentation_seed}:{shift}:{first}\n".encode())
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            model(torch.from_numpy(np.stack(xs)).to(device)), torch.tensor(ys, device=device)
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach())
        record["updates"] = update
        if time.monotonic() - start > 1200:
            record.update(status="budget_exhausted", elapsed_seconds=time.monotonic() - start)
            atomic_json(output / "run.json", record)
            return
        if update % STEPS_PER_EVENT:
            continue
        scheduler.step()
        pred = evaluate_features(model, val_features, device, batch=4)
        metrics = {}
        for domain, indices in (("beatport", range(126)), ("fmak", range(126, len(val_ids)))):
            idx = list(indices)
            p, y = pred[idx], refs[idx]
            exact, sym = score_mirex(p.tolist(), y.tolist())
            asym = score_mirex(p.tolist(), y.tolist(), symmetric=False)[1]
            metrics[domain] = {
                "tracks": len(idx),
                "exact_correct": int(np.sum(p == y)),
                "symmetric_units10": round(sym * len(idx) * 10),
                "asymmetric_units10": round(asym * len(idx) * 10),
                "exact": exact,
                "mirex": sym,
                "mirex_asymmetric": asym,
                "ids": [val_ids[i] for i in idx],
                "predictions": p.tolist(),
            }
        key = selection_key(metrics)
        if best is None or key > best:
            best = key
            record.update(best_update=update, best_metrics=metrics)
            save_checkpoint(
                output / "best.pt",
                model,
                experiment=VERSION,
                arm=arm,
                seed=SEED,
                plan_sha256=digest(plan_bytes),
                selected_update=update,
            )
        record["history"].append(
            {
                "update": update,
                "train_ce": loss_sum / STEPS_PER_EVENT,
                "selection_exact": float(key[0]),
                "selection_mirex": float(key[1]),
            }
        )
        loss_sum = 0.0
        record.update(elapsed_seconds=time.monotonic() - start)
        if time.monotonic() - start > 1200:
            record.update(status="budget_exhausted", elapsed_seconds=time.monotonic() - start)
            atomic_json(output / "run.json", record)
            return
        atomic_json(output / "run.json", record)
    record.update(
        status="complete",
        bp_trace_sha256=trace.hexdigest(),
        checkpoint_sha256=audio_digest(output / "best.pt"),
        coverage={
            d: {
                "draws": sum(c.values()),
                "unique_ids": len(c),
                "pool_ids": len(plan["bp_fit"] if d == "beatport" else fma),
                "minimum_draws_observed": min(c.values()),
                "maximum_draws": max(c.values()),
                "used_feature_frames_including_repeats": used_frames[d],
            }
            for d, c in coverage.items()
        },
    )
    atomic_json(output / "run.json", record)


def pilot_gate(restricted: dict[str, Any], expanded: dict[str, Any]) -> dict[str, Any]:
    if restricted["arm"] != "restricted" or expanded["arm"] != "expanded":
        raise ValueError("Arm mismatch")
    for row in (restricted, expanded):
        if (
            row["status"] != "complete"
            or row["updates"] != TOTAL_UPDATES
            or row["confirmation_used"]
            or row["calibration_used"]
            or len(row["history"]) != EVENTS
        ):
            raise ValueError("Both arms must complete fixed updates without confirmation")
    for name in (
        "plan_sha256",
        "feature_manifest_sha256",
        "initial_state_sha256",
        "bp_trace_sha256",
        "source_sha256",
        "runtime_versions",
        "device",
        "config",
    ):
        if restricted[name] != expanded[name] or not restricted[name]:
            raise ValueError("Pairing/provenance mismatch:" + name)

    for domain, expected_count in (("beatport", 126), ("fmak", 250)):
        a, b = restricted["best_metrics"][domain], expanded["best_metrics"][domain]
        if (
            a["tracks"] != expected_count
            or b["tracks"] != expected_count
            or a["ids"] != b["ids"]
            or len(a["ids"]) != expected_count
            or len(set(a["ids"])) != expected_count
        ):
            raise ValueError("Selection ID ordering/counts differ")

    def rate(row: dict[str, Any], domain: str, field: str) -> Fraction:
        m = row["best_metrics"][domain]
        return Fraction(m[field], m["tracks"] * (10 if field.endswith("units10") else 1))

    fma_gain = rate(expanded, "fmak", "exact_correct") - rate(restricted, "fmak", "exact_correct")
    nondecline = all(
        rate(expanded, d, k) >= rate(restricted, d, k)
        for d in ("beatport",)
        for k in ("exact_correct", "symmetric_units10", "asymmetric_units10")
    )
    return {
        "status": "PILOT_SCREEN_NOT_RELEASE_EVIDENCE",
        "confirmation_earned": fma_gain >= Fraction(2, 100) and nondecline,
        "fma_exact_gain": float(fma_gain),
        "bp_exact_mirex_nondecline": nondecline,
        "fma_symmetric_mirex_gain_report_only": float(
            rate(expanded, "fmak", "symmetric_units10")
            - rate(restricted, "fmak", "symmetric_units10")
        ),
        "fma_asymmetric_mirex_gain_report_only": float(
            rate(expanded, "fmak", "asymmetric_units10")
            - rate(restricted, "fmak", "asymmetric_units10")
        ),
        "confirmation_scored": False,
        "separate_root_review_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "train", "gate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--bp-manifest", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--permit", type=Path)
    parser.add_argument("--arm", choices=("restricted", "expanded"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--restricted-run", type=Path)
    parser.add_argument("--expanded-run", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        result = make_plan(args.study, args.bp_manifest, args.labels)
    elif args.command == "gate":
        result = pilot_gate(
            json.loads(args.restricted_run.read_bytes()), json.loads(args.expanded_run.read_bytes())
        )
    else:
        train(args.plan, args.features, args.arm, args.output, args.permit, args.device)
        return
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    args.output.write_bytes(gzip.compress(raw, mtime=0) if args.output.suffix == ".gz" else raw)


if __name__ == "__main__":
    main()
