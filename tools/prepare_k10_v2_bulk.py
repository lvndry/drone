"""Normal full pilot preparation, reusing the frozen per-track first16 implementation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from tools.prepare_k10_v2_pilot import CONTRACT, digest, one, read_json, selected_rows


def canonical_audit(rows: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    fitting = set(plan["bp_fit"] + plan["expanded_fma_fit"])
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[row["canonical_model_input_sha256"]].append(row)
    cross = []
    conflicts = []
    duplicates = []
    for key, records in sorted(families.items()):
        roles = {"fit" if r["id"] in fitting else "selection" for r in records}
        refs = {r["ref"] for r in records}
        witness = {
            "canonical_sha256": key,
            "ids": [r["id"] for r in records],
            "roles": sorted(roles),
            "refs": sorted(refs),
        }
        if len(roles) > 1:
            cross.append(witness)
        if len(refs) > 1:
            conflicts.append(witness)
        if len(records) > 1:
            duplicates.append(witness)
    return {
        "rows": len(rows),
        "unique_canonical_inputs": len(families),
        "cross_fit_selection_families": cross,
        "conflicting_target_families": conflicts,
        "duplicate_families": duplicates,
        "complete": not cross and not conflicts,
        "no_role_or_label_repair": True,
    }


def carry_first16(
    seed: dict[str, Any], seed_path: Path, plan: dict[str, Any], output: Path
) -> list[dict[str, Any]]:
    import numpy as np

    from drone.features import N_TOTAL, audio_digest, cqt_path

    rows = seed["rows"]
    if len(rows) != 16 or seed["complete"] or seed["canonical_identity_contract"] != CONTRACT:
        raise ValueError("Expected exact bounded first16 receipt")
    result = []
    for row in rows:
        identity = row["id"]
        if (
            identity not in plan["train_feature_bindings"]
            or row["sha256"] != plan["train_feature_bindings"][identity]
            or row["ref"] != plan["expected_refs"][identity]
        ):
            raise ValueError("First16 identity/label binding differs")
        if audio_digest(row["path"]) != row["sha256"]:
            raise ValueError("First16 source container changed")
        source = (seed_path.parent / row["cache"]).resolve()
        if not source.is_relative_to(seed_path.parent.resolve()):
            raise ValueError("First16 cache escapes source root")
        if audio_digest(source) != row["cache_sha256"]:
            raise ValueError("First16 cache content changed")
        array = np.load(source, allow_pickle=False, mmap_mode="r")
        if (
            array.dtype != np.float16
            or array.shape != (N_TOTAL, row["frames"])
            or not np.isfinite(array).all()
        ):
            raise ValueError("First16 cache shape/dtype/finiteness differs")
        target = Path(cqt_path(output, row["sha256"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError("No overwrite of carried cache")
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
        result.append(
            {
                **row,
                "cache": str(target.relative_to(output)),
                "carried_first16": True,
                "first16_manifest_sha256": digest(seed_path),
            }
        )
    return result


def main() -> None:
    from drone.features import FEATURE_SPEC
    from drone.keys import LABEL_SCHEMA
    from experiments.k10_v2_pilot import validate_feature_manifest

    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plan",
        "labels",
        "config",
        "permit",
        "reuse-manifest",
        "first16-manifest",
        "first16-controller",
        "fma-audio",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan)
    config = read_json(args.config)
    permit = read_json(args.permit)
    if (
        permit.get("allow_full_pilot_preprocessing") is not True
        or permit.get("allow_training") is not False
        or permit.get("config_sha256") != digest(args.config)
        or permit.get("plan_sha256") != digest(args.plan)
    ):
        raise ValueError("Separate root-reviewed bulk-only permit required")
    checks = {
        "plan": args.plan,
        "labels": args.labels,
        "reuse_manifest": args.reuse_manifest,
        "first16_manifest": args.first16_manifest,
        "first16_controller": args.first16_controller,
    }
    if any(digest(path) != config["inputs"][name] for name, path in checks.items()):
        raise ValueError("Frozen bulk input witness changed")
    if digest(Path(one.__code__.co_filename)) != config["per_track_source_sha256"]:
        raise ValueError("Numerical per-track implementation changed")
    if digest(Path(__file__)) != config["bulk_source_sha256"]:
        raise ValueError("Bulk driver changed")
    root = Path(__file__).resolve().parents[1]
    if any(
        digest(root / name) != expected for name, expected in plan["bound_source_sha256"].items()
    ):
        raise ValueError("Bound physical/model sources changed")
    labels = read_json(args.labels)
    if (
        labels["labels"] != plan["expected_refs"]
        or not labels["csv_selected_labels_match_prepared_refs"]
    ):
        raise ValueError("Pinned numeric-label binding differs")
    controller = read_json(args.first16_controller)
    seed = read_json(args.first16_manifest)
    if (
        controller["exit"]
        or controller["cap_violation"] is not None
        or seed["source_sha256"] != config["per_track_source_sha256"]
        or seed["plan_sha256"] != digest(args.plan)
    ):
        raise ValueError("First16 source/terminal receipt invalid")
    if args.output.exists():
        raise ValueError("Fresh bulk output root required")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    completed = []
    try:
        completed = carry_first16(seed, args.first16_manifest, plan, args.output)
        remaining = sorted(set(plan["train_feature_bindings"]) - {r["id"] for r in completed})
        original = read_json(args.reuse_manifest)
        requested = selected_rows(plan, original, remaining, args.fma_audio)
        reuse = {
            r["id"]: {**r, "absolute_cache": str(args.reuse_manifest.parent / r["cache"])}
            for r in original["rows"]
            if r["id"] in remaining
        }
        with ProcessPoolExecutor(max_workers=config["caps"]["workers"]) as pool:
            for row in pool.map(one, [(r, reuse.get(r["id"]), args.output) for r in requested]):
                completed.append(row)
                if len(completed) % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "completed": len(completed),
                                "required": len(plan["train_feature_bindings"]),
                                "elapsed_seconds": time.monotonic() - started,
                            }
                        ),
                        flush=True,
                    )
                if time.monotonic() - started > config["caps"]["wall_seconds"]:
                    raise TimeoutError("Bulk cooperative wall cap")
        if len(completed) != 5641 or {r["id"] for r in completed} != set(
            plan["train_feature_bindings"]
        ):
            raise ValueError("Full pilot coverage incomplete")
        audit = canonical_audit(completed, plan)
        (args.output / "canonical-audit.json").write_text(
            json.dumps(audit, sort_keys=True, indent=2) + "\n"
        )
        manifest = {
            "status": "VALIDATING_NOT_TRAINABLE",
            "complete": False,
            "feature_spec": FEATURE_SPEC,
            "label_schema": LABEL_SCHEMA,
            "canonical_identity_contract": CONTRACT,
            "label_export_sha256": plan["label_export_sha256"],
            "rows": completed,
            "plan_sha256": digest(args.plan),
            "config_sha256": digest(args.config),
            "source_sha256": digest(Path(__file__)),
            "per_track_source_sha256": config["per_track_source_sha256"],
            "carried_first16": 16,
            "elapsed_seconds": time.monotonic() - started,
            "actions": dict(
                Counter(
                    "carried_first16"
                    if r.get("carried_first16")
                    else "reused_v1"
                    if r["reused_cache"]
                    else "new_cqt"
                    for r in completed
                )
            ),
        }
        if not audit["complete"]:
            (args.output / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n"
            )
            raise ValueError(
                "Actual canonical cross-role/target conflict;no repair or role changes"
            )
        candidate = {
            **manifest,
            "complete": True,
            "status": "FULL_PILOT_PREPARED_NOT_TRAINING_AUTHORIZED",
        }
        validate_feature_manifest(plan, candidate)
        (args.output / "manifest.json").write_text(
            json.dumps(candidate, sort_keys=True, indent=2) + "\n"
        )
    except Exception:
        (args.output / "failed-partial.json").write_text(
            json.dumps(
                {"status": "FAILED_NOT_TRAINABLE", "complete": False, "rows": completed},
                sort_keys=True,
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
