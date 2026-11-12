"""Metadata-only cache inventory summary and fixed first preprocessing batch."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


def build(plan: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    rows = inventory["rows"]
    if {r["id"]: r["sha256"] for r in rows} != plan["train_feature_bindings"]:
        raise ValueError("Cache inventory and reviewed pilot identities differ")
    missing = [r for r in rows if not r["existing"]]
    fit = set(plan["expanded_fma_fit"])
    new_fit = sorted(
        (r for r in missing if r["id"] in fit),
        key=lambda r: (r["duration_seconds_metadata"], r["id"]),
    )
    selected = []
    for quartile in range(4):
        pool = new_fit[len(new_fit) * quartile // 4 : len(new_fit) * (quartile + 1) // 4]
        order = sorted(
            pool,
            key=lambda r: hashlib.sha256(
                ("v2-first-preprocess-20261112:" + r["id"]).encode()
            ).hexdigest(),
        )
        chosen = order[:3]
        if quartile == 3:
            longest = pool[-1]
            chosen = order[:2] + [longest] if longest not in order[:2] else order[:3]
        selected.extend(chosen)
    # Four reused fitting examples measure canonical decode/hash even when CQT is reused.
    for ids in (set(plan["bp_fit"]), set(plan["restricted_fma_fit"])):
        pool = sorted(
            (r for r in rows if r["id"] in ids and r["existing"]),
            key=lambda r: hashlib.sha256(
                ("v2-cached-preprocess-20261112:" + r["id"]).encode()
            ).hexdigest(),
        )
        selected.extend(pool[:2])
    if len(selected) != 16 or len({r["id"] for r in selected}) != 16:
        raise ValueError("Fixed first-batch construction failed")
    bytes_missing = sum(
        (int(r["duration_seconds_metadata"] * 22050 / 4096) + 2) * 208 * 2 + 128 for r in missing
    )
    categories = {
        "bp": set(plan["bp_fit"] + plan["bp_selection_guard"]),
        "expanded_fma_fit": fit,
        "fma_inner_selection": set(plan["fma_inner_selection"]),
    }
    return {
        "status": "PROPOSED_PREPROCESS_ONLY_NOT_LAUNCHED",
        "membership_rule": "12 missing fitting CQT,3 per duration quartile with longest forced in upper quartile;4 cached fit examples(2BP/2restrictedFMA);fixed SHA ranks,no labels/scores",
        "counts": {
            name: {
                "required": len(ids),
                "existing": sum(bool(r["existing"]) for r in rows if r["id"] in ids),
                "missing": sum(not r["existing"] for r in rows if r["id"] in ids),
            }
            for name, ids in categories.items()
        },
        "inventory_summary": inventory["summary"],
        "projected_missing_float16_cache_bytes": bytes_missing,
        "projection_method": "208float16bins * (floor(metadata_duration*22050/4096)+2) +128header;not measured frames",
        "first_batch": selected,
        "first_batch_caps": {
            "workers": 2,
            "cpus": [10, 11],
            "native_threads": 1,
            "wall_seconds": 900,
            "cpu_seconds": 1800,
            "new_cache_bytes": 134217728,
            "aggregate_rss_bytes": 4294967296,
            "wav_scratch_bytes": 268435456,
        },
        "canonical_identity_required_for_all_pilot_rows": True,
        "calibration_or_confirmation_audio_allowed": False,
        "bulk_projection_rule": "Measure cold/warm wall+CPU and bytes by duration stratum on first16;sum per-stratum seconds-per-audio-second over missing+CQT-reuse inventories;include all5641 canonical decodes. No bulk start from old latency proxy alone.",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.plan.read_bytes()
    plan = json.loads(gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw)
    invraw = args.inventory.read_bytes()
    inventory = json.loads(gzip.decompress(invraw) if invraw.startswith(b"\x1f\x8b") else invraw)
    result = build(plan, inventory)
    result.update(
        plan_sha256=hashlib.sha256(raw).hexdigest(),
        inventory_sha256=hashlib.sha256(invraw).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
