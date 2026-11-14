"""Metadata-only complete-window extraction plan; no audio, encoder or head execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.myna_cache import (
    AUGMENTATION_METHOD,
    BASE_RESAMPLER,
    ENCODER_SHA,
    PINS,
    PLAN_SCHEMA,
    base_sample_count,
    checked_bytes,
    checked_json,
    group_rows,
    load_cache,
    rate_geometry,
    validate_manifest,
    validate_waveform_lineage,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/studies/k10-v2-pilot/data/canonical-eligibility-v2"


def project(
    labels: dict[str, int],
    groups: list[dict[str, str]],
    prepared: dict[str, Any],
) -> dict[str, Any]:
    families = {row["id"]: row for row in groups}
    tracks = []
    for source in prepared["rows"]:
        identity = source["id"]
        family = families[identity]
        role = family["logical_role"]
        samples_24k = source["samples"]
        samples_16k = base_sample_count(samples_24k)
        frames = samples_16k // 512 + 1
        windows, tail = divmod(frames, 196)
        tracks.append(
            {
                "id": identity,
                "domain": family["domain"],
                "role": role,
                "group": family["derived_uncertainty_group"],
                "target": labels[identity],
                "source_sha256": source["sha256"],
                "pcm_sha256": source["waveform_sha256"],
                "physical_pcm_path": source["cache"],
                "physical_pcm_identity": source["cache_identity"],
                "samples_24k": samples_24k,
                "samples_16k": samples_16k,
                "mel_frames": frames,
                "complete_windows": windows,
                "discarded_frames": tail,
                "views": [
                    {
                        "shift": shift,
                        **rate_geometry(samples_16k, shift),
                        "raw_samples": None,
                        "trimmed_samples": None,
                        "padded_samples": None,
                        "dtype": "float32",
                        "shape": [rate_geometry(samples_16k, shift)["windows"], 384],
                        "window_order": "all-complete-196-frame-windows-in-order",
                        "path": "embeddings/"
                        + identity.replace(":", "-")
                        + f"-shift{shift:+d}.npy",
                        "bytes": None,
                        "sha256": None,
                    }
                    for shift in (range(-6, 7) if role == "fit" else [0])
                ],
            }
        )
    return {
        "schema_version": PLAN_SCHEMA,
        "augmentation_method": AUGMENTATION_METHOD,
        "base_resampler": BASE_RESAMPLER,
        "complete": False,
        "status": "METADATA_PROJECTION_NOT_EXTRACTED_NOT_TRAINABLE",
        "label_schema": "tonic-times-two-plus-minor-v1",
        "identities": {
            "encoder_sha256": ENCODER_SHA,
            "encoder_source_sha256": None,
            "environment_sha256": None,
            "frontend_sha256": None,
            "augmentation_sha256": None,
            "qualification_sha256": None,
        },
        "lineage": {
            name: PINS[name]
            for name in ("plan", "labels", "groups", "waveforms", "prepared_waveforms")
        },
        "tracks": tracks,
    }


def prepare(canonical_path: Path, prepared_path: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=False, exist_ok=False)

    def watch() -> None:
        return None

    canonical = checked_json(canonical_path, PINS["waveforms"], watch)
    physical = checked_json(prepared_path, PINS["prepared_waveforms"], watch)
    validate_waveform_lineage(canonical, physical)
    plan = checked_json(DATA / "plan-v2.json.gz", PINS["plan"], watch)
    labels = checked_json(DATA / "numeric-labels-v2.json.gz", PINS["labels"], watch)["labels"]
    groups = group_rows(checked_bytes(DATA / "uncertainty-groups-v2.csv.gz", PINS["groups"], watch))
    projection = project(labels, groups, physical)
    contract = validate_manifest(
        projection,
        plan,
        labels,
        groups,
        physical["rows"],
        output,
        projection["identities"],
        metadata_only=True,
    )
    try:
        load_cache(contract, contract.embedding_bytes, watch)
    except ValueError as error:
        if "metadata-only projection" not in str(error):
            raise
    else:
        raise RuntimeError("Metadata projection must never activate loading/training")
    path = output / "expected-cache-plan.json"
    path.write_text(json.dumps(projection, indent=2, allow_nan=False) + "\n")
    receipt = {
        "status": "METADATA_DRY_VALIDATION_PASSED_NOT_EXTRACTION_QUALIFICATION",
        "plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "plan_bytes": path.stat().st_size,
        "source_sha256": {
            str(source_path.relative_to(ROOT)): hashlib.sha256(source_path.read_bytes()).hexdigest()
            for source_path in (Path(__file__).resolve(), ROOT / "experiments/myna_cache.py")
        },
        "lineage": projection["lineage"],
        "tracks": len(projection["tracks"]),
        "views": len(contract.views),
        "fitting_views": sum(view.role == "fit" for view in contract.views),
        "selection_views": sum(view.role == "selection" for view in contract.views),
        "fitting_vectors": contract.fit_windows,
        "selection_vectors": contract.selection_windows,
        "embedding_payload_bytes": contract.embedding_bytes,
        "canonical_roles": dict(Counter(row["role"] for row in canonical["rows"])),
        "physical_roles": dict(Counter(row["role"] for row in physical["rows"])),
        "role_aliases": {"fit": "fit", "validation": "selection"},
        "group_counts": {
            role: len(
                {row["derived_uncertainty_group"] for row in groups if row["logical_role"] == role}
            )
            for role in ("fit", "selection")
        },
        "projection_loading_refused": True,
        "physical_pcm_opened": False,
        "model_loaded": False,
        "torch_imported": "torch" in sys.modules,
        "cuda_initialized": bool(sys.modules["torch"].cuda.is_initialized())
        if "torch" in sys.modules
        else False,
        "augmentation_or_blob_identities_invented": False,
    }
    (output / "dry-validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.canonical, args.prepared, args.output), indent=2))
