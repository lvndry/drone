"""Bounded pilot-only canonical decode/CQT preparation. Requires explicit preprocessing permit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

CONTRACT = "drone.features.decode_mp3/read_wav_mono/resample_linear:22050:float32le-v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw)


def selected_rows(
    plan: dict[str, Any], bp: dict[str, Any], identities: list[str], fma_audio: Path
) -> list[dict[str, Any]]:
    allowed = plan["train_feature_bindings"]
    if len(identities) != len(set(identities)) or not set(identities) <= set(allowed):
        raise ValueError(
            "Only unique reviewed pilot fitting/inner IDs;no calibration/confirmation audio"
        )
    bp_rows = {r["id"]: r for r in bp["rows"] if r["domain"] == "beatport" and r["id"] in allowed}
    fitting = set(plan["bp_fit"] + plan["expanded_fma_fit"])
    rows = []
    for identity in identities:
        domain = identity.split(":")[0]
        path = (
            bp_rows[identity]["path"]
            if domain == "beatport"
            else str(fma_audio / (identity.split(":")[1] + ".mp3"))
        )
        rows.append(
            {
                "id": identity,
                "domain": domain,
                "role": "train" if identity in fitting else "validation",
                "path": path,
                "sha256": allowed[identity],
                "ref": plan["expected_refs"][identity],
            }
        )
    return rows


def one(task: tuple[dict[str, Any], dict[str, Any] | None, Path]) -> dict[str, Any]:
    import numpy as np

    from drone.features import (
        N_TOTAL,
        SR,
        audio_digest,
        cqt,
        cqt_path,
        decode_mp3,
        read_wav_mono,
        resample_linear,
    )

    row, reuse, output = task
    started = time.monotonic()
    if audio_digest(row["path"]) != row["sha256"]:
        raise ValueError("Changed source audio")
    destination = Path(cqt_path(output, row["sha256"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("Fresh destination required;no silent overwrite")
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        wav = Path(scratch) / "canonical.wav"
        decode_mp3(row["path"], wav)
        samples, rate = read_wav_mono(wav)
        samples = resample_linear(samples, rate, SR)
        if not len(samples) or not np.isfinite(samples).all():
            raise ValueError("Invalid canonical waveform")
        canonical = hashlib.sha256(np.ascontiguousarray(samples, dtype="<f4").tobytes()).hexdigest()
        decoded = time.monotonic()
        if reuse is not None:
            source = Path(reuse["absolute_cache"])
            if audio_digest(source) != reuse["cache_sha256"]:
                raise ValueError("Changed reusable CQT")
            array = np.load(source, allow_pickle=False, mmap_mode="r")
            if (
                array.dtype != np.float16
                or array.ndim != 2
                or array.shape[0] != N_TOTAL
                or array.shape[1] < 8
                or not np.isfinite(array).all()
            ):
                raise ValueError("Invalid reusable physical CQT")
            try:
                os.link(source, destination)
            except OSError:
                shutil.copyfile(source, destination)
        else:
            array = cqt(samples).astype(np.float16)
            if array.shape[1] < 8 or not np.isfinite(array).all():
                raise ValueError("Invalid new physical CQT")
            pending = Path(scratch) / "feature.npy"
            np.save(pending, array)
            os.replace(pending, destination)
    return {
        **row,
        "cache": str(destination.relative_to(output)),
        "frames": array.shape[1],
        "cache_sha256": audio_digest(destination),
        "canonical_model_input_sha256": canonical,
        "canonical_samples": len(samples),
        "reused_cache": reuse is not None,
        "decode_hash_seconds": decoded - started,
        "cqt_or_reuse_seconds": time.monotonic() - decoded,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    from drone.features import FEATURE_SPEC
    from drone.keys import LABEL_SCHEMA

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "batch", "permit", "reuse-manifest", "fma-audio", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan)
    batch = read_json(args.batch)
    permit = read_json(args.permit)
    if (
        permit.get("allow_preprocessing") is not True
        or permit.get("plan_sha256") != digest(args.plan)
        or permit.get("batch_sha256") != digest(args.batch)
    ):
        raise ValueError("Hash-bound root preprocessing permit required")
    if digest(args.reuse_manifest) != plan["sources"]["bp_feature_manifest_sha256"]:
        raise ValueError("Frozen reuse/BP source differs")
    if batch.get("plan_sha256") != digest(args.plan):
        raise ValueError("Preprocessing batch belongs to another pilot plan")
    reuse_manifest = read_json(args.reuse_manifest)
    identities = [r["id"] for r in batch["first_batch"]]
    if len(identities) != 16:
        raise ValueError("This reviewed stage is first16 only;bulk needs measured follow-up")
    rows = selected_rows(plan, reuse_manifest, identities, args.fma_audio)
    if args.output.exists():
        raise ValueError("Fresh preparation root required")
    args.output.mkdir(parents=True)
    reuse = {
        r["id"]: {**r, "absolute_cache": str(args.reuse_manifest.parent / r["cache"])}
        for r in reuse_manifest["rows"]
        if r["id"] in identities
    }
    started = time.monotonic()
    completed = []
    try:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for result in pool.map(one, [(r, reuse.get(r["id"]), args.output) for r in rows]):
                completed.append(result)
                if time.monotonic() - started > 900:
                    raise TimeoutError("Bounded first-batch wall cap")
        manifest = {
            "status": "BOUNDED_BATCH_ONLY_NOT_TRAINABLE",
            "complete": False,
            "feature_spec": FEATURE_SPEC,
            "label_schema": LABEL_SCHEMA,
            "canonical_identity_contract": CONTRACT,
            "label_export_sha256": plan["label_export_sha256"],
            "rows": completed,
            "plan_sha256": digest(args.plan),
            "batch_sha256": digest(args.batch),
            "source_sha256": digest(Path(__file__)),
            "elapsed_seconds": time.monotonic() - started,
        }
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )
    except Exception:
        (args.output / "failed-partial.json").write_text(
            json.dumps({"status": "FAILED_NOT_TRAINABLE", "rows": completed}, sort_keys=True) + "\n"
        )
        raise


if __name__ == "__main__":
    main()
