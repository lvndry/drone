"""Immutable full frozen-K10 activations, exact-size preflight and reload proof."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import platform
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from drone.features import FEATURE_SPEC, N_CORE, audio_digest
from drone.keys import LABEL_SCHEMA
from experiments.ema import read_reference
from experiments.k10_multidomain import FROZEN_SHA256, domain_metrics, validate_rows
from experiments.octave import device_setup
from experiments.small import atomic_json
from experiments.temporal_model import VERSION

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "experiments/studies/k10-temporal"
PROTOCOL = STUDY / "PROTOCOL.md"
REVIEWED = ROOT / "experiments/studies/k10-capacity/data/reviewed-sources.json"
SOURCE_FILES = (
    "experiments/temporal_model.py",
    "experiments/temporal_cache.py",
    "experiments/temporal_study.py",
    "experiments/temporal_cpu.py",
    "experiments/k10_multidomain.py",
    "experiments/paired.py",
    "experiments/ema.py",
    "experiments/octave.py",
    "experiments/small.py",
    "drone/export.py",
    "drone/features.py",
    "drone/keys.py",
    "pyproject.toml",
    "uv.lock",
)


class FrozenEncoder(nn.Module):
    """Structural typing for the verified archived KeyCNN implementation."""

    net: nn.Sequential
    head: nn.Linear


def identity() -> dict[str, Any]:
    return dict(
        version=VERSION,
        source_sha256={p: audio_digest(ROOT / p) for p in SOURCE_FILES},
        protocol_sha256=audio_digest(PROTOCOL),
        torch=str(torch.__version__),
        numpy=np.__version__,
        python=platform.python_version(),
        platform=platform.platform(),
        cuda_build=torch.version.cuda,
    )


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def npy_bytes(shape: tuple[int, ...]) -> int:
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, dict(descr="<f4", fortran_order=False, shape=shape)
    )
    return len(header.getvalue()) + 4 * int(np.prod(shape))


def preflight(
    rows: list[dict[str, Any]], base: dict[str, Any], available: int
) -> tuple[dict[str, Any], dict[str, int]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        tokens = row["frames"] // 8
        if tokens < 1:
            raise ValueError("Need at least8 frames for every recording")
        records.append(
            {
                k: row[k]
                for k in (
                    "id",
                    "role",
                    "domain",
                    "ref",
                    "sha256",
                    "frames",
                    "cache",
                    "cache_sha256",
                )
            }
            | dict(
                cqt_shape=[208, row["frames"]],
                hidden_shape=[64, 24, tokens],
                hidden="hidden/" + hashlib.sha256(row["id"].encode()).hexdigest() + ".npy",
                hidden_sha256="0" * 64,
                hidden_npy_bytes=npy_bytes((64, 24, tokens)),
            )
        )
    manifest = dict(base, rows=records, canonical_logits_sha256="0" * 64, complete=True)
    payload = 6144 * sum(r["hidden_shape"][2] for r in records)
    if payload > 4 * 1024**3:
        raise MemoryError(
            "Exact FP32 hidden payload exceeds prospective4GiB ceiling; stop, never truncate"
        )
    hidden_files = sum(r["hidden_npy_bytes"] for r in records)
    size: dict[str, int] = dict(
        hidden_payload_bytes=payload,
        npy_header_bytes=hidden_files - payload,
        canonical_logits_bytes=npy_bytes((len(rows), 24)),
        manifest_bytes=len(json_bytes(manifest)),
        largest_atomic_temporary_bytes=max(r["hidden_npy_bytes"] for r in records),
        evidence_reserve_bytes=2 * 1024**2,
    )
    for domain in ("beatport", "fmak"):
        for role in ("train", "validation"):
            selected = [r for r in records if r["domain"] == domain and r["role"] == role]
            size[f"{domain}_{role}_recordings"] = len(selected)
            size[f"{domain}_{role}_tokens"] = sum(r["hidden_shape"][2] for r in selected)
            size[f"{domain}_{role}_payload_bytes"] = 6144 * size[f"{domain}_{role}_tokens"]
    size["committed_cache_bytes"] = (
        hidden_files + size["canonical_logits_bytes"] + size["manifest_bytes"]
    )
    size["reserved_disk_bytes"] = (
        size["committed_cache_bytes"]
        + size["largest_atomic_temporary_bytes"]
        + size["evidence_reserve_bytes"]
    )
    if available < size["reserved_disk_bytes"]:
        raise OSError(
            "Insufficient disk for exact full cache plus atomic temporary and evidence reserve"
        )
    return manifest, size


def atomic_npy(path: Path, value: np.ndarray) -> None:
    pending = path.with_suffix(".pending")
    path.parent.mkdir(parents=True, exist_ok=True)
    with pending.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(pending, path)


def verify_original(source: Path, expected: dict[str, str], check: Callable[[], None]) -> None:
    for name, digest in expected.items():
        check()
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or audio_digest(path) != digest:
            raise ValueError(f"Archived source differs: {name}")


def load_encoder(checkpoint: Path, source: Path, run: dict[str, Any], device: str) -> FrozenEncoder:
    if audio_digest(checkpoint) != run["checkpoint_sha256"] or N_CORE != 192:
        raise ValueError("Canonical checkpoint/core identity changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    md = payload.get("metadata", {})
    if (
        payload.get("format_version") != 2
        or payload.get("feature_spec") != FEATURE_SPEC
        or payload.get("label_schema") != LABEL_SCHEMA
        or payload.get("crop") != 320
        or payload.get("model_config") != {"width": 1.0}
        or any(md.get(k) != run[k] for k in ("seed", "arm", "source_sha256"))
        or md.get("epoch") != run["best_epoch"]
    ):
        raise ValueError("Canonical checkpoint schema/selection/source differs")
    spec = importlib.util.spec_from_file_location("_temporal_frozen_k10", source / "drone/model.py")
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import verified archived model")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    constructor = cast(Callable[..., FrozenEncoder], getattr(module, "KeyCNN"))
    model = constructor(width=1.0).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Canonical encoder must remain FP32, no AMP/quantization")
    if sum(p.numel() for p in model.parameters()) != 109128:
        raise ValueError("Wrong canonical encoder size")
    return model


def reconstruct(model: FrozenEncoder, hidden: torch.Tensor) -> torch.Tensor:
    return model.head(model.net[-1](hidden).flatten(1))


def within_tolerance(actual: np.ndarray, reference: np.ndarray) -> bool:
    return bool(
        actual.shape == reference.shape
        and np.isfinite(actual).all()
        and np.isfinite(reference).all()
        and np.all(
            np.abs(actual.astype(np.float64) - reference.astype(np.float64))
            <= 1e-5 + 1e-5 * np.abs(reference.astype(np.float64))
        )
        and np.array_equal(actual.argmax(-1), reference.argmax(-1))
    )


def validate_cache(cache: Path, check: Callable[[], None]) -> tuple[dict[str, Any], np.ndarray]:
    manifest = json.loads((cache / "manifest.json").read_bytes())
    proof = json.loads((cache / "reconstruction.json").read_bytes())
    if (
        not manifest.get("complete")
        or manifest["identity"] != identity()
        or proof.get("status") != "complete"
        or proof.get("recordings") != 2582
        or proof.get("argmax_disagreements") != 0
        or not proof.get("all_logits_within_tolerance")
        or proof["manifest_sha256"] != audio_digest(cache / "manifest.json")
        or proof["canonical_logits_sha256"] != manifest["canonical_logits_sha256"]
    ):
        raise ValueError("Cache/source/reconstruction proof changed or incomplete")
    original = cache / "cqt-manifest.json"
    if audio_digest(original) != manifest["feature_manifest_sha256"]:
        raise ValueError("Original data/label manifest changed")
    source_rows = json.loads(original.read_bytes())["rows"]
    if len(source_rows) != 2582 or len(manifest["rows"]) != 2582:
        raise ValueError("Wrong reconstruction cohort size")
    for row, source in zip(manifest["rows"], source_rows, strict=True):
        if any(
            row[k] != source[k]
            for k in ("id", "role", "domain", "ref", "sha256", "frames", "cache", "cache_sha256")
        ):
            raise ValueError("Cached recording label/identity differs from original manifest")
        if row["hidden_shape"] != [64, 24, row["frames"] // 8]:
            raise ValueError("Hidden time/layout differs from original manifest")
    canonical = cache / "canonical-logits.npy"
    if audio_digest(canonical) != manifest["canonical_logits_sha256"]:
        raise ValueError("Canonical logits changed")
    logits = np.load(canonical, allow_pickle=False)
    if (
        logits.shape != (2582, 24)
        or logits.dtype != np.dtype("<f4")
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Invalid canonical logits")
    for row in manifest["rows"]:
        check()
        path = (cache / row["hidden"]).resolve()
        if not path.is_relative_to(cache.resolve()) or audio_digest(path) != row["hidden_sha256"]:
            raise ValueError("Hidden cache hash/path changed")
        h = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            h.dtype != np.dtype("<f4")
            or list(h.shape) != row["hidden_shape"]
            or not h.flags.c_contiguous
            or not np.isfinite(h).all()
        ):
            raise ValueError("Hidden shape/layout/finite identity changed")
    return manifest, logits


def extract(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError("Use fresh cache output")
    args.output.mkdir(parents=True)
    start = time.monotonic()

    def check() -> None:
        if time.monotonic() - start > 900:
            raise TimeoutError("Frozen15-minute extraction/reconstruction budget exceeded")

    original = json.loads(REVIEWED.read_bytes())["archived"]
    verify_original(args.reference_source, original["sources"], check)
    run = read_reference(args.reference_run, "joint", args.seed)
    if run["source_sha256"] != original["sources"]:
        raise ValueError("Canonical reference must be reviewed source673")
    raw = args.data / "manifest.json"
    data = json.loads(raw.read_bytes())
    if (
        not data["complete"]
        or data["feature_spec"] != FEATURE_SPEC
        or data["label_schema"] != LABEL_SCHEMA
        or data["frozen_manifest_sha256"] != FROZEN_SHA256
        or audio_digest(raw) != run["feature_manifest_sha256"]
    ):
        raise ValueError("Canonical quarantined data/feature identity changed")
    rows = data["rows"]
    validate_rows(rows)
    if [
        sum(r["domain"] == d and r["role"] == role for r in rows)
        for d in ("beatport", "fmak")
        for role in ("train", "validation")
    ] != [1230, 126, 976, 250]:
        raise ValueError("Wrong2206/376 development identities")
    for role, field in (("train", "fit_ids"), ("validation", "validation_ids")):
        if [r["id"] for r in rows if r["role"] == role] != run[field]:
            raise ValueError(f"Canonical {field} changed")
    own = identity()
    if args.seed != 13:
        from experiments.temporal_study import require_pilot

        require_pilot(args.pilot, own)
    base = dict(
        version=VERSION,
        identity=own,
        encoder_seed=args.seed,
        canonical_config=run["config"],
        checkpoint_sha256=run["checkpoint_sha256"],
        reference_run_sha256=audio_digest(args.reference_run),
        archived_source_sha256=original["sources"],
        feature_manifest_sha256=audio_digest(raw),
        feature_spec=FEATURE_SPEC,
        label_schema=LABEL_SCHEMA,
    )
    manifest, size = preflight(rows, base, shutil.disk_usage(args.output).free)
    size["source_manifest_bytes"] = raw.stat().st_size
    size["protocol_bytes"] = PROTOCOL.stat().st_size
    size["reserved_disk_bytes"] += size["source_manifest_bytes"] + size["protocol_bytes"]
    size["committed_cache_bytes"] += size["source_manifest_bytes"] + size["protocol_bytes"]
    if shutil.disk_usage(args.output).free < size["reserved_disk_bytes"]:
        raise OSError("Insufficient disk for exact source manifest/protocol overhead")
    atomic_json(args.output / "preflight.json", size)
    atomic_json(args.output / "identity.json", base)
    (args.output / "PROTOCOL.md").write_bytes(PROTOCOL.read_bytes())
    (args.output / "cqt-manifest.json").write_bytes(raw.read_bytes())
    device_setup(args.device)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    model = load_encoder(
        args.reference_run.with_name("best.pt"), args.reference_source, run, args.device
    )
    direct = np.empty((2582, 24), dtype="<f4")
    bit_identical = 0
    max_error = 0.0
    for i, row in enumerate(manifest["rows"]):
        check()
        source = (args.data / row["cache"]).resolve()
        if (
            not source.is_relative_to(args.data.resolve())
            or audio_digest(source) != row["cache_sha256"]
        ):
            raise ValueError("Original CQT content/path changed")
        cqt = np.load(source, mmap_mode="r", allow_pickle=False)
        if (
            cqt.dtype != np.float16
            or list(cqt.shape) != row["cqt_shape"]
            or not np.isfinite(cqt).all()
        ):
            raise ValueError("Original CQT shape/dtype/finite changed")
        x = torch.from_numpy(np.asarray(cqt[8:200], dtype=np.float32)[None]).to(args.device)
        with torch.inference_mode():
            hidden = model.net[:-2](x.unsqueeze(1))
            canonical = model(x).cpu().numpy()[0]
        h = np.ascontiguousarray(hidden.cpu().numpy()[0], dtype="<f4")
        if list(h.shape) != row["hidden_shape"] or not np.isfinite(h).all():
            raise ValueError("Hidden extraction layout/finite failure")
        path = args.output / row["hidden"]
        atomic_npy(path, h)
        if path.stat().st_size != row["hidden_npy_bytes"]:
            raise ValueError("NPY actual header/payload differs from exact preflight")
        row["hidden_sha256"] = audio_digest(path)
        # Mandatory disk reload, unchanged AdaptiveAvgPool2d and original head.
        loaded = np.load(path, allow_pickle=False)
        with torch.inference_mode():
            rebuilt = (
                reconstruct(model, torch.from_numpy(loaded[None]).to(args.device)).cpu().numpy()[0]
            )
        if not within_tolerance(rebuilt, canonical):
            raise ValueError(f"Cache reconstruction integrity failure:{row['id']}")
        bit_identical += int(rebuilt.tobytes() == canonical.tobytes())
        max_error = max(max_error, float(np.max(np.abs(rebuilt - canonical))))
        direct[i] = canonical
    val = [i for i, r in enumerate(rows) if r["role"] == "validation"]
    val_rows = [rows[i] for i in val]
    refs = np.array([r["ref"] for r in val_rows])
    if domain_metrics(direct[val].argmax(1), refs, val_rows) != run["domain_metrics"]:
        raise ValueError("Canonical validation predictions/metrics fail archived reproduction")
    atomic_npy(args.output / "canonical-logits.npy", direct)
    manifest["canonical_logits_sha256"] = audio_digest(args.output / "canonical-logits.npy")
    if len(json_bytes(manifest)) != size["manifest_bytes"]:
        raise ValueError("Manifest actual bytes differ from exact preflight")
    atomic_json(args.output / "manifest.json", manifest)
    check()
    atomic_json(
        args.output / "reconstruction.json",
        dict(
            version=VERSION,
            status="complete",
            recordings=2582,
            all_logits_within_tolerance=True,
            argmax_disagreements=0,
            bit_identical=bit_identical,
            max_abs_error=max_error,
            manifest_sha256=audio_digest(args.output / "manifest.json"),
            canonical_logits_sha256=manifest["canonical_logits_sha256"],
            elapsed_seconds=time.monotonic() - start,
            aggregate_started_epoch=time.time() - (time.monotonic() - start),
            memory=dict(
                allocator_fraction=0.04,
                required_free_bytes=8 * 1024**3,
                free_bytes=free,
                total_bytes=total,
                gpu_name=torch.cuda.get_device_name(0),
                cudnn_version=torch.backends.cudnn.version(),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32,
                actual_cap_bytes=int(total * 0.04),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            ),
        ),
    )


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-run", type=Path, required=True)
    p.add_argument("--reference-source", type=Path, required=True)
    p.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    p.add_argument("--seed", type=int, choices=(13, 37, 73), default=13)
    p.add_argument("--pilot", type=Path)
    args = p.parse_args()
    try:
        extract(args)
    except (ValueError, RuntimeError, TimeoutError, OSError, MemoryError) as error:
        if args.output.is_dir():
            atomic_json(
                args.output / "failure.json",
                dict(
                    version=VERSION,
                    status="integrity_or_resource_inconclusive",
                    error=str(error),
                    promoted=False,
                ),
            )
        raise


if __name__ == "__main__":
    main()
