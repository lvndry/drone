"""Label-blind K10 chunk parity and isolated cached-feature CPU measurements.

Run as `python -m tools.evaluate_chunked`. Only diagnose imports Torch.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import signal
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import onnxruntime as ort

from experiments.chunked import (
    ATOL,
    FEATURE_SPEC,
    RTOL,
    ChunkedK10,
    compare_logits,
    core_windows,
    cpu_session,
    sha256,
)


def budget_exceeded(signum: int, frame: FrameType | None) -> None:
    raise TimeoutError("Stage exceeded the declared 600 CPU-second budget")


def source_provenance() -> dict[str, str]:
    return {
        "runtime_source_sha256": sha256(Path(__file__).parents[1] / "experiments/chunked.py"),
        "evaluator_source_sha256": sha256(Path(__file__)),
    }


def thread_environment() -> dict[str, str | None]:
    return {
        k: os.environ.get(k)
        for k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }


def load_rows(manifest: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest.read_text())
    if payload.get("feature_spec") != FEATURE_SPEC or not payload.get("complete"):
        raise ValueError("Incomplete/incompatible feature manifest")
    # Whitelist metadata fields: ground-truth ref never enters evaluation records.
    fields = ("id", "domain", "cache", "cache_sha256", "frames", "sha256")
    rows = [{k: r[k] for k in fields} for r in payload["rows"] if r["role"] == "validation"]
    if len(rows) != 376:
        raise ValueError("Expected exactly 376 public validation tracks")
    identities = {(r["domain"], str(r["id"])) for r in rows}
    if len(identities) != len(rows):
        raise ValueError("Duplicate validation identity")
    return rows


def load_features(manifest: Path, row: dict[str, Any]) -> np.ndarray:
    path = manifest.parent / row["cache"]
    if sha256(path) != row["cache_sha256"]:
        raise ValueError(f"Changed cache: {row['domain']}/{row['id']}")
    padded = np.load(path, mmap_mode="r", allow_pickle=False)
    if padded.shape != (208, row["frames"]) or padded.dtype != np.float16:
        raise ValueError("Expected unchanged padded FP16 CQT cache")
    return np.ascontiguousarray(padded[8:200][None], dtype=np.float32)


def selected_rows(rows: list[dict[str, Any]], selection: Path) -> list[dict[str, Any]]:
    payload = json.loads(selection.read_text())
    chosen = payload if isinstance(payload, list) else payload.get("rows", payload.get("tracks"))
    if not isinstance(chosen, list) or len(chosen) != 20:
        raise ValueError("Expected exactly 20 existing selected public tracks")
    matched = []
    for item in chosen:
        matches = [r for r in rows if r["sha256"] == item["sha"] and r["domain"] == "fmak"]
        if len(matches) != 1 or matches[0]["frames"] != item["frames"]:
            raise ValueError("Selection does not uniquely match public FMAK validation cache")
        matched.append(matches[0])
    if len({r["sha256"] for r in matched}) != 20:
        raise ValueError("Selection contains duplicate audio")
    return matched


def validate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    runtime = ChunkedK10(args.bundle, args.core_outputs)
    full = cpu_session(args.full_model)
    records = []
    for row in rows:
        x = load_features(args.manifest, row)
        baseline = np.asarray(full.run(["logits"], {"features": x})[0])
        reduced = runtime.full_prefix_logits(x)
        chunked = runtime.predict(x)
        records.append(
            {
                "id": row["id"],
                "domain": row["domain"],
                "frames": row["frames"],
                "cache_sha256": row["cache_sha256"],
                "full_prefix_vs_full_ort": compare_logits(reduced, baseline),
                "chunked_vs_full_ort": compare_logits(chunked, baseline),
            }
        )
        if len(records) % 20 == 0:
            print(f"Validated {len(records)}/{len(rows)}", flush=True)
    passed = all(
        all(r[key][gate] for gate in ("finite", "allclose", "argmax_agrees"))
        for r in records
        for key in ("full_prefix_vs_full_ort", "chunked_vs_full_ort")
    )
    summary = {}
    for key in ("full_prefix_vs_full_ort", "chunked_vs_full_ort"):
        summary[key] = {
            "argmax_disagreements": sum(not r[key]["argmax_agrees"] for r in records),
            "numerical_failures": sum(not r[key]["allclose"] for r in records),
            "max_abs_error": max(r[key]["max_abs_error"] or 0.0 for r in records),
        }
    return {"passed": passed, "tracks": len(records), "summary": summary, "records": records}


def diagnose(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    from drone.checkpoints import load_checkpoint
    from tools.export_chunked import HiddenPrefix

    runtime = ChunkedK10(args.bundle, args.core_outputs)
    if args.checkpoint is None or sha256(args.checkpoint) != runtime.metadata["checkpoint_sha256"]:
        raise ValueError("Diagnosis requires the exact source checkpoint")
    torch.set_num_threads(1)
    model, _ = load_checkpoint(args.checkpoint)
    prefix = HiddenPrefix(model.net).eval()
    full = cpu_session(args.full_model)
    records = []
    for row in rows:
        x = load_features(args.manifest, row)
        with torch.inference_mode():
            expected = model(torch.from_numpy(x)).numpy()
            expected_hidden = prefix(torch.from_numpy(x)).numpy()
        actual_hidden = runtime.hidden(x)
        stitched_max = 0.0
        stitched_close = True
        for q0, q1, a, b in core_windows(x.shape[-1], args.core_outputs):
            local = runtime.hidden(np.ascontiguousarray(x[:, :, a:b]))
            kept = local[:, :, :, q0 - a // 8 : q1 - a // 8]
            reference = actual_hidden[:, :, :, q0:q1]
            stitched_max = max(stitched_max, float(np.max(np.abs(kept - reference))))
            stitched_close &= bool(np.allclose(kept, reference, atol=ATOL, rtol=RTOL))
        records.append(
            {
                "id": row["id"],
                "domain": row["domain"],
                "torch_vs_full_ort": compare_logits(
                    np.asarray(full.run(["logits"], {"features": x})[0]), expected
                ),
                "torch_vs_ort_hidden_max_abs_error": float(
                    np.max(np.abs(expected_hidden - actual_hidden))
                ),
                "torch_vs_ort_hidden_allclose": bool(
                    np.allclose(expected_hidden, actual_hidden, atol=ATOL, rtol=RTOL)
                ),
                "stitched_vs_full_hidden_max_abs_error": stitched_max,
                "stitched_vs_full_hidden_allclose": stitched_close,
            }
        )
    return {"diagnostic_only": True, "tracks": len(records), "records": records}


def benchmark(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if args.validation_report is None or args.selection is None:
        raise ValueError("Benchmark requires a passing validation report and existing selection")
    report = json.loads(args.validation_report.read_text())
    expected = {
        "mode": "validate",
        "passed": True,
        "tracks": 376,
        "manifest_sha256": sha256(args.manifest),
        "full_model_sha256": sha256(args.full_model),
        "contract_sha256": sha256(args.bundle / "contract.json"),
        "core_outputs": args.core_outputs,
        "atol": ATOL,
        "rtol": RTOL,
        **source_provenance(),
    }
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("Validation report does not pass for this exact experiment")
    chosen = selected_rows(rows, args.selection)
    # Do not instantiate the unused model: separate process peaks must remain meaningful.
    if args.bench_mode == "chunked":
        runtime = ChunkedK10(args.bundle, args.core_outputs)
        predict = runtime.predict
    else:
        full = cpu_session(args.full_model)

        def predict(x: np.ndarray) -> np.ndarray:
            return np.asarray(full.run(["logits"], {"features": x})[0])

    scale = 1 / 1024 if sys.platform == "darwin" else 1
    loaded_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    records = []
    for row in chosen:
        x = load_features(args.manifest, row)
        predict(x)  # One explicitly reported warmup per track.
        elapsed = []
        for _ in range(20):
            start = time.perf_counter()
            logits = predict(x)
            elapsed.append(time.perf_counter() - start)
        records.append(
            {
                "id": row["id"],
                "sha256": row["sha256"],
                "frames": row["frames"],
                "cache_sha256": row["cache_sha256"],
                "seconds": elapsed,
                "median_seconds": float(np.median(elapsed)),
                "logits": logits.tolist(),
            }
        )
        del x
    result = {
        "bench_mode": args.bench_mode,
        "records": records,
        "tracks": len(records),
        "selection_sha256": sha256(args.selection),
        "repetitions": 20,
        "warmups_per_track": 1,
        "runtime_loaded_peak_rss_kib": loaded_rss,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale,
        "median_track_median_seconds": float(np.median([r["median_seconds"] for r in records])),
        "note": "Cached features only; excludes decoder/CQT. Pair separate fresh full/chunked processes.",
    }
    if args.full_benchmark_report is not None:
        if args.bench_mode != "chunked":
            raise ValueError("Paired comparison requires chunked mode")
        baseline = json.loads(args.full_benchmark_report.read_text())
        for key, value in {
            "mode": "benchmark",
            "bench_mode": "full",
            "torch_imported": False,
            "full_model_sha256": sha256(args.full_model),
            "manifest_sha256": sha256(args.manifest),
            "contract_sha256": sha256(args.bundle / "contract.json"),
            "selection_sha256": sha256(args.selection),
            "repetitions": 20,
            "core_outputs": args.core_outputs,
            "platform": platform.platform(),
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "thread_environment": thread_environment(),
            **source_provenance(),
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
        }.items():
            if baseline.get(key) != value:
                raise ValueError(f"Incompatible full benchmark: {key}")
        paired = {r["sha256"]: r for r in baseline["records"]}
        if len(paired) != 20 or set(paired) != {r["sha256"] for r in records}:
            raise ValueError("Benchmark identities differ")
        ratios = [r["median_seconds"] / paired[r["sha256"]]["median_seconds"] for r in records]
        comparison = {
            "median_paired_latency_ratio": float(np.median(ratios)),
            "paired_peak_rss_reduction": baseline["peak_process_rss_kib"]
            / result["peak_process_rss_kib"],
            "historical_peak_rss_reduction": 535 * 1024 / result["peak_process_rss_kib"],
            "full_benchmark_sha256": sha256(args.full_benchmark_report),
        }
        result["comparison"] = comparison
        result["performance_target_met"] = (
            comparison["median_paired_latency_ratio"] <= 1.1
            and comparison["paired_peak_rss_reduction"] >= 2
            and comparison["historical_peak_rss_reduction"] >= 2
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "diagnose", "benchmark"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--full-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--full-benchmark-report", type=Path)
    parser.add_argument("--bench-mode", choices=("full", "chunked"), default="chunked")
    parser.add_argument("--core-outputs", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new report path; previous evidence must not be overwritten")
    signal.signal(signal.SIGPROF, budget_exceeded)
    signal.setitimer(signal.ITIMER_PROF, 600)
    start = time.process_time()
    result: dict[str, Any] = {
        "mode": args.mode,
        "atol": ATOL,
        "rtol": RTOL,
        "core_outputs": args.core_outputs,
        "manifest_sha256": sha256(args.manifest),
        "full_model_sha256": sha256(args.full_model),
        "contract_sha256": sha256(args.bundle / "contract.json"),
        "platform": platform.platform(),
        "cpu_budget_seconds": 600,
        "threads": 1,
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "thread_environment": thread_environment(),
        "checkpoint_sha256": json.loads((args.bundle / "contract.json").read_text())[
            "checkpoint_sha256"
        ],
        "prefix_sha256": sha256(args.bundle / "prefix.onnx"),
        "head_sha256": sha256(args.bundle / "head.onnx"),
        **source_provenance(),
        "labels_scored": False,
        "promotion": False,
    }
    try:
        rows = load_rows(args.manifest)
        operation = {"validate": validate, "diagnose": diagnose, "benchmark": benchmark}[args.mode]
        result.update(operation(args, rows))
        if args.mode != "diagnose" and "torch" in sys.modules:
            raise RuntimeError("Isolation failed: Torch imported")
        result["torch_imported"] = "torch" in sys.modules
    except (Exception, KeyboardInterrupt) as error:
        result.update(passed=False, error=f"{type(error).__name__}: {error}")
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
    result["cpu_seconds"] = time.process_time() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    if "error" in result or (args.mode == "validate" and not result.get("passed")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
