"""Fresh-process cached-feature memory per track; no Torch or accuracy scoring."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.chunked import ATOL, RTOL, ChunkedK10, cpu_session, sha256
from tools.research.contracts import (
    load_cached_features,
    load_validation_tracks,
    select_performance_tracks,
)
from tools.research.evidence import ArtifactSet, source_provenance
from tools.research.measurements import (
    peak_rss_mib,
    runtime_environment,
    time_prediction,
)


def rss_mib() -> float:
    """Compatibility alias; process-unit conversion lives with other measurements."""
    return peak_rss_mib()


def check_validation(args: argparse.Namespace) -> None:
    ArtifactSet(args.manifest, args.bundle, args.full_model).require_validation(
        args.validation_report
    )


def measure(args: argparse.Namespace) -> dict[str, Any]:
    chosen = select_performance_tracks(load_validation_tracks(args.manifest), args.selection)
    row = chosen[args.index]
    before = rss_mib()
    if args.mode == "full":
        session = cpu_session(args.full_model)

        def predict(x: np.ndarray) -> np.ndarray:
            return np.asarray(session.run(["logits"], {"features": x})[0])

    else:
        runtime = ChunkedK10(args.bundle, 64)
        predict = runtime.predict
    loaded = rss_mib()
    x = load_cached_features(args.manifest, row)
    features_loaded = rss_mib()
    timing = time_prediction(predict, x)
    if "torch" in sys.modules:
        raise RuntimeError("Torch entered the isolated process")
    return dict(
        id=row.id,
        sha256=row.sha256,
        cache_sha256=row.cache_sha256,
        frames=row.frames,
        duration_proxy_seconds=row.frames * 4096 / 22050,
        mode=args.mode,
        before_session_peak_mib=before,
        loaded_session_peak_mib=loaded,
        loaded_features_peak_mib=features_loaded,
        process_peak_mib=rss_mib(),
        median_seconds=timing.median_seconds,
        seconds=timing.seconds,
        logits=timing.logits.tolist(),
        torch_imported=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "selection", "bundle", "full-model", "validation-report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("full", "chunked"))
    parser.add_argument("--index", type=int, choices=range(20))
    args = parser.parse_args()
    check_validation(args)
    if args.mode:
        if args.index is None:
            parser.error("Child mode requires --index")
        print(json.dumps(measure(args), allow_nan=False))
        return
    if args.output is None or args.output.exists():
        parser.error("Choose a new --output path")
    start = time.monotonic()
    records = []
    common = []
    for name in ("manifest", "selection", "bundle", "full-model", "validation-report"):
        common.extend([f"--{name}", str(getattr(args, name.replace("-", "_")))])
    for index in range(20):
        for mode in ("full", "chunked"):
            remaining = 600 - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError("600-second total wall budget exhausted")
            child = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.benchmark_k10_memory_curve",
                    *common,
                    "--mode",
                    mode,
                    "--index",
                    str(index),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=min(60, remaining),
            )
            records.append(json.loads(child.stdout))
        print(f"Measured {index + 1}/20 track pairs", flush=True)
    pairs = [records[i : i + 2] for i in range(0, 40, 2)]
    parity = all(
        np.allclose(a["logits"], b["logits"], atol=ATOL, rtol=RTOL)
        and np.argmax(a["logits"]) == np.argmax(b["logits"])
        for a, b in pairs
    )
    report = dict(
        status="complete",
        records=records,
        fresh_processes=40,
        repetitions=20,
        warmups_per_track=1,
        paired_parity_passed=bool(parity),
        **runtime_environment(),
        implementation=source_provenance(),
        source_sha256=sha256(Path(__file__)),
        runtime_sha256=sha256(Path(__file__).parents[1] / "experiments/chunked.py"),
        validation_report_sha256=sha256(args.validation_report),
        selection_sha256=sha256(args.selection),
        manifest_sha256=sha256(args.manifest),
        full_model_sha256=sha256(args.full_model),
        contract_sha256=sha256(args.bundle / "contract.json"),
        elapsed_seconds=time.monotonic() - start,
        accuracy_scored=False,
        limitations="Process RSS high-water per track; includes input features and Python/ORT. Excludes audio decode/CQT and native Euphoria. Duration is a CQT-frame proxy.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not parity:
        raise RuntimeError("Fresh-process paired parity failed; retain report for diagnosis")


if __name__ == "__main__":
    main()
