"""Export validated, label-blind FP32 inputs for the native K10 research runner."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, TypedDict

import numpy as np
import onnxruntime as ort

from experiments.chunked import FEATURE_SPEC, LABEL_SCHEMA, cpu_session, sha256
from tools.research.contracts import (
    load_cached_features,
    load_validation_tracks,
    select_performance_tracks,
)
from tools.research.evidence import ArtifactSet, source_provenance, support_sources
from tools.research.measurements import runtime_environment


class NativeFixtureRow(TypedDict):
    id: str
    domain: str
    frames: int
    path: str
    sha256: str
    audio_sha256: str
    cache_sha256: str
    source_sha256: str
    reference_logits: list[float]
    selection20: bool


@dataclass(frozen=True)
class ExportRequest:
    artifacts: ArtifactSet
    selection: Path
    validation_report: Path
    output: Path


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def export_fixtures(request: ExportRequest) -> dict[str, Any]:
    started = time.monotonic()
    if request.output.exists():
        raise FileExistsError(f"Choose a new fixture directory: {request.output}")
    request.artifacts.require_validation(request.validation_report)
    if ort.__version__ != "1.31.1":
        raise ValueError("Native comparison requires the pinned ONNX Runtime 1.31.1")
    tracks = load_validation_tracks(request.artifacts.manifest)
    chosen = select_performance_tracks(tracks, request.selection)
    selected = {(track.domain, track.id) for track in chosen}
    contract = json.loads((request.artifacts.bundle / "contract.json").read_bytes())
    session = cpu_session(request.artifacts.full_model)
    request.output.mkdir(parents=True)
    (request.output / "features").mkdir()
    rows: list[NativeFixtureRow] = []
    manifest: dict[str, Any] = dict(
        format="k10-core-f32le-v1",
        complete=False,
        feature_spec=FEATURE_SPEC,
        label_schema=LABEL_SCHEMA,
        layout="bands,time",
        dtype="float32-le",
        bands=192,
        full_model_sha256=sha256(request.artifacts.full_model),
        checkpoint_sha256=contract["checkpoint_sha256"],
        reference_onnxruntime=ort.__version__,
        validation_report_sha256=sha256(request.validation_report),
        export_tool_sha256=sha256(Path(__file__)),
        support_source_sha256=support_sources(),
        source_provenance=source_provenance(),
        feature_manifest_sha256=sha256(request.artifacts.manifest),
        selection_sha256=sha256(request.selection),
        environment=runtime_environment(),
        cpu_wall_budget_seconds=600,
        accuracy_scored=False,
        rows=rows,
        source_sha256_definition="Each row source_sha256 is the original padded FP16 cache hash; raw-file sha256 identifies its widened core.",
    )
    path = request.output / "manifest.json"
    write_manifest(path, manifest)
    for index, track in enumerate(tracks):
        if time.monotonic() - started > 600:
            raise TimeoutError("Native fixture export exceeded its 600-second wall budget")
        features = load_cached_features(request.artifacts.manifest, track)
        logits = np.asarray(session.run(["logits"], {"features": features})[0])
        if logits.shape != (1, 24) or logits.dtype != np.float32 or not np.isfinite(logits).all():
            raise ValueError("Full model did not produce finite FP32 [1,24] logits")
        relative = f"features/{index:06}.f32le"
        raw = request.output / relative
        with raw.open("xb") as handle:
            handle.write(features[0].astype("<f4", copy=False).tobytes(order="C"))
        rows.append(
            dict(
                id=track.id,
                domain=track.domain,
                frames=track.frames,
                path=relative,
                sha256=sha256(raw),
                audio_sha256=track.sha256,
                cache_sha256=track.cache_sha256,
                source_sha256=track.cache_sha256,
                reference_logits=logits[0].tolist(),
                selection20=(track.domain, track.id) in selected,
            )
        )
        if (index + 1) % 20 == 0:
            manifest["elapsed_seconds"] = time.monotonic() - started
            write_manifest(path, manifest)
            print(f"Exported {index + 1}/{len(tracks)} native inputs", flush=True)
    if "torch" in sys.modules:
        raise RuntimeError("Torch entered the label-blind fixture exporter")
    if time.monotonic() - started > 600:
        raise TimeoutError("Native fixture export exceeded its 600-second wall budget")
    manifest.update(
        complete=True,
        elapsed_seconds=time.monotonic() - started,
        tracks=len(rows),
        selected_tracks=sum(row["selection20"] for row in rows),
        torch_imported=False,
    )
    write_manifest(path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "selection", "bundle", "full-model", "validation-report", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, budget_exceeded)
    signal.signal(signal.SIGPROF, budget_exceeded)
    signal.setitimer(signal.ITIMER_REAL, 600)
    signal.setitimer(signal.ITIMER_PROF, 600)
    try:
        report = export_fixtures(
            ExportRequest(
                ArtifactSet(args.manifest, args.bundle, args.full_model),
                args.selection,
                args.validation_report,
                args.output,
            )
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_PROF, 0)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


def budget_exceeded(signum: int, frame: FrameType | None) -> None:
    raise TimeoutError("Native fixture export exceeded its 600-second CPU/wall budget")


if __name__ == "__main__":
    main()
