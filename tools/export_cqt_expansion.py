"""Label-blind, one-track reference producer for fixed public native-CQT validation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soxr

from experiments.chunked import FEATURE_SPEC, cpu_session, sha256
from tools.export_cqt_goldens import Trace, load_features_module, source_identity, write_json
from tools.research.contracts import load_validation_tracks


def plan_identity(case: dict[str, Any]) -> dict[str, Any]:
    return dict(
        stages=[
            {
                key: stage[key]
                for key in ("index", "sr", "n_fft", "hop", "frequencies", "pad_mode", "basis_shape")
            }
            | {key: stage[key]["sha256"] for key in ("indptr", "indices", "basis")}
            for stage in case["stages"]
        ],
        resamplings=[
            {
                key: entry[key]
                for key in (
                    "index",
                    "after_stage",
                    "orig_sr",
                    "target_sr",
                    "scale",
                    "fix",
                    "res_type",
                )
            }
            for entry in case["resamplings"]
        ],
    )


def initialize(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError("Immutable study output already exists")
    reference = json.loads(args.goldens.read_text())
    if reference.get("complete") is not True or reference.get("tracks") != 9:
        raise ValueError("Require complete fixed-nine goldens")
    common = plan_identity(reference["cases"][0])
    if any(plan_identity(case) != common for case in reference["cases"]):
        raise ValueError("Nine reference plans disagree on invariant constants")
    tracks = sorted(
        load_validation_tracks(args.data_manifest), key=lambda track: (track.domain, track.id)
    )
    raw = json.loads(args.data_manifest.read_text())["rows"]
    paths = {row["id"]: row["path"] for row in raw if row["role"] == "validation"}
    if sha256(args.full_model) != reference["full_model_sha256"]:
        raise ValueError("Changed authoritative model")
    features = load_features_module(args.feature_source)
    payload = dict(
        format="k10-cqt-expansion-study-v1",
        complete=True,
        tracks=376,
        feature_spec=FEATURE_SPEC,
        full_model_sha256=reference["full_model_sha256"],
        source_manifest_sha256=sha256(args.data_manifest),
        golden_manifest_sha256=sha256(args.goldens),
        protocol_sha256=sha256(args.protocol),
        producer_source_sha256=sha256(Path(__file__)),
        golden_export_source_sha256=sha256(Path(__file__).with_name("export_cqt_goldens.py")),
        feature_source_sha256=sha256(args.feature_source),
        source_identity=source_identity(features),
        versions={
            name: importlib.metadata.version(name)
            for name in ("librosa", "soxr", "numpy", "scipy", "onnxruntime")
        },
        common_plan=common,
        normalization_policy="Capture each track's actual wavelet-length divisors anew; no silence statistics reused",
        accuracy_scored=False,
        rows=[track.as_dict() | {"path": paths[track.id]} for track in tracks],
    )
    write_json(args.output, payload)


def produce(args: argparse.Namespace) -> None:
    study = json.loads(args.study.read_text())
    if (
        study.get("format") != "k10-cqt-expansion-study-v1"
        or study.get("complete") is not True
        or study.get("tracks") != 376
        or len(study["rows"]) != 376
    ):
        raise ValueError("Expected immutable fixed376 study")
    if importlib.metadata.version("librosa") != "1.0.0" or soxr.__version__ != "1.1.0":
        raise ValueError("Unexpected reference frontend versions")
    if (
        sha256(Path(__file__)) != study["producer_source_sha256"]
        or sha256(args.feature_source) != study["feature_source_sha256"]
        or sha256(args.full_model) != study["full_model_sha256"]
    ):
        raise ValueError("Changed producer/frontend/model source")
    if not 0 <= args.row_index < 376:
        raise ValueError("Bad fixed study index")
    row = study["rows"][args.row_index]
    audio = Path(row["path"])
    if sha256(audio) != row["sha256"]:
        raise ValueError("Changed public source audio")
    if args.spool.exists():
        raise FileExistsError("Choose a fresh case spool")
    args.spool.mkdir()
    features = load_features_module(args.feature_source)
    decoded = args.spool / "decoded.wav"
    features.decode_mp3(audio, decoded)
    pcm, rate = features.read_wav_mono(decoded)
    pcm = features.resample_linear(pcm, rate, 22050)
    decoded.unlink()
    with Trace(args.spool, "reference", True) as trace:
        padded = features.cqt(pcm)
    if padded.shape != (208, row["frames"]):
        raise ValueError("Fresh reference shape differs from fixed metadata")
    if (
        plan_identity(dict(stages=trace.stages, resamplings=trace.resamplings))
        != study["common_plan"]
    ):
        raise ValueError("Track changed invariant kernel/resampler plan")
    pcm_artifact = trace.artifact("reference.pcm", pcm, "<f4")
    del pcm
    core = np.ascontiguousarray(padded.astype(np.float16).astype(np.float32)[8:200])
    del padded
    core_artifact = trace.artifact("reference.core", core, "<f4")
    session = cpu_session(args.full_model)
    logits = np.asarray(session.run(["logits"], {"features": core[None]})[0])[0]
    if not np.isfinite(core).all() or not np.isfinite(logits).all() or "torch" in sys.modules:
        raise ValueError("Nonfinite reference or unexpected Torch dependency")
    case = dict(
        id=row["id"],
        domain=row["domain"],
        public=True,
        frames=row["frames"],
        audio_sha256=row["sha256"],
        cache_sha256=row["cache_sha256"],
        pcm=pcm_artifact,
        core=core_artifact,
        reference_logits=logits.tolist(),
        stages=trace.stages,
        resamplings=trace.resamplings,
        normalization=trace.length_calls[-1],
    )
    result = dict(
        format="k10-cqt-expansion-case-v1",
        complete=True,
        tracks=1,
        expected_study_tracks=376,
        study_manifest_sha256=sha256(args.study),
        row_index=args.row_index,
        feature_spec=FEATURE_SPEC,
        sample_rate=22050,
        hop=4096,
        bins=208,
        core_slice=[8, 200],
        full_model_sha256=study["full_model_sha256"],
        cases=[case],
        torch_imported=False,
    )
    write_json(args.spool / "manifest.json", result)
    print(
        json.dumps(dict(id=row["id"], frames=row["frames"], pcm_sha256=pcm_artifact["sha256"])),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("initialize")
    for name in ("goldens", "data-manifest", "protocol", "feature-source", "full-model", "output"):
        init.add_argument(f"--{name}", type=Path, required=True)
    case = commands.add_parser("produce")
    for name in ("study", "spool", "feature-source", "full-model"):
        case.add_argument(f"--{name}", type=Path, required=True)
    case.add_argument("--row-index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "initialize":
        initialize(args)
    else:
        produce(args)


if __name__ == "__main__":
    main()
