"""Conditional, predeclared one-thread full cached-inference deployment screen."""

from __future__ import annotations

import argparse
import copy
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from drone.features import audio_digest
from experiments.small import atomic_json
from experiments.temporal_cache import (
    FrozenEncoder,
    identity,
    load_encoder,
    reconstruct,
    validate_cache,
    verify_original,
    within_tolerance,
)
from experiments.temporal_study import load_head


class FullModel(nn.Module):
    def __init__(self, encoder: FrozenEncoder, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.backbone = encoder.net[:-2]
        self.residual = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x.unsqueeze(1))
        lengths = torch.ones(hidden.shape[0], dtype=torch.long, device=x.device) * hidden.shape[-1]
        return reconstruct(self.encoder, hidden) + self.residual(hidden, lengths)


def export_full(model: FullModel, output: Path) -> None:
    dummy = torch.zeros(1, 192, 320)
    from drone.export import _TimeMean

    exported_model = copy.deepcopy(model)
    exported_model.encoder.net[-1] = _TimeMean()
    torch.onnx.export(
        exported_model,
        (dummy,),
        str(output),
        input_names=["cqt"],
        output_names=["logits"],
        dynamo=True,
        opset_version=18,
        dynamic_shapes=({2: torch.export.Dim("frames", min=8)},),
        external_data=False,
    )


def screen(args: argparse.Namespace) -> None:
    result = json.loads((args.study / "result.json").read_bytes())
    if (
        result["status"] != "complete"
        or not result["gates"]["candidate_pass"]
        or result["identity"] != identity()
    ):
        raise ValueError("CPU screen requires passing complete O candidate under frozen identity")
    if args.output.exists():
        raise ValueError("Use fresh CPU output")
    args.output.mkdir(parents=True)
    start = time.monotonic()
    aggregate = json.loads((args.cache / "reconstruction.json").read_bytes())[
        "aggregate_started_epoch"
    ]

    def check() -> None:
        if time.monotonic() - start > 600 or time.time() - aggregate > 3600:
            raise TimeoutError("Fixed10-minute CPU/60-minute aggregate budget exhausted")

    manifest, _ = validate_cache(args.cache, check)
    expected = dict(
        identity=identity(),
        arm="O",
        encoder_seed=manifest["encoder_seed"],
        cache_manifest_sha256=audio_digest(args.cache / "manifest.json"),
        selected_epoch=result["arms"]["O"]["best_epoch"],
        selection=result["arms"]["O"]["selected_criterion"],
    )
    if audio_digest(args.study / "best-O.pt") != result["arms"]["O"]["checkpoint_sha256"]:
        raise ValueError("Selected O artifact changed")
    verify_original(args.reference_source, manifest["archived_source_sha256"], check)
    run = json.loads(args.reference_run.read_bytes())
    if audio_digest(args.reference_run) != manifest["reference_run_sha256"]:
        raise ValueError("Canonical reference changed")
    torch.set_num_threads(1)
    head = load_head(args.study / "best-O.pt", expected).eval()
    encoder = load_encoder(
        args.reference_run.with_name("best.pt"), args.reference_source, run, "cpu"
    )
    model = FullModel(encoder, head).eval()
    validation = [r for r in manifest["rows"] if r["role"] == "validation"]
    selected_path = args.study / "validation-O.json"
    if audio_digest(selected_path) != result["arms"]["O"]["validation_sha256"]:
        raise ValueError("Selected O logits artifact changed")
    selected = json.loads(selected_path.read_bytes())
    if selected["ids"] != [r["id"] for r in validation] or selected["refs"] != [
        r["ref"] for r in validation
    ]:
        raise ValueError("Selected O prediction identities changed")
    selected_logits = np.asarray(selected["logits"], dtype=np.float32)
    if selected_logits.shape != (376, 24):
        raise ValueError("Selected O logits shape changed")
    atomic_json(
        args.output / "measurement-identity.json",
        dict(
            study_result_sha256=audio_digest(args.study / "result.json"),
            ids=[r["id"] for r in validation],
            threads=1,
            warmups_per_track=1,
            repetitions_per_track=5,
            order="interleaved baseline then full; no decode/CQT/cache I/O in timers",
            percentile="numpy linear",
            frontend="identical cached CQT; PyTorch baseline/full timing; ONNX full parity only",
            ort=dict(
                intra_op_num_threads=1,
                inter_op_num_threads=1,
                execution_mode="sequential",
                spinning=False,
            ),
        ),
    )
    # Export only after qualification. Export failure is an inconclusive CPU screen,
    # never permission to change the frozen accuracy architecture or gate.
    import onnxruntime as ort

    export_full(model, args.output / "full.onnx")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    session = ort.InferenceSession(
        str(args.output / "full.onnx"), options, providers=["CPUExecutionProvider"]
    )
    records = []
    max_error = 0.0
    for index, row in enumerate(validation):
        check()
        source = (args.data / row["cache"]).resolve()
        if (
            not source.is_relative_to(args.data.resolve())
            or audio_digest(source) != row["cache_sha256"]
        ):
            raise ValueError("CQT content changed")
        x = torch.from_numpy(
            np.asarray(np.load(source, mmap_mode="r", allow_pickle=False)[8:200], dtype=np.float32)[
                None
            ]
        )
        n = row["hidden_shape"][2]
        hidden = torch.from_numpy(np.load(args.cache / row["hidden"], allow_pickle=False)[None])
        lengths = torch.tensor([n])
        with torch.inference_mode():
            reference = model(x).numpy()
            if not within_tolerance(reference, selected_logits[index : index + 1]):
                raise ValueError(
                    "CPU full prediction differs from selected cached-GPU head logits/argmax"
                )
            encoder(x)
            head(hidden, lengths)  # one untimed warmup of every measured path
            exported = np.asarray(session.run(None, {"cqt": x.numpy()})[0])
            if not within_tolerance(exported, reference):
                raise ValueError("Exported full logits/argmax parity failed")
            max_error = max(max_error, float(np.max(np.abs(exported - reference))))
            baseline = []
            full = []
            absolute_head = []
            for _ in range(5):
                check()
                t = time.perf_counter()
                encoder(x)
                baseline.append(time.perf_counter() - t)
                t = time.perf_counter()
                model(x)
                full.append(time.perf_counter() - t)
                t = time.perf_counter()
                head(hidden, lengths)
                absolute_head.append(time.perf_counter() - t)
        records.append(
            dict(
                id=row["id"],
                baseline_seconds=baseline,
                full_seconds=full,
                head_seconds=absolute_head,
            )
        )
    b = np.array([np.median(r["baseline_seconds"]) for r in records])
    f = np.array([np.median(r["full_seconds"]) for r in records])
    ratio = (f - b) / b
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )
    report: dict[str, Any] = dict(
        status="complete",
        records=records,
        median_increment=float(np.median(ratio)),
        p95_increment=float(np.percentile(ratio, 95, method="linear")),
        rss_bytes=int(rss),
        rss_scope="Whole CPU screen including export/ORT; not isolated model/head memory",
        parity=dict(recordings=len(records), argmax_disagreements=0, max_abs_error=max_error),
        elapsed_seconds=time.monotonic() - start,
        onnx_sha256=audio_digest(args.output / "full.onnx"),
        promoted=False,
    )
    report["deployment_screen_pass"] = (
        report["median_increment"] <= 0.10 and report["p95_increment"] <= 0.20
    )
    check()
    atomic_json(args.output / "result.json", report)


def main() -> None:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--reference-run", type=Path, required=True)
    p.add_argument("--reference-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        screen(args)
    except (ValueError, RuntimeError, TimeoutError, OSError, MemoryError) as error:
        if args.output.is_dir():
            atomic_json(
                args.output / "failure.json",
                dict(status="cpu_inconclusive", error=str(error), promoted=False),
            )
        raise


if __name__ == "__main__":
    main()
