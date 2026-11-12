"""Fixed nine-case identical-PCM Mac diagnostic; no decoding or SVML."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from tools import run_cqt_expansion as guard

EXPECTED = {
    "librosa": "1.0.0",
    "numpy": "2.5.4",
    "scipy": "1.18.1",
    "soxr": "1.1.0",
    "onnxruntime": "1.31.1",
}


def export(root: Path, feature_source: Path) -> None:
    import numpy as np

    from experiments.chunked import cpu_session
    from tools.export_cqt_goldens import Trace, load_features_module, source_identity

    canonical = root / "canonical"
    source = json.loads((canonical / "manifest.json").read_bytes())
    if source.get("complete") is not True or len(source["cases"]) != 9:
        raise ValueError("fixed nine membership")
    if (
        guard.sha256(root / "full.onnx")
        != "35fc1bfe5d67f7aed2c46221c15519d0f3d5822351932b5bd9899d47d48cb613"
    ):
        raise ValueError("released full model identity")
    output = root / "mac-reference"
    output.mkdir()
    features = load_features_module(feature_source)
    session = cpu_session(root / "full.onnx")
    result: dict[str, Any] = dict(
        format="k10-cqt-goldens-v1",
        complete=False,
        cases=[],
        full_model_sha256=source["full_model_sha256"],
        sources=source_identity(features),
    )
    for original in source["cases"]:
        pcm_path = canonical / original["pcm"]["path"]
        if guard.sha256(pcm_path) != original["pcm"]["sha256"]:
            raise ValueError("canonical PCM changed")
        pcm = np.fromfile(pcm_path, dtype="<f4")
        with Trace(output, original["id"], original["public"]) as trace:
            padded = features.cqt(pcm)
        unobserved = features.cqt(pcm)
        if not np.array_equal(padded, unobserved):
            raise ValueError("observation changes Mac reference")
        core = np.ascontiguousarray(padded.astype(np.float16).astype(np.float32)[8:200])
        logits = np.asarray(session.run(["logits"], {"features": core[None]})[0])[0]
        linux_path = canonical / original["core"]["path"]
        if guard.sha256(linux_path) != original["core"]["sha256"]:
            raise ValueError("canonical core changed")
        linux_core = np.fromfile(linux_path, dtype="<f4").reshape(192, original["frames"])
        model_only = np.asarray(session.run(["logits"], {"features": linux_core[None]})[0])[0]
        for current, old in zip(trace.stages, original["stages"], strict=True):
            for name in ("sr", "n_fft", "hop", "basis_shape", "pad_mode"):
                if current.get(name) != old.get(name):
                    raise ValueError(f"structural plan change {name}")
        for current, old in zip(trace.resamplings, original["resamplings"], strict=True):
            for name in (
                "after_stage",
                "orig_sr",
                "target_sr",
                "scale",
                "fix",
                "res_type",
                "input_samples",
            ):
                if current[name] != old[name]:
                    raise ValueError(f"structural resampling change {name}")
        case = dict(
            original,
            pcm=trace.artifact("reference.pcm", pcm, "<f4"),
            core=trace.artifact("reference.core", core, "<f4"),
            stages=trace.stages,
            resamplings=trace.resamplings,
            normalization=trace.length_calls[-1],
            reference_logits=logits.tolist(),
            canonical_linux_logits=original["reference_logits"],
            canonical_core_mac_logits=model_only.tolist(),
        )
        result["cases"].append(case)
        guard.write_json(output / "manifest.json", result)
    result.update(
        complete=True,
        tracks=9,
        versions=EXPECTED,
        platform=platform.platform(),
        source_sha256=guard.sha256(Path(__file__)),
        canonical_manifest_sha256=guard.sha256(canonical / "manifest.json"),
    )
    guard.write_json(output / "manifest.json", result)


def compare(values: list[float], reference: list[float]) -> dict[str, Any]:
    import numpy as np

    actual = np.asarray(values, dtype=np.float64)
    expected = np.asarray(reference, dtype=np.float64)
    if actual.shape != (24,) or expected.shape != (24,):
        raise ValueError("logit shape")
    delta = np.abs(actual - expected)
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    same_key = bool(np.argmax(actual) == np.argmax(expected))
    return dict(
        passed=finite and same_key and bool(np.all(delta <= 1e-5 + 1e-5 * np.abs(expected))),
        finite=finite,
        same_key=same_key,
        max_absolute=float(delta.max()),
        logits=actual.tolist(),
        reference=expected.tolist(),
    )


def analyze(root: Path) -> None:
    import numpy as np

    stage = root / "execution"
    mac = json.loads((root / "mac-reference/manifest.json").read_bytes())
    linux = json.loads((root / "canonical/manifest.json").read_bytes())
    native = json.loads((stage / "native.json").read_bytes())
    comparisons = []
    for index, (m, linux_case, n) in enumerate(
        zip(mac["cases"], linux["cases"], native["cases"], strict=True)
    ):
        if not m["id"] == linux_case["id"] == n["id"]:
            raise ValueError("case identity")
        logits = n["iterations"][0]["predictions"][0]["logits"]
        gates = dict(
            native_vs_mac=compare(logits, m["reference_logits"]),
            mac_vs_linux=compare(m["reference_logits"], linux_case["reference_logits"]),
            native_vs_linux=compare(logits, linux_case["reference_logits"]),
            canonical_core_mac_vs_linux=compare(
                m["canonical_core_mac_logits"], linux_case["reference_logits"]
            ),
        )
        a = np.fromfile(stage / "native-cores" / f"{index:04d}.f32", dtype="<f4")
        b = np.fromfile(root / "mac-reference" / m["core"]["path"], dtype="<f4")
        c = np.fromfile(root / "canonical" / linux_case["core"]["path"], dtype="<f4")
        feature = {}
        for name, ref in (("mac", b), ("linux", c)):
            delta = a.astype(np.float64) - ref.astype(np.float64)
            feature[name] = dict(
                max_absolute=float(np.abs(delta).max()),
                rms=float(np.sqrt(np.mean(delta * delta))),
                mismatches=int(np.count_nonzero(a != ref)),
                values=int(a.size),
            )
        comparisons.append(dict(id=m["id"], gates=gates, features=feature))
    guard.write_json(
        stage / "comparisons.json",
        dict(
            complete=True,
            tracks=9,
            cases=comparisons,
            passed=all(g["passed"] for row in comparisons for g in row["gates"].values()),
            source_sha256=guard.sha256(Path(__file__)),
            native_report_sha256=guard.sha256(stage / "native.json"),
            mac_manifest_sha256=guard.sha256(root / "mac-reference/manifest.json"),
            linux_manifest_sha256=guard.sha256(root / "canonical/manifest.json"),
        ),
    )


def darwin_tree(pid: int) -> tuple[int, float]:
    rows = subprocess.check_output(["ps", "-A", "-o", "pid=,ppid=,rss=,time="], text=True)
    records = {}
    for row in rows.splitlines():
        fields = row.split()
        if len(fields) != 4:
            continue
        parts = fields[3].split(":")
        seconds = sum(float(value) * 60**index for index, value in enumerate(reversed(parts)))
        records[int(fields[0])] = (int(fields[1]), int(fields[2]) * 1024, seconds)
    selected = {pid}
    while True:
        expanded = selected | {key for key, value in records.items() if value[0] in selected}
        if expanded == selected:
            break
        selected = expanded
    return sum(records[key][1] for key in selected if key in records), sum(
        records[key][2] for key in selected if key in records
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--feature-source", type=Path, required=True)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--ort", type=Path)
    args = parser.parse_args()
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    if versions != EXPECTED or platform.machine() != "arm64":
        raise ValueError("exact-version/arm64 preflight failed")
    if args.command == "export":
        export(args.root, args.feature_source)
        return
    stage = args.root / "execution"
    stage.mkdir()
    (stage / "logs").mkdir()
    setattr(guard, "process_tree", darwin_tree)
    guard.CPU_LIMIT, guard.WALL_LIMIT = 600.0, 900.0
    supervisor = guard.Supervisor(
        argparse.Namespace(
            stage_root=stage,
            budget_root=args.root,
            deadline="2026-11-12T08:00:00+00:00",
            math_library=None,
            source_root=Path(__file__).parents[1],
        )
    )
    supervisor.environment["VECLIB_MAXIMUM_THREADS"] = "1"
    export_command = [
        sys.executable,
        "-m",
        "tools.cqt_mac_stage",
        "export",
        "--root",
        str(args.root),
        "--feature-source",
        str(args.feature_source),
    ]
    binding = dict(
        versions=versions,
        platform=platform.platform(),
        machine=platform.machine(),
        deadline="2026-11-12T08:00:00+00:00",
        source_sha256=guard.sha256(Path(__file__)),
        guard_sha256=guard.sha256(Path(guard.__file__)),
        exporter_sha256=guard.sha256(Path(__file__).with_name("export_cqt_goldens.py")),
        feature_source_sha256=guard.sha256(args.feature_source),
        native_sha256=guard.sha256(args.native),
        ort_sha256=guard.sha256(args.ort),
        soxr_sha256=guard.sha256(args.root / "libcqt_soxr_split.dylib"),
        protocol_sha256=guard.sha256(
            Path(__file__).parents[1] / "experiments/studies/native-cqt-next/MAC_PROTOCOL.md"
        ),
        canonical_manifest_sha256=guard.sha256(args.root / "canonical/manifest.json"),
        thread_environment={
            name: supervisor.environment[name]
            for name in (*guard.THREADS, "VECLIB_MAXIMUM_THREADS")
        },
    )
    guard.write_json(stage / "identity.json", binding)
    supervisor.run(export_command, "reference")
    core_output = stage / "native-cores"
    command = [
        str(args.native),
        "--mode",
        "qualify",
        "--backend",
        "full",
        "--manifest",
        str(args.root / "mac-reference/manifest.json"),
        "--full-model",
        str(args.root / "full.onnx"),
        "--ort-library",
        str(args.ort),
        "--soxr-wrapper",
        str(args.root / "libcqt_soxr_split.dylib"),
        "--soxr-sha256",
        guard.sha256(args.root / "libcqt_soxr_split.dylib"),
        "--core-output-directory",
        str(core_output),
        "--output",
        str(stage / "native.json"),
    ]
    code = supervisor.run(command, "native", True)
    analyze(args.root)
    guard.write_json(
        stage / "resources.json",
        dict(
            exit_code=code,
            resources=supervisor.resources,
            peak_owned_tree_rss_bytes=supervisor.peak_resident,
            native_sha256=guard.sha256(args.native),
            source_sha256=guard.sha256(Path(__file__)),
        ),
    )


if __name__ == "__main__":
    main()
