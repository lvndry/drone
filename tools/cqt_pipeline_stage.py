"""Frozen composed CQT qualification and conditional paired timing supervisor."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import resource
import shutil
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from tools import run_cqt_expansion as guard


def sha(path: Path) -> str:
    return guard.sha256(path)


def load_features(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("pipeline_original_features", path)
    if spec is None or spec.loader is None:
        raise ValueError("feature source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def produce(config: dict[str, Any], index: int, output: Path) -> None:
    import numpy as np

    study = json.loads(Path(config["study"]).read_bytes())
    row = study["rows"][index]
    canonical_path = Path(config["canonical_root"]) / "cases" / f"{index:04d}-reference.json"
    canonical_report = json.loads((Path(config["canonical_root"]) / "report.json").read_bytes())
    if sha(canonical_path) != canonical_report["cases"][index]["reference_sha256"]:
        raise ValueError("canonical reference record changed")
    canonical = json.loads(canonical_path.read_bytes())["cases"][0]
    if canonical["id"] != row["id"] or sha(Path(row["path"])) != row["sha256"]:
        raise ValueError("fixed audio identity")
    output.mkdir()
    features = load_features(Path(config["feature_source"]))
    wav = output / "decode.wav"
    features.decode_mp3(Path(row["path"]), wav)
    pcm, rate = features.read_wav_mono(wav)
    pcm = features.resample_linear(pcm, rate, 22050)
    wav.unlink()
    pcm = np.ascontiguousarray(pcm, dtype="<f4")
    pcm_path = output / "pcm.f32"
    pcm_path.write_bytes(pcm.tobytes())
    if sha(pcm_path) != canonical["pcm"]["sha256"]:
        raise ValueError("decoded PCM differs from fixed canonical PCM")
    golden_path = Path(config["goldens"])
    golden = json.loads(golden_path.read_bytes())["cases"][0]
    stages = []
    for stage_index, (stage, template) in enumerate(
        zip(canonical["stages"], golden["stages"], strict=True)
    ):
        stage = dict(stage)
        for name in ("basis", "indices", "indptr"):
            artifact = template[name]
            if stage[name]["sha256"] != artifact["sha256"]:
                raise ValueError("invariant kernel changed")
            source = golden_path.parent / artifact["path"]
            if sha(source) != artifact["sha256"]:
                raise ValueError("kernel bytes")
            path = output / f"stage{stage_index}-{name}"
            shutil.copyfile(source, path)
            stage[name] = dict(artifact, path=path.name)
        stages.append(stage)
    case = dict(canonical, stages=stages, pcm=dict(canonical["pcm"], path=pcm_path.name))
    guard.write_json(
        output / "manifest.json",
        dict(
            format="k10-composed-case-v1",
            complete=True,
            full_model_sha256=study["full_model_sha256"],
            cases=[case],
        ),
    )


def python_benchmark(config: dict[str, Any], manifest: Path, output: Path) -> None:
    import numpy as np

    from experiments.chunked import cpu_session

    case = json.loads(manifest.read_bytes())["cases"][0]
    pcm_path = manifest.parent / case["pcm"]["path"]
    if sha(pcm_path) != case["pcm"]["sha256"]:
        raise ValueError("PCM input")
    pcm = np.fromfile(pcm_path, dtype="<f4")
    features = load_features(Path(config["feature_source"]))
    __import__("librosa")
    session = cpu_session(Path(config["full_model"]))
    loaded = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/self/status").read_text().splitlines()
                if line.startswith("VmRSS:")
            )
        )
        * 1024
    )
    expected = np.asarray(case["reference_logits"], dtype=np.float64)
    seconds = []
    predictions = []
    for iteration in range(21):
        started = time.perf_counter()
        core = np.ascontiguousarray(features.cqt(pcm).astype(np.float16).astype(np.float32)[8:200])
        logits = np.asarray(session.run(["logits"], {"features": core[None]})[0])[0]
        duration = time.perf_counter() - started
        valid = (
            np.isfinite(core).all()
            and np.isfinite(logits).all()
            and hashlib.sha256(core.tobytes()).hexdigest() == case["core"]["sha256"]
            and bool(
                np.all(
                    np.abs(logits.astype(np.float64) - expected) <= 1e-5 + 1e-5 * np.abs(expected)
                )
            )
            and int(np.argmax(logits)) == int(np.argmax(expected))
        )
        if not valid:
            raise ValueError("Python timed path parity failed")
        if iteration:
            seconds.append(duration)
        predictions.append(logits.tolist())
    if "torch" in sys.modules:
        raise RuntimeError("Torch entered the timed Python worker")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    guard.write_json(
        output,
        dict(
            passed=True,
            id=case["id"],
            seconds=seconds,
            logits=predictions,
            loaded_rss_bytes=loaded,
            peak_rss_bytes=usage.ru_maxrss * 1024,
            cpu_seconds=usage.ru_utime + usage.ru_stime,
            torch_imported="torch" in sys.modules,
        ),
    )


def native_command(
    config: dict[str, Any],
    manifest: Path,
    output: Path,
    backend: str,
    qualification: Path | None = None,
) -> list[str]:
    command = [
        config["native"],
        "--mode",
        "benchmark" if qualification else "qualify",
        "--backend",
        backend,
        "--manifest",
        str(manifest),
        "--full-model",
        config["full_model"],
        "--bundle",
        config["bundle"],
        "--ort-library",
        config["ort_library"],
        "--soxr-wrapper",
        config["soxr_wrapper"],
        "--soxr-sha256",
        sha(Path(config["soxr_wrapper"])),
        "--math-library",
        config["math_library"],
        "--math-sha256",
        sha(Path(config["math_library"])),
        "--exact-core",
        "--output",
        str(output),
    ]
    if qualification:
        command.extend(["--qualification-report", str(qualification)])
    return command


def identity(config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        f"{key}_sha256": sha(Path(config[key]))
        for key in (
            "native",
            "ort_library",
            "soxr_wrapper",
            "math_library",
            "full_model",
            "feature_source",
            "fixtures",
            "study",
            "goldens",
        )
    }
    result.update(
        native_sha256=result.pop("native_sha256"),
        ort_sha256=result.pop("ort_library_sha256"),
        soxr_sha256=result.pop("soxr_wrapper_sha256"),
        math_sha256=result.pop("math_library_sha256"),
        evaluator_sha256=sha(Path(__file__)),
    )
    result["guard_supervisor_source_sha256"] = sha(Path(guard.__file__))
    result["cpu_session_source_sha256"] = sha(Path(__file__).parents[1] / "experiments/chunked.py")
    result["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("librosa", "numpy", "scipy", "soxr", "onnxruntime")
    }
    expected = {
        "librosa": "1.0.0",
        "numpy": "2.5.4",
        "scipy": "1.18.1",
        "soxr": "1.1.0",
        "onnxruntime": "1.31.1",
    }
    if result["versions"] != expected:
        raise ValueError("pinned Python environment changed")
    result["thread_environment"] = {name: "1" for name in guard.THREADS}
    result["bundle"] = {
        name: sha(Path(config["bundle"]) / name)
        for name in ("prefix.onnx", "head.onnx", "contract.json")
    }
    return result


def supervise(config_path: Path, phase: str) -> None:
    config = json.loads(config_path.read_bytes())
    stage = Path(config["root"]) / phase
    stage.mkdir()
    for name in ("logs", "cases"):
        (stage / name).mkdir()
    args = argparse.Namespace(
        stage_root=stage,
        budget_root=Path(config["root"]),
        math_library=Path(config["math_library"]),
        deadline=config["deadline"],
        source_root=Path(config["source_root"]),
    )
    guard.CPU_LIMIT = 1200.0 if phase == "qualification" else 1800.0
    guard.WALL_LIMIT = 1500.0 if phase == "qualification" else 2100.0
    supervisor = guard.Supervisor(args)
    binding = identity(config)
    binding["configuration_sha256"] = sha(config_path)
    binding["protocol_sha256"] = sha(Path(config["protocol"]))
    binding["frozen_sources"] = config["frozen_sources"]
    study = json.loads(Path(config["study"]).read_bytes())
    if study["complete"] is not True or len(study["rows"]) != 376:
        raise ValueError("fixed376 membership")
    qualification = Path(config["root"]) / "qualification/report.json"
    if phase == "timing":
        previous = json.loads(qualification.read_bytes())
        if (
            previous.get("complete") is not True
            or previous.get("passed") is not True
            or previous["identity"] != binding
        ):
            raise ValueError("same-source passing composed qualification required")
        fixtures = json.loads(Path(config["fixtures"]).read_bytes())
        chosen = {(r["domain"], r["id"]) for r in fixtures["rows"] if r["selection20"]}
        indices = [i for i, row in enumerate(study["rows"]) if (row["domain"], row["id"]) in chosen]
        if len(indices) != 20:
            raise ValueError("fixed20 selection")
    else:
        indices = list(range(376))
    report: dict[str, Any] = dict(
        format="k10-composed-stage-v1",
        complete=False,
        passed=False,
        identity=binding,
        tracks=376 if phase == "qualification" else 20,
        cases=[],
        phase=phase,
    )

    def checkpoint() -> None:
        report.update(
            resources=supervisor.resources,
            child_cpu_seconds=guard.child_cpu(),
            wall_seconds=time.monotonic() - supervisor.started,
            peak_owned_tree_rss_bytes=supervisor.peak_resident,
            storage_bytes=guard.disk_bytes(Path(config["root"])),
        )
        guard.write_json(stage / "progress.json", report)

    try:
        if phase == "qualification":
            result = stage / "nine.json"
            supervisor.run(
                ["taskset", "-c", "4,5"]
                + native_command(config, Path(config["goldens"]), result, "both"),
                "nine",
            )
            if json.loads(result.read_bytes())["passed"] is not True:
                raise ValueError("nine gate")
        for position, index in enumerate(indices):
            spool = stage / f"spool-{index:04d}"
            supervisor.run(
                [
                    "taskset",
                    "-c",
                    "4,5",
                    config["python"],
                    "-m",
                    "tools.cqt_pipeline_stage",
                    "produce",
                    "--config",
                    str(config_path),
                    "--index",
                    str(index),
                    "--output",
                    str(spool),
                ],
                f"{index:04d}-pcm",
            )
            manifest = spool / "manifest.json"
            meta = json.loads(manifest.read_bytes())
            guard.write_json(stage / "cases" / f"{index:04d}-input.json", meta)
            results = {}
            modes = ["both"] if phase == "qualification" else ["python", "full", "chunked"]
            if phase == "timing":
                offset = position % 3
                modes = modes[offset:] + modes[:offset]
            for backend in modes:
                output = stage / "cases" / f"{index:04d}-{backend}.json"
                if backend == "python":
                    command = [
                        config["python"],
                        "-m",
                        "tools.cqt_pipeline_stage",
                        "python-benchmark",
                        "--config",
                        str(config_path),
                        "--manifest",
                        str(manifest),
                        "--output",
                        str(output),
                    ]
                else:
                    command = native_command(
                        config,
                        manifest,
                        output,
                        backend,
                        qualification if phase == "timing" else None,
                    )
                code = supervisor.run(
                    ["taskset", "-c", "2" if phase == "timing" else "4,5"] + command,
                    f"{index:04d}-{backend}",
                    True,
                )
                if not output.exists():
                    raise ValueError(f"{backend} failed without report: exit{code}")
                data = json.loads(output.read_bytes())
                results[backend] = dict(passed=data["passed"], report_sha256=sha(output))
                if code not in (0, 1) or (code == 0) != (data["passed"] is True):
                    raise ValueError("exit/status mismatch")
            passed = all(r["passed"] for r in results.values())
            report["cases"].append(
                dict(
                    index=index,
                    id=study["rows"][index]["id"],
                    pcm_sha256=meta["cases"][0]["pcm"]["sha256"],
                    passed=passed,
                    results=results,
                )
            )
            if not passed and not (stage / "first-failure").exists():
                spool.rename(stage / "first-failure")
            else:
                shutil.rmtree(spool)
            checkpoint()
        report.update(complete=True, passed=all(row["passed"] for row in report["cases"]))
        checkpoint()
        guard.write_json(stage / "report.json", report)
    except BaseException as error:
        report["error"] = str(error)
        checkpoint()
        guard.write_json(stage / "failed.json", report)
        raise
    if not report["passed"]:
        raise RuntimeError("composed gate failed; no timing qualification")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["produce", "python-benchmark", "qualification", "timing"]
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    if args.command == "produce":
        produce(config, args.index, args.output)
    elif args.command == "python-benchmark":
        python_benchmark(config, args.manifest, args.output)
    else:
        supervise(args.config, args.command)


if __name__ == "__main__":
    main()
