"""Fixed0308 frontend observations only; never initialize ORT or load a model.

Execution still requires separately reviewed runtime/source/build and UTC limits.
This module supplies no decoder, subprocess launcher or inference path.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

MANIFEST = "462205de24ff451e040afe45a3409cfc3fa9b74afc4f37d1725024cee59dc1fc"
PCM = "0b1234cb5e3c1ec2573d379660d7078ee7bffb1db029b87ffa58605836e2dde0"
CORE = "4024b6238ef874f7f6e44a3355f384b4d2bdc881ada425d544107a5f7516023e"
HISTORICAL_RUNTIME = "cda8d97ed71069866768ef0de37296f5bdc3492f49e605f1c62aca3a211d4ea6"
GAPS = ["scipy_fft_backend", "jit_runtime", "system_math", "rustfft_dispatch"]
LIMIT = 64 * 1024**2


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def bound(path: Path, expected: str) -> bytes:
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError("Bound artifact differs: " + str(path))
    return raw


class Writer:
    """One diagnostic's immutable blobs; identical observations share storage."""

    def __init__(self, root: Path) -> None:
        if root.is_symlink():
            raise ValueError("Trace output symlink")
        self.root = root.resolve()
        (self.root / "objects").mkdir(parents=True, exist_ok=True)
        if any(p.is_symlink() for p in self.root.rglob("*")):
            raise ValueError("Trace output symlink")
        self.size = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def array(self, values: NDArray[Any], dtype: DTypeLike) -> dict[str, Any]:
        array = np.asarray(values, dtype=dtype, order="C")
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite observation")
        raw = array.tobytes(order="C")
        hash_ = digest(raw)
        relative = "objects/" + hash_
        path = self.root / relative
        if path.exists():
            bound(path, hash_)
        else:
            # Other owned files (including a future task-local JIT cache) count too.
            self.size = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
            if self.size + len(raw) > LIMIT - 262144:
                raise ValueError("64MiB shared trace-output cap")
            with path.open("xb") as stream:
                stream.write(raw)
            self.size += len(raw)
        return dict(
            path=relative,
            sha256=hash_,
            dtype=array.dtype.str,
            shape=list(array.shape),
            bytes=len(raw),
        )


class NumpyProxy:
    """Module-local hooks return original ufunc results without changing layout."""

    def __init__(self, writer: Writer, records: dict[str, Any], normalization: bool) -> None:
        self.writer, self.records, self.normalization = writer, records, normalization

    def __getattr__(self, name: str) -> object:
        return getattr(np, name)

    def sqrt(self, values: NDArray[Any] | float) -> NDArray[Any] | np.floating[Any]:
        result = np.sqrt(values)
        if self.normalization and isinstance(values, np.ndarray) and values.shape == (208, 1):
            if "terminal.scales" in self.records:
                raise ValueError("Repeated final normalization")
            self.records["terminal.lengths"] = self.writer.array(values[8:200, 0], "<f8")
            self.records["terminal.scales"] = self.writer.array(result[8:200, 0], "<f8")
        return result

    def abs(self, values: NDArray[Any]) -> NDArray[Any]:
        result = np.abs(values)
        if not self.normalization:
            if values.shape != (208, 399):
                raise ValueError("Unexpected final coefficient geometry")
            self.records["terminal.normalized"] = self.writer.array(values[8:200], "<c8")
            self.records["terminal.magnitude"] = self.writer.array(result[8:200], "<f4")
        return result

    def log1p(self, values: NDArray[Any]) -> NDArray[Any]:
        result = np.log1p(values)
        if not self.normalization:
            self.records["terminal.log1p"] = self.writer.array(result[8:200], "<f4")
        return result


def runtime(proof_path: Path, expected: str) -> tuple[dict[str, Any], ModuleType]:
    proof = json.loads(bound(proof_path, expected))
    if proof["format"] != "cqt-boundary-runtime-proof-v1" or proof["historical_gaps"] != GAPS:
        raise ValueError("Explicit historical closure gaps required")
    if proof["trace_source_sha256"] != digest(Path(__file__).read_bytes()):
        raise ValueError("Actual trace source changed")
    for name, version in {"numba": "0.68.1", "llvmlite": "0.50.0"}.items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Current JIT version changed: " + name)
    old = json.loads(bound(Path(proof["historical_runtime"]), HISTORICAL_RUNTIME))
    for name, version in old["versions"].items():
        if name != "onnxruntime" and importlib.metadata.version(name) != version:
            raise ValueError("Package version changed: " + name)
    for name, info in old["sources"].items():
        bound(Path(proof["source_paths"][name]), info["file_sha256"])
    for name, info in old["binary_modules"].items():
        if not name.startswith("onnxruntime"):
            module = importlib.import_module(name)
            if module.__file__ is None:
                raise ValueError("Missing actual module file")
            bound(Path(module.__file__), info["sha256"])
    bound(Path(sys.executable), old["executable_sha256"])
    required = {
        "scipy.fft._basic_backend",
        "scipy.fft._duccfft.basic",
        "scipy.fft._duccfft.pyduccfft",
        "numba",
        "numba._helperlib",
        "llvmlite",
    }
    if required != set(proof["current_modules"]):
        raise ValueError("Missing prospective DUCC/JIT pins")
    threads = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    if any(proof["dispatch_environment"].get(name) != "1" for name in threads):
        raise ValueError("Numeric thread controls must be explicitly one")
    for name, info in proof["current_modules"].items():
        module = importlib.import_module(name)
        if module.__file__ != info["path"]:
            raise ValueError("Module path changed: " + name)
        bound(Path(info["path"]), info["sha256"])
    for info in proof["jit_libraries"]:
        bound(Path(info["path"]), info["sha256"])
    if not proof["jit_libraries"]:
        raise ValueError("Explicit LLVM/JIT native-library pins required")
    if any(os.environ.get(k) != v for k, v in proof["dispatch_environment"].items()):
        raise ValueError("Dispatch environment changed")
    actual_sources = {
        "cqt": "librosa.core.constantq",
        "vqt": "librosa.core.constantq",
        "filter_fft": "librosa.core.constantq",
        "response": "librosa.core.constantq",
        "stft": "librosa.core.spectrum",
        "resample": "librosa.core.audio",
        "soxr_resample": "soxr",
    }
    for name, module_name in actual_sources.items():
        actual = importlib.import_module(module_name).__file__
        if actual is None or Path(actual).resolve() != Path(proof["source_paths"][name]).resolve():
            raise ValueError("Actual source path changed: " + name)
        bound(Path(actual), old["sources"][name]["file_sha256"])
    spec = importlib.util.spec_from_file_location(
        "boundary_frozen_features", proof["source_paths"]["features"]
    )
    if spec is None or spec.loader is None:
        raise ValueError("Feature source cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return proof, module


def execute(manifest: Path, output: Path, proof_path: Path, proof_sha: str) -> None:
    proof, features = runtime(proof_path, proof_sha)
    value = json.loads(bound(manifest, MANIFEST))
    if len(value["cases"]) != 1 or value["cases"][0]["id"] != "fmak:091711":
        raise ValueError("Fixed0308 only")
    case = value["cases"][0]
    pcm_path = (manifest.parent / case["pcm"]["path"]).resolve()
    if not pcm_path.is_relative_to(manifest.parent.resolve()):
        raise ValueError("PCM escapes fixed input root")
    pcm = np.frombuffer(bound(pcm_path, PCM), dtype="<f4")
    if pcm.shape != (1633354,) or not np.isfinite(pcm).all():
        raise ValueError("Fixed PCM geometry/finiteness")
    writer, records = Writer(output), {}
    if "onnxruntime" in sys.modules or "torch" in sys.modules:
        raise ValueError("Frontend-only process required")
    report = output / "python-trace.json"
    if report.exists():
        raise ValueError("Immutable trace report exists")
    # Import only here: tests of storage/proxies never invoke a transform.
    import librosa.core.constantq as cq

    support = importlib.util.find_spec("tools.export_cqt_goldens")
    if support is None or support.origin is None:
        raise ValueError("Missing trace support source")
    if Path(support.origin).resolve() != Path(proof["trace_support"]["path"]).resolve():
        raise ValueError("Trace support path changed")
    bound(Path(support.origin), proof["trace_support"]["sha256"])
    from tools.export_cqt_goldens import Trace

    class CompleteTrace(Trace):
        def artifact(self, name: str, values: NDArray[Any], dtype: DTypeLike) -> dict[str, Any]:
            return writer.array(values, dtype)

        def observed(self, name: str, values: NDArray[Any], dtype: str) -> dict[str, Any]:
            record = self.artifact(name, values, dtype)
            record["probes"] = super().observed(name, values, dtype).get("probes", [])
            return record

    original_stack = getattr(cq, "__trim_stack")

    def stack(responses: list[NDArray[Any]], bins: int, dtype: DTypeLike) -> NDArray[Any]:
        result = original_stack(responses, bins, dtype)
        if result.shape == (208, 399):
            records["terminal.pre_normalization"] = writer.array(result[8:200], "<c8")
        return result

    with contextlib.ExitStack() as cleanup:
        old_np, old_feature_np = cq.np, features.np
        cleanup.callback(setattr, cq, "np", old_np)
        cleanup.callback(setattr, features, "np", old_feature_np)
        cleanup.callback(setattr, cq, "__trim_stack", original_stack)
        setattr(cq, "np", NumpyProxy(writer, records, True))
        setattr(features, "np", NumpyProxy(writer, records, False))
        setattr(cq, "__trim_stack", stack)
        with CompleteTrace(output, "python", True) as trace:
            padded = features.cqt(pcm)
    core = np.ascontiguousarray(padded.astype(np.float16).astype(np.float32)[8:200])
    records["terminal.core"] = writer.array(core, "<f4")
    records["terminal.fp16"] = writer.array(core, "<f2")
    preserved = digest(core.tobytes()) == CORE
    if "onnxruntime" in sys.modules or "torch" in sys.modules:
        raise ValueError("Unexpected inference/training module import")
    if (
        len(trace.stages) != 9
        or len(trace.resamplings) != 7
        or sum(len(stage["response"]["probes"]) for stage in trace.stages) != 27
        or sum(len(stage["output"]["probes"]) for stage in trace.resamplings) != 21
    ):
        raise ValueError("Actual stage coverage differs")
    for actual, captured in zip(trace.stages, case["stages"], strict=True):
        for field in ("basis", "indices", "indptr"):
            if actual[field]["sha256"] != captured[field]["sha256"]:
                raise ValueError("Captured CSR changed: " + field)
        for field in ("n_fft", "hop", "basis_shape"):
            if actual[field] != captured[field]:
                raise ValueError("Stage geometry changed: " + field)
    expected = {
        "terminal.pre_normalization",
        "terminal.lengths",
        "terminal.scales",
        "terminal.normalized",
        "terminal.magnitude",
        "terminal.log1p",
        "terminal.core",
        "terminal.fp16",
    }
    if set(records) != expected:
        raise ValueError("Terminal observations incomplete")
    encoded = (
        json.dumps(
            dict(
                format="cqt-boundary-trace-v1",
                implementation="python",
                case="fmak:091711",
                manifest_sha256=MANIFEST,
                runtime_proof_sha256=proof_sha,
                source_sha256=digest(Path(__file__).read_bytes()),
                records=records,
                stages=trace.stages,
                resamplings=trace.resamplings,
                historical_gaps=proof["historical_gaps"],
                core_sha256=digest(core.tobytes()),
                archived_core_preserved=preserved,
                no_model_or_ort_invoked="onnxruntime" not in sys.modules
                and "torch" not in sys.modules,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    with report.open("x") as stream:
        stream.write(encoded)
    if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > LIMIT:
        raise ValueError("Final trace output cap")
    if not preserved:
        raise ValueError(
            "Instrumentation/build/runtime changed archived Mac core; no causal admission"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output", "runtime-proof"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--runtime-proof-sha256", required=True)
    args = parser.parse_args()
    execute(args.manifest, args.output, args.runtime_proof, args.runtime_proof_sha256)


if __name__ == "__main__":
    main()
