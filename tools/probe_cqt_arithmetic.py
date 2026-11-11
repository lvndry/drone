"""First-failure-only arithmetic observation; never a new parity candidate."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import importlib
import io
import json
from pathlib import Path
from typing import Any

import librosa
import librosa.core.constantq as cq
import numpy as np
from scipy.sparse import csr_matrix

from experiments.chunked import sha256
from tools.export_cqt_goldens import Trace, load_features_module, source_identity, write_json


def checked_array(root: Path, artifact: dict[str, Any], dtype: str) -> np.ndarray:
    path = (root / artifact["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or sha256(path) != artifact["sha256"]:
        raise ValueError("Changed/escaped frozen diagnostic input")
    return np.fromfile(path, dtype=dtype).reshape(artifact["shape"])


def runtime_context() -> dict[str, Any]:
    core = importlib.import_module("numpy._core._multiarray_umath")
    binary = Path(core.__file__ or "")
    runtime = io.StringIO()
    with contextlib.redirect_stdout(runtime):
        np.show_runtime()
    libc = ctypes.CDLL(None)
    libc.gnu_get_libc_version.restype = ctypes.c_char_p
    glibc: bytes = libc.gnu_get_libc_version()
    dependencies = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = Path(line.split()[-1])
        if path.name.startswith(("libm.so", "libc.so")) and path.is_file():
            dependencies[str(path)] = sha256(path)
    return dict(
        numpy_version=np.__version__,
        numpy_git_revision=np.version.git_revision,
        numpy_binary=dict(path=str(binary), sha256=sha256(binary)),
        cpu_features=core.__cpu_features__,
        cpu_baseline=core.__cpu_baseline__,
        cpu_dispatch=core.__cpu_dispatch__,
        ufunc_dispatch=np.lib.introspect.opt_func_info(func_name="log1p|absolute"),
        numpy_build=np.__config__.show(mode="dicts"),
        numpy_runtime=runtime.getvalue(),
        glibc_version=glibc.decode(),
        math_libraries=dependencies,
        source_urls=[
            "https://raw.githubusercontent.com/numpy/numpy/v2.5.4/numpy/_core/src/umath/loops_unary_complex.dispatch.c.src",
            "https://raw.githubusercontent.com/numpy/numpy/v2.5.4/numpy/_core/src/umath/loops_umath_fp.dispatch.c.src",
        ],
    )


def prepare(args: argparse.Namespace) -> None:
    original = json.loads((args.failure_spool / "manifest.json").read_text())
    case = original["cases"][0]
    native = json.loads(args.native_report.read_text())
    if (
        case["id"] != "beatport:3662506"
        or original["row_index"] != 72
        or native["passed"] is not False
        or native["cases"][0]["features"]["fp16_mismatches"] != 6
    ):
        raise ValueError("Only the frozen first failure and its six mismatches are authorized")
    if any(
        not entry.get("response", entry.get("resample"))["full_hash_equal"]
        for entry in native["cases"][0]["stage_diagnostics"]["resamplings"]
    ):
        raise ValueError("First failure did not establish exact projection/resampling")
    if args.output.exists():
        raise FileExistsError("New immutable arithmetic directory required")
    args.output.mkdir()
    pcm = checked_array(args.failure_spool, case["pcm"], "<f4")
    reference = checked_array(args.failure_spool, case["core"], "<f4")
    features = load_features_module(args.feature_source)
    responses: list[np.ndarray] = []
    normalized: list[np.ndarray] = []
    with Trace(args.output, "first-failure", True) as trace:
        original_response = getattr(cq, "__cqt_response")
        original_cqt = librosa.cqt

        def response_capture(
            y: np.ndarray,
            n_fft: int,
            hop: int,
            basis: csr_matrix,
            mode: str,
            window: str = "ones",
            phase: bool = True,
            dtype: object = None,
        ) -> np.ndarray:
            value = original_response(
                y, n_fft, hop, basis, mode, window=window, phase=phase, dtype=dtype
            )
            responses.append(value.copy(order="K"))
            return value

        def normalized_capture(
            y: np.ndarray,
            *,
            sr: float,
            hop_length: int,
            fmin: float,
            n_bins: int,
            bins_per_octave: int,
        ) -> np.ndarray:
            value = original_cqt(
                y,
                sr=sr,
                hop_length=hop_length,
                fmin=fmin,
                n_bins=n_bins,
                bins_per_octave=bins_per_octave,
            )
            normalized.append(value.copy(order="K"))
            return value

        trace.patch(cq, "__cqt_response", response_capture)
        trace.patch(librosa, "cqt", normalized_capture)
        padded = features.cqt(pcm)
    if len(normalized) != 1 or len(responses) != 9:
        raise ValueError("Unexpected actual private-call layout")
    if [stage["response"]["sha256"] for stage in trace.stages] != [
        stage["response"]["sha256"] for stage in case["stages"]
    ] or [entry["output"]["sha256"] for entry in trace.resamplings] != [
        entry["output"]["sha256"] for entry in case["resamplings"]
    ]:
        raise ValueError("Reobserved responses/resampling changed")
    coeffs = normalized[0]
    magnitude = np.abs(coeffs)
    logged = np.log1p(magnitude)
    if not np.array_equal(logged.view(np.uint32), padded.view(np.uint32)):
        raise ValueError("Vectorized arithmetic observation differs from original feature function")
    current_core = np.ascontiguousarray(padded.astype(np.float16).astype(np.float32)[8:200])
    if current_core.tobytes() != reference.tobytes():
        raise ValueError("Reobserved core differs from frozen reference bytes")
    pre = getattr(cq, "__trim_stack")(responses, 208, np.complex64)
    result = dict(
        format="k10-cqt-first-failure-arithmetic-input-v1",
        id=case["id"],
        frames=case["frames"],
        expected_features=native["cases"][0]["features"],
        original_reference_sha256=sha256(args.failure_spool / "manifest.json"),
        original_native_report_sha256=sha256(args.native_report),
        pcm_sha256=case["pcm"]["sha256"],
        producer_source_sha256=sha256(Path(__file__)),
        source_identity=source_identity(features),
        runtime=runtime_context(),
        array_layout=dict(
            coefficients_strides=list(coeffs.strides),
            coefficients_c_contiguous=bool(coeffs.flags.c_contiguous),
            coefficients_f_contiguous=bool(coeffs.flags.f_contiguous),
            magnitude_strides=list(magnitude.strides),
            logged_strides=list(logged.strides),
        ),
        pre=trace.artifact("arithmetic.pre", pre[8:200], "<c8"),
        normalized=trace.artifact("arithmetic.normalized", coeffs[8:200], "<c8"),
        magnitude=trace.artifact("arithmetic.magnitude", magnitude[8:200], "<f4"),
        logged=trace.artifact("arithmetic.logged", logged[8:200], "<f4"),
        reference=trace.artifact("arithmetic.reference", reference, "<f4"),
        lengths=trace.artifact(
            "arithmetic.lengths", np.asarray(trace.length_calls[-1]["lengths"])[8:200], "<f8"
        ),
        reobserved_all_responses_exact=True,
        reobserved_all_resamplers_exact=True,
    )
    write_json(args.output / "input.json", result)


def scalar_bits(value: np.float32) -> dict[str, Any]:
    converted = np.asarray(value, dtype=np.float32)
    return dict(value=float(converted), bits=f"{int(converted.view(np.uint32)):08x}")


def finish(args: argparse.Namespace) -> None:
    source = json.loads(args.input.read_text())
    result = json.loads(args.result.read_text())
    if result["input_sha256"] != sha256(args.input):
        raise ValueError("Arithmetic result input changed")
    root = args.input.parent
    normalized = checked_array(root, source["normalized"], "<c8")
    magnitude = checked_array(root, source["magnitude"], "<f4")
    for row in result["positions"]:
        index = (row["core_bin"], row["frame"])
        value = normalized[index]
        mag = magnitude[index]
        row["numpy_scalar_builtin_abs"] = scalar_bits(abs(value))
        row["numpy_scalar_ufunc_abs"] = scalar_bits(np.abs(value))
        row["numpy_length_one_array_abs"] = scalar_bits(
            np.abs(np.array([value], dtype=np.complex64))[0]
        )
        row["numpy_scalar_ufunc_log1p_on_reference_magnitude"] = scalar_bits(np.log1p(mag))
        row["numpy_length_one_array_log1p_on_reference_magnitude"] = scalar_bits(
            np.log1p(np.array([mag], dtype=np.float32))[0]
        )
    result.update(
        reference_runtime=source["runtime"],
        array_layout=source["array_layout"],
        original_reference_sha256=source["original_reference_sha256"],
        original_native_report_sha256=source["original_native_report_sha256"],
        pcm_sha256=source["pcm_sha256"],
        producer_source_sha256=source["producer_source_sha256"],
        finisher_source_sha256=sha256(Path(__file__)),
    )
    if args.output.exists():
        raise FileExistsError("Immutable finished diagnostic already exists")
    write_json(args.output, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("prepare")
    for name in ("failure-spool", "native-report", "feature-source", "output"):
        first.add_argument(f"--{name}", type=Path, required=True)
    second = commands.add_parser("finish")
    for name in ("input", "result", "output"):
        second.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        finish(args)


if __name__ == "__main__":
    main()
