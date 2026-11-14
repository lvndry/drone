"""Freeze actual librosa CQT calls and identical-PCM feasibility goldens, label-blind."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import signal
import subprocess
import sys
import time
import wave
from collections.abc import Callable
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any, Literal

import librosa.core.constantq as constantq
import numpy as np
import soxr
from numpy.typing import DTypeLike
from scipy.sparse import csr_matrix

from experiments.chunked import FEATURE_SPEC, LABEL_SCHEMA, cpu_session, sha256
from tools.research.contracts import load_validation_tracks, select_performance_tracks
from tools.research.measurements import runtime_environment

SAMPLE_RATE = 22050
HOP = 4096
PUBLIC = (
    ("fmak:110544", "77bea21b8fc4e58323d5d48953b9ce4e6b2ec0c3452663fe0f2510a9ef2f18c9"),
    ("fmak:107354", "22e809a87ebb977793214280cfa0953639abe4377800581d9ab24d89013faf16"),
    ("fmak:087373", "25d906f7a7f0b19f340972fa8b1e882c13ceb89ed4cd83487c5c3bf74255d755"),
)


def write_json(path: Path, value: object) -> None:
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def load_features_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("frozen_cqt_features", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load original feature source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.FEATURE_SPEC != FEATURE_SPEC:
        raise ValueError("Unexpected feature contract")
    return module


def synthetic_cases() -> list[tuple[str, np.ndarray]]:
    lengths = [7 * HOP, 8 * HOP + 1, 15 * HOP - 1, 16 * HOP, 17 * HOP + 17, 31 * HOP + 127]
    silence = np.zeros(lengths[0], dtype=np.float64)
    impulse = np.zeros(lengths[1], dtype=np.float64)
    impulse[[0, lengths[1] // 2, lengths[1] - 1]] = [0.8, -0.4, 0.6]

    def tones(length: int, frequencies: list[float]) -> np.ndarray:
        seconds = np.arange(length, dtype=np.float64) / SAMPLE_RATE
        return sum(0.2 * np.sin(2 * np.pi * frequency * seconds) for frequency in frequencies)

    major = [130.8127826502993, 164.81377845643496, 195.99771799087463]
    minor = [110.0, 130.8127826502993, 164.81377845643496]
    noise = np.random.Generator(np.random.PCG64(20261111)).uniform(-0.2, 0.2, lengths[5])
    return list(
        zip(
            ("silence", "impulses", "tone", "major", "minor", "noise"),
            (
                silence,
                impulse,
                tones(lengths[2], [440.0]),
                tones(lengths[3], major),
                tones(lengths[4], minor),
                noise,
            ),
            strict=True,
        )
    )


def pcm16_wav(path: Path, samples: np.ndarray) -> None:
    quantized = np.clip(np.rint(samples * 32768), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(quantized.tobytes())


class Trace:
    """Observe actual private calls; never regenerate their numerical constants."""

    def __init__(self, root: Path, case: str, public: bool) -> None:
        self.root = root
        self.case = case
        self.public = public
        self.directory = root / "cases" / case
        self.directory.mkdir(parents=True)
        self.stages: list[dict[str, Any]] = []
        self.resamplings: list[dict[str, Any]] = []
        self.filters: list[dict[str, Any]] = []
        self.length_calls: list[dict[str, Any]] = []
        self.originals: list[tuple[object, str, Callable[..., Any]]] = []

    def artifact(self, name: str, values: np.ndarray, dtype: str) -> dict[str, Any]:
        path = self.directory / name
        array = np.asarray(values, dtype=np.dtype(dtype), order="C")
        raw = array.tobytes(order="C")
        path.write_bytes(raw)
        return dict(
            path=str(path.relative_to(self.root)),
            dtype=dtype,
            shape=list(array.shape),
            sha256=hashlib.sha256(raw).hexdigest(),
            bytes=len(raw),
        )

    def observed(self, name: str, values: np.ndarray, dtype: str) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.dtype(dtype), order="C")
        if not self.public:
            return self.artifact(name, array, dtype)
        # Full public intermediate arrays can be large. Preserve their digest
        # and exact first/middle/final probes; final PCM/features remain complete.
        count = array.shape[-1]
        width = min(16 if np.iscomplexobj(array) else 256, count)
        starts = sorted({0, max(0, (count - width) // 2), max(0, count - width)})
        return dict(
            dtype=dtype,
            shape=list(array.shape),
            sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            probes=[
                dict(
                    start=start,
                    artifact=self.artifact(
                        f"{name}.probe{index}", array[..., start : start + width], dtype
                    ),
                )
                for index, start in enumerate(starts)
            ],
        )

    def patch(self, owner: object, name: str, replacement: Callable[..., Any]) -> None:
        self.originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def __enter__(self) -> Trace:
        original_filter = getattr(constantq, "__vqt_filter_fft")
        original_response = getattr(constantq, "__cqt_response")
        original_stft = constantq.stft
        original_resample = constantq.audio.resample
        original_lengths = constantq.filters.wavelet_lengths

        def filter_call(
            sr: float,
            freqs: np.ndarray,
            filter_scale: float,
            norm: float,
            sparsity: float,
            hop_length: int | None = None,
            window: str = "hann",
            gamma: float = 0.0,
            dtype: object = np.complex64,
            alpha: np.ndarray | None = None,
        ) -> tuple[csr_matrix, int, np.ndarray]:
            output = original_filter(
                sr,
                freqs,
                filter_scale,
                norm,
                sparsity,
                hop_length=hop_length,
                window=window,
                gamma=gamma,
                dtype=dtype,
                alpha=alpha,
            )
            self.filters.append(
                dict(
                    sr=float(sr),
                    frequencies=freqs.tolist(),
                    n_fft=int(output[1]),
                )
            )
            return output

        def response_call(
            y: np.ndarray,
            n_fft: int,
            hop: int,
            basis: csr_matrix,
            mode: str,
            window: str = "ones",
            phase: bool = True,
            dtype: object = None,
        ) -> np.ndarray:
            index = len(self.stages)
            stage = dict(
                index=index,
                sr=self.filters[-1]["sr"],
                n_fft=int(n_fft),
                hop=int(hop),
                frequencies=self.filters[-1]["frequencies"],
                pad_mode=mode,
                input=self.observed(f"stage{index}.pcm", y, "<f4"),
                basis_shape=list(basis.shape),
                indptr=self.artifact(f"stage{index}.indptr", basis.indptr, "<u4"),
                indices=self.artifact(f"stage{index}.indices", basis.indices, "<u4"),
                basis=self.artifact(f"stage{index}.basis", basis.data, "<c8"),
            )
            self.stages.append(stage)
            output = original_response(
                y, n_fft, hop, basis, mode, window=window, phase=phase, dtype=dtype
            )
            stage["response"] = self.observed(f"stage{index}.response", output, "<c8")
            return output

        def stft_call(
            y: np.ndarray,
            *,
            n_fft: int,
            hop_length: int,
            window: str,
            pad_mode: Literal["constant"],
            dtype: DTypeLike | None = None,
        ) -> np.ndarray:
            output = original_stft(
                y, n_fft=n_fft, hop_length=hop_length, window=window, pad_mode=pad_mode, dtype=dtype
            )
            self.stages[-1]["stft"] = self.observed(
                f"stage{len(self.stages) - 1}.stft", output, "<c8"
            )
            return output

        def resample_call(
            y: np.ndarray,
            *,
            orig_sr: float,
            target_sr: float,
            res_type: str = "soxr_hq",
            scale: bool = False,
            fix: bool = True,
        ) -> np.ndarray:
            output = original_resample(
                y, orig_sr=orig_sr, target_sr=target_sr, res_type=res_type, scale=scale, fix=fix
            )
            index = len(self.resamplings)
            self.resamplings.append(
                dict(
                    index=index,
                    after_stage=len(self.stages) - 1,
                    orig_sr=float(orig_sr),
                    target_sr=float(target_sr),
                    scale=scale,
                    fix=fix,
                    res_type=res_type,
                    input_samples=int(y.shape[-1]),
                    output=self.observed(f"resample{index}.pcm", output, "<f4"),
                )
            )
            return output

        def length_call(
            *,
            freqs: np.ndarray,
            sr: float,
            window: str = "hann",
            filter_scale: float = 1.0,
            gamma: float | None = None,
            alpha: np.ndarray | None = None,
        ) -> tuple[np.ndarray, float]:
            output = original_lengths(
                freqs=freqs,
                sr=sr,
                window=window,
                filter_scale=filter_scale,
                gamma=gamma,
                alpha=alpha,
            )
            if len(freqs) == 208:
                self.length_calls.append(
                    dict(sr=float(sr), lengths=output[0].tolist(), cutoff=float(output[1]))
                )
            return output

        self.patch(constantq, "__vqt_filter_fft", filter_call)
        self.patch(constantq, "__cqt_response", response_call)
        self.patch(constantq, "stft", stft_call)
        self.patch(constantq.audio, "resample", resample_call)
        self.patch(constantq.filters, "wavelet_lengths", length_call)
        return self

    def __exit__(self, *exception: object) -> None:
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)


def source_identity(features: ModuleType) -> dict[str, Any]:
    functions = {
        "features": features.cqt,
        "decode": features.decode_mp3,
        "read_wav": features.read_wav_mono,
        "cqt": constantq.cqt,
        "vqt": constantq.vqt,
        "filter_fft": getattr(constantq, "__vqt_filter_fft"),
        "response": getattr(constantq, "__cqt_response"),
        "stft": constantq.stft,
        "resample": constantq.audio.resample,
        "soxr_resample": soxr.resample,
    }
    return {
        name: dict(
            path=inspect.getsourcefile(function),
            function_sha256=hashlib.sha256(inspect.getsource(function).encode()).hexdigest(),
            file_sha256=sha256(Path(inspect.getsourcefile(function) or "")),
        )
        for name, function in functions.items()
    }


def export(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("Choose a new immutable golden directory")
    if importlib.metadata.version("librosa") != "1.0.0" or soxr.__version__ != "1.1.0":
        raise ValueError("Require actual installed librosa1.0.0/soxr1.1.0")
    if sha256(args.full_model) != args.full_model_sha256:
        raise ValueError("Changed authoritative full ONNX")
    feature_source = load_features_module(args.feature_source)
    sources = source_identity(feature_source)
    tracks = load_validation_tracks(args.data_manifest)
    selection = sorted(
        select_performance_tracks(tracks, args.selection),
        key=lambda track: (track.frames, track.id),
    )
    chosen = [selection[index] for index in (0, 10, 19)]
    if tuple((track.id, track.sha256) for track in chosen) != PUBLIC:
        raise ValueError("Fixed public membership changed")
    rows = json.loads(args.data_manifest.read_text())["rows"]
    public_paths = {row["id"]: Path(row["path"]) for row in rows if row["role"] == "validation"}
    args.output.mkdir(parents=True)
    sources_dir = args.output / "source"
    sources_dir.mkdir()
    session = cpu_session(args.full_model)
    manifest: dict[str, Any] = dict(
        format="k10-cqt-goldens-v1",
        complete=False,
        feature_spec=FEATURE_SPEC,
        label_schema=LABEL_SCHEMA,
        sample_rate=SAMPLE_RATE,
        hop=HOP,
        bins=208,
        core_slice=[8, 200],
        final_logit_atol=1e-5,
        final_logit_rtol=1e-5,
        full_model_sha256=args.full_model_sha256,
        protocol_sha256=sha256(args.protocol),
        export_tool_sha256=sha256(Path(__file__)),
        source_identity=sources,
        versions={
            name: importlib.metadata.version(name)
            for name in ("librosa", "numpy", "scipy", "soxr", "onnxruntime")
        },
        soxr_embedded_version=soxr.__libsoxr_version__,
        ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
        environment=runtime_environment(),
        accuracy_scored=False,
        public_files=3,
        intermediate_policy="complete synthetic arrays; public hashes plus first/middle/end probes",
        cases=[],
    )
    inputs: list[tuple[str, Path, bool, str]] = []
    for name, samples in synthetic_cases():
        path = sources_dir / f"{name}.wav"
        pcm16_wav(path, samples)
        inputs.append((name, path, False, sha256(path)))
    for track in chosen:
        path = public_paths[track.id]
        if sha256(path) != track.sha256:
            raise ValueError("Changed public source audio")
        inputs.append((track.id.replace(":", "-"), path, True, track.sha256))
    started = time.monotonic()
    for name, path, public, audio_hash in inputs:
        decoded = args.output / f"{name}.decoded.wav"
        feature_source.decode_mp3(path, decoded)
        pcm, rate = feature_source.read_wav_mono(decoded)
        pcm = feature_source.resample_linear(pcm, rate, SAMPLE_RATE)
        decoded.unlink()
        with Trace(args.output, name, public) as trace:
            padded_float32 = feature_source.cqt(pcm)
        if padded_float32.shape[0] != 208 or padded_float32.shape[1] < 8:
            raise ValueError("Unexpected feature shape")
        core = np.ascontiguousarray(padded_float32.astype(np.float16).astype(np.float32)[8:200])
        logits = np.asarray(session.run(["logits"], {"features": core[None]})[0])
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite reference logits")
        case = dict(
            id=name,
            public=public,
            audio_sha256=audio_hash,
            pcm=trace.artifact("reference.pcm", pcm, "<f4"),
            padded_float32=trace.artifact("reference.padded", padded_float32, "<f4"),
            core=trace.artifact("reference.core", core, "<f4"),
            frames=core.shape[-1],
            reference_logits=logits[0].tolist(),
            stages=trace.stages,
            resamplings=trace.resamplings,
            normalization=trace.length_calls[-1],
        )
        manifest["cases"].append(case)
        manifest["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "manifest.json", manifest)
        size = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
        if size > 2 * 1024**3:
            raise RuntimeError("Golden storage budget exceeded")
        print(f"Exported {name}: {core.shape[-1]}frames, {size}bytes", flush=True)
    if "torch" in sys.modules:
        raise RuntimeError("Torch entered reference exporter")
    manifest.update(
        complete=True,
        tracks=len(inputs),
        torch_imported=False,
        elapsed_seconds=time.monotonic() - started,
    )
    write_json(args.output / "manifest.json", manifest)
    return manifest


def budget_exceeded(signum: int, frame: FrameType | None) -> None:
    raise TimeoutError("CQT golden export exceeded 1200 CPU/wall seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data-manifest",
        "selection",
        "feature-source",
        "full-model",
        "protocol",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--full-model-sha256", required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, budget_exceeded)
    signal.signal(signal.SIGPROF, budget_exceeded)
    signal.setitimer(signal.ITIMER_REAL, 1200)
    signal.setitimer(signal.ITIMER_PROF, 1200)
    try:
        result = export(args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_PROF, 0)
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
