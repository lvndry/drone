"""CPU-only observation of the preserved pitch recipe; no gate changes or model load."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


def rubberband(
    signals: Tensor, shift: int, config: dict[str, Any]
) -> tuple[Tensor, list[dict[str, Any]]]:
    """Apply frozen external RubberBand settings independently to each mono signal."""
    import numpy as np
    import torch

    if shift == 0:
        return signals.clone(), [{"identity_bypass": True, "samples": signals.shape[-1]}]
    outputs = []
    records = []
    r3 = None
    if config.get("r3_source"):
        spec = importlib.util.spec_from_file_location(
            "qualified_r3_adapter", config["r3_source"]["path"]
        )
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load R3 adapter")
        r3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(r3)
    for signal in signals:
        command = [
            config["ffmpeg"]["path"],
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-filter_complex_threads",
            "1",
            "-f",
            "f32le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-af",
            "rubberband=tempo=1:pitch="
            + format(2 ** (shift / 12), ".17g")
            + ":transients=crisp:detector=compound:phase=laminar:window=standard"
            + ":smoothing=off:formant=shifted:pitchq=quality:channels=apart",
            "-f",
            "f32le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "pipe:1",
        ]
        engine_details: dict[str, Any] = {}
        if r3 is not None:
            raw, engine_details = r3.shift_r3(signal, shift, config["rubberband_library"])
            if raw.numel() != signal.numel():
                raise ValueError(f"Offline R3 changed length: {raw.numel()} != {signal.numel()}")
        else:
            completed = subprocess.run(
                command, input=signal.numpy().tobytes(), capture_output=True, check=True, timeout=20
            )
            raw = torch.from_numpy(np.frombuffer(completed.stdout, dtype="<f4").copy())
        if raw.numel() < 32000:
            raise ValueError("Unexpectedly short RubberBand output")
        active = torch.nonzero(raw.abs() > 1e-5).flatten()
        records.append(
            {
                "samples_before_length_fix": raw.numel(),
                "engine_details": engine_details,
                "input_samples": signal.numel(),
                "tail_trimmed_samples": max(0, raw.numel() - signal.numel()),
                "tail_zero_padded_samples": max(0, signal.numel() - raw.numel()),
                "active_support_threshold": 1e-5,
                "first_active_sample": int(active[0]) if active.numel() else None,
                "last_active_sample": int(active[-1]) if active.numel() else None,
                "first_second_rms": float(raw[:16000].square().mean().sqrt()),
                "last_second_rms": float(raw[-16000:].square().mean().sqrt()),
            }
        )
        if raw.numel() < signal.numel():
            raw = torch.nn.functional.pad(raw, (0, signal.numel() - raw.numel()))
        outputs.append(raw[: signal.numel()])
    return torch.stack(outputs), records


def stages(signals: Tensor, pitch_function: Callable[[Tensor, int], Tensor]) -> dict[str, Any]:
    """Locate attenuation at fixed +/-3 shifts; preserve the original algorithm."""
    import numpy as np
    import torch

    phase_vocoder = importlib.import_module("torchaudio.functional").phase_vocoder
    resample_poly = importlib.import_module("scipy.signal").resample_poly
    original_rms = signals[:, 16000:-16000].square().mean(-1).sqrt()

    def observe(signal: torch.Tensor) -> dict[str, Any]:
        center = signal[:, 16000:-16000]
        magnitude = torch.fft.rfft(center * torch.hann_window(center.shape[-1])).abs()
        return {
            "samples": signal.shape[-1],
            "finite": bool(torch.isfinite(signal).all()),
            "rms_ratio": (center.square().mean(-1).sqrt() / original_rms).tolist(),
            "pure_tone_global_peak_hz": [
                int(magnitude[row].argmax()) * 16000 / center.shape[-1] for row in range(4)
            ],
        }

    window = torch.hann_window(2048)
    spectrum = torch.stft(signals, 2048, 512, 2048, window, return_complex=True)
    identity = torch.istft(spectrum, 2048, 512, 2048, window, length=signals.shape[-1])
    identity_error = float((identity - signals).abs().max())
    results = [
        {"stage": "stft_istft_identity", **observe(identity), "max_abs_error": identity_error}
    ]
    for shift in (-3, 3):
        rate = 2 ** (-shift / 12)
        stretched_spec = phase_vocoder(
            spectrum, rate, torch.linspace(0, math.pi * 512, 1025)[..., None]
        )
        stretched = torch.istft(
            stretched_spec,
            2048,
            512,
            2048,
            window,
            length=round(signals.shape[-1] / rate),
        )
        resampled = torch.from_numpy(
            np.ascontiguousarray(
                resample_poly(
                    stretched.numpy(),
                    16000,
                    int(16000 / rate),
                    axis=-1,
                    window=("kaiser", 5.0),
                    padtype="constant",
                )
            )
        )
        results.append({"stage": "after_phase_vocoder", "semitones": shift, **observe(stretched)})
        results.append(
            {"stage": "after_polyphase_resample", "semitones": shift, **observe(resampled)}
        )
        final = resampled[..., : signals.shape[-1]]
        if final.shape[-1] < signals.shape[-1]:
            final = torch.nn.functional.pad(final, (0, signals.shape[-1] - final.shape[-1]))
        existing = pitch_function(signals, shift)
        if not torch.equal(existing, final):
            raise ValueError("Stage instrumentation changed the preserved method")
    return {
        "status": "diagnostic_complete",
        "stages": results,
        "model_loaded": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "gate_changed": False,
        "instrumentation_matches_preserved_method": True,
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Diagnostic requires CUDA hidden")
    source = Path(config["pitch_source"]["path"])
    spec = importlib.util.spec_from_file_location("preserved_pitch_recipe", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot inspect preserved recipe")
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    recipe.verify_file(config["pitch_source"])
    inventory = json.loads(recipe.verify_file(config["runtime_inventory"]).read_text())
    for item in inventory["files"]:
        recipe.verify_file(
            {
                "path": str(Path(config["environment"]) / item["path"]),
                "sha256": item["sha256"],
            }
        )
    for item in config.get("external_files", []):
        recipe.verify_file(item)
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(13)
    torch.use_deterministic_algorithms(True)
    grid = torch.arange(16000 * 12, dtype=torch.float32) / 16000
    frequencies = [41.2034, 55.0, 110.0, 440.0]
    tones = [0.2 * torch.sin(2 * math.pi * hz * grid) for hz in frequencies]
    chord_hz = [55.0, 69.2957, 82.4069]
    chord = torch.stack([0.08 * torch.sin(2 * math.pi * hz * grid) for hz in chord_hz]).sum(0)
    signals = torch.stack([*tones, chord])
    stage_result = (
        stages(signals, recipe.pitch_shift_cpu) if config.get("stage_diagnostic") else None
    )
    if stage_result is not None and not config.get("rubberband"):
        return stage_result
    rows = []
    for steps in range(-6, 7):
        boundary_records: list[dict[str, Any]] = []
        if config.get("rubberband"):
            shifted, boundary_records = rubberband(signals, steps, config)
        else:
            shifted = recipe.pitch_shift_cpu(signals, steps)
        center = shifted[:, 16000:-16000]
        spectra = torch.fft.rfft(center * torch.hann_window(center.shape[-1])).abs()
        errors = []
        for row, hz in [*enumerate(frequencies), *((4, hz) for hz in chord_hz)]:
            target = hz * 2 ** (steps / 12)
            nominal = target * center.shape[-1] / 16000
            left, right = int(nominal) - 3, int(nominal) + 4
            peak = left + int(spectra[row, left:right].argmax())
            y = spectra[row, peak - 1 : peak + 2].clamp_min(1e-20).log()
            offset = float(0.5 * (y[0] - y[2]) / (y[0] - 2 * y[1] + y[2]))
            frequency = (peak + offset) * 16000 / center.shape[-1]
            errors.append({"source_hz": hz, "error_cents": 1200 * math.log2(frequency / target)})
        rms_ratio = (
            center.square().mean(-1).sqrt() / signals[:, 16000:-16000].square().mean(-1).sqrt()
        )
        rows.append(
            {
                "semitones": steps,
                "boundary_records": boundary_records,
                "rms_ratio": rms_ratio.tolist(),
                "tones": errors,
                "shape_matches": shifted.shape == signals.shape,
                "finite": bool(torch.isfinite(shifted).all()),
                "zero_shift_identity": steps != 0 or bool(torch.equal(shifted, signals)),
                "pure_tone_global_peak_hz": [
                    int(spectra[row].argmax()) * 16000 / center.shape[-1]
                    for row in range(len(frequencies))
                ],
                "passes_original_amplitude_gate": bool(
                    ((rms_ratio >= 0.5) & (rms_ratio <= 1.5)).all()
                ),
                "passes_original_pitch_gate": all(abs(e["error_cents"]) <= 5 for e in errors),
            }
        )
        Path(config["progress_path"]).write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    return {
        "status": "diagnostic_complete",
        "existing_method_stage_diagnostic": stage_result,
        "candidate_method": (
            "rubberband_offline_r3"
            if config.get("r3_source")
            else "ffmpeg_rubberband_quality"
            if config.get("rubberband")
            else "preserved_phase_vocoder_polyphase"
        ),
        "rows": rows,
        "qualification_pass": all(
            r["passes_original_amplitude_gate"]
            and r["passes_original_pitch_gate"]
            and r["finite"]
            and r["shape_matches"]
            and r["zero_shift_identity"]
            for r in rows
        ),
        "model_loaded": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "gate_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = json.loads(parser.parse_args().config.read_text())
    result = run(config)
    Path(config["result_path"]).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
