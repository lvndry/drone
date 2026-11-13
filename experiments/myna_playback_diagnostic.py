"""One prospective synthetic qualification of the declared playback-rate intervention."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


def center_ten(signal: Tensor) -> Tensor:
    start = (signal.numel() - 160000) // 2
    if start < 0:
        raise ValueError("Insufficient support for the fixed ten-second analysis")
    return signal[start : start + 160000]


def powers(signal: Tensor, expected: list[float]) -> dict[str, Any]:
    import torch

    center = center_ten(signal)
    spectrum = torch.fft.rfft(center * torch.hann_window(center.numel())).abs().square()
    frequencies = torch.fft.rfftfreq(center.numel(), d=1 / 16000)
    total = spectrum.sum()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("Invalid spectral power")
    masks = [
        (frequencies >= hz * 2 ** (-1 / 24)) & (frequencies <= hz * 2 ** (1 / 24))
        for hz in expected
    ]
    ratios = [float(spectrum[mask].sum() / total) for mask in masks]
    union = float(spectrum[torch.stack(masks).any(0)].sum() / total)
    return {
        "union_fraction": union,
        "component_fractions": ratios,
        "passed": union >= 0.90 and all(r >= 0.20 for r in ratios),
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CUDA must remain hidden")
    spec = importlib.util.spec_from_file_location("bound_myna_util", config["pitch_source"]["path"])
    if spec is None or spec.loader is None:
        raise ValueError("Missing verified utility source")
    util = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(util)
    util.verify_file(config["pitch_source"])
    for item in config["source_files"]:
        util.verify_file(item)
    inventory = json.loads(util.verify_file(config["runtime_inventory"]).read_text())
    for item in inventory["files"]:
        util.verify_file(
            {"path": str(Path(config["environment"]) / item["path"]), "sha256": item["sha256"]}
        )
    spec = importlib.util.spec_from_file_location(
        "bound_playback_recipe", config["augmentation_source"]["path"]
    )
    if spec is None or spec.loader is None:
        raise ValueError("Missing augmentation implementation")
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(13)
    torch.use_deterministic_algorithms(True)
    grid = torch.arange(16000 * 24, dtype=torch.float32) / 16000

    def chord(frequencies: list[float], amplitude: float = 0.08) -> Tensor:
        return torch.stack(
            [amplitude * torch.sin(2 * math.pi * hz * grid) for hz in frequencies]
        ).sum(0)

    fixtures = {
        "41Hz": [41.2034],
        "55Hz": [55.0],
        "110Hz": [110.0],
        "440Hz": [440.0],
        "A_major": [55.0, 69.2957, 82.4069],
        "A_minor": [55.0, 65.4064, 82.4069],
    }
    rows = []
    for name, source_frequencies in fixtures.items():
        original = chord(source_frequencies, 0.2 if len(source_frequencies) == 1 else 0.08)
        original_rms = center_ten(original).square().mean().sqrt()
        for shift in range(-6, 7):
            shifted, metadata = recipe.shift_playback(original, shift)
            expected = [hz * 2 ** (shift / 12) for hz in source_frequencies]
            finite = bool(torch.isfinite(shifted).all())
            exact_length = shifted.numel() == metadata["output_samples"]
            zero_identity = shift != 0 or bool(torch.equal(original, shifted))
            rms = float(center_ten(shifted).square().mean().sqrt() / original_rms)
            result: dict[str, Any] = {
                "fixture": name,
                "semitones": shift,
                "geometry": metadata,
                "finite": finite,
                "exact_length": exact_length,
                "zero_identity": zero_identity,
                "rms_ratio": rms,
            }
            passed = finite and exact_length and zero_identity and 0.5 <= rms <= 1.5
            if len(expected) == 1:
                center = center_ten(shifted)
                magnitude = torch.fft.rfft(center * torch.hann_window(center.numel())).abs()
                peak = int(magnitude.argmax())
                if not 0 < peak < len(magnitude) - 1:
                    raise ValueError("Invalid global tone peak")
                y = magnitude[peak - 1 : peak + 2].clamp_min(1e-20).log()
                offset = float(0.5 * (y[0] - y[2]) / (y[0] - 2 * y[1] + y[2]))
                hz = (peak + offset) / 10
                cents = 1200 * math.log2(hz / expected[0])
                result.update(global_peak_hz=hz, error_cents=cents)
                passed = passed and abs(cents) <= 5
            else:
                measured = powers(shifted, expected)
                correct = powers(chord(expected), expected)
                wrong = powers(chord([hz * 2 ** (1 / 12) for hz in expected]), expected)
                deleted = powers(chord([expected[0], expected[2]]), expected)
                controls = correct["passed"] and not wrong["passed"] and not deleted["passed"]
                result.update(
                    energy=measured,
                    correct_control=correct,
                    wrong_control=wrong,
                    missing_middle_control=deleted,
                    controls_pass=bool(controls),
                )
                passed = passed and bool(measured["passed"]) and bool(controls)
            result["passed"] = bool(passed)
            rows.append(result)
            Path(config["progress_path"]).write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    return {
        "status": "diagnostic_complete",
        "method": "pitch-by-playback-rate-polyphase-v1",
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
        "model_loaded": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "changes_tempo_and_duration": True,
        "paper_replication": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = json.loads(parser.parse_args().config.read_text())
    result = run(config)
    Path(config["result_path"]).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
