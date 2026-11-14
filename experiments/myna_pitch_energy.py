"""Prospective exploratory v2 diagnostic; preserve the failed local-peak v1 result."""

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


def energy(signal: Tensor, frequencies: list[float]) -> dict[str, Any]:
    """Central ten-second Hann spectrum; all ratios use total spectral power."""
    import torch

    center = signal[16000:-16000]
    spectrum = torch.fft.rfft(center * torch.hann_window(center.numel())).abs().square()
    bins = torch.fft.rfftfreq(center.numel(), d=1 / 16000)
    total = spectrum.sum()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("Invalid diagnostic spectral power")
    masks = [
        (bins >= frequency_hz * 2 ** (-1 / 24)) & (bins <= frequency_hz * 2 ** (1 / 24))
        for frequency_hz in frequencies
    ]
    union = torch.stack(masks).any(dim=0)
    band_ratios = [float(spectrum[mask].sum() / total) for mask in masks]
    union_ratio = float(spectrum[union].sum() / total)
    return {
        "union_total_power_fraction": union_ratio,
        "component_total_power_fractions": band_ratios,
        "passes_energy_rule": union_ratio >= 0.90 and all(value >= 0.20 for value in band_ratios),
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Synthetic diagnostic requires CUDA hidden")
    spec = importlib.util.spec_from_file_location("myna_bound_util", config["pitch_source"]["path"])
    if spec is None or spec.loader is None:
        raise ValueError("Missing verified utilities")
    util = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(util)
    util.verify_file(config["pitch_source"])
    for item in config["source_files"] + config["external_files"]:
        util.verify_file(item)
    prior = json.loads(util.verify_file(config["prior_r3_result"]).read_text())
    if (
        prior["qualification_pass"] is not False
        or prior["candidate_method"] != "rubberband_offline_r3"
    ):
        raise ValueError("Prior R3 FAIL-v1 identity differs")
    prior_rows = prior["rows"]
    if len(prior_rows) != 13 or sorted(row["semitones"] for row in prior_rows) != list(
        range(-6, 7)
    ):
        raise ValueError("Prior pure-tone coverage differs")
    for row in prior_rows:
        if (
            not all(abs(item["error_cents"]) <= 5 for item in row["tones"][:4])
            or len(row["tones"][:4]) != 4
            or not all(0.5 <= ratio <= 1.5 for ratio in row["rms_ratio"][:4])
            or len(row["rms_ratio"][:4]) != 4
            or not all(
                row[key] is True for key in ("shape_matches", "finite", "zero_shift_identity")
            )
        ):
            raise ValueError("Prior individual-tone engineering gate failed")
    inventory = json.loads(util.verify_file(config["runtime_inventory"]).read_text())
    for item in inventory["files"]:
        util.verify_file(
            {"path": str(Path(config["environment"]) / item["path"]), "sha256": item["sha256"]}
        )
    spec = importlib.util.spec_from_file_location("myna_bound_r3", config["r3_source"]["path"])
    if spec is None or spec.loader is None:
        raise ValueError("Missing R3 adapter")
    rubberband_r3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rubberband_r3)
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(13)
    torch.use_deterministic_algorithms(True)
    grid = torch.arange(16000 * 12, dtype=torch.float32) / 16000

    def chord(frequencies: list[float]) -> Tensor:
        return torch.stack(
            [0.08 * torch.sin(2 * math.pi * frequency_hz * grid) for frequency_hz in frequencies]
        ).sum(0)

    rows = []
    for mode, source_frequencies in (
        ("A_major", [55.0, 69.2957, 82.4069]),
        ("A_minor", [55.0, 65.4064, 82.4069]),
    ):
        original = chord(source_frequencies)
        original_rms = original[16000:-16000].square().mean().sqrt()
        for shift in range(-6, 7):
            expected = [frequency_hz * 2 ** (shift / 12) for frequency_hz in source_frequencies]
            correct = energy(chord(expected), expected)
            wrong = energy(
                chord([frequency_hz * 2 ** (1 / 12) for frequency_hz in expected]), expected
            )
            deleted = energy(chord([expected[0], expected[2]]), expected)
            controls_pass = (
                correct["passes_energy_rule"]
                and not wrong["passes_energy_rule"]
                and not deleted["passes_energy_rule"]
            )
            shifted, metadata = rubberband_r3.shift_r3(
                original, shift, config["rubberband_library"]
            )
            exact_length = shifted.shape == original.shape
            finite = bool(torch.isfinite(shifted).all())
            zero_identity = shift != 0 or bool(torch.equal(shifted, original))
            if not exact_length or not finite:
                raise ValueError("R3 violated existing length/finiteness contract")
            amplitude_ratio = float(shifted[16000:-16000].square().mean().sqrt() / original_rms)
            measured = energy(shifted, expected)
            rows.append(
                {
                    "mode": mode,
                    "semitones": shift,
                    "expected_hz": expected,
                    "analytic_correct": correct,
                    "wrong_semitone_control": wrong,
                    "missing_middle_control": deleted,
                    "controls_pass": bool(controls_pass),
                    "r3": measured,
                    "rms_ratio": amplitude_ratio,
                    "finite": finite,
                    "exact_length": exact_length,
                    "zero_identity": zero_identity,
                    "adapter": metadata,
                    "passes_v2": bool(
                        controls_pass
                        and measured["passes_energy_rule"]
                        and 0.5 <= amplitude_ratio <= 1.5
                        and zero_identity
                    ),
                }
            )
            Path(config["progress_path"]).write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    return {
        "status": "diagnostic_complete",
        "protocol": "exploratory_pitch_energy_v2",
        "prior_v1_remains_failed": True,
        "prior_result_sha256": config["prior_r3_result"]["sha256"],
        "passes_v2": all(row["passes_v2"] for row in rows),
        "rows": rows,
        "rule": {
            "band_half_width_cents": 50,
            "union_minimum_total_power": 0.90,
            "component_minimum_total_power": 0.20,
        },
        "engineering_tolerance_not_perceptual_standard": True,
        "model_loaded": False,
        "cuda_initialized": torch.cuda.is_initialized(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    config = json.loads(parser.parse_args().config.read_text())
    result = run(config)
    Path(config["result_path"]).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
