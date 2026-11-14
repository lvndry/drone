"""Myna embeddings for audio files at several playback-rate pitch shifts.

Reuses the qualified Myna encoder, mel frontend and playback-rate shift from the
earlier extraction, without its supervision machinery: decode to 24 kHz mono,
resample to 16 kHz, shift, cut 196-frame mel windows, encode each window.
Writes <name>-shift<+s>.npy holding (windows, 384) float32.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from environment import required_path

MYNA_RELEASE = required_path("DRONE_MYNA")
MYNA_SOURCE = MYNA_RELEASE / "myna.py"
MYNA_WEIGHTS = MYNA_RELEASE / "model.safetensors"
MYNA_CONFIG = MYNA_RELEASE / "config.json"
QUALIFIED = required_path("DRONE_MYNA_FRONTEND")
WINDOW_FRAMES = 196
ENCODER_BATCH = 16


def load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def decode_24k(audio_path: str) -> np.ndarray:
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        audio_path,
        "-ac",
        "1",
        "-ar",
        "24000",
        "-f",
        "f32le",
        "-",
    ]
    return np.frombuffer(
        subprocess.run(command, check=True, capture_output=True).stdout, dtype="<f4"
    ).copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True, help="JSON list of [name, audio_path, [shifts]]")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    from safetensors.torch import load_file  # ty: ignore[import-not-found]
    from torchaudio.transforms import Resample  # ty: ignore[import-not-found]

    myna = load_module("myna_encoder", MYNA_SOURCE)
    frontend = load_module("myna_frontend", QUALIFIED / "myna_qualification.py")
    augmentation = load_module("myna_augmentation", QUALIFIED / "myna_resampling.py")
    architecture = json.loads(MYNA_CONFIG.read_text())
    architecture["n_samples"] = 100000
    device = torch.device("cuda:0")
    model = myna.Myna(myna.MynaConfig(**architecture))
    model.load_state_dict(load_file(str(MYNA_WEIGHTS), device="cpu"), strict=True)
    model.eval().requires_grad_(False).to(device)
    mel = myna.MynaPreprocessor().mel_spec.cpu().eval()
    resample = Resample(
        24000, 16000, resampling_method="sinc_interp_hann", lowpass_filter_width=6, rolloff=0.99
    )
    output_directory = Path(arguments.out)
    output_directory.mkdir(parents=True, exist_ok=True)
    items = json.loads(Path(arguments.items).read_text())
    for item_index, (name, audio_path, shifts) in enumerate(items):
        pending = [
            shift
            for shift in shifts
            if not (output_directory / f"{name}-shift{shift:+d}.npy").exists()
        ]
        if not pending:
            continue
        with torch.inference_mode():
            base = resample(torch.from_numpy(decode_24k(audio_path)))
            for shift in pending:
                shifted, _ = augmentation.shift_playback(base, shift)
                windows = []
                for left, right, offset, count in frontend.mel_blocks(
                    shifted.numel(), WINDOW_FRAMES
                ):
                    if count == WINDOW_FRAMES:
                        windows.append(
                            mel(shifted.unsqueeze(0)[..., left:right])[..., offset : offset + count]
                        )
                embeddings = []
                for start in range(0, len(windows), ENCODER_BATCH):
                    batch = torch.stack(windows[start : start + ENCODER_BATCH]).to(device)
                    embeddings.append(model(batch).float().cpu())
                array = (
                    torch.cat(embeddings).numpy()
                    if embeddings
                    else np.zeros((0, 384), dtype=np.float32)
                )
                np.save(output_directory / f"{name}-shift{shift:+d}.npy", array)
        if item_index % 50 == 0:
            print(f"{item_index}/{len(items)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
