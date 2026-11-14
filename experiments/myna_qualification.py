"""Tensor-only Myna qualification; no reference keys, checkpoint search, or task fit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor
    from torch import device as TorchDevice


def mel_blocks(samples: int, block_frames: int = 196) -> list[tuple[int, int, int, int]]:
    """Return waveform slices and central-frame selections preserving global reflection."""
    if samples <= 1024 or block_frames < 1:
        raise ValueError("Reflect-padded frontend needs more than 1024 samples")
    total = samples // 512 + 1
    result = []
    for start in range(0, total, block_frames):
        count = min(block_frames, total - start)
        left = max(0, start * 512 - 1024)
        right = min(samples, (start + count - 1) * 512 + 1024)
        if right - left <= 1024:
            left = max(0, left - 512)
        result.append((left, right, start - left // 512, count))
    return result


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def verify_file(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    if path.is_symlink() or not path.is_file() or sha256(path) != spec["sha256"]:
        raise ValueError("Bound input identity changed")
    return path


def compare(reference: Tensor, candidate: Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    import torch

    if (
        reference.shape != candidate.shape
        or not torch.isfinite(reference).all()
        or not torch.isfinite(candidate).all()
    ):
        raise ValueError("Invalid numerical comparison")
    difference = (reference - candidate).abs()
    allowed = atol + rtol * reference.abs()
    result = {
        "values": reference.numel(),
        "max_abs_error": float(difference.max()),
        "max_normalized_error": float((difference / allowed).max()),
        "outside_tolerance": int((difference > allowed).sum()),
    }
    if result["outside_tolerance"]:
        raise ValueError(f"Numerical parity failed: {result}")
    return result


def pitch_shift_cpu(signal: Tensor, steps: int) -> Tensor:
    """Declared local augmentation: phase vocoder plus Kaiser polyphase resampling."""
    import numpy as np
    import torch

    if steps not in range(-6, 7) or signal.device.type != "cpu":
        raise ValueError("Expected CPU waveform and a declared semitone shift")
    if steps == 0:
        return signal.clone()
    phase_vocoder = importlib.import_module("torchaudio.functional").phase_vocoder
    resample_poly = importlib.import_module("scipy.signal").resample_poly
    rate = 2 ** (-steps / 12)
    down = int(16000 / rate)
    if not 11313 <= down <= 22627:
        raise ValueError("Resampling ratio outside frozen bound")
    window = torch.hann_window(2048)
    spectrum = torch.stft(signal, 2048, 512, 2048, window, return_complex=True)
    advance = torch.linspace(0, math.pi * 512, 1025)[..., None]
    stretched_spectrum = phase_vocoder(spectrum, rate, advance)
    del spectrum
    stretched = torch.istft(
        stretched_spectrum,
        2048,
        512,
        2048,
        window,
        length=round(signal.shape[-1] / rate),
    )
    del stretched_spectrum
    converted = resample_poly(
        stretched.numpy(),
        16000,
        down,
        axis=-1,
        window=("kaiser", 5.0),
        padtype="constant",
    )
    result = torch.from_numpy(np.ascontiguousarray(converted))
    length = signal.shape[-1]
    if result.shape[-1] < length:
        result = torch.nn.functional.pad(result, (0, length - result.shape[-1]))
    return result[..., :length].contiguous()


def run_advanced(
    config: dict[str, Any],
    model: Callable[[Tensor], Tensor],
    mel: Callable[[Tensor], Tensor],
    device: TorchDevice,
    watch: Callable[[], None],
) -> list[dict[str, Any]]:
    """Numerically qualify declared pitch views and composed extraction, without labels."""
    import numpy as np
    import torch

    Resample = importlib.import_module("torchaudio.transforms").Resample
    resample = Resample(24000, 16000, lowpass_filter_width=6, rolloff=0.99)
    phases: list[dict[str, Any]] = []

    def record(name: str, before: float, result: object) -> None:
        phases.append({"phase": name, "wall_seconds": time.monotonic() - before, "result": result})
        Path(config["progress_path"]).write_text(json.dumps({"phases": phases}, indent=2) + "\n")

    def embed(features: Tensor) -> Tensor:
        complete = features.shape[-1] // 196
        inputs = features[..., : complete * 196].reshape(1, 128, complete, 196)
        inputs = inputs.permute(2, 0, 1, 3).contiguous()
        with torch.inference_mode():
            return torch.cat(
                [
                    model(inputs[start : start + 16].to(device)).cpu()
                    for start in range(0, complete, 16)
                ]
            )

    admission = config["augmentation_qualification"]
    previous_result = json.loads(verify_file(admission["result"]).read_text())
    previous_controller = json.loads(verify_file(admission["controller"]).read_text())
    previous_config_path = verify_file(admission["config"])
    previous_config = json.loads(previous_config_path.read_text())
    if (
        previous_result.get("passed") is not True
        or previous_result.get("method") != "pitch-by-playback-rate-polyphase-v1"
        or previous_controller.get("status") != "complete"
        or previous_controller.get("config_sha256") != sha256(previous_config_path)
        or previous_config["runtime_inventory"]["sha256"] != config["runtime_inventory"]["sha256"]
        or previous_config["augmentation_source"]["sha256"]
        != config["augmentation_source"]["sha256"]
    ):
        raise ValueError("No passed qualification for this exact augmentation source")
    spec = importlib.util.spec_from_file_location(
        "pinned_playback_augmentation", verify_file(config["augmentation_source"])
    )
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load qualified playback augmentation")
    augmentation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(augmentation)

    for row in config["advanced_pcm"]:
        if row["role"] != "fit":
            raise ValueError("Only fixed fitting PCM permitted")
        path = verify_file({"path": row["cache"], "sha256": row["waveform_sha256"]})
        signal = torch.from_numpy(np.fromfile(path, dtype="<f4").copy()).reshape(1, -1)
        if signal.shape[-1] != row["samples"] or not torch.isfinite(signal).all():
            raise ValueError("Invalid fitting PCM")
        original = resample(signal)
        del signal
        for steps in row["shifts"]:
            before = time.monotonic()
            watch()
            shifted_mono, view_geometry = augmentation.shift_playback(original[0], steps)
            shifted = shifted_mono.unsqueeze(0)
            del shifted_mono
            pitch_seconds = time.monotonic() - before
            if (
                shifted.shape[-1] != view_geometry["output_samples"]
                or not torch.isfinite(shifted).all()
            ):
                raise ValueError("Invalid playback-rate fitting PCM")
            feature_start = time.monotonic()
            with torch.inference_mode():
                whole = mel(shifted)
                whole_frontend_seconds = time.monotonic() - feature_start
                blocked_start = time.monotonic()
                blocked = torch.cat(
                    [
                        mel(shifted[..., left:right])[..., offset : offset + count]
                        for left, right, offset, count in mel_blocks(shifted.shape[-1])
                    ],
                    dim=-1,
                )
                blocked_frontend_seconds = time.monotonic() - blocked_start
                if whole.shape[-1] != view_geometry["mel_frames"]:
                    raise ValueError("Playback-rate mel geometry differs")
                mel_proof = compare(whole, blocked, atol=config["atol"], rtol=config["rtol"])
                frontend_seconds = time.monotonic() - feature_start
                # Warm up the actual full-batch and remainder shapes, without timing them.
                complete = whole.shape[-1] // 196
                warm_windows = whole[..., : complete * 196].reshape(1, 128, complete, 196)
                warm_windows = warm_windows.permute(2, 0, 1, 3).contiguous()
                model(warm_windows[: min(16, complete)].to(device))
                if complete > 16 and complete % 16:
                    model(warm_windows[-(complete % 16) :].to(device))
                del warm_windows
                torch.cuda.synchronize(device)
                encoder_start = time.monotonic()
                expected = embed(whole)
                torch.cuda.synchronize(device)
                encoder_seconds = time.monotonic() - encoder_start
                observed = embed(blocked)
                embedding_proof = compare(
                    expected, observed, atol=config["atol"], rtol=config["rtol"]
                )
            record(
                "pitch_and_composed_extraction",
                before,
                {
                    "id": row["id"],
                    "semitones": steps,
                    "samples_16k": shifted.shape[-1],
                    "windows": expected.shape[0],
                    "discarded_frames": whole.shape[-1] % 196,
                    "pitch_seconds": pitch_seconds,
                    "view_geometry": view_geometry,
                    "whole_frontend_seconds": whole_frontend_seconds,
                    "blocked_frontend_seconds": blocked_frontend_seconds,
                    "both_frontends_seconds": frontend_seconds,
                    "encoder_batch16_seconds": encoder_seconds,
                    "mel_parity": mel_proof,
                    "embedding_parity": embedding_proof,
                },
            )
            del shifted, whole, blocked, expected, observed
        del original
    if config.get("head_benchmark"):
        before = time.monotonic()
        head_source = verify_file(config["head_source"])
        spec = importlib.util.spec_from_file_location("pinned_myna_head", head_source)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load exact trainer hot path")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        torch.manual_seed(42)
        head = helper.make_head().to(device)
        optimizer = helper.make_optimizer(head)
        rng = np.random.default_rng(42)
        values = rng.standard_normal((16384, 384), dtype=np.float32)
        targets = (np.arange(len(values)) % 24).astype(np.int64)
        weights = np.linspace(0.25, 1.75, len(values), dtype=np.float32)
        order = rng.permutation(len(values))[: 110 * 64].reshape(110, 64)

        def step(indices: np.ndarray) -> float:
            # Match trainer CPU gathers, transfers, checks, optimizer and scalar synchronization.
            return helper.train_batch(
                head, optimizer, values[indices], targets[indices], weights[indices], watch
            )

        for indices in order[:10]:
            step(indices)
        torch.cuda.synchronize(device)
        timed_start = time.monotonic()
        losses = [step(indices) for indices in order[10:]]
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - timed_start
        if not all(math.isfinite(value) for value in losses):
            raise ValueError("Nonfinite steady-state head update")
        record(
            "steady_head_training_path",
            before,
            {
                "warmup_updates": 10,
                "timed_updates": 100,
                "batch_size": 64,
                "timed_wall_seconds": elapsed,
                "updates_per_second": 100 / elapsed,
                "synthetic_only": True,
                "includes_disk_admission": False,
                "includes_cpu_gather_transfer_checks_and_scalar_sync": True,
                "helper_sha256": config["head_source"]["sha256"],
                "parameters": sum(parameter.numel() for parameter in head.parameters()),
            },
        )
        before = time.monotonic()
        counts = config["selection_window_counts"]
        if len(counts) != 376 or sum(counts) != 10892 or min(counts) < 1:
            raise ValueError("Synthetic evaluation geometry differs")
        selected = values[: sum(counts)]
        logits_by_batch = {}
        timings = {}
        for batch in (64, 1):
            # Warm the actual helper path before measuring all original-window-sized synthetic inputs.
            list(helper.window_logits(head, selected[:batch], batch, watch))
            torch.cuda.synchronize(device)
            started = time.monotonic()
            aggregate = []
            offset = 0
            for count in counts:
                aggregate.append(
                    helper.mean_logits(
                        helper.window_logits(head, selected[offset : offset + count], batch, watch),
                        count,
                    )
                )
                offset += count
            logits_by_batch[batch] = np.stack(aggregate)
            torch.cuda.synchronize(device)
            timings[str(batch)] = time.monotonic() - started
        proof = compare(
            torch.from_numpy(logits_by_batch[1]),
            torch.from_numpy(logits_by_batch[64]),
            atol=config["atol"],
            rtol=config["rtol"],
        )
        if not np.array_equal(
            np.argmax(logits_by_batch[1], axis=1), np.argmax(logits_by_batch[64], axis=1)
        ):
            raise ValueError("Head batch parity changed synthetic aggregate class")
        record(
            "head_evaluation_path",
            before,
            {
                "synthetic_windows": len(selected),
                "batch_wall_seconds": timings,
                "batch_parity": proof,
                "synthetic_tracks": len(counts),
                "reference_accuracy_scored": False,
            },
        )
    return phases


def run(config: dict[str, Any]) -> dict[str, Any]:
    # All inputs are bound before importing any ML runtime.
    source = verify_file(config["myna_source"])
    weights = verify_file(config["weights"])
    original_config = json.loads(verify_file(config["myna_config"]).read_text())
    inventory = json.loads(verify_file(config["runtime_inventory"]).read_text())
    environment = Path(config["environment"])
    for item in inventory["files"]:
        path = environment / item["path"]
        if (
            path.is_symlink()
            or path.stat().st_size != item["bytes"]
            or sha256(path) != item["sha256"]
        ):
            raise ValueError("Runtime bytes changed after provisioning")
    for item in config["source_files"]:
        verify_file(item)
    import numpy as np
    import torch

    load_file = importlib.import_module("safetensors.torch").load_file
    from torch import nn

    Resample = importlib.import_module("torchaudio.transforms").Resample

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(13)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("One explicitly visible CUDA device required")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    uuid = str(properties.uuid)
    if uuid.removeprefix("GPU-") != config["gpu_uuid"].removeprefix("GPU-"):
        raise ValueError("Visible CUDA device differs")
    torch.cuda.set_per_process_memory_fraction(
        config["allocator_bytes"] / properties.total_memory, device
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    phases: list[dict[str, Any]] = []

    def watch() -> None:
        for process in config["protected_processes"]:
            stat = Path(f"/proc/{process['pid']}/stat")
            if (
                not stat.exists()
                or stat.read_text().rsplit(")", 1)[1].split()[19] != process["start_ticks"]
            ):
                raise RuntimeError("Protected inference process changed")
        free, _ = torch.cuda.mem_get_info(device)
        if free < config["min_free_gpu_bytes"]:
            raise RuntimeError("Global GPU reserve violated")
        if time.monotonic() - started > config["wall_seconds"]:
            raise RuntimeError("Worker wall budget exceeded")

    def record(name: str, began: float, result: object) -> None:
        torch.cuda.synchronize(device)
        phases.append({"phase": name, "wall_seconds": time.monotonic() - began, "result": result})
        Path(config["progress_path"]).write_text(json.dumps({"phases": phases}, indent=2) + "\n")
        watch()

    watch()
    began = time.monotonic()
    spec = importlib.util.spec_from_file_location("drone_pinned_myna", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load pinned Myna module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_config["n_samples"] = 100000
    model = module.Myna(module.MynaConfig(**original_config))
    if model.config.n_frames != 196 or model.n_params != 21375872:
        raise ValueError("Unexpected released architecture or frame count")
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    model.eval().requires_grad_(False).to(device)
    # Upstream preprocessor is a plain object, not a registered nn.Module child.
    # Keep this frontend explicitly on CPU while encoder parameters live on CUDA.
    mel = model.preprocessor.mel_spec.cpu()
    record("strict_load", began, {"parameters": model.n_params, "frames_per_window": 196})

    if config.get("advanced_pcm"):
        phases.extend(run_advanced(config, model, mel, device, watch))
        watch()
        return {
            "status": "complete",
            "phases": phases,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "accuracy_scored": False,
            "trained_weights_retained": False,
            "pitch_method": "pitch-by-playback-rate-polyphase-v1",
            "paper_pitch_implementation_parity_claimed": False,
        }

    def streamed(signal: Tensor) -> Tensor:
        return torch.cat(
            [
                mel(signal[..., left:right])[..., offset : offset + count]
                for left, right, offset, count in mel_blocks(signal.shape[-1])
            ],
            dim=-1,
        )

    # Exact tensor boundary is shared by the released frontend and our execution adapter.
    synthetic = torch.arange(16000 * 600 + 137, dtype=torch.float32) / 16000
    synthetic = (
        0.2 * torch.sin(2 * math.pi * 220 * synthetic)
        + 0.1 * torch.sin(2 * math.pi * 330 * synthetic)
    ).reshape(1, -1)
    synthetic[0, [0, 100352, 9600000]] += 0.25
    began = time.monotonic()
    with torch.inference_mode():
        reference = mel(synthetic)
        blocked = streamed(synthetic)
        boundary = compare(reference, blocked, atol=config["atol"], rtol=config["rtol"])
    record("600_second_mel_boundaries", began, boundary)
    del reference, blocked, synthetic

    began = time.monotonic()
    stereo = torch.stack((torch.linspace(-0.5, 0.5, 24000 * 3 + 7), torch.zeros(24000 * 3 + 7)))
    mono = stereo.mean(dim=0, keepdim=True)
    resample = Resample(
        orig_freq=24000,
        new_freq=16000,
        resampling_method="sinc_interp_hann",
        lowpass_filter_width=6,
        rolloff=0.99,
        beta=None,
    )
    converted = resample(mono)
    if (
        converted.shape != (1, math.ceil(mono.shape[-1] * 2 / 3))
        or not torch.isfinite(converted).all()
    ):
        raise ValueError("Downmix/resample contract failed")
    record(
        "tensor_downmix_resample",
        began,
        {
            "input_samples": mono.shape[-1],
            "output_samples": converted.shape[-1],
            "decoder_parity_claimed": False,
        },
    )

    waveforms: list[tuple[str, Tensor]] = []
    for row in config["fitting_pcm"]:
        if row["role"] != "fit":
            raise ValueError("Only fixed fitting PCM admitted")
        path = verify_file({"path": row["cache"], "sha256": row["waveform_sha256"]})
        if path.stat().st_size != row["samples"] * 4:
            raise ValueError("PCM size mismatch")
        signal = torch.from_numpy(np.fromfile(path, dtype="<f4").copy()).reshape(1, -1)
        if not torch.isfinite(signal).all():
            raise ValueError("Nonfinite fitting waveform")
        waveforms.append((row["id"], resample(signal)))
    # Synthetic encoder batches include silence and deterministic noise.
    generator = torch.Generator().manual_seed(13)
    signal = torch.randn((1, 100352 * 32), generator=generator) * 0.05
    signal[:, :100352] = 0
    waveforms.insert(0, ("synthetic_32_windows", signal))
    with torch.inference_mode():
        for identity, signal in waveforms:
            watch()
            began = time.monotonic()
            whole = mel(signal)
            blocks = streamed(signal)
            mel_proof = compare(whole, blocks, atol=config["atol"], rtol=config["rtol"])
            complete = whole.shape[-1] // 196
            if complete < 1:
                raise ValueError("No complete window; padding is not authorized")
            inputs = (
                whole[:, :, : complete * 196]
                .reshape(1, 128, complete, 196)
                .permute(2, 0, 1, 3)
                .contiguous()
            )
            outputs: dict[int, Tensor] = {}
            timings: dict[int, float] = {}
            for batch in (1, 8, 16):
                torch.cuda.synchronize(device)
                before = time.monotonic()
                chunks = []
                for offset in range(0, complete, batch):
                    watch()
                    chunks.append(model(inputs[offset : offset + batch].to(device)).cpu())
                torch.cuda.synchronize(device)
                timings[batch] = time.monotonic() - before
                outputs[batch] = torch.cat(chunks)
            if outputs[1].shape != (complete, 384):
                raise ValueError("Unexpected embedding shape")
            proofs = {
                str(batch): compare(
                    outputs[1], outputs[batch], atol=config["atol"], rtol=config["rtol"]
                )
                for batch in (8, 16)
            }
            record(
                "encoder_batches",
                began,
                {
                    "id": identity,
                    "samples_16k": signal.shape[-1],
                    "complete_windows": complete,
                    "discarded_mel_frames": whole.shape[-1] % 196,
                    "mel_parity": mel_proof,
                    "batch_parity": proofs,
                    "batch_wall_seconds": timings,
                },
            )

    began = time.monotonic()
    torch.manual_seed(13)
    head = nn.Sequential(
        nn.Linear(384, 4096),
        nn.ReLU(),
        nn.Dropout(0.99),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 24),
    ).to(device)
    if sum(parameter.numel() for parameter in head.parameters()) != 18456600:
        raise ValueError("Head parameter contract differs")
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-4, weight_decay=1e-5)
    inputs = torch.randn(64, 384, device=device)
    targets = torch.arange(64, device=device) % 24
    losses = []
    for _ in range(3):
        watch()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(head(inputs), targets)
        loss.backward()
        if not torch.isfinite(loss) or any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in head.parameters()
        ):
            raise ValueError("Nonfinite synthetic training step")
        optimizer.step()
        losses.append(float(loss.detach()))
    record("three_synthetic_head_updates", began, {"losses": losses, "task_training": False})
    watch()
    return {
        "status": "complete",
        "phases": phases,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": properties.name,
        "wall_seconds_after_import": time.monotonic() - started,
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "accuracy_scored": False,
        "trained_weights_retained": False,
        "pitch_shift_qualification": "pending_separate_all_13_views",
        "decoder_contract": "existing_verified_mono_24k_ffmpeg_PCM_then_pinned_torchaudio_16k_resample",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config["result_path"])
    if output.exists():
        raise FileExistsError(output)
    result = run(config)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
