"""Bounded qualified Myna extraction; isolated runtime, no head fitting or key scoring."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import queue
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def verify(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    require(path.is_file() and not path.is_symlink(), "Missing or symlinked input")
    require(digest(path) == spec["sha256"], "Input identity changed")
    return path


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Cannot import qualified source")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def owned_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "Escaping output path")
    result = root / path
    require(result.resolve().is_relative_to(root.resolve()), "Escaping output symlink")
    return result


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def validate_projection(plan: dict[str, Any], physical: dict[str, Any]) -> None:
    """Check the sealed, previously dry-validated plan against actual prepared PCM lineage."""
    require(plan["schema_version"] == "drone-myna-rate-window-plan-v2", "Wrong projection")
    require(plan["complete"] is False, "Expected unexecuted projection")
    require(plan["augmentation_method"] == "pitch-by-playback-rate-polyphase-v1", "Wrong recipe")
    require(physical["complete"] is True, "Incomplete PCM preparation")
    require(physical["canonical_manifest_sha256"] == plan["lineage"]["waveforms"], "PCM lineage")
    require(physical["plan_sha256"] == plan["lineage"]["plan"], "Plan lineage")
    require(physical["labels_sha256"] == plan["lineage"]["labels"], "Label lineage")
    rows = {row["id"]: row for row in physical["rows"]}
    require(len(rows) == len(physical["rows"]) == 5638, "Duplicate/missing physical rows")
    tracks = plan["tracks"]
    require(len(tracks) == 5638 and {t["id"] for t in tracks} == set(rows), "Track membership")
    require(Counter(t["role"] for t in tracks) == {"fit": 5262, "selection": 376}, "Role counts")
    paths: set[str] = set()
    for track in tracks:
        row = rows[track["id"]]
        require(
            row["role"] == {"fit": "fit", "selection": "validation"}[track["role"]], "Role alias"
        )
        for key, other in [
            ("target", "target"),
            ("samples_24k", "samples"),
            ("pcm_sha256", "waveform_sha256"),
            ("source_sha256", "sha256"),
            ("physical_pcm_path", "cache"),
            ("physical_pcm_identity", "cache_identity"),
        ]:
            require(track[key] == row[other], f"Physical metadata differs: {key}")
        require(row["bytes"] == row["samples"] * 4, "PCM byte geometry")
        shifts = list(range(-6, 7)) if track["role"] == "fit" else [0]
        require([v["shift"] for v in track["views"]] == shifts, "Missing/extra/reordered views")
        for view in track["views"]:
            path = view["path"]
            require(path not in paths, "Duplicate blob path")
            paths.add(path)
            require(
                not Path(path).is_absolute() and ".." not in Path(path).parts, "Unsafe blob path"
            )
            require(view["shape"] == [view["windows"], 384] and view["windows"] > 0, "View shape")
            require(
                all(
                    view[k] is None
                    for k in ("raw_samples", "trimmed_samples", "padded_samples", "bytes", "sha256")
                ),
                "Projection contains fabricated observations",
            )


def projected_output_bound(plan: dict[str, Any]) -> int:
    """Payload, conservative NPY headers/receipts, and bounded metadata/progress overhead."""
    views = [view for track in plan["tracks"] for view in track["views"]]
    require(
        bool(views) and all(type(v["windows"]) is int and v["windows"] > 0 for v in views),
        "Invalid projected output geometry",
    )
    payload = sum(view["windows"] for view in views) * 384 * 4
    return payload + len(views) * (256 + 2048) + 2 * 64 * 1024**2 + 1024**2


class ViewBuffer:
    """Ordered complete-window accumulator; rejects missing/repeated/reordered batches."""

    def __init__(self, windows: int) -> None:
        import numpy as np

        self.values = np.empty((windows, 384), dtype=np.float32)
        self.next = 0

    def append(self, start: int, values: NDArray[np.float32]) -> None:
        import numpy as np

        require(start == self.next, "Window order/coverage changed")
        require(
            values.dtype == np.float32 and values.ndim == 2 and values.shape[1] == 384,
            "Embedding shape/dtype changed",
        )
        require(
            0 < len(values) <= 16 and start + len(values) <= len(self.values), "Invalid batch size"
        )
        require(bool(np.isfinite(values).all()), "Nonfinite embeddings")
        self.values[start : start + len(values)] = values
        self.next += len(values)

    def save(self, path: Path) -> dict[str, Any]:
        import numpy as np

        require(self.next == len(self.values), "Incomplete view cannot be published")
        require(not path.exists(), "Refusing to replace cache blob")
        temporary = path.with_name(path.name + ".pending")
        with temporary.open("xb") as stream:
            np.save(stream, self.values, allow_pickle=False)
        temporary.replace(path)
        return {"bytes": path.stat().st_size, "sha256": digest(path)}


def run(config: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    bound = {
        name: verify(config[name])
        for name in (
            "myna_source",
            "weights",
            "myna_config",
            "runtime_inventory",
            "qualification_source",
            "augmentation_source",
            "qualification_admission",
            "prepared_waveforms",
            "expected_cache_plan",
        )
    }
    for item in config["source_files"]:
        verify(item)
    require(
        any(
            Path(item["path"]).resolve() == Path(__file__).resolve()
            for item in config["source_files"]
        ),
        "Unbound executing extractor",
    )
    inventory = json.loads(bound["runtime_inventory"].read_text())
    environment = Path(config["environment"])
    for item in inventory["files"]:
        path = environment / item["path"]
        require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == item["bytes"]
            and digest(path) == item["sha256"],
            "Runtime bytes changed",
        )
    plan = json.loads(bound["expected_cache_plan"].read_text())
    physical = json.loads(bound["prepared_waveforms"].read_text())
    require(
        config["prepared_waveforms"]["sha256"] == plan["lineage"]["prepared_waveforms"],
        "Prepared source identity differs",
    )
    validate_projection(plan, physical)
    require(
        projected_output_bound(plan) <= config["output_bytes"],
        "Projected cache exceeds owned output budget",
    )
    admission = json.loads(bound["qualification_admission"].read_text())
    require(
        admission["complete"] is True
        and all(
            admission[k] is True
            for k in (
                "all_pitch_views_passed",
                "long_track_composed_parity_passed",
                "sustained_head_cost_measured",
            )
        ),
        "Unqualified extraction",
    )
    identities = config["identities"]
    require(
        admission["identities"]
        == {k: v for k, v in identities.items() if k != "qualification_sha256"},
        "Admission identity mismatch",
    )
    bindings = {
        "encoder_sha256": "weights",
        "encoder_source_sha256": "myna_source",
        "environment_sha256": "runtime_inventory",
        "frontend_sha256": "qualification_source",
        "augmentation_sha256": "augmentation_source",
        "qualification_sha256": "qualification_admission",
    }
    require(
        all(identities[k] == config[v]["sha256"] for k, v in bindings.items()),
        "Source/identity mismatch",
    )
    require(
        admission["augmentation_method"] == plan["augmentation_method"], "Admission recipe mismatch"
    )
    require(
        config["encoder_batch"] == 16
        and config["queue_batches"] == 4
        and len(config["cpu_worker_cores"]) == 4,
        "Unqualified concurrency",
    )
    root = Path(config["output_root"]).resolve()
    require(root.is_dir(), "Guardian must create owned output root")
    (root / "embeddings").mkdir(exist_ok=False)
    for name in ("result_path", "progress_path", "cache_manifest_path"):
        require(Path(config[name]).resolve().is_relative_to(root), "Output escapes task root")
    import numpy as np
    import torch

    load_file = importlib.import_module("safetensors.torch").load_file
    Resample = importlib.import_module("torchaudio.transforms").Resample

    require(torch.get_default_dtype() == torch.float32, "Base resampler requires float32 default")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(13)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1, "One visible GPU required"
    )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    require(
        str(properties.uuid).removeprefix("GPU-") == config["gpu_uuid"].removeprefix("GPU-"),
        "Wrong visible GPU",
    )
    torch.cuda.set_per_process_memory_fraction(
        config["allocator_bytes"] / properties.total_memory, device
    )
    torch.cuda.reset_peak_memory_stats(device)
    getattr(os, "sched_setaffinity")(0, {config["cpus"][-1]})
    stop = threading.Event()

    def cpu_watch() -> None:
        if stop.is_set() or time.monotonic() - started > config["wall_seconds"]:
            raise RuntimeError("Extraction cancelled or wall budget reached")

    def watch() -> None:
        cpu_watch()
        for process in config["protected_processes"]:
            stat = Path(f"/proc/{process['pid']}/stat")
            require(
                stat.exists()
                and stat.read_text().rsplit(")", 1)[1].split()[19] == process["start_ticks"],
                "Protected inference process changed",
            )
        require(
            torch.cuda.mem_get_info(device)[0] >= config["min_free_gpu_bytes"],
            "GPU reserve breached",
        )

    watch()
    module = load_module("drone_qualified_myna", bound["myna_source"])
    frontend = load_module("drone_qualified_frontend", bound["qualification_source"])
    augmentation = load_module("drone_qualified_augmentation", bound["augmentation_source"])
    architecture = json.loads(bound["myna_config"].read_text())
    architecture["n_samples"] = 100000
    model = module.Myna(module.MynaConfig(**architecture))
    require(
        model.config.n_frames == 196 and model.n_params == 21375872,
        "Released model geometry changed",
    )
    model.load_state_dict(load_file(str(bound["weights"]), device="cpu"), strict=True)
    model.eval().requires_grad_(False).to(device)
    local = threading.local()
    cores = iter(config["cpu_worker_cores"])
    core_lock = threading.Lock()
    messages: queue.Queue[tuple[str, int, int, Any]] = queue.Queue(maxsize=4)

    def initialize() -> None:
        with core_lock:
            core = next(cores)
        getattr(os, "sched_setaffinity")(0, {core})
        local.mel = module.MynaPreprocessor().mel_spec.cpu().eval()
        local.resample = Resample(
            24000, 16000, resampling_method="sinc_interp_hann", lowpass_filter_width=6, rolloff=0.99
        )

    def put(message: tuple[str, int, int, Any]) -> None:
        while True:
            cpu_watch()
            try:
                messages.put(message, timeout=0.25)
                return
            except queue.Full:
                continue

    def produce(index: int) -> None:
        track = plan["tracks"][index]
        path = Path(track["physical_pcm_path"])
        stat = path.stat()
        identity = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        require(
            not path.is_symlink() and identity == track["physical_pcm_identity"],
            "PCM identity changed",
        )
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == track["pcm_sha256"], "PCM hash changed")
        after = path.stat()
        require(
            [after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns]
            == identity,
            "PCM changed during read",
        )
        values = np.frombuffer(raw, dtype="<f4").copy()
        del raw
        require(
            values.shape == (track["samples_24k"],) and bool(np.isfinite(values).all()),
            "Invalid PCM",
        )
        with torch.inference_mode():
            base = local.resample(torch.from_numpy(values))
            del values
            require(base.numel() == track["samples_16k"], "Observed base sample count differs")
            for view in track["views"]:
                cpu_watch()
                began = time.monotonic()
                shifted, geometry = augmentation.shift_playback(base, view["shift"])
                pitch_seconds = time.monotonic() - began
                require(
                    shifted.numel() == view["samples_16k"]
                    and geometry["complete_windows"] == view["windows"],
                    "Observed augmentation geometry differs",
                )
                blocks: list[Any] = []
                window_start = 0
                mel_seconds = 0.0
                for left, right, offset, count in frontend.mel_blocks(shifted.numel(), 196):
                    if count != 196:
                        continue
                    cpu_watch()
                    began = time.monotonic()
                    block = local.mel(shifted.unsqueeze(0)[..., left:right])[
                        ..., offset : offset + count
                    ]
                    require(
                        tuple(block.shape) == (1, 128, 196) and bool(torch.isfinite(block).all()),
                        "Invalid mel window",
                    )
                    blocks.append(block)
                    mel_seconds += time.monotonic() - began
                    if len(blocks) == 16:
                        batch = torch.stack(blocks).contiguous().numpy()
                        put(("batch", index, view["shift"], (window_start, batch)))
                        window_start += len(blocks)
                        blocks.clear()
                if blocks:
                    batch = torch.stack(blocks).contiguous().numpy()
                    put(("batch", index, view["shift"], (window_start, batch)))
                    window_start += len(blocks)
                require(window_start == view["windows"], "Missing complete frontend windows")
                put(
                    (
                        "view",
                        index,
                        view["shift"],
                        {
                            "raw_samples": shifted.numel(),
                            "trimmed_samples": 0,
                            "padded_samples": 0,
                            "pitch_seconds": pitch_seconds,
                            "mel_seconds": mel_seconds,
                        },
                    )
                )
                del shifted
        put(("track", index, 0, None))

    total_windows = sum(v["windows"] for t in plan["tracks"] for v in t["views"])
    completed_tracks = completed_views = completed_windows = 0
    encoder_seconds = pitch_seconds_total = mel_seconds_total = 0.0
    buffers: dict[tuple[int, int], ViewBuffer] = {}
    views = {(i, v["shift"]): v for i, track in enumerate(plan["tracks"]) for v in track["views"]}
    futures: dict[int, Future[None]] = {}
    next_track = 0
    receipt_path = root / "view-receipts.jsonl"
    executor = ThreadPoolExecutor(max_workers=4, initializer=initialize)
    try:
        for _ in range(4):
            futures[next_track] = executor.submit(produce, next_track)
            next_track += 1
        with receipt_path.open("x") as receipts, torch.inference_mode():
            while completed_tracks < len(plan["tracks"]):
                watch()
                for future in futures.values():
                    if future.done() and future.exception() is not None:
                        future.result()
                try:
                    kind, index, shift, payload = messages.get(timeout=0.5)
                except queue.Empty:
                    continue
                key = (index, shift)
                if kind == "batch":
                    window_start, batch = payload
                    require(
                        batch.dtype == np.float32 and batch.shape[1:] == (1, 128, 196),
                        "Encoder input contract",
                    )
                    began = time.monotonic()
                    output = model(torch.from_numpy(batch).to(device)).cpu().numpy()
                    encoder_seconds += time.monotonic() - began
                    if key not in buffers:
                        buffers[key] = ViewBuffer(views[key]["windows"])
                    buffers[key].append(window_start, output)
                elif kind == "view":
                    view = views[key]
                    view.update(buffers.pop(key).save(owned_path(root, view["path"])))
                    for name in ("raw_samples", "trimmed_samples", "padded_samples"):
                        view[name] = payload[name]
                    pitch_seconds_total += payload["pitch_seconds"]
                    mel_seconds_total += payload["mel_seconds"]
                    completed_views += 1
                    completed_windows += view["windows"]
                    receipts.write(
                        json.dumps(
                            {
                                "id": plan["tracks"][index]["id"],
                                "shift": shift,
                                **payload,
                                "sha256": view["sha256"],
                                "bytes": view["bytes"],
                            }
                        )
                        + "\n"
                    )
                    receipts.flush()
                elif kind == "track":
                    futures.pop(index).result()
                    completed_tracks += 1
                    if next_track < len(plan["tracks"]):
                        futures[next_track] = executor.submit(produce, next_track)
                        next_track += 1
                    elapsed = time.monotonic() - started
                    atomic_json(
                        Path(config["progress_path"]),
                        {
                            "status": "EXTRACTING",
                            "completed_tracks": completed_tracks,
                            "total_tracks": len(plan["tracks"]),
                            "completed_views": completed_views,
                            "completed_windows": completed_windows,
                            "total_windows": total_windows,
                            "elapsed_seconds": elapsed,
                            "observed_windows_per_second": completed_windows / elapsed,
                            "remaining_seconds_at_observed_rate": elapsed
                            * (total_windows - completed_windows)
                            / completed_windows,
                            "encoder_seconds": encoder_seconds,
                            "summed_worker_pitch_seconds": pitch_seconds_total,
                            "summed_worker_mel_seconds": mel_seconds_total,
                        },
                    )
                else:
                    raise ValueError("Unknown pipeline event")
        require(
            not buffers and completed_windows == total_windows and completed_views == len(views),
            "Incomplete cache coverage",
        )
        plan.update(
            schema_version="drone-myna-rate-window-cache-v2",
            complete=True,
            status="COMPLETE_QUALIFIED_EMBEDDINGS_NO_HEAD_TRAINING",
            identities=identities,
        )
        atomic_json(Path(config["cache_manifest_path"]), plan)
        require(
            Path(config["cache_manifest_path"]).stat().st_size <= 64 * 1024**2,
            "Cache manifest exceeds consumer bound",
        )
        result = {
            "complete": True,
            "tracks": completed_tracks,
            "views": completed_views,
            "windows": completed_windows,
            "embedding_payload_bytes": total_windows * 384 * 4,
            "elapsed_seconds": time.monotonic() - started,
            "encoder_seconds": encoder_seconds,
            "summed_worker_pitch_seconds": pitch_seconds_total,
            "summed_worker_mel_seconds": mel_seconds_total,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "manifest_sha256": digest(Path(config["cache_manifest_path"])),
            "head_training": False,
            "key_accuracy_scored": False,
        }
        atomic_json(Path(config["result_path"]), result)
        atomic_json(Path(config["progress_path"]), result)
        return result
    finally:
        stop.set()
        executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
