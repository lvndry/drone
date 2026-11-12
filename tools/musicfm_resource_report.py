"""Verify lossless MusicFM provisioning/resource evidence and regenerate its figure."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

STUDY = Path(__file__).parents[1] / "experiments/studies/musicfm-preflight"


def load_raw(root: Path) -> dict[str, str]:
    index = json.loads((root / "data/runtime-index.json").read_bytes())
    packed = (root / "data" / index["path"]).read_bytes()
    if hashlib.sha256(packed).hexdigest() != index["sha256"]:
        raise ValueError("compressed evidence integrity")
    raw = gzip.decompress(packed)
    if hashlib.sha256(raw).hexdigest() != index["decoded_sha256"]:
        raise ValueError("decoded evidence integrity")
    files = json.loads(raw)["files"]
    if len(files) != index["files"]:
        raise ValueError("raw evidence count")
    for record in files.values():
        if hashlib.sha256(record["raw_utf8"].encode()).hexdigest() != record["sha256"]:
            raise ValueError("raw file integrity")
    return {name: record["raw_utf8"] for name, record in files.items()}


def summarize(root: Path = STUDY) -> dict[str, Any]:
    raw = load_raw(root)
    failed = json.loads(raw["v1/provision-report.json"])
    provision = json.loads(raw["v2/provision-report.json"])
    imports = json.loads(raw["v2/imports.json"])
    layout = json.loads(raw["v2/checkpoint-layout.json"])
    config_raw = raw["v2/resource-review-config-84685ce.json"]
    config = json.loads(config_raw)
    approval = json.loads(raw["v2/resource-root-approval-7520c831.json"])
    effective_raw = raw["v2/resource-84685ce-v1/effective-config.json"]
    supervisor = json.loads(raw["v2/resource-84685ce-v1/supervisor.json"])
    worker = json.loads(raw["v2/resource-84685ce-v1/worker-report.json"])
    index = json.loads((root / "data/runtime-index.json").read_bytes())
    if failed["complete"] or "pinned-checkpoint-download" in {
        row["label"] for row in failed["commands"]
    }:
        raise ValueError("preserved failed import gate")
    if (
        "CUDA version 13.2" not in raw["v1/provision/logs/cpu-imports.log"]
        or "CUDA version 13.0" not in raw["v1/provision/logs/cpu-imports.log"]
    ):
        raise ValueError("missing original CUDA-build failure")
    if not provision["complete"] or not imports["passed"] or not layout["passed"]:
        raise ValueError("CPU provisioning/layout gate")
    if raw["v2/baseline.json"] != raw["v2/baseline-after.json"]:
        raise ValueError("base environment changed")
    baseline = json.loads(raw["v2/baseline.json"])["protected"]
    if (
        imports["protected"] != baseline
        or imports["protected_after"] != baseline
        or baseline["cuda_initialized"]
    ):
        raise ValueError("Torch/NumPy/CUDA import protection")
    wheels = json.loads(raw["v2/selected-wheels.json"])["wheels"]
    audio = [row for row in wheels if row["name"] == "torchaudio"]
    if (
        len(audio) != 1
        or audio[0]["version"] != "2.11.0+cpu"
        or any(row["name"] in {"torch", "numpy"} for row in wheels)
    ):
        raise ValueError("exact CPU build/no-shadow policy")
    config_sha = hashlib.sha256(config_raw.encode()).hexdigest()
    if (
        config_sha != index["approved_config_sha256"]
        or config_sha != supervisor["config_sha256"]
        or config_sha != approval["approved_config_sha256"]
    ):
        raise ValueError("approval/config binding")
    if hashlib.sha256(effective_raw.encode()).hexdigest() != worker["effective_config_sha256"]:
        raise ValueError("effective config binding")
    if (
        not supervisor["complete"]
        or supervisor["exit_code"] != 0
        or not worker["complete"]
        or worker["accuracy_scored"]
    ):
        raise ValueError("resource completion/scope")
    if (
        worker["tool_sha256"] != config["tool_sha256"]
        or worker["tool_sha256"] != supervisor["tool_sha256"]
        or worker["checkpoint_sha256"] != layout["checkpoint_sha256"]
    ):
        raise ValueError("executed source/checkpoint binding")
    started = datetime.fromisoformat(approval["created_utc"])
    if (
        not datetime.fromisoformat(config["start_utc"])
        <= started
        < started + timedelta(seconds=supervisor["wall_seconds"])
        <= datetime.fromisoformat(config["deadline_utc"])
    ):
        raise ValueError("authorized execution window")
    protected = {str(key): value for key, value in supervisor["protected_processes"].items()}
    if protected != config["protected_vllm_start_times"]:
        raise ValueError("protected process identities")
    steps = worker["steps"]
    if (
        [step["index"] for step in steps] != list(range(21))
        or worker["shape_contract"]["encoder_frames"] != 300
        or worker["parameter_dtypes"] != ["torch.float32"]
    ):
        raise ValueError("fixed resource shape/precision/steps")
    if not all(math.isfinite(step["seconds"]) and step["seconds"] > 0 for step in steps):
        raise ValueError("invalid durations")
    observed = supervisor["observations"]
    free = min(row["gpu"]["free_mib"] for row in observed)
    nvml = max(row["worker_nvml_mib"] for row in observed)
    rss = max(row["tree_rss_bytes"] for row in observed)
    if (
        free < config["reserve_mib"]
        or nvml > config["requested_cap_mib"] + 1024
        or rss > 8 * 1024**3
        or supervisor["wall_seconds"] > 600
        or supervisor["aggregate_child_cpu_seconds"] > 600
    ):
        raise ValueError("fixed resource envelope")
    active = worker["estimated_active_state_floor"]["active_parameters"]
    if worker["estimated_active_state_floor"]["total_bytes"] != 4 * worker[
        "parameter_count"
    ] + 12 * active or any(
        step["gradient_bytes"] != 4 * active or step["optimizer_tensor_bytes"] < 8 * active
        for step in steps
    ):
        raise ValueError("state storage arithmetic")
    return {
        "format": "musicfm-resource-summary-v1",
        "source_commit": index["measured_source"]["v2_and_resource"],
        "config_sha256": config_sha,
        "model_parameters_including_new_head": worker["parameter_count"],
        "checkpoint_state_elements": sum(
            row["elements"] for row in layout["totals_by_dtype"].values()
        ),
        "checkpoint_state_tensors": layout["state_tensor_count"],
        "active_optimizer_elements": active,
        "analytical_active_state_bytes": worker["estimated_active_state_floor"]["total_bytes"],
        "first_step_seconds": steps[0]["seconds"],
        "steady_step_median_seconds": statistics.median(step["seconds"] for step in steps[1:]),
        "peak_allocated_bytes": max(step["peak_allocated_bytes"] for step in steps),
        "peak_reserved_bytes": max(step["peak_reserved_bytes"] for step in steps),
        "maximum_sampled_worker_nvml_mib": nvml,
        "minimum_sampled_global_free_mib": free,
        "maximum_sampled_tree_rss_bytes": rss,
        "resource_wall_seconds": supervisor["wall_seconds"],
        "resource_child_cpu_seconds": supervisor["aggregate_child_cpu_seconds"],
        "steps": steps,
        "failed_provision": {
            key: failed[key]
            for key in ("child_cpu_seconds", "wall_seconds", "peak_tree_rss_bytes", "storage_bytes")
        },
        "successful_provision": {
            key: provision[key]
            for key in ("child_cpu_seconds", "wall_seconds", "peak_tree_rss_bytes", "storage_bytes")
        },
        "wheel_count": len(wheels),
        "kernel_cache_warning": raw["v2/resource-84685ce-v1/worker.log"],
        "accuracy_scored": False,
    }


def plot(root: Path, result: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "musicfm-resource-study-v1"
    steps = result["steps"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    axes[0].bar(
        range(21), [step["seconds"] * 1000 for step in steps], color=["#bb6b36"] + ["#317c86"] * 20
    )
    axes[0].axhline(
        result["steady_step_median_seconds"] * 1000,
        color="#254754",
        linestyle="--",
        label="Steady median 51.0 ms",
    )
    axes[0].set_ylabel("Instrumented update duration (ms)")
    axes[0].set_xlabel("Update 0 = first AdamW state allocation; 1–20 = steady")
    axes[0].legend(frameon=False)
    for field, label, color in (
        ("peak_allocated_bytes", "Torch peak allocated", "#317c86"),
        ("peak_reserved_bytes", "Torch peak reserved", "#b67b3f"),
    ):
        axes[1].plot(
            range(21),
            [step[field] / 1024**3 for step in steps],
            marker=".",
            label=label,
            color=color,
        )
    axes[1].axhline(
        result["maximum_sampled_worker_nvml_mib"] / 1024,
        label="Sampled NVML maximum 6.50 GiB",
        linestyle="--",
        color="#5d5676",
    )
    axes[1].set_ylabel("GPU memory (GiB)")
    axes[1].set_xlabel("Fixed 12 s synthetic microbatch, FP32 weights/BF16 autocast")
    axes[1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.85))
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "MusicFM resource feasibility: one first update + 20 steady updates\nServing processes remained live; no task accuracy or isolated-throughput claim",
        fontsize=12,
    )
    figure.savefig(root / "resource-probe.png", dpi=180)
    figure.savefig(root / "resource-probe.svg", metadata={"Date": None})
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    arguments = parser.parse_args()
    summary = summarize()
    (STUDY / "data/runtime-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    if arguments.plot:
        plot(STUDY, summary)
