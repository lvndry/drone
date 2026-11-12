"""Verify and plot the three preserved GPU1 qualifications; never run a model."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/studies/k10-capacity"
GIB = 1024**3
PINS = {
    "v1": (
        "c4dbda8c2cd0af09d3d68cb4ddd378e67002e9e7",
        "6162bbf8bdf4b1e447dafa7f43f783e0e947234ba65c2329825832c04ab016f2",
    ),
    "v2": (
        "a1b6cc3497f064ebf306c79cf0b0b20b3cd31ade",
        "2511ca1160ac59b3bca086fd47d51ebe16094ffe9aa7d71f812f063fb7b4b679",
    ),
    "v3": (
        "57e0fac2b63a242e2d96d86f6fcc0be4d8831bc0",
        "132978eae5395599d7e9ed0da0a12ef62ad7fd5f13646bb06d9147076577728c",
    ),
}
ARCHIVES = {
    "v1": "594e35f54950fbb8d583f353c4a93741c0e47a240be4eeaecb08bf1dc745eaf2",
    "v2": "c79ad8bbb88481cb98500e8fd5350f93a32f9b13fbb4c7c29df637daa95a41e8",
    "v3": "effe7f099a4ada5a081b9004c4c1fb0c8ac2b2558c0892116479163a2252654b",
}
INITIAL = {
    "baseline": "bb227d5d27d0f28c0065de14468a4ada92d79944ccbcb9da6a68c97e64daa219",
    "candidate": "4fc58182dc884d7e9e9b948e72737742628be6b6a0685472b42808b0aed715d4",
}
PREFIX = "54d0c895089289238293d49d211a4a8ce26bfe4b79e3d47419a2fc73a51d5b4a"
BACKEND = dict(
    global_deterministic_algorithms=False,
    deterministic_warn_only=False,
    cudnn_deterministic=True,
    cudnn_benchmark=False,
    cudnn_allow_tf32=False,
    matmul_allow_tf32=False,
    default_dtype="torch.float32",
    autocast_enabled=False,
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def document(files: dict[str, str], name: str) -> dict[str, Any]:
    return json.loads(files[name])


def workload(config: dict[str, Any], result: dict[str, Any], model: str) -> None:
    require(result["complete"] and result["model"] == model, "Incomplete model resource result")
    require(
        result["initial_state_sha256"] == INITIAL[model]
        and result["first_two_blocks_sha256"] == PREFIX,
        "Initialization/prefix differs",
    )
    updates = result["update_with_transition_seconds"]
    require(
        len(updates) == 3 and all(math.isfinite(t) and t > 0 for t in updates),
        "Three updates required",
    )
    require(
        result["backend_policy"] == config["backend_policy"] == BACKEND, "Backend policy differs"
    )
    expected = {
        "selection376": (376, sum(r["frames"] for r in config["selection"]), 284),
        "synthetic_envelope4": (4, 714, 4),
    }
    for label, counts in expected.items():
        phase = result["evaluation"][label]
        require(
            tuple(phase[k] for k in ("tracks", "frames", "batches")) == counts,
            "True-length phase coverage differs",
        )
        require(
            math.isfinite(phase["wall_seconds"]) and phase["wall_seconds"] > 0, "Invalid phase time"
        )
    require(
        result["allocator_bytes"] == 2 * GIB
        and 0 < result["peak_allocated_bytes"] <= result["peak_reserved_bytes"] <= 2 * GIB,
        "Torch allocator cap differs",
    )
    require(
        all(result[k] is False for k in ("task_fit", "predictions_retained", "weights_retained")),
        "Task evidence/weights unexpectedly retained",
    )


def trace(files: dict[str, str], config: dict[str, Any]) -> list[dict[str, Any]]:
    samples = [json.loads(line) for line in files["run-v1/samples.jsonl"].splitlines()]
    require(bool(samples), "No resource samples")
    policy = config["policy"]
    require(
        all(a["epoch"] <= b["epoch"] for a, b in zip(samples, samples[1:])), "Trace order differs"
    )
    for row in samples:
        require(
            all(
                math.isfinite(row[k]) and row[k] >= 0
                for k in (
                    "epoch",
                    "owned_bytes",
                    "free_bytes",
                    "gpu0_free_bytes",
                    "rss_bytes",
                    "cpu_seconds",
                )
            ),
            "Nonfinite trace",
        )
        require(
            row["owned_bytes"] <= policy["owned_gpu_bytes"]
            and row["free_bytes"] >= policy["early_abort_bytes"]
            and row["gpu0_free_bytes"] >= policy["gpu0_reserve_bytes"]
            and row["rss_bytes"] <= policy["rss_bytes"]
            and row["cpu_seconds"] <= policy["cpu_seconds"],
            "Observed resource cap failed",
        )
        require(row["observed_hard_floor_overshoot_bytes"] == 0, "Observed hard-floor overshoot")
        require(row["free_bytes"] <= row["total_bytes"], "Impossible free-memory sample")
    for model in ("baseline", "candidate"):
        launch = f"run-v1/{model}-launch.json"
        if launch not in files:
            continue
        start = document(files, launch)["epoch"]
        stop = (
            document(files, f"run-v1/{model}-released.json")["epoch"]
            if f"run-v1/{model}-released.json" in files
            else samples[-1]["epoch"]
        )
        active = [r for r in samples if start <= r["epoch"] <= stop]
        require(
            bool(active)
            and all(
                r["poll_gap_seconds"] <= 0.2 and r["sensor_completion_gap_seconds"] <= 0.2
                for r in active
            ),
            "Active trace stale",
        )
    return samples


def check(study: Path = STUDY) -> dict[str, Any]:
    index = json.loads((study / "data/resolution-gpu1-index.json").read_bytes())
    compressed = (study / "data" / index["bundle"]).read_bytes()
    raw = gzip.decompress(compressed)
    require(
        sha(compressed) == index["bundle_sha256"] and sha(raw) == index["decoded_sha256"],
        "Bundle byte hash differs",
    )
    packet = json.loads(raw)
    require(
        set(packet["attempts"]) == set(index["attempts"]) == set(PINS), "Attempt membership differs"
    )
    require(
        packet["format"] == index["format"] == "k10-resolution-gpu1-raw-v1", "Wrong bundle format"
    )
    configs = {}
    for version, (commit, cfgsha) in PINS.items():
        files, binding = packet["attempts"][version], index["attempts"][version]
        require(set(files) == set(binding["files"]), "Evidence file set differs")
        for name, value in files.items():
            require(
                sha(value.encode()) == binding["files"][name]["sha256"]
                and len(value.encode()) == binding["files"][name]["bytes"],
                "Raw file hash differs",
            )
        config = configs[version] = document(files, "sealed-config.json")
        require(sha(files["sealed-config.json"].encode()) == cfgsha, "Frozen config differs")
        require(
            binding["source_commit"] == commit
            and binding["source_sha256"] == config["source_sha256"],
            "Source binding differs",
        )
        staged = document(files, "staging-readback.json")
        require(
            staged["source_commit"] == commit
            and staged["config_sha256"] == cfgsha
            and staged["archive_sha256"] == binding["archive_sha256"] == ARCHIVES[version],
            "Archive receipt differs",
        )
        for name, field in (
            ("runtime.json", "runtime_sha256"),
            ("protected.json", "protected_sha256"),
            ("launch-reviewed-probe.sh", "launch_script_sha256"),
        ):
            require(sha(files[name].encode()) == staged[field], "Runtime/launch receipt differs")
        require(document(files, "runtime.json") == config["runtime"], "Runtime metadata differs")
        require(document(files, "protected.json") == config["protected"], "Protected starts differ")
        approval = document(files, "approval-final.json")
        require(
            approval["authorized"] is True
            and approval["allow_resource_probe"] is True
            and approval["allow_training"] is False
            and approval["config_sha256"] == cfgsha
            and approval["source_commit"] == commit
            and approval["source_sha256"] == config["source_sha256"]
            and approval["archive_sha256"] == ARCHIVES[version]
            and approval["not_before_epoch"] == config["not_before_epoch"]
            and approval["deadline_epoch"] == config["deadline_epoch"],
            "Approval/source differs",
        )
        require(
            files["run-v1/config.json"] == files["sealed-config.json"]
            and files["run-v1/approval.json"] == files["approval-final.json"],
            "Executed bytes differ",
        )
        controller, terminal = (
            document(files, "run-v1/controller.json"),
            document(files, "terminal-receipt.json"),
        )
        for name, expected in terminal["raw_files_sha256"].items():
            require(sha(files[name].encode()) == expected, "Terminal raw-file binding differs")
        require(
            terminal["protected8_live_and_start_matched"]
            and terminal["owned_nvml_allocation_absent"],
            "Terminal protected/recovery proof missing",
        )
        require(
            controller["cleanup_error"] is None and controller["task_fit"] is False,
            "Cleanup/task-fit scope differs",
        )
        epoch = document(files, "actual-launch.json")["epoch"]
        require(
            config["not_before_epoch"] <= epoch < config["deadline_epoch"]
            and config["deadline_epoch"] - config["not_before_epoch"] == 300
            and epoch + controller["wall_seconds"] < config["deadline_epoch"] + 0.5,
            "Execution window exceeded",
        )
        samples = trace(files, config)
        first_worker = document(files, "run-v1/baseline-launch.json")["epoch"]
        require(
            first_worker - epoch >= 30
            and all(r["free_bytes"] >= 9 * GIB for r in samples if r["epoch"] < first_worker),
            "Passive admission differs",
        )
        peaks = controller["peaks"]
        require(
            peaks["owned_gpu_bytes"] == max(r["owned_bytes"] for r in samples)
            and peaks["min_gpu1_free_bytes"] == min(r["free_bytes"] for r in samples)
            and peaks["rss_bytes"] <= 4 * GIB
            and peaks["cpu_seconds"] <= 1200
            and controller["wall_seconds"] <= 300,
            "Controller peak/budget differs",
        )
        if version != "v3":
            needle = "Wrong visible GPU" if version == "v1" else "adaptive_avg_pool2d_backward_cuda"
            require(
                not controller["complete"]
                and needle in files["run-v1/baseline.log"]
                and "run-v1/candidate-launch.json" not in files
                and "run-v1/baseline/progress.json" not in files,
                "Setup-failure classification differs",
            )
        else:
            require(
                controller["complete"] and controller["error"] is None, "Canonical probe incomplete"
            )
            for model in INITIAL:
                result = document(files, f"run-v1/{model}/result.json")
                require(result["config_sha256"] == cfgsha, "Result config differs")
                workload(config, result, model)
                require(
                    document(files, f"run-v1/{model}/progress.json")["optimizer_commits"] == 3,
                    "Optimizer-commit receipt differs",
                )
                require(
                    document(files, f"run-v1/{model}-released.json")["released"] is True,
                    "Per-model release proof missing",
                )
    require(
        all(
            c["policy"] == configs["v1"]["policy"]
            and c["selection"] == configs["v1"]["selection"]
            and c["runtime"] == configs["v1"]["runtime"]
            for c in configs.values()
        ),
        "Resource/runtime/shape rescue occurred",
    )
    require(
        len(configs["v3"]["selection"]) == 376
        and len({r["id"] for r in configs["v3"]["selection"]}) == 376,
        "Selection identity differs",
    )
    require(len(configs["v3"]["source_sha256"]) == 18, "Source closure size differs")
    for name in (
        "experiments/k10_gpu1_resource.py",
        "tools/run_k10_gpu1_resource.py",
        "experiments/k10_resolution.py",
    ):
        require(
            sha((ROOT / name).read_bytes()) == configs["v3"]["source_sha256"][name],
            "Current qualification/model source changed from V3",
        )
    return packet


def plot(packet: dict[str, Any]) -> None:
    from tools.plot_research import BLUE, ORANGE, canvas, plt, save

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "svg.hashsalt": "k10-gpu1-v1"}
    )
    fig, axes = canvas(
        "GPU1 qualification: two setup failures, then bounded feasibility",
        3,
        "Synthetic data only · 30-second passive admission precedes workers · shaded spans delimit worker observation intervals.\n"
        "Full traces retained; sampled memory cannot bound sub-poll bursts or serving tail latency. V3 is not accuracy or ordinary-PC evidence.",
    )
    for axis, version, title in zip(
        axes,
        PINS,
        ("V1: UUID representation", "V2: global determinism", "V3: canonical CUDA policy"),
        strict=True,
    ):
        files = packet["attempts"][version]
        samples = [json.loads(line) for line in files["run-v1/samples.jsonl"].splitlines()]
        epoch = document(files, "actual-launch.json")["epoch"]
        x = [r["epoch"] - epoch for r in samples]
        axis.plot(x, [r["free_bytes"] / GIB for r in samples], color=BLUE, label="GPU1 free")
        axis.plot(
            x, [r["owned_bytes"] / GIB for r in samples], color=ORANGE, label="Owned GPU, sampled"
        )
        axis.axhline(6.5, color=BLUE, linestyle=":", linewidth=1, label="Free early-abort floor")
        axis.axhline(2.75, color=ORANGE, linestyle=":", linewidth=1, label="Owned GPU ceiling")
        for model in INITIAL:
            if f"run-v1/{model}-launch.json" not in files:
                continue
            start = document(files, f"run-v1/{model}-launch.json")["epoch"] - epoch
            end = (
                document(files, f"run-v1/{model}-released.json")["epoch"] - epoch
                if f"run-v1/{model}-released.json" in files
                else x[-1]
            )
            axis.axvspan(start, end, color="#64748b", alpha=0.10)
            axis.text(
                (start + end) / 2,
                4.2 if model == "baseline" else 3.3,
                "K10" if model == "baseline" else "Grid",
                ha="center",
                fontsize=8,
            )
        axis.set(
            title=title,
            xlabel="Seconds after guardian launch",
            ylabel="GiB",
            ylim=(0, 10.3),
            xlim=(0, 50),
        )
    axes[0].legend(frameon=False, loc="center left", fontsize=7)
    save(fig, "k10-capacity", "resolution-gpu1-qualifications")
    import matplotlib

    inputs = {
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "shared_helpers_sha256": sha((ROOT / "tools/plot_research.py").read_bytes()),
        "index_sha256": sha((STUDY / "data/resolution-gpu1-index.json").read_bytes()),
        "matplotlib_version": matplotlib.__version__,
    }
    for suffix in ("png", "svg"):
        inputs[f"figure_{suffix}_sha256"] = sha(
            (STUDY / f"figures/resolution-gpu1-qualifications.{suffix}").read_bytes()
        )
    (STUDY / "figures/resolution-gpu1-inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    packet = check()
    if args.plot:
        plot(packet)
    print("Verified three GPU1 attempts; only V3 completed synthetic qualification.")


if __name__ == "__main__":
    main()
