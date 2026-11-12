"""Verify immutable disposable MusicFM cost evidence; no Torch, audio or task scoring."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

STUDY = Path(__file__).parents[1] / "experiments/studies/musicfm-key-pilot"
LABELS = ("original-ef3a735", "cachefix-051e4ad")
LIMIT = 1e-5
INITIAL = "d278e0e197d21d98f8a570be7fbb6195e79247c5dbfada2686c34c6e502a507d"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_raw(study: Path) -> tuple[dict[str, str], dict[str, Any]]:
    index = json.loads((study / "data/cost-index.json").read_bytes())
    packed = (study / "data" / index["path"]).read_bytes()
    require(digest(packed) == index["sha256"], "compressed evidence integrity")
    raw = gzip.decompress(packed)
    require(digest(raw) == index["decoded_sha256"], "decoded evidence integrity")
    files = json.loads(raw)["files"]
    require(len(files) == index["files"], "evidence file count")
    for entry in files.values():
        require(digest(entry["raw_utf8"].encode()) == entry["sha256"], "raw file integrity")
    return {name: entry["raw_utf8"] for name, entry in files.items()}, index


def paired_vectors(result: dict[str, Any]) -> dict[str, Any]:
    proof = result["composed_batch_proof"]
    reference, candidate = proof["reference_logits"], proof["candidate_logits"]
    metadata = proof["per_song"]
    require(
        len(reference) == len(candidate) == len(metadata) == proof["tracks"] == 376,
        "complete 376 vectors required",
    )
    schedules = [result[k]["domain_schedule"] for k in ("validation", "batched_validation")]
    require(schedules[0] == schedules[1] == proof["domain_schedule"], "domain schedule differs")
    schedule = proof["domain_schedule"]
    require(
        [(r["domain"], r["tracks"]) for r in schedule] == [("beatport", 126), ("fmak", 250)],
        "domain membership counts",
    )
    ids = [identity for row in schedule for identity in row["ids"]]
    require(len(ids) == len(set(ids)) == 376, "duplicate or missing vector IDs")
    require(ids == [row["id"] for row in metadata], "vector context order")
    for panel in ("validation", "batched_validation"):
        require(
            [(r["id"], r["samples"], r["windows"]) for r in result[panel]["per_song"]]
            == [(r["id"], r["samples"], r["windows"]) for r in metadata],
            "actual B1/B16 window contexts differ",
        )
    rows = []
    disagreements = 0
    for meta, left, right in zip(metadata, reference, candidate, strict=True):
        require(
            len(left) == len(right) == 24 and all(math.isfinite(v) for v in left + right),
            "24 finite logits required",
        )
        samples = meta["samples"]
        require(type(samples) is int and samples > 1024, "actual sample count")
        windows, tail = divmod(samples, 288000)
        require(meta["windows"] == windows + bool(tail), "complete-song coverage")
        errors = [abs(a - b) for a, b in zip(left, right, strict=True)]
        worst = max(range(24), key=errors.__getitem__)
        disagreements += max(range(24), key=left.__getitem__) != max(
            range(24), key=right.__getitem__
        )
        rows.append(
            {
                "id": meta["id"],
                "samples": samples,
                "windows": meta["windows"],
                "tail_samples": tail,
                "max_absolute_error": max(errors),
                "failed": max(errors) > LIMIT,
                "worst_class": worst,
                "reference_logit": left[worst],
                "candidate_logit": right[worst],
            }
        )
    maximum = max(r["max_absolute_error"] for r in rows)
    passed = maximum <= LIMIT and disagreements == 0
    require(sum(r["windows"] for r in rows) == proof["windows"] == 5893, "window total")
    require(
        maximum == proof["max_absolute_logit_error"]
        and disagreements == proof["argmax_disagreements"]
        and passed == proof["passed"],
        "reported composed gate disagrees with vectors",
    )
    require(proof["label_blind"] and proof["task_scores_computed"] is False, "label-blind scope")
    return {
        "passed": passed,
        "max_absolute_error": maximum,
        "argmax_disagreements": disagreements,
        "failed_tracks": sum(r["failed"] for r in rows),
        "per_song": rows,
    }


def summarize(study: Path = STUDY) -> dict[str, Any]:
    raw, index = load_raw(study)
    measured: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        result = json.loads(raw[label + "/result.json"])
        config = json.loads(raw[label + "/sealed-config.json"])
        controller = json.loads(raw[label + "/controller.json"])
        terminal = json.loads(raw[label + "/cost-terminal-receipt.json"])
        config_sha = digest(raw[label + "/sealed-config.json"].encode())
        result_sha = digest(raw[label + "/result.json"].encode())
        require(
            result_sha == index["authoritative_result_sha256"][label] == terminal["result_sha256"],
            "authoritative result binding",
        )
        require(
            config_sha == result["provenance"]["config_sha256"] == terminal["config_sha256"],
            "config binding",
        )
        require(
            digest(raw[label + "/controller.json"].encode()) == terminal["controller_sha256"],
            "controller binding",
        )
        require(
            result["status"] == "DISPOSABLE_COST_ONLY"
            and result["synthetic_target_updates"] == 3
            and result["accumulation"] == 32
            and len(result["update_seconds"]) == 3
            and result["task_scores_computed"] is False
            and result["weights_saved_or_reused"] is False,
            "disposable-only scope",
        )
        require(
            result["provenance"]["initial_state_sha256"]
            == index["initial_state_sha256"]
            == INITIAL,
            "original initialization binding",
        )
        require(
            controller["worker_exit"] == 0
            and controller["state"] == "COMPLETE_NO_RAM_OWNERSHIP"
            and terminal["allow_training"] is False
            and terminal["protected_starts_verified"],
            "terminal/scope/protected process receipt",
        )
        require(result["provenance"]["source_sha256"] == config["source_sha256"], "source binding")
        require(
            result["provenance"]["input_provider"] == config["input_provider"] == "disk"
            and result["provenance"]["eval_batch"] == config["eval_batch_size"] == 16,
            "measured provider/evaluation path",
        )
        proof = paired_vectors(result)
        require(
            proof["passed"] == terminal["passed"]
            and proof["max_absolute_error"] == terminal["max_absolute_logit_error"],
            "independent terminal proof",
        )
        steady = statistics.mean(result["update_seconds"][1:])
        require(all(math.isfinite(v) and v > 0 for v in result["update_seconds"]), "update times")
        projection = (
            result["update_seconds"][0]
            + 8279 * steady
            + 120 * result["batched_validation"]["elapsed_seconds"]
        )
        projection += result["cache_verify_seconds"] + result["model_load_seconds"]
        projection += (
            result["validation"]["elapsed_seconds"]
            + result["batched_validation"]["elapsed_seconds"]
        )
        measured[label] = {
            "source_commit": index["executed_sources"][label],
            "result_sha256": result_sha,
            "config_sha256": config_sha,
            "initial_state_sha256": INITIAL,
            "update_seconds": result["update_seconds"],
            "steady_two_mean_seconds": steady,
            "training_pipeline_seconds": result["training_pipeline_seconds"],
            "input_wait_seconds_per_update": result["input_wait_seconds_per_update"],
            "validation_b1_seconds": result["validation"]["elapsed_seconds"],
            "validation_b16_seconds": result["batched_validation"]["elapsed_seconds"],
            "validation_both_cpu_seconds": result["validation_cpu_seconds"],
            "cache_verify_seconds": result["cache_verify_seconds"],
            "model_load_seconds": result["model_load_seconds"],
            "worker_elapsed_seconds": result["elapsed_seconds"],
            "controller_wall_seconds": controller["wall_seconds"],
            "controller_aggregate_cpu_seconds": controller["peaks"]["cpu"],
            "controller_tree_peak_rss_bytes": controller["peaks"]["rss"],
            "cuda_allocated_bytes": result["max_cuda_allocated"],
            "cuda_reserved_bytes": result["max_cuda_reserved"],
            "model_steady_plus_validation_projection_hours": projection / 3600,
            "projection_measured": False,
            "composed_gate": proof,
        }
        configs[label] = config
    old, fixed = (configs[label] for label in LABELS)
    for field in (
        "plan_sha256",
        "labels_sha256",
        "cache_manifest_sha256",
        "identity_overlay_sha256",
        "disk_cache_manifest_sha256",
        "weight_sha256",
        "module_receipts",
        "eval_batch_size",
    ):
        require(old[field] == fixed[field], "comparison invariant changed: " + field)
    require(set(old["source_sha256"]) == set(fixed["source_sha256"]), "source closure differs")
    changed = [
        name
        for name in old["source_sha256"]
        if old["source_sha256"][name] != fixed["source_sha256"][name]
    ]
    require(changed == ["experiments/musicfm_teacher.py"], "unexpected numerical source change")
    require(
        measured[LABELS[0]]["composed_gate"]["passed"] is False
        and measured[LABELS[1]]["composed_gate"]["passed"] is True,
        "preserved failed gate/corrected pass",
    )
    stopped = json.loads(raw["controlled-stop/controlled-stop-terminal-receipt.json"])
    stop = json.loads(raw["controlled-stop/root-controlled-stop.json"])
    require(
        digest(raw["controlled-stop/controller.json"].encode()) == stopped["controller_sha256"],
        "controlled-stop controller binding",
    )
    require(
        stopped["status"] == "ROOT_CONTROLLED_STOP_INCOMPLETE_NO_COST_OR_TRAINING"
        and stopped["result_absent"]
        and stopped["no_completed_update_or_validation_claim"]
        and stopped["worker_exit"] == -15
        and stopped["protected_starts_verified"],
        "controlled stop must remain incomplete",
    )
    require(
        digest(raw["controlled-stop/root-controlled-stop.json"].encode())
        == stopped["root_stop_sha256"]
        and stop["no_fit_or_retry_authorized"],
        "controlled stop binding",
    )
    return {
        "format": "musicfm-cost-summary-v1",
        "executions": measured,
        "controlled_stop": {
            "complete": False,
            "wall_seconds": stopped["wall_seconds"],
            "worker_exit": stopped["worker_exit"],
        },
        "real_task_accuracy_measured": False,
        "training_authorized_by_this_report": False,
    }


def plot(summary: dict[str, Any], study: Path = STUDY) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "svg.hashsalt": "musicfm-cost-v1"}
    )
    old, fixed = (summary["executions"][label] for label in LABELS)
    colors = ("#a66346", "#2d7169")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.65), layout="constrained")
    for offset, result, color, label in zip(
        (-0.17, 0.17), (old, fixed), colors, ("Original cache", "Corrected cache"), strict=True
    ):
        axes[0].bar(
            [v + offset for v in range(3)],
            result["update_seconds"],
            width=0.32,
            color=color,
            label=label,
        )
    axes[0].set(
        xticks=[0, 1, 2],
        xticklabels=["First", "Second", "Third"],
        ylabel="Seconds / 32-microbatch update",
        title="Three disposable synthetic-target updates",
    )
    axes[0].legend(fontsize=8)
    axes[1].bar(
        [0, 1],
        [fixed["validation_b1_seconds"], fixed["validation_b16_seconds"]],
        color=["#637a99", colors[1]],
    )
    for i, value in enumerate((fixed["validation_b1_seconds"], fixed["validation_b16_seconds"])):
        axes[1].text(i, value + 4, f"{value:.2f}s", ha="center")
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["B1", "B1 frontend / B16 encoder"],
        ylabel="Seconds / complete 376-song pass",
        title="Corrected-cache run: 5,893 whole-song windows",
        ylim=(0, 1.2 * max(fixed["validation_b1_seconds"], fixed["validation_b16_seconds"])),
    )
    ordered = sorted(
        old["composed_gate"]["per_song"], key=lambda r: r["max_absolute_error"], reverse=True
    )
    other = {r["id"]: r for r in fixed["composed_gate"]["per_song"]}
    for rows, color, label in (
        (ordered, colors[0], "Original: failed"),
        ([other[r["id"]] for r in ordered], colors[1], "Corrected: passed"),
    ):
        axes[2].scatter(
            range(1, 377),
            [max(r["max_absolute_error"], 1e-10) for r in rows],
            s=8,
            alpha=0.7,
            color=color,
            label=label,
        )
    axes[2].axhline(LIMIT, color="#444444", linestyle="--", linewidth=1, label="Fixed 1e-5 gate")
    axes[2].set(
        yscale="log",
        xlabel="Same 376 songs, sorted by original error",
        ylabel="Maximum absolute pooled-logit error",
        title="B1 vs B16: zero changed keys in both",
    )
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.savefig(study / "cost-results.png", dpi=180, metadata={"Software": "MusicFM cost report"})
    fig.savefig(
        study / "cost-results.svg", metadata={"Date": None, "Creator": "MusicFM cost report"}
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    summary = summarize()
    if args.plot:
        plot(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
