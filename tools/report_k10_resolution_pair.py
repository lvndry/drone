"""Reproduce recorded pair arithmetic and figures, with no model/audio execution."""

from __future__ import annotations

import argparse
import gzip
import json
import runpy
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from tools.report_k10_gpu1 import BACKEND, INITIAL, require, sha

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/studies/k10-capacity"
SOURCE = "72ac56507ebfab8f3a2a4d1ce1e572ccfc5fd3bd"
CONFIG = "e8c03fff868d55cbbc28e9d971a4f9abab22a5a7667896630cda3dad1a3bfbad"
ARCHIVE = "966eb60d4ad82e0ba9e9aa8478c75e7e3d1cbd6cd9376160ced86576c9fad47d"


def validate_arm(result: dict[str, Any], model: str, config: dict[str, Any]) -> None:
    require(
        result["updates"] == 8280 and result["events"] == len(result["history"]) == 120,
        "Incomplete fixed horizon",
    )
    require(result["initial_state_sha256"] == INITIAL[model], "Initial state differs")
    require(
        result["provenance"]["config_sha256"] == CONFIG
        and result["provenance"]["source_sha256"] == config["source_sha256"]
        and result["provenance"]["backend_flags"] == BACKEND,
        "Source/backend binding differs",
    )
    require(
        [r["update"] for r in result["history"]] == list(range(69, 8281, 69)),
        "Selection cadence differs",
    )

    def key(row: dict[str, Any]) -> tuple[int, int, int]:
        a, b = row["metrics"]["beatport"], row["metrics"]["fmak"]
        return (
            250 * a["exact_correct"] + 126 * b["exact_correct"],
            250 * a["symmetric_units10"] + 126 * b["symmetric_units10"],
            -row["update"],
        )

    best = max(result["history"], key=key)
    require(
        best["update"] == result["best_update"] and best["metrics"] == result["best_metrics"],
        "Exact-first earliest selection differs",
    )
    proof = result["selected_reload_proof"]
    require(
        proof["strict_state_load"]
        and proof["scores_identical"]
        and proof["argmax_disagreements"] == 0
        and proof["tracks"] == 376
        and proof["frames"] == 372532
        and proof["checkpoint_sha256"] == result["best_checkpoint_sha256"],
        "Reload proof differs",
    )
    require(
        not result["confirmation_used"] and not result["calibration_used"], "Reserved data used"
    )


def check() -> dict[str, Any]:
    path = STUDY / "data/resolution-pair-raw.tar.gz"
    require(sha(path.read_bytes()) == ARCHIVE, "Raw archive differs")
    with tarfile.open(path) as archive:
        files = {}
        for member in archive.getmembers():
            require(
                member.isfile()
                and not Path(member.name).is_absolute()
                and ".." not in Path(member.name).parts
                and member.name.endswith((".json", ".jsonl", ".log", ".sh")),
                "Unexpected raw member",
            )
            stream = archive.extractfile(member)
            require(stream is not None, "Missing archive member")
            assert stream is not None
            files[member.name] = stream.read()

    def load(name: str) -> dict[str, Any]:
        return json.loads(files[name])

    audit = load("post-run-receipt.json")
    for name, record in audit["raw_files"].items():
        require(
            sha(files[name]) == record["sha256"] and len(files[name]) == record["bytes"],
            "Recorded file hash differs",
        )
    config, controller, pair = (
        load(n) for n in ("sealed-config.json", "run-v1/controller.json", "run-v1/result.json")
    )
    require(
        sha(files["sealed-config.json"]) == CONFIG and audit["source_commit"] == SOURCE,
        "Executed source/config differs",
    )
    require(
        controller["complete"]
        and controller["error"] is None
        and controller["cleanup_error"] is None
        and audit["all8protected_start_match"]
        and audit["guardian_absent"]
        and all(audit["workers_absent"].values())
        and audit["gpu"]["owned_bytes"] == 0,
        "Terminal/recovery proof differs",
    )
    arms = {model: load(f"run-v1/{model}/result.json") for model in INITIAL}
    for model, result in arms.items():
        require(
            sha(files[f"run-v1/{model}/result.json"]) == pair["arms"][model],
            "Arm result binding differs",
        )
        validate_arm(result, model, config)
    for field in (
        "coverage",
        "actual_frames",
        "common_draw_pitch_sha256",
        "temporal_sha256",
        "bp_augmented_sha256",
    ):
        require(arms["baseline"][field] == arms["candidate"][field], "Paired exposure differs")
    require(
        len(arms["baseline"]["coverage"]) == 5262 and all(audit["paired_traces"].values()),
        "Incomplete fitting coverage",
    )
    # Reuse the hash-bound shared score contract without importing drone.__init__/Torch.
    require(
        sha((ROOT / "drone/keys.py").read_bytes()) == config["source_sha256"]["drone/keys.py"],
        "Scoring source differs",
    )
    score = runpy.run_path(str(ROOT / "drone/keys.py"))["score_mirex"]
    records = pair["error_movements"]["selected_per_ID"]
    by_id = {r["id"]: r for r in records}
    canonical = {r["id"]: r for r in config["rows"]}
    require(
        len(by_id) == len(records) == 376
        and all(r["ref"] == canonical[i]["ref"] for i, r in by_id.items()),
        "Selected references differ from admitted metadata",
    )
    for domain, count in (("beatport", 126), ("fmak", 250)):
        ids = arms["baseline"]["best_metrics"][domain]["ids"]
        require(len(ids) == len(set(ids)) == count, "Selection denominator differs")
        values = []
        for model in INITIAL:
            metrics = arms[model]["best_metrics"][domain]
            predictions = metrics["predictions"]
            field = "control_prediction" if model == "baseline" else "candidate_prediction"
            require(
                predictions == [by_id[i][field] for i in ids], "Paired prediction binding differs"
            )
            refs = [by_id[i]["ref"] for i in ids]
            exact, symmetric = score(predictions, refs)
            asymmetric = score(predictions, refs, symmetric=False)[1]
            require(
                metrics["ids"] == ids
                and sum(a == b for a, b in zip(predictions, refs, strict=True))
                == metrics["exact_correct"]
                and np.allclose(
                    [exact, symmetric, asymmetric],
                    [metrics["exact"], metrics["mirex"], metrics["mirex_asymmetric"]],
                    atol=1e-14,
                    rtol=0,
                ),
                "Selected scores differ",
            )
            values.append(
                np.array(
                    [
                        [p == r, score([p], [r])[1], score([p], [r], symmetric=False)[1]]
                        for p, r in zip(predictions, refs, strict=True)
                    ],
                    dtype=float,
                )
            )
        delta = values[1] - values[0]
        names = sorted({by_id[i]["uncertainty_group"] for i in ids})
        groups = [
            [n for n, i in enumerate(ids) if by_id[i]["uncertainty_group"] == name]
            for name in names
        ]
        sums, sizes = (
            np.array([delta[g].sum(0) for g in groups]),
            np.array([len(g) for g in groups]),
        )
        rng = np.random.default_rng(20261112)
        draws = [
            sums[take].sum(0) / sizes[take].sum()
            for take in (rng.integers(0, len(names), len(names)) for _ in range(10000))
        ]
        bounds = np.quantile(draws, [0.025, 0.975], axis=0)
        recorded = pair["uncertainty"]["domains"][domain]
        require(recorded["clusters"] == len(names), "Cluster count differs")
        for n, metric in enumerate(("exact", "symmetric", "asymmetric")):
            require(
                np.allclose(bounds[:, n], recorded["metrics"][metric]["ci95"], atol=1e-14, rtol=0)
                and np.isclose(
                    delta[:, n].mean(), recorded["metrics"][metric]["delta"], atol=1e-14, rtol=0
                ),
                "Clustered interval differs",
            )
    gate = pair["continuation_gate"]
    historical_path = STUDY.parent / "k10-v2-pilot/data/results/expanded.json.gz"
    require(
        sha(historical_path.read_bytes()) == config["historical_gpu_control_sha256"],
        "Historical baseline artifact differs",
    )
    historical = json.loads(gzip.decompress(historical_path.read_bytes()))["best_metrics"]
    gaps = {
        d: {
            k: arms["baseline"]["best_metrics"][d][k] - historical[d][k]
            for k in ("exact_correct", "symmetric_units10", "asymmetric_units10")
        }
        for d in ("beatport", "fmak")
    }
    require(gaps == gate["historical_gaps"], "Historical guard arithmetic differs")
    require(
        gate["FMA_exact_gain"] == -15
        and not gate["resolution_gate"]
        and not gate["historical_GPU_control_noninferiority_guard"]
        and not gate["continuation_earned"]
        and not gate["promotion_authorized"],
        "Failed gate changed",
    )
    samples = [json.loads(line) for line in files["run-v1/samples.jsonl"].splitlines()]
    active = {
        model: [
            r
            for r in samples
            if load(f"run-v1/{model}-launch.json")["epoch"]
            <= r["epoch"]
            <= load(f"run-v1/{model}-released.json")["epoch"]
        ]
        for model in INITIAL
    }
    cadence = {
        model: {
            k: max(row[k] for row in rows)
            for k in ("poll_gap_seconds", "sensor_completion_gap_seconds")
        }
        for model, rows in active.items()
    }
    require(
        all(v <= 0.2 for row in cadence.values() for v in row.values()),
        "Active monitoring gap exceeded",
    )
    return dict(
        arms=arms,
        pair=pair,
        controller=controller,
        cadence=cadence,
        source_sha256=config["source_sha256"],
        raw_files_verified=len(audit["raw_files"]),
    )


def plot(data: dict[str, Any]) -> None:
    from tools.plot_research import BLUE, ORANGE, canvas, plt, save

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "resolution-pair-v1"}
    )
    fig, axes = canvas(
        "Finer frequency grid: the fixed accuracy gate failed",
        2,
        "Fresh matched GPU pair · seed 13 · 8,280 updates and 120 selection events per arm · reused adaptive development data.\n"
        "Intervals: 10,000 metadata-cluster bootstrap draws. FMA: 250/100 groups; BP: 126 record proxies, artist independence unknown.",
    )
    for model, color, label in (("baseline", BLUE, "K10"), ("candidate", ORANGE, "Finer grid")):
        arm = data["arms"][model]
        history = arm["history"]
        axes[0].plot(
            [r["update"] for r in history],
            [
                50 * (r["metrics"]["beatport"]["exact"] + r["metrics"]["fmak"]["exact"])
                for r in history
            ],
            color=color,
            label=label,
        )
        axes[0].axvline(arm["best_update"], color=color, linestyle="--", alpha=0.6)
    axes[0].set(
        xlabel="Optimizer updates",
        ylabel="Equal-domain selection exact (%)",
        title="Same horizon; selected checkpoints differ",
    )
    axes[0].legend(frameon=False)
    for i, domain in enumerate(("beatport", "fmak")):
        row = data["pair"]["uncertainty"]["domains"][domain]["metrics"]["exact"]
        value = 100 * row["delta"]
        lo, hi = [100 * x for x in row["ci95"]]
        axes[1].errorbar(
            i, value, yerr=[[value - lo], [hi - value]], fmt="o", capsize=5, color=BLUE
        )
        axes[1].annotate(
            f"{value:+.2f} pp", (i, value), xytext=(12, 0), textcoords="offset points", va="center"
        )
    axes[1].axhline(0, color="#64748b", linewidth=1)
    axes[1].hlines(2, 0.7, 1.3, color=ORANGE, linestyle="--", label="FMA continuation hurdle")
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Beatport", "FMAK"],
        ylabel="Candidate − K10 exact (percentage points)",
        title="Descriptive paired accuracy change",
        xlim=(-0.5, 1.6),
        ylim=(-11, 9),
    )
    axes[1].legend(frameon=False, fontsize=9)
    save(fig, "k10-capacity", "resolution-pair-accuracy")
    (STUDY / "figures/resolution-pair-inputs.json").write_text(
        json.dumps(
            dict(
                generator_sha256=sha(Path(__file__).read_bytes()),
                archive_sha256=ARCHIVE,
                source_commit=SOURCE,
            ),
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    data = check()
    if args.plot:
        plot(data)
    print(
        "Verified 24 raw files, both 8280/120/reload proofs, scores and clustered intervals; no continuation."
    )


if __name__ == "__main__":
    main()
