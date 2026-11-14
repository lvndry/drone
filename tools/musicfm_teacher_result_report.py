"""Reproduce the completed MusicFM pilot report from metadata, never models or audio."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import tarfile
from pathlib import Path
from typing import Any

from experiments.k10_v2_pilot import selection_key
from tools.musicfm_teacher_cost_report import INITIAL, LIMIT, STUDY, digest, require

DOMAINS = (("beatport", 126), ("fmak", 250))


def verify_bn_budget(report: dict[str, Any], study: Path = STUDY) -> dict[str, Any]:
    """Keep technical qualification separate from a never-opened accuracy result."""
    path = study / "data/bn-diagnostic/terminal-metadata.tar.gz"
    require(
        digest(path.read_bytes())
        == "8aabe764ce6a49ce1615bb526c9390a3628d4690b0153b1afb716d1b043fb1c1",
        "BN terminal metadata changed",
    )
    with tarfile.open(path) as archive:

        def read(name: str) -> dict[str, Any]:
            member = archive.extractfile(name)
            require(member is not None, "Missing BN evidence: " + name)
            assert member is not None
            return json.load(member)

        config = read("sealed-config.json")
        controller = read("run-v1/controller.json")
        result = read("run-v1/worker/result.json")
        qualification = read("run-v1/worker/qualification.json")
        admission = read("run-v1/worker/admission.json")
    require(
        config["source_commit"] == "6ea7ab810d2c42dde1ed761b7b723c41d30ab6d0",
        "Unexpected diagnostic source",
    )
    require(
        controller["complete"]
        and controller["error"] is None
        and controller["cleanup_error"] is None
        and controller["worker_exit"] == 0,
        "BN owner did not retire successfully",
    )
    require(
        result["status"] == "BN_DIAGNOSTIC_BUDGET_INCONCLUSIVE_NO_SCORING"
        and result["selection_scored"] is False
        and result["training_performed"] is False
        and result["promotion_authorized"] is False,
        "Do not reinterpret budget qualification as an accuracy result",
    )
    require(
        result["qualification"] == qualification and result["admission"] == admission,
        "BN result and separate receipts disagree",
    )
    projected = 512 * sum(qualification["site_max_seconds"])
    require(
        len(qualification["site_max_seconds"]) == 18
        and math.isclose(projected, admission["projected_calibration_seconds"], abs_tol=1e-9),
        "Calibration projection mismatch",
    )
    remaining = admission["remaining_seconds"]
    required = admission["required_seconds"]
    overhead = (
        required - 2 * projected - 2 * report["prospective_complete_comparison_seconds"] - 120
    )
    require(0 <= overhead <= controller["wall_seconds"], "Inconsistent measured overhead")
    require(
        required > remaining and admission["fits_budget"] is False,
        "Budget refusal must retain the declared safety factors",
    )
    require(
        qualification["max_absolute_difference"] == 0.0
        and qualification["correctness_prefixes"] == 36
        and qualification["full_traces"] == 2
        and qualification["timing_prefixes"] == 288,
        "Unexpected qualification evidence",
    )
    require(
        controller["wall_seconds"] <= config["caps"]["wall_seconds"]
        and controller["peaks"]["cpu"] <= config["caps"]["cpu_seconds"]
        and controller["peaks"]["rss"] <= config["caps"]["rss_bytes"],
        "Observed BN resource cap breach",
    )
    return dict(
        status=result["status"],
        required_seconds=required,
        remaining_seconds=remaining,
        projected_calibration_seconds=projected,
        wall_seconds=controller["wall_seconds"],
        peaks=controller["peaks"],
        correctness_prefixes=36,
        timing_prefixes=288,
        accuracy_opened=False,
    )


def load_evidence(study: Path = STUDY) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check stored bytes before parsing their independently preserved documents."""
    index = json.loads((study / "data/terminal-index.json").read_bytes())
    packed = (study / "data" / index["path"]).read_bytes()
    require(digest(packed) == index["sha256"], "compressed terminal evidence integrity")
    decoded = gzip.decompress(packed)
    require(digest(decoded) == index["decoded_sha256"], "decoded terminal evidence integrity")
    files = json.loads(decoded)["files"]
    require(set(files) == set(index["files"]), "terminal evidence inventory")
    documents = {}
    for name, entry in files.items():
        require(
            digest(entry["raw_utf8"].encode()) == entry["sha256"] == index["files"][name],
            "terminal document integrity: " + name,
        )
        if name.endswith(".json"):
            documents[name] = json.loads(entry["raw_utf8"])
    for name in ("config", "result"):
        require(
            index["files"][name + ".json"] == index["authoritative_" + name + "_sha256"],
            "authoritative " + name + " identity",
        )
    controls = {}
    for arm, entry in index["controls"].items():
        packed_control = (study / entry["path"]).read_bytes()
        require(
            digest(packed_control)
            == entry["sha256"]
            == documents["config.json"]["controls"][arm]["sha256"],
            "sealed comparator identity",
        )
        controls[arm] = json.loads(gzip.decompress(packed_control))
    documents["controls"] = controls
    return documents, index


def check_metrics(metrics: dict[str, Any], expected: dict[str, Any]) -> None:
    for domain, count in DOMAINS:
        row = metrics[domain]
        require(row["tracks"] == count, "metric track count")
        require(row["ids"] == expected[domain]["ids"], "metric membership or order")
        require(len(set(row["ids"])) == count, "unique metric identities")
        require(
            len(row["predictions"]) == count
            and all(
                type(prediction) is int and 0 <= prediction < 24
                for prediction in row["predictions"]
            ),
            "24-class prediction inventory",
        )
        for numerator, field, scale in (
            ("exact_correct", "exact", 1),
            ("symmetric_units10", "mirex", 10),
            ("asymmetric_units10", "mirex_asymmetric", 10),
        ):
            value = row[numerator]
            require(type(value) is int and 0 <= value <= count * scale, "metric numerator")
            require(
                math.isclose(row[field], value / (count * scale), rel_tol=0, abs_tol=1e-12),
                "metric fraction consistency",
            )


def check_proof(result: dict[str, Any]) -> dict[str, float | int]:
    proof = result["selected_teacher_composed_batch_proof"]
    reference = proof["reference_logits"]
    candidate = proof["candidate_logits"]
    songs = proof["per_song"]
    require(
        len(reference) == len(candidate) == len(songs) == proof["tracks"] == 376,
        "complete selected-checkpoint proof",
    )
    ids = [identity for domain, _ in DOMAINS for identity in result["best_metrics"][domain]["ids"]]
    predictions = [
        prediction
        for domain, _ in DOMAINS
        for prediction in result["best_metrics"][domain]["predictions"]
    ]
    require([song["id"] for song in songs] == ids, "proof membership order")
    require(
        [entry["domain"] for entry in proof["domain_schedule"]] == [name for name, _ in DOMAINS],
        "proof domain order",
    )
    for schedule, (domain, count) in zip(proof["domain_schedule"], DOMAINS, strict=True):
        require(
            schedule["tracks"] == count
            and schedule["ids"] == result["best_metrics"][domain]["ids"],
            "proof domain membership",
        )
    maximum = 0.0
    disagreements = 0
    for song, left, right, prediction in zip(songs, reference, candidate, predictions, strict=True):
        require(
            len(left) == len(right) == 24 and all(math.isfinite(logit) for logit in left + right),
            "finite 24-logit proof",
        )
        samples = song["samples"]
        require(type(samples) is int and samples > 1024, "proof waveform length")
        windows, tail = divmod(samples, 288000)
        require(song["windows"] == windows + bool(tail), "proof complete-song contexts")
        require(math.isfinite(song["seconds"]) and song["seconds"] > 0, "proof B1 timing")
        reference_class, candidate_class = (
            max(range(24), key=vector.__getitem__) for vector in (left, right)
        )
        require(reference_class == prediction, "selected metric prediction matches B1 proof")
        disagreements += reference_class != candidate_class
        maximum = max(
            maximum,
            max(abs(first - second) for first, second in zip(left, right, strict=True)),
        )
    require(
        sum(song["windows"] for song in songs) == proof["windows"] == 5893, "proof window total"
    )
    require(maximum == proof["max_absolute_logit_error"] <= LIMIT, "selected proof logit tolerance")
    require(
        disagreements == proof["argmax_disagreements"] == 0 and proof["passed"] is True,
        "selected proof decision",
    )
    return {
        "max_absolute_error": maximum,
        "disagreements": disagreements,
        "b1_seconds": sum(song["seconds"] for song in songs),
    }


def verify(documents: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
    """Validate recorded claims; this does not independently rescore reference labels."""
    config, result, controller, audit = (
        documents[name + ".json"] for name in ("config", "result", "controller", "audit")
    )
    require(
        documents["success.json"]["result_sha256"] == index["files"]["result.json"],
        "success binds actual result bytes",
    )
    require(
        documents["launch.json"]["config_sha256"]
        == result["provenance"]["config_sha256"]
        == index["files"]["config.json"],
        "launch/result configuration identity",
    )
    require(result["provenance"]["source_sha256"] == config["source_sha256"], "executed source map")
    require(result["provenance"]["initial_state_sha256"] == INITIAL, "initial state continuity")
    require(result["status"] == "COMPLETE_FIXED_PILOT_NOT_RELEASE_EVIDENCE", "terminal task status")
    require(
        result["updates"] == 8280 and result["events"] == len(result["history"]) == 120,
        "complete fixed horizon",
    )
    require(
        [event["update"] for event in result["history"]] == list(range(69, 8281, 69)),
        "selection cadence",
    )
    best = result["best_metrics"]
    b16_seconds = []
    proof_songs = result["selected_teacher_composed_batch_proof"]["per_song"]
    for event in result["history"]:
        check_metrics(event["metrics"], best)
        require(math.isfinite(event["train_ce"]) and event["train_ce"] >= 0, "finite training loss")
        exact, mirex = selection_key(event["metrics"])
        require(
            event["selection_exact"] == float(exact) and event["selection_mirex"] == float(mirex),
            "exact-first selection values",
        )
        contexts = [
            song for domain, _ in DOMAINS for song in event["validation"][domain]["per_song"]
        ]
        require(
            [(song["id"], song["samples"], song["windows"]) for song in contexts]
            == [(song["id"], song["samples"], song["windows"]) for song in proof_songs],
            "history/proof complete-song context identity",
        )
        timings = [event["validation"][domain]["elapsed_seconds"] for domain, _ in DOMAINS]
        require(
            all(math.isfinite(seconds) and seconds > 0 for seconds in timings),
            "finite B16 event timing",
        )
        b16_seconds.append(sum(timings))
    selected = max(result["history"], key=lambda event: selection_key(event["metrics"]))
    require(
        selected["update"] == result["best_update"] and selected["metrics"] == best,
        "earliest exact-first selected checkpoint",
    )
    checks = {}
    for arm, control in documents["controls"].items():
        check_metrics(control["best_metrics"], best)
        baseline = control["best_metrics"]
        checks[arm] = {
            "fma_exact_plus5": best["fmak"]["exact_correct"]
            >= baseline["fmak"]["exact_correct"] + 5,
            "bp_nondecline": all(
                best["beatport"][field] >= baseline["beatport"][field]
                for field in ("exact_correct", "symmetric_units10", "asymmetric_units10")
            ),
        }
    gate = result["continuation_gate"]
    require(gate["checks_against_each_control"] == checks, "control gate recomputation")
    require(
        gate["continuation_earned"]
        == all(all(arm_checks.values()) for arm_checks in checks.values()),
        "continuation decision",
    )
    require(
        not any(
            (
                result["calibration_used"],
                result["confirmation_used"],
                gate["confirmation_scored"],
                gate["release_or_distillation_authorized"],
            )
        ),
        "no calibration, confirmation, release or distillation",
    )
    require(
        controller["state"] == "COMPLETE_NO_RAM_OWNERSHIP"
        and controller["worker_exit"] == 0
        and controller["worker_started"] is True,
        "controller completion",
    )
    require(
        len(audit["failures"]) == 2
        and all(
            not failure["exists"] and not failure["symlink"]
            for failure in audit["failures"].values()
        ),
        "independent absence of both failure artifacts",
    )
    require(
        len(audit["original_processes"]) == 2
        and all(process is None for process in audit["original_processes"].values()),
        "original job retirement",
    )
    protected = {record["pid"]: record for record in audit["protected"]}
    require(
        set(protected) == {start["pid"] for start in config["protected_starts"]},
        "protected process inventory",
    )
    for expected in config["protected_starts"]:
        actual = protected[expected["pid"]]
        require(
            actual["preserved"]
            and actual["actual"]["pid"] == expected["pid"]
            and actual["actual"]["start_ticks"]
            == actual["expected_start"]
            == expected["start_ticks"]
            and actual["actual"]["state"] != "Z",
            "protected process identity/live state",
        )
    for kind in ("best", "final"):
        checkpoint = audit["checkpoints"][kind + ".pt"]
        require(
            checkpoint["sha256"]
            == checkpoint["expected_sha256"]
            == result[kind + "_checkpoint_sha256"]
            and checkpoint["matches"]
            and checkpoint["stable"]
            and checkpoint["identity_before"] == checkpoint["identity_after"],
            "independent actual checkpoint byte audit",
        )
    resources = {
        "wall_seconds": controller["wall_seconds"],
        "cpu_seconds": controller["peaks"]["cpu"],
        "rss_bytes": controller["peaks"]["rss"],
        "output_bytes": audit["output_bytes"],
    }
    for name, value in resources.items():
        require(
            math.isfinite(value) and 0 < value <= config["caps"][name],
            "recorded resource cap: " + name,
        )
    require(
        0
        < result["max_cuda_allocated"]
        <= result["max_cuda_reserved"]
        <= config["allocator_bytes"],
        "recorded Torch memory bounds",
    )
    proof = check_proof(result)
    return {
        "executed_source": index["executed_source"],
        "best_update": result["best_update"],
        "best_event": result["best_update"] // 69,
        "updates": result["updates"],
        "continuation_earned": gate["continuation_earned"],
        "control_checks": checks,
        "metrics": {
            "teacher": best,
            **{arm: control["best_metrics"] for arm, control in documents["controls"].items()},
        },
        "history": [
            {
                "update": event["update"],
                "train_ce": event["train_ce"],
                **{
                    domain: {
                        metric: event["metrics"][domain][metric] for metric in ("exact", "mirex")
                    }
                    for domain, _ in DOMAINS
                },
            }
            for event in result["history"]
        ],
        "resources": {
            **resources,
            "cuda_allocated_bytes": result["max_cuda_allocated"],
            "cuda_reserved_bytes": result["max_cuda_reserved"],
        },
        "caps": config["caps"],
        "proof": proof,
        "b16_event_seconds": b16_seconds,
        "b16_max_seconds": max(b16_seconds),
        "prospective_complete_comparison_seconds": 2 * proof["b1_seconds"] + max(b16_seconds),
        "audit_captured_utc": audit["captured_utc"],
    }


def plot(report: dict[str, Any], directory: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "musicfm-terminal-v1",
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    blue = "#2563eb"
    orange = "#c2410c"
    gray = "#64748b"

    def save(figure: Figure, name: str, caption: str) -> None:
        figure.text(0.06, 0.035, caption, fontsize=9, color="#475569")
        figure.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.20, wspace=0.30)
        for suffix in ("png", "svg"):
            metadata = {"Date": None} if suffix == "svg" else None
            figure.savefig(
                directory / f"{name}.{suffix}", dpi=170, metadata=metadata, bbox_inches="tight"
            )
            if suffix == "svg":
                path = directory / f"{name}.{suffix}"
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
                )
        plt.close(figure)

    history = report["history"]
    updates = [event["update"] for event in history]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    figure.suptitle(
        "Full MusicFM fit: early selection, no continuation earned", x=0.06, ha="left", fontsize=16
    )
    axes[0].plot(updates, [event["train_ce"] for event in history], color=gray)
    axes[0].set(title="Training cross-entropy", ylabel="Mean across preceding 69 updates")
    for axis, metric, title in zip(
        axes[1:], ("exact", "mirex"), ("Exact accuracy", "Symmetric MIREX"), strict=True
    ):
        for domain, label, color in (
            ("beatport", "Beatport · 126", blue),
            ("fmak", "FMA · 250", orange),
        ):
            axis.plot(
                updates,
                [100 * event[domain][metric] for event in history],
                label=label,
                color=color,
            )
        axis.set(title=title, ylabel="Percent")
        axis.legend(frameon=False, fontsize=9, loc="lower right")
    for axis in axes:
        axis.axvline(report["best_update"], color="#0f172a", ls="--", lw=1)
        axis.set_xlabel("Optimizer update")
        axis.grid(axis="y", alpha=0.15)
    save(
        figure,
        "teacher-learning",
        "120 recorded selection events · dashed line: selected update 828 / event 12 · full horizon 8,280 updates",
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    figure.suptitle(
        "Selected teacher trails both matched K10 controls", x=0.06, ha="left", fontsize=16
    )
    arms = ("restricted", "expanded", "teacher")
    names = ("K10 restricted", "K10 expanded", "MusicFM selected")
    for axis, (domain, count) in zip(axes, DOMAINS, strict=True):
        values = [100 * report["metrics"][arm][domain]["exact"] for arm in arms]
        axis.barh(names, values, color=(gray, blue, orange), height=0.55)
        for row, (arm, value) in enumerate(zip(arms, values, strict=True)):
            correct = report["metrics"][arm][domain]["exact_correct"]
            axis.text(
                value - 1.3,
                row,
                f"{correct}/{count} · {value:.2f}%",
                va="center",
                ha="right",
                fontsize=9,
                color="white",
            )
        threshold = 100 * (89 / 126 if domain == "beatport" else 181 / 250)
        axis.axvline(threshold, color="#0f172a", ls="--", lw=1)
        axis.set(
            xlim=(0, 88),
            xlabel="Exact accuracy (%)",
            title="Beatport" if domain == "beatport" else "FMA",
        )
        axis.invert_yaxis()
    figure.subplots_adjust(wspace=0.45)
    save(
        figure,
        "teacher-controls",
        "Dashed: BP exact nondecline (89/126); FMA +5 versus each control (181/250). BP also requires both weighted scores.",
    )

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    figure.suptitle(
        "Observed whole-job cost and recorded evaluation timings", x=0.06, ha="left", fontsize=16
    )
    resources = report["resources"]
    hours = [resources[field] / 3600 for field in ("wall_seconds", "cpu_seconds")]
    axes[0].barh(("Wall", "Owned CPU"), hours, color=(blue, gray))
    for row, value in enumerate(hours):
        axes[0].text(value + 0.15, row, f"{value:.2f} h", va="center")
    axes[0].set(xlim=(0, 12), xlabel="Hours", title="Guardian + worker scope")
    memories = [
        resources[field] / 2**30
        for field in ("rss_bytes", "cuda_allocated_bytes", "cuda_reserved_bytes")
    ]
    axes[1].barh(
        ("Host RSS", "CUDA allocated", "CUDA reserved"), memories, color=(gray, blue, orange)
    )
    for row, value in enumerate(memories):
        axes[1].text(value + 0.12, row, f"{value:.2f}", va="center", fontsize=9)
    axes[1].set(xlim=(0, 14), xlabel="GiB", title="Recorded peaks")
    axes[2].plot(updates, report["b16_event_seconds"], color=blue, lw=1.1)
    axes[2].axhline(
        report["proof"]["b1_seconds"], color=orange, ls="--", label="Selected B1 proof: 208.28 s"
    )
    axes[2].set(
        xlabel="Optimizer update",
        ylabel="Seconds / 376 tracks",
        ylim=(0, 235),
        title="120 B16 selection events",
    )
    axes[2].legend(frameon=False, fontsize=8, loc="lower right")
    save(
        figure,
        "teacher-cost",
        "Contended server; no model-only or product CPU benchmark. B1 proof is a separate phase, not a paired speedup measurement.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    report = verify(*load_evidence())
    if args.plot:
        plot(report, STUDY / "figures")
    concise = {
        key: value
        for key, value in report.items()
        if key not in ("history", "metrics", "b16_event_seconds")
    }
    concise["bn_diagnostic"] = verify_bn_budget(report)
    print(json.dumps(concise, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
