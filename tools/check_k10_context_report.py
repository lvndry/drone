"""Verify completed context-study evidence and reproduce metadata analysis; no inference."""

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from experiments.k10_context_pair import compare, metrics, movements, uncertainty
from experiments.k10_v2_pilot import selection_key

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/studies/k10-context/data"
CONFIG = "da13456c0bb8d992b81a73e422ce4cf74bc8c1b8afb126adb4cbcdfc55f72925"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def checked_inputs(data: Path = DATA) -> dict[str, Any]:
    index = json.loads((data / "archive-index.json").read_bytes())
    result: dict[str, Any] = {}
    for name, binding in index["files"].items():
        compressed = (data / name).read_bytes()
        raw = gzip.decompress(compressed)
        assert sha(compressed) == binding["sha256"], name
        assert sha(raw) == binding["decoded_sha256"], name
        assert len(raw) == binding["decoded_bytes"], name
        if name.endswith(".json.gz"):
            result[name.removesuffix(".json.gz")] = json.loads(raw)
    source = gzip.decompress((data / "executed-context-source.py.gz").read_bytes())
    assert sha(source) == result["pair-config"]["source_sha256"]["experiments/k10_context_pair.py"]
    # Current publication source can add runtime adapters; the analysis functions must be identical.
    for name in ["metrics", "compare", "movements", "uncertainty"]:

        def function(raw: bytes) -> str:
            return ast.dump(
                next(
                    n
                    for n in ast.parse(raw).body
                    if isinstance(n, ast.FunctionDef) and n.name == name
                ),
                include_attributes=False,
            )

        assert function(source) == function((ROOT / "experiments/k10_context_pair.py").read_bytes())
    assert index["files"]["pair-config.json.gz"]["decoded_sha256"] == CONFIG
    result["_index"] = index
    return result


def verify(v: dict[str, Any]) -> dict[str, Any]:
    a, b, pair, config, post = (
        v[k] for k in ["arm-320", "arm-65", "pair-result", "pair-config", "postflight"]
    )
    assert post["config_sha256"] == CONFIG
    assert post["source_hashes"] == config["source_sha256"]
    assert all(x is None for x in post["processes"].values())
    assert post["failures"] == []
    assert all(
        x["actual"]["start_ticks"] == x["expected"]["start_ticks"] for x in post["protected"]
    )
    historical_path = ROOT / "experiments/studies/k10-v2-pilot/data/results/expanded.json.gz"
    assert sha(historical_path.read_bytes()) == config["historical_gpu_control_sha256"]
    assert post["historical_sha256"] == config["historical_gpu_control_sha256"]
    old = json.loads(gzip.decompress(historical_path.read_bytes()))
    rows = {r["id"]: r for r in config["rows"]}
    group_path = (
        ROOT
        / "experiments/studies/k10-v2-pilot/data/canonical-eligibility-v2/uncertainty-groups-v2.csv.gz"
    )
    assert sha(group_path.read_bytes()) == config["groups_sha256"]
    groups = {
        r["id"]: r["derived_uncertainty_group"]
        for r in csv.DictReader(io.StringIO(gzip.decompress(group_path.read_bytes()).decode()))
    }
    refs = {i: r["ref"] for i, r in rows.items()}
    for n, arm in [(320, a), (65, b)]:
        assert arm["provenance"]["config_sha256"] == CONFIG
        assert len(arm["coverage"]) == 5262 and sum(arm["coverage"].values()) == 8280 * 32
        assert arm["provenance"]["source_sha256"] == config["source_sha256"]
        for kind in ["best", "final"]:
            assert post["checkpoints"][f"{n}/{kind}"]["sha256"] == arm[f"{kind}_checkpoint_sha256"]
        proof = arm["selected_reload_proof"]
        assert proof["checkpoint_sha256"] == arm["best_checkpoint_sha256"]
        assert proof["frames"] == 372532
        assert len(arm["history"]) == 120
        assert [e["update"] for e in arm["history"]] == list(range(69, 8281, 69))
        selected = max(arm["history"], key=lambda e: selection_key(e["metrics"]))
        assert selected["update"] == arm["best_update"]
        assert selected["metrics"] == arm["best_metrics"]
        for d, m in arm["best_metrics"].items():
            assert metrics(m["ids"], m["predictions"], refs) == m, d
    assert a["bp_augmented_sha256"] == old["bp_trace_sha256"]
    assert a["temporal_sha256"] != b["temporal_sha256"]
    assert compare(a, b, old) == pair["gate"]
    assert uncertainty(a, b, refs, groups) == pair["uncertainty"]
    assert movements(a, b, rows, groups) == pair["paired_errors_and_duration"]
    controller = v["controller"]
    assert controller["complete"] is True and controller["error"] is None
    assert controller["worker_exit"] == 0
    assert controller["wall_seconds"] <= 21600 and controller["peaks"]["cpu"] <= 40000
    assert controller["peaks"]["rss"] <= 8 * 1024**3
    assert controller["teacher_state"]["state"] == "VERIFIED_LEGITIMATE_TERMINAL"
    assert controller["teacher_state"]["events"] == 120
    for n in [320, 65]:
        raw_index = v["_index"]
        assert (
            controller["completed_arm_records"][str(n)]
            == raw_index["files"][f"arm-{n}.json.gz"]["decoded_sha256"]
        )
    cost = v["cost-result"]
    assert v["_index"]["files"]["cost-result.json.gz"]["decoded_sha256"] == config["cost_sha256"]
    assert (
        v["_index"]["files"]["cost-controller.json.gz"]["decoded_sha256"]
        == config["cost_controller_sha256"]
    )
    assert v["cost-controller"]["complete"] is True and v["cost-controller"]["worker_exit"] == 0
    assert v["cost-controller"]["error"] is None
    assert cost["status"] and cost["task_scores_computed"] is False
    assert cost["weights_retained"] is False
    assert [x["crop_frames"] for x in cost["arms"]] == [320, 65]
    for arm in cost["arms"]:
        assert arm["optimizer_commits"] == 3 and arm["batch"] == 32
        assert arm["validation"]["tracks"] == 376 and arm["validation"]["frames"] == 372532
        assert arm["task_scores_computed"] is False and arm["predictions_retained"] is False
    return {
        "gate": pair["gate"],
        "uncertainty": pair["uncertainty"],
        "movement_counts": pair["paired_errors_and_duration"]["domains"],
        "resource": {"wall_seconds": controller["wall_seconds"], "peaks": controller["peaks"]},
    }


def check() -> dict[str, Any]:
    return verify(checked_inputs())


def main() -> None:
    print(json.dumps(check(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
