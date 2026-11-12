"""Synthetic-only driver integration: no upstream model, decoder, audio or GPU work."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

import experiments.musicfm_pilot as pilot
from experiments.k10_v2_pilot import Draw
from experiments.musicfm_inputs import ordered_batches
from experiments.musicfm_teacher import KeyTeacher
from tools.musicfm_crop_io import save


class FakeProvider:
    def __init__(self) -> None:
        self.submitted = 0
        self.resolved = 0
        self.peak_pending = 0

    def submit(self, requests: list[tuple[str, int, int]]) -> list[Future[Any]]:
        self.submitted += 1
        self.peak_pending = max(self.peak_pending, self.submitted - self.resolved)
        result = []
        for identity, start, length in requests:
            f: Future[Any] = Future()
            f.set_result(np.full(1200, 0.25, dtype="<f4"))
            result.append(f)
        return result

    def resolve(self, ticket: list[Future[Any]]) -> list[Any]:
        self.resolved += 1
        return [f.result() for f in ticket]


def test_ordered_prefetch_preserves_requests_and_only_one_next_batch() -> None:
    rows = {"b": {"samples": 288000}, "f": {"samples": 288000}}
    source = [
        [Draw("b" if i % 2 else "f", "domain", 32 * u + i) for i in range(32)] for u in range(3)
    ]
    provider = FakeProvider()
    seen = []
    for draws, arrays in ordered_batches(source, rows, provider):
        seen.append(draws)
        assert len(arrays) == 32
        assert provider.submitted - provider.resolved <= 1
        arrays.clear()
    assert seen == source
    assert provider.submitted == provider.resolved == 3
    assert provider.peak_pending <= 1


def toy() -> KeyTeacher:
    return cast(KeyTeacher, torch.nn.Linear(1, 1))


def patch_training(monkeypatch: pytest.MonkeyPatch) -> tuple[list[list[int]], list[bool]]:
    targets: list[list[int]] = []
    evaluations = []
    monkeypatch.setattr(
        pilot, "make_optimizer", lambda model: torch.optim.AdamW(model.parameters())
    )

    def update(
        model: KeyTeacher,
        optimizer: torch.optim.Optimizer,
        examples: list[tuple[torch.Tensor, int]],
    ) -> float:
        targets.append([y for _, y in examples])
        optimizer.step()
        return 1.0

    monkeypatch.setattr(pilot, "accumulated_update", update)
    monkeypatch.setattr(pilot, "finite_state", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def panel(
        model: KeyTeacher,
        ids: list[str],
        rows: dict[str, dict[str, Any]],
        watch: Callable[[], None],
        *,
        collect_predictions: bool,
        collect_logits: bool = False,
        batch_size: int = 1,
    ) -> dict[str, Any]:
        evaluations.append(collect_predictions)
        return {
            "tracks": len(ids),
            "windows": len(ids),
            "elapsed_seconds": 0.1,
            "per_song": [],
            **({"predictions": [0] * len(ids)} if collect_predictions else {}),
        }

    monkeypatch.setattr(pilot, "evaluate_panel", panel)
    return targets, evaluations


def test_disposable_probe_uses96synthetic_targets_and_no_task_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets, evals = patch_training(monkeypatch)
    ids = [str(i) for i in range(16)]
    rows = {i: {"role": "fit", "samples": 288000, "target": 999} for i in ids}
    result = pilot.probe(toy(), rows, ids, ["selection"] * 376, lambda: None, FakeProvider())  # type: ignore[arg-type]
    assert targets == [[((32 * u + i) % 24) for i in range(32)] for u in range(3)]
    assert evals == [False]
    assert result["synthetic_target_updates"] == 3
    assert not result["task_scores_computed"] and not result["weights_saved_or_reused"]
    assert "predictions" not in result["validation"]


def test_real_loop_completes8280updates120events_and_earliest_tie(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets, evals = patch_training(monkeypatch)
    saves = []
    monkeypatch.setattr(pilot, "save_or_replace", lambda *_: None)

    def checkpoint(
        path: Path, model: KeyTeacher, provenance: dict[str, Any], watch: Callable[[], None]
    ) -> str:
        saves.append((path.name, provenance))
        return path.name

    monkeypatch.setattr(pilot, "checkpoint", checkpoint)
    plan = {
        "bp_fit": ["b"],
        "expanded_fma_fit": ["f"],
        "bp_selection_guard": ["vb"],
        "fma_inner_selection": ["vf"],
    }
    rows = {"b": {"samples": 288000}, "f": {"samples": 288000}}
    result = pilot.fit(
        toy(),
        rows,
        plan,
        {"b": 0, "f": 1, "vb": 0, "vf": 0},
        tmp_path,
        {},
        lambda: None,
        FakeProvider(),
    )  # type: ignore[arg-type]
    assert len(targets) == 8280 and all(len(batch) == 32 for batch in targets)
    assert result["events"] == 120 and len(evals) == 240
    assert result["best_update"] == 69  # All exact ties retain first checkpoint.
    assert [name for name, _ in saves] == ["best.pt", "final.pt"]
    assert sum(result["coverage"].values()) == 264960
    assert not result["calibration_used"] and not result["confirmation_used"]


def test_five_exact_gate_against_each_control_is_not_weakened() -> None:
    def metric(n: int) -> dict[str, Any]:
        return {
            "ids": ["x"],
            "exact_correct": n,
            "symmetric_units10": 10 * n,
            "asymmetric_units10": 10 * n,
        }

    controls = {
        "restricted": {"best_metrics": {"beatport": metric(86), "fmak": metric(172)}},
        "expanded": {"best_metrics": {"beatport": metric(89), "fmak": metric(176)}},
    }
    result = pilot.continuation_gate({"beatport": metric(89), "fmak": metric(180)}, controls)
    assert not result["continuation_earned"]
    assert pilot.continuation_gate({"beatport": metric(89), "fmak": metric(181)}, controls)[
        "continuation_earned"
    ]


def test_guardian_rejects_probe_with_training_only_authority(tmp_path: Path) -> None:
    from tools.run_musicfm_teacher import supervise

    config = tmp_path / "config.json"
    approval = tmp_path / "approval.json"
    save(config, {"mode": "probe"})
    save(approval, {"authorized": True, "config_sha256": pilot.sha(config), "allow_training": True})
    with pytest.raises(ValueError, match="mode"):
        supervise(config, approval, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def metadata_fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = {
        "bp_fit": [f"beatport:{i}" for i in range(1229)],
        "expanded_fma_fit": [f"fmak:{i}" for i in range(4033)],
        "bp_selection_guard": [f"beatport:{i}" for i in range(1229, 1355)],
        "fma_inner_selection": [f"fmak:{i}" for i in range(4033, 4283)],
    }
    rows = []
    for key in plan:
        for identity in plan[key]:
            rows.append(
                {
                    "id": identity,
                    "role": "fit" if key.endswith("fit") else "validation",
                    "target": 0,
                    "samples": 1200,
                    "bytes": 4800,
                }
            )
    return plan, {"labels": {r["id"]: 0 for r in rows}}, {"complete": True, "rows": rows}


@pytest.mark.parametrize(
    "change", ["wrong-role", "wrong-label", "duplicate", "reserved-extra", "train-too-short"]
)
def test_metadata_contract_rejects_stale_or_reserved_inputs(change: str) -> None:
    plan, labels, manifest = metadata_fixture()
    if change == "wrong-role":
        manifest["rows"][0]["role"] = "validation"
    elif change == "wrong-label":
        manifest["rows"][0]["target"] = 1
    elif change == "duplicate":
        manifest["rows"][-1] = manifest["rows"][0].copy()
    elif change == "reserved-extra":
        manifest["rows"].append({**manifest["rows"][0], "id": "confirmation:reserved"})
    else:
        manifest["rows"][0].update(samples=1199, bytes=1199 * 4)
    with pytest.raises(ValueError):
        pilot.validate_inputs(plan, labels, manifest)


def test_metadata_contract_retains_all_approved_rows() -> None:
    plan, labels, manifest = metadata_fixture()
    assert len(pilot.validate_inputs(plan, labels, manifest)) == 5638


def test_canonical_bp_trace_matches_interleaved_stream_without_comparing_old_format() -> None:
    import hashlib

    from experiments.musicfm_sampling import batches

    bp, fm = ["b1", "b2"], ["f1", "f2", "f3"]
    expected, _ = pilot.canonical_bp_trace(bp, fm, updates=100)
    actual = hashlib.sha256()
    for batch in batches(bp, fm, updates=100):
        for draw in batch:
            if draw.domain == "beatport":
                actual.update(f"{draw.identity}:{draw.augmentation_seed}\n".encode())
    assert actual.hexdigest() == expected


def test_cost_binding_rejects_changed_source_data_or_evaluation() -> None:
    config = {
        "source_sha256": {"driver": "frozen"},
        "input_provider": "disk",
        "weight_sha256": "original",
        "plan_sha256": "plan",
        "labels_sha256": "labels",
        "cache_manifest_sha256": "canonical",
        "identity_overlay_sha256": "physical",
        "disk_cache_manifest_sha256": "disk",
        "ram_config_sha256": "decoder",
        "module_receipts": {"torch": "pinned"},
        "canonical_bp_trace_sha256": "trace",
        "eval_batch_size": 16,
    }
    provenance = {
        **{
            k: v
            for k, v in config.items()
            if k not in {"weight_sha256", "eval_batch_size", "ram_config_sha256"}
        },
        "pretrained_checkpoint_sha256": "original",
        "decoder_reference_sha256": "decoder",
        "eval_batch": 16,
        "seed_after_strict_load": 13,
        "config_sha256": "different-mode-window-budget-is-allowed",
        "initial_state_sha256": "separately-checked-at-load",
    }
    pilot.bind_cost_provenance(config, {"provenance": provenance})
    for field in provenance.keys() - {"config_sha256", "initial_state_sha256"}:
        changed = {**provenance, field: "different"}
        with pytest.raises(ValueError, match="provenance differs"):
            pilot.bind_cost_provenance(config, {"provenance": changed})
    with pytest.raises(ValueError, match="provenance differs"):
        pilot.bind_cost_provenance(config, {"provenance": {}})
