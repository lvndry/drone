from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from experiments.ema import (
    DECAY,
    VERSION,
    assert_reproduced,
    best_shadow,
    bn_schedule,
    make_shadow,
    method_gate,
    preserve_rng,
    recalibrate_bn,
    schedule_batches,
    summarize,
    update_shadow,
)


def test_ema_first_update_uses_theta_zero_and_never_changes_source_or_buffers() -> None:
    bn = nn.BatchNorm1d(2)
    model = nn.Sequential(nn.Linear(2, 2), bn)
    shadow = make_shadow(model)
    initial = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        for value in model.parameters():
            value.add_(2)
        assert bn.running_mean is not None
        bn.running_mean.fill_(7)
    source = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    update_shadow(shadow, model)
    for name, value in shadow.named_parameters():
        torch.testing.assert_close(value, DECAY * initial[name] + (1 - DECAY) * source[name])
    assert torch.equal(shadow.state_dict()["1.running_mean"], initial["1.running_mean"])
    selected = best_shadow(shadow, model)
    assert torch.equal(selected["1.running_mean"], source["1.running_mean"])
    assert all(torch.equal(value, source[name]) for name, value in model.state_dict().items())
    assert torch.equal(torch.get_rng_state(), rng)
    with preserve_rng():
        torch.rand(3)
    assert torch.equal(torch.get_rng_state(), rng)


def fixture_rows() -> list[dict[str, Any]]:
    return [
        dict(id=f"beatport:{i}", sha256=f"hash{i}", role="train", domain="beatport")
        for i in range(34)
    ]


def test_bn_schedule_is_fixed_fitting_only_centered_and_keeps_partial_batch() -> None:
    rows = fixture_rows()
    arrays = [np.ones((208, 300 if i % 2 else 340), dtype=np.float16) for i in range(34)]
    rows[-1]["role"] = "validation"
    fitting = list(range(33))
    schedule = bn_schedule(rows, arrays, fitting)
    assert schedule == bn_schedule(rows, arrays, fitting)
    expected = np.random.default_rng(20261111).permutation(fitting).tolist()
    assert [r["index"] for r in schedule] == expected
    assert all(r["index"] != 33 and r["shift"] == 0 for r in schedule)
    assert all(r["start"] == (10 if r["index"] % 2 == 0 else 0) for r in schedule)
    assert all(r["audio_sha256"] == rows[r["index"]]["sha256"] for r in schedule)
    batches = list(schedule_batches(arrays, schedule))
    assert [tuple(x.shape) for x in batches] == [(32, 192, 320), (1, 192, 320)]
    for position, row in enumerate(schedule):
        feature = batches[position // 32][position % 32]
        assert torch.all(feature[:, : row["length"]] == 1)
        assert torch.all(feature[:, row["length"] :] == 0)
    with pytest.raises(ValueError, match="fitting rows only"):
        bn_schedule(rows, arrays, list(range(34)))
    with pytest.raises(ValueError, match="unique"):
        bn_schedule(rows, arrays, [0, 0])


class ObservedBN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout2d(0.9)
        self.bn = nn.BatchNorm2d(1)
        self.observed: list[tuple[bool, bool, bool]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.observed.append((self.training, self.dropout.training, self.bn.training))
        return self.bn(self.dropout(x.unsqueeze(1)))


def test_bn_recalibration_updates_copy_only_with_dropout_off_and_rng_preserved() -> None:
    source = ObservedBN()
    source.train()
    assert source.bn.running_mean is not None
    source.bn.running_mean.fill_(99)
    original = copy.deepcopy(source.state_dict())
    rows = fixture_rows()[:2]
    arrays = [np.full((208, 320), 3.0, dtype=np.float16) for _ in rows]
    schedule = bn_schedule(rows, arrays, [0, 1])
    rng = torch.get_rng_state().clone()
    result = recalibrate_bn(source, arrays, schedule, "cpu", time.monotonic(), expected_bn=1)
    assert source.observed == [] and source.training and source.dropout.training
    assert all(torch.equal(value, original[name]) for name, value in source.state_dict().items())
    assert result.observed == [(False, False, True)]
    assert result.bn.running_mean is not None
    assert torch.equal(result.bn.running_mean, torch.tensor([3.0]))
    assert result.bn.num_batches_tracked == 1
    assert result.bn.momentum == 0.1 and not result.bn.training
    assert torch.equal(torch.get_rng_state(), rng)
    with pytest.raises(ValueError, match="Expected 6"):
        recalibrate_bn(source, arrays, schedule, "cpu", time.monotonic())


def metric_outputs(gain: float = 0.02) -> dict[str, Any]:
    baseline = {d: dict(mirex=0.5, exact=0.4) for d in ("beatport", "fmak")}
    candidate = copy.deepcopy(baseline)
    candidate["fmak"]["mirex"] += gain
    return {
        "raw": dict(metrics=copy.deepcopy(baseline)),
        "raw_bn": dict(metrics=baseline),
        "ema_bn": dict(metrics=candidate),
    }


def test_gate_requires_accuracy_constraints_not_only_fmak_gain() -> None:
    outputs = metric_outputs()
    assert method_gate(outputs)["qualifies_for_confirmation"]
    outputs["ema_bn"]["metrics"]["beatport"]["mirex"] -= 0.001
    assert not method_gate(outputs)["qualifies_for_confirmation"]
    outputs = metric_outputs()
    outputs["ema_bn"]["metrics"]["fmak"]["exact"] -= 0.001
    assert not method_gate(outputs)["qualifies_for_confirmation"]
    assert not method_gate(metric_outputs(0.009))["qualifies_for_confirmation"]


def raw_record() -> dict[str, Any]:
    return dict(
        arm="joint",
        seed=13,
        config={"lr": 0.001},
        feature_manifest_sha256="frozen",
        fit_ids=["beatport:1"],
        validation_ids=["fmak:2"],
        history=[dict(epoch=1, loss=1.23, mirex=0.5)],
        best_epoch=1,
        early_stopped=True,
        domain_metrics={"fmak": {"predictions": [0]}},
    )


@pytest.mark.parametrize(
    "key",
    [
        "config",
        "fit_ids",
        "validation_ids",
        "history",
        "best_epoch",
        "early_stopped",
        "domain_metrics",
    ],
)
def test_raw_reproduction_rejects_any_changed_history_identity_or_prediction(key: str) -> None:
    reference = raw_record()
    current = copy.deepcopy(reference)
    current[key] = "changed"
    with pytest.raises(ValueError, match=key):
        assert_reproduced(current, reference)


def test_raw_reproduction_is_exact_on_weights_and_ignores_elapsed_metadata() -> None:
    reference, current = raw_record(), raw_record()
    current["elapsed_seconds"] = 123
    state = {"w": torch.ones(2)}
    assert_reproduced(current, reference, current_state=state, reference_state=state)
    with pytest.raises(ValueError, match="checkpoint state"):
        assert_reproduced(
            current, reference, current_state={"w": torch.ones(2) + 1e-7}, reference_state=state
        )


def confirmation_result(seed: int, gain: float) -> dict[str, Any]:
    outputs = metric_outputs(gain)
    return dict(
        seed=seed,
        artifact_version=VERSION,
        status="complete",
        raw_reproduced=True,
        identity={
            key: "same"
            for key in (
                "source_sha256",
                "config",
                "feature_manifest_sha256",
                "protocol_sha256",
                "bn_schedule_sha256",
            )
        },
        method_gate=method_gate(outputs),
        outputs=outputs,
        bp_metrics={d: dict(mirex=0.5, exact=0.4) for d in ("beatport", "fmak")},
    )


def test_three_seed_gate_keeps_original_two_point_hurdle_and_rejects_lucky_pilot() -> None:
    results = [
        confirmation_result(13, 0.03),
        confirmation_result(37, 0.001),
        confirmation_result(73, 0.001),
    ]
    assert not summarize(results)["qualifies_for_external_freeze"]
    results = [confirmation_result(s, 0.015) for s in (13, 37, 73)]
    result = summarize(results)
    assert result["method_gate"]["qualifies_for_confirmation"]
    assert not result["original_hurdle_pass"]
    assert not result["qualifies_for_external_freeze"]
    results = [confirmation_result(s, 0.03) for s in (13, 37, 73)]
    assert summarize(results)["qualifies_for_external_freeze"]
    results[1]["identity"]["protocol_sha256"] = "changed"
    with pytest.raises(ValueError, match="protocol_sha256"):
        summarize(results)


def trajectory(with_probe: bool) -> tuple[dict[str, torch.Tensor], list[float], torch.Tensor]:
    torch.manual_seed(13)
    model = nn.Sequential(nn.Linear(2, 3), nn.Dropout(0.2), nn.Linear(3, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    shadow = make_shadow(model) if with_probe else None
    losses = []
    for _ in range(3):
        data, target = torch.randn(4, 2), torch.randint(0, 2, (4,))
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()
        if shadow is not None:
            update_shadow(shadow, model)
            best_shadow(shadow, model)
        losses.append(float(loss.detach()))
    return copy.deepcopy(model.state_dict()), losses, torch.get_rng_state()


def test_probe_hooks_leave_optimizer_trajectory_loss_and_rng_identical() -> None:
    ordinary, ordinary_losses, ordinary_rng = trajectory(False)
    hooked, hooked_losses, hooked_rng = trajectory(True)
    assert ordinary_losses == hooked_losses
    assert torch.equal(ordinary_rng, hooked_rng)
    assert all(torch.equal(value, hooked[name]) for name, value in ordinary.items())


def test_final_scoring_reproduces_raw_and_writes_three_comparable_outputs(tmp_path: Path) -> None:
    from argparse import Namespace

    from drone.checkpoints import save_checkpoint
    from drone.features import window
    from drone.model import KeyCNN
    from experiments.ema import EmaProbe
    from experiments.k10_multidomain import domain_metrics
    from train import evaluate_features

    torch.set_num_threads(1)
    model = KeyCNN()
    out, archive = tmp_path / "out", tmp_path / "archive"
    out.mkdir()
    archive.mkdir()
    save_checkpoint(out / "best.pt", model)
    save_checkpoint(archive / "best.pt", model)
    rows = [
        dict(id=f"{domain}:1", domain=domain, role="train", sha256=domain, ref=0)
        for domain in ("beatport", "fmak")
    ]
    arrays = [np.ones((208, 16), dtype=np.float16) for _ in rows]
    features = [window(a).astype(np.float32) for a in arrays]
    refs = np.array([0, 0])
    metrics = domain_metrics(evaluate_features(model, features, "cpu", batch=4), refs, rows)
    record = raw_record()
    record.update(domain_metrics=metrics)
    probe = object.__new__(EmaProbe)
    probe.args = Namespace(
        output=out,
        device="cpu",
        seed=13,
        reference_checkpoint=None,
        reference_run=archive / "run.json",
    )
    probe.reference = copy.deepcopy(record)
    probe.bp = dict(domain_metrics=metrics)
    probe.shadow = make_shadow(model)
    probe.arrays = arrays
    probe.schedule = bn_schedule(rows, arrays, [0, 1])
    probe.identity = dict(artifact_version=VERSION)
    probe.updates = 1
    probe.save_best(model)
    result = probe.finish(record, features, refs, rows, time.monotonic())
    assert result["status"] == "complete"
    assert result["raw_reproduced"] and result["checkpoint_state_checked"]
    assert set(result["outputs"]) == {"raw", "raw_bn", "ema_bn"}
    for name in result["outputs"]:
        data = json.loads((out / result["outputs"][name]["artifact"]).read_bytes())
        assert data["ids"] == [r["id"] for r in rows]
        assert data["refs"] == refs.tolist()
        assert np.asarray(data["logits"]).shape == (2, 24)
    changed = copy.deepcopy(record)
    changed["history"][0]["loss"] += 1e-12
    failed = probe.finish(changed, features, refs, rows, time.monotonic())
    assert failed["status"] == "reproduction_or_protocol_failed"
    assert not failed["raw_reproduced"]
    assert not failed["method_gate"]["qualifies_for_confirmation"]


def test_preflight_requires_passing_canonical_pilot_and_rejects_changed_confirmation(
    tmp_path: Path,
) -> None:
    from argparse import Namespace

    from drone.features import audio_digest
    from experiments.ema import PROTOCOL, EmaProbe

    reviewed = json.loads(Path("experiments/ema-reviewed-sources.json").read_bytes())
    rows = [
        dict(id=f"{domain}:{i}", domain=domain)
        for domain, count in (("beatport", 1230), ("fmak", 976))
        for i in range(count)
    ]
    record = dict(
        config={},
        feature_manifest_sha256="frozen",
        fit_ids=[r["id"] for r in rows],
        validation_ids=["fixed"],
        source_sha256=reviewed["reviewed_base"],
    )
    probe = object.__new__(EmaProbe)
    probe.rows = rows
    probe.args = Namespace(arm="joint", seed=37, pilot_run=None)
    probe.reference = copy.deepcopy(record)
    probe.reference["source_sha256"] = reviewed["archived"]
    probe.bp = copy.deepcopy(probe.reference)
    probe.bp["fit_ids"] = record["fit_ids"][:1230]
    with pytest.raises(ValueError, match="passing canonical"):
        probe.startup_checks(record, list(range(2206)))
    pilot: dict[str, Any] = dict(
        status="complete",
        seed=13,
        artifact_version=VERSION,
        raw_reproduced=True,
        method_gate=dict(qualifies_for_confirmation=True),
        identity={**record, "protocol_sha256": audio_digest(PROTOCOL)},
    )
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(json.dumps(pilot))
    probe.args.pilot_run = pilot_path
    probe.startup_checks(record, list(range(2206)))
    pilot["identity"]["source_sha256"] = {"changed": "source"}
    pilot_path.write_text(json.dumps(pilot))
    with pytest.raises(ValueError, match="Confirmation changed source_sha256"):
        probe.startup_checks(record, list(range(2206)))


def test_bn_control_cannot_hide_accuracy_decline_against_ordinary_raw_joint() -> None:
    results = [confirmation_result(seed, 0.03) for seed in (13, 37, 73)]
    for result in results:
        result["outputs"]["raw"]["metrics"]["beatport"] = dict(mirex=0.6, exact=0.5)
    summary = summarize(results)
    assert summary["method_gate"]["qualifies_for_confirmation"]
    assert summary["original_hurdle_pass"]
    assert not summary["raw_accuracy_safeguard"]["passes"]
    assert not summary["qualifies_for_external_freeze"]
