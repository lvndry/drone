from __future__ import annotations

import copy
import json
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from drone.features import audio_digest
from drone.keys import key_relation, transpose_key
from experiments.structured import (
    HARD_LOSS,
    PROTOCOL,
    REVIEWED,
    SOFT_LOSS,
    VERSION,
    StructuredProbe,
    check_pilot,
    compare_metrics,
    confusion_movements,
    require_pair,
    summarize,
    target_matrix,
)


def test_targets_match_relation_contract_normalize_and_are_transposition_equivariant() -> None:
    matrix = target_matrix()
    assert matrix.shape == (24, 24) and torch.isfinite(matrix).all()
    torch.testing.assert_close(matrix.sum(1), torch.ones(24))
    for reference in range(24):
        for prediction in range(24):
            expected = {
                "exact": 0.8,
                "fifth": 0.2 * 0.5 / 1.5,
                "relative": 0.2 * 0.3 / 1.5,
                "parallel": 0.2 * 0.2 / 1.5,
                "other": 0.0,
            }[key_relation(prediction, reference)]
            assert float(matrix[reference, prediction]) == pytest.approx(expected)
            for shift in range(12):
                assert (
                    matrix[transpose_key(reference, shift), transpose_key(prediction, shift)]
                    == matrix[reference, prediction]
                )
        assert torch.count_nonzero(matrix[reference]) == 5
    # C major gives weight to G/F major, A minor and C minor, never C# major.
    assert matrix[0, 14] > 0 and matrix[0, 10] > 0
    assert matrix[0, 19] > 0 and matrix[0, 1] > 0 and matrix[0, 2] == 0


def trajectory(with_table: bool) -> tuple[dict[str, torch.Tensor], list[float], torch.Tensor]:
    torch.manual_seed(13)
    model = nn.Sequential(nn.Linear(2, 24), nn.Dropout(0.2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    if with_table:
        target_matrix()  # Control identity/provenance construction must consume no RNG.
    losses = []
    for _ in range(3):
        data = torch.randn(4, 2)
        labels = torch.randint(0, 24, (4,))
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(data), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return copy.deepcopy(model.state_dict()), losses, torch.get_rng_state()


def test_target_metadata_leaves_hard_ce_trajectory_and_rng_exact() -> None:
    baseline, baseline_loss, baseline_rng = trajectory(False)
    control, control_loss, control_rng = trajectory(True)
    assert baseline_loss == control_loss
    assert torch.equal(baseline_rng, control_rng)
    assert all(torch.equal(value, control[key]) for key, value in baseline.items())


def test_soft_target_loss_and_gradient_use_the_full_declared_distribution() -> None:
    logits = torch.randn(3, 24, requires_grad=True)
    labels = torch.tensor([0, 3, 22])
    targets = target_matrix()[labels]
    loss = nn.functional.cross_entropy(logits, targets)
    expected = -(targets * nn.functional.log_softmax(logits, 1)).sum(1).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(logits.grad, (logits.detach().softmax(1) - targets) / 3)


def metrics(fmak: float = 0.5) -> dict[str, Any]:
    return {
        "beatport": dict(mirex=0.5, mirex_asymmetric=0.5, exact=0.4),
        "fmak": dict(mirex=fmak, mirex_asymmetric=fmak, exact=0.4),
    }


def reference_record() -> dict[str, Any]:
    reviewed = json.loads(REVIEWED.read_bytes())
    return dict(
        status="complete",
        arm="joint",
        seed=13,
        config={"lr": 0.001},
        parameters=109128,
        source_sha256=reviewed["archived"],
        feature_manifest_sha256="fixed",
        fit_ids=[f"beatport:{index}" for index in range(1230)]
        + [f"fmak:{index}" for index in range(976)],
        validation_ids=[f"beatport:v{index}" for index in range(126)]
        + [f"fmak:v{index}" for index in range(250)],
        history=[dict(epoch=1, loss=1.0, mirex=0.5)],
        best_epoch=1,
        early_stopped=True,
        calibration_used=False,
        final_holdout_used=False,
        domain_metrics=metrics(),
        elapsed_seconds=1.0,
    )


def probe_args(tmp_path: Path) -> tuple[Namespace, dict[str, Any]]:
    reference = reference_record()
    archived = tmp_path / "archive"
    archived.mkdir()
    joint = archived / "joint.json"
    bp_path = archived / "bp.json"
    joint.write_text(json.dumps(reference))
    beatport_run = copy.deepcopy(reference)
    beatport_run.update(arm="control", fit_ids=reference["fit_ids"][:1230])
    bp_path.write_text(json.dumps(beatport_run))
    torch.save({"state_dict": {}}, archived / "best.pt")
    output = tmp_path / "control-seed13"
    output.mkdir()
    args = Namespace(
        structured_control=True,
        structured_targets=False,
        seed=13,
        arm="joint",
        reference_run=joint,
        reference_bp=bp_path,
        reference_checkpoint=None,
        paired_control=None,
        pilot_run=None,
        output=output,
        device="cpu",
    )
    current = copy.deepcopy(reference)
    current["source_sha256"] = json.loads(REVIEWED.read_bytes())["reviewed_base"]
    current["source_sha256"]["experiments/k10_multidomain.py"] = "reviewed-hooks"
    return args, current


def test_candidate_requires_completed_exact_control_before_fit(tmp_path: Path) -> None:
    args, control = probe_args(tmp_path)
    StructuredProbe(args, control)
    control_path = args.output / "run.json"
    control_path.write_text(json.dumps(control))
    result_path = args.output / "structured-result.json"
    result = dict(
        artifact_version=VERSION,
        kind="control",
        seed=13,
        status="complete",
        control_reproduced=False,
        run_sha256=audio_digest(control_path),
        identity=dict(protocol_sha256=audio_digest(PROTOCOL)),
    )
    result_path.write_text(json.dumps(result))
    args.structured_targets = True
    args.structured_control = False
    args.paired_control = control_path
    args.output = tmp_path / "candidate-seed13"
    args.output.mkdir()
    candidate = copy.deepcopy(control)
    with pytest.raises(ValueError, match="exactly reproduced fresh paired control"):
        StructuredProbe(args, candidate)
    result["control_reproduced"] = True
    result_path.write_text(json.dumps(result))
    probe = StructuredProbe(args, candidate)
    assert probe.kind == "candidate" and candidate["loss_spec"] == SOFT_LOSS
    assert control["loss_spec"] == HARD_LOSS
    require_pair(control, candidate)
    candidate["config"]["lr"] = 0.002
    with pytest.raises(ValueError, match="beyond loss: config"):
        require_pair(control, candidate)


def test_pilot_gate_blocks_confirmation_changes_and_failed_pilots(tmp_path: Path) -> None:
    args, record = probe_args(tmp_path)
    with pytest.raises(ValueError, match="passing canonical"):
        check_pilot(None, record)
    pilot: dict[str, Any] = dict(
        artifact_version=VERSION,
        kind="candidate",
        seed=13,
        status="complete",
        control_reproduced=True,
        identity={**record, "protocol_sha256": audio_digest(PROTOCOL)},
        method_gate=dict(qualifies_for_confirmation=True),
    )
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(pilot))
    check_pilot(path, record)
    pilot["method_gate"]["qualifies_for_confirmation"] = False
    path.write_text(json.dumps(pilot))
    with pytest.raises(ValueError, match="passing seed13"):
        check_pilot(path, record)
    pilot["method_gate"]["qualifies_for_confirmation"] = True
    pilot["identity"]["protocol_sha256"] = "changed"
    path.write_text(json.dumps(pilot))
    with pytest.raises(ValueError, match="protocol"):
        check_pilot(path, record)


def test_method_gate_rejects_exact_and_beatport_losses() -> None:
    baseline = metrics()
    candidate = metrics(0.53)
    assert compare_metrics(baseline, candidate)["qualifies_for_confirmation"]
    candidate["beatport"]["mirex"] -= 0.001
    assert not compare_metrics(baseline, candidate)["qualifies_for_confirmation"]
    candidate = metrics(0.53)
    candidate["fmak"]["exact"] -= 0.001
    assert not compare_metrics(baseline, candidate)["qualifies_for_confirmation"]
    assert not compare_metrics(baseline, metrics(0.509))["qualifies_for_confirmation"]


def result(seed: int, gain: float) -> dict[str, Any]:
    return dict(
        artifact_version=VERSION,
        kind="candidate",
        seed=seed,
        status="complete",
        control_reproduced=True,
        metrics=metrics(0.5 + gain),
        control_metrics=metrics(),
        bp_metrics=metrics(),
        archived_joint_metrics=metrics(),
        identity=dict(
            source_sha256={},
            config={},
            feature_manifest_sha256="fixed",
            protocol_sha256="fixed",
            loss_spec=SOFT_LOSS,
            targets=target_matrix().tolist(),
        ),
        method_gate=compare_metrics(metrics(), metrics(0.5 + gain)),
    )


def test_confirmation_keeps_original_hurdle_all_seeds_and_joint_safeguard() -> None:
    results = [result(seed, 0.015) for seed in (13, 37, 73)]
    summary = summarize(results)
    assert summary["method_gate"]["qualifies_for_confirmation"]
    assert not summary["original_hurdle_pass"] and not summary["qualifies_for_external_freeze"]
    results = [result(13, 0.03), result(37, 0.001), result(73, 0.001)]
    assert not summarize(results)["qualifies_for_external_freeze"]
    results = [result(seed, 0.03) for seed in (13, 37, 73)]
    assert summarize(results)["qualifies_for_external_freeze"]
    for item in results:
        item["archived_joint_metrics"]["beatport"]["exact"] += 0.01
    assert not summarize(results)["raw_joint_safeguard"]["passes"]
    assert not summarize(results)["qualifies_for_external_freeze"]
    results[1]["identity"]["source_sha256"] = {"changed": "source"}
    with pytest.raises(ValueError, match="source_sha256"):
        summarize(results)


def test_final_artifacts_replay_control_then_score_loss_only_candidate(tmp_path: Path) -> None:
    from drone.checkpoints import save_checkpoint
    from drone.features import window
    from drone.model import KeyCNN
    from experiments.k10_multidomain import domain_metrics
    from train import evaluate_features

    torch.set_num_threads(1)
    args, record = probe_args(tmp_path)
    model = KeyCNN()
    save_checkpoint(args.output / "best.pt", model)
    save_checkpoint(args.reference_run.with_name("best.pt"), model)
    rows = [
        dict(id=f"{domain}:val", domain=domain, sha256=domain) for domain in ("beatport", "fmak")
    ]
    features = [window(np.ones((208, 16), dtype=np.float16)).astype(np.float32) for _ in rows]
    references = np.array([0, 0])
    record["domain_metrics"] = domain_metrics(
        evaluate_features(model, features, "cpu", 4), references, rows
    )
    args.reference_run.write_text(
        json.dumps({**record, "source_sha256": json.loads(REVIEWED.read_bytes())["archived"]})
    )
    beatport_run = json.loads(args.reference_bp.read_bytes())
    beatport_run["domain_metrics"] = record["domain_metrics"]
    args.reference_bp.write_text(json.dumps(beatport_run))
    control = StructuredProbe(args, record)
    (args.output / "run.json").write_text(json.dumps(record))
    random_state = torch.get_rng_state().clone()
    completed = control.finish(record, features, references, rows, time.monotonic())
    assert completed["status"] == "complete" and completed["checkpoint_state_checked"]
    assert completed["control_reproduced"] and torch.equal(torch.get_rng_state(), random_state)
    artifact = json.loads((args.output / completed["validation_artifact"]).read_bytes())
    assert np.array(artifact["logits"]).shape == (2, 24)
    assert (
        artifact["ids"] == [row["id"] for row in rows] and artifact["refs"] == references.tolist()
    )
    assert artifact["audio_sha256"] == [row["sha256"] for row in rows]
    candidate_args = copy.copy(args)
    candidate_args.structured_targets = True
    candidate_args.structured_control = False
    candidate_args.paired_control = args.output / "run.json"
    candidate_args.output = tmp_path / "candidate-seed13"
    candidate_args.output.mkdir()
    candidate_record = copy.deepcopy(record)
    candidate = StructuredProbe(candidate_args, candidate_record)
    (candidate_args.output / "run.json").write_text(json.dumps(candidate_record))
    save_checkpoint(candidate_args.output / "best.pt", model)
    completed_candidate = candidate.finish(
        candidate_record, features, references, rows, time.monotonic()
    )
    assert completed_candidate["status"] == "complete"
    assert not completed_candidate["method_gate"]["qualifies_for_confirmation"]
    changed = copy.deepcopy(record)
    changed["history"][0]["loss"] += 1e-12
    failed = control.finish(changed, features, references, rows, time.monotonic())
    assert (
        failed["status"] == "reproduction_or_protocol_failed" and not failed["control_reproduced"]
    )


def test_confusion_movements_show_exact_demotions_and_both_fifth_conventions() -> None:
    rows = [dict(id="beatport:1", domain="beatport"), dict(id="fmak:1", domain="fmak")]
    control = {
        "beatport": dict(ids=["beatport:1"], predictions=[0]),
        "fmak": dict(ids=["fmak:1"], predictions=[10]),
    }
    moves = confusion_movements(control, np.array([14, 0]), np.array([0, 0]), rows)
    assert moves["beatport"]["repo_symmetric_fifths"]["exact_demotions"] == 1
    assert moves["fmak"]["repo_symmetric_fifths"]["exact_promotions"] == 1
    # F major over C major is the descending fifth: credited only by repo convention.
    assert moves["fmak"]["repo_symmetric_fifths"]["row_control_column_candidate"][1][0] == 1
    assert (
        moves["fmak"]["existing_asymmetric_ascending_fifth"]["row_control_column_candidate"][4][0]
        == 1
    )
    control["fmak"]["ids"] = ["wrong"]
    with pytest.raises(ValueError, match="identities"):
        confusion_movements(control, np.array([14, 0]), np.array([0, 0]), rows)
