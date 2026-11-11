from __future__ import annotations

import copy
import json
import shutil
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from drone.checkpoints import load_checkpoint, save_checkpoint
from drone.features import audio_digest
from drone.model import KeyCNN, params_count
from experiments.capacity import PROTOCOL, REVIEWED, VERSION, CapacityProbe, memory_guard
from experiments.ema import evaluate_logits
from experiments.paired import compare, ema_evidence_math, summarize, training_math, verify_control
from experiments.small import source_manifest


def test_width2_checkpoint_restores_declared_shape_and_parameter_count(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    model = KeyCNN(width=2.0).eval()
    path = tmp_path / "wide.pt"
    save_checkpoint(path, model, width=2.0)
    restored, payload = load_checkpoint(path)
    assert params_count(model) == params_count(restored) == 361080
    assert payload["model_config"] == {"width": 2.0}
    x = torch.zeros(1, 192, 16)
    with torch.inference_mode():
        assert torch.equal(model(x), restored(x))
    assert params_count(KeyCNN()) == 109128


def test_only_explicit_scoring_budget_extension_permits_long_candidate() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(192 * 8, 24))
    features = [np.zeros((192, 8), dtype=np.float32)]
    with pytest.raises(TimeoutError):
        evaluate_logits(model, features, "cpu", time.monotonic() - 1300)
    logits = evaluate_logits(model, features, "cpu", time.monotonic() - 1300, budget_seconds=2700)
    assert logits.shape == (1, 24)


def test_memory_guard_requires12gib_before_raising_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (11 * 1024**3, 96 * 1024**3))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", calls.append)
    with pytest.raises(RuntimeError, match="12GiB"):
        memory_guard("cuda:0")
    assert calls == []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (12 * 1024**3, 96 * 1024**3))
    memory = memory_guard("cuda:0")
    assert calls == [0.08] and memory["allocator_fraction"] == 0.08
    assert memory["allocator_cap_bytes"] == int(96 * 1024**3 * 0.08)


def test_math_audit_matches_all_frozen_controls_and_detects_recipe_changes() -> None:
    source = Path("experiments/k10_multidomain.py").read_text()
    audit = training_math(source)
    reviewed = json.loads(REVIEWED.read_bytes())
    assert all(audit == value["training_math"] for value in reviewed.values())
    for before, after in (
        ("lr=0.001", "lr=0.002"),
        ("range(1, 121)", "range(1, 61)"),
        ("epoch - best_epoch >= 25", "epoch - best_epoch >= 20"),
        ("rng.integers(-4, 5)", "rng.integers(-3, 4)"),
    ):
        assert before in source
        assert training_math(source.replace(before, after)) != audit


def metrics(fmak: float = 0.5) -> dict[str, Any]:
    return {
        "beatport": dict(mirex=0.5, mirex_asymmetric=0.5, exact=0.4),
        "fmak": dict(mirex=fmak, mirex_asymmetric=fmak, exact=0.4),
    }


def fixture_control(tmp_path: Path) -> tuple[Namespace, dict[str, Any], dict[str, Any]]:
    reviewed = json.loads(REVIEWED.read_bytes())
    archive, control_dir, output = (
        tmp_path / "archive",
        tmp_path / "control13",
        tmp_path / "candidate13",
    )
    for path in (archive, control_dir, output):
        path.mkdir()
    model = KeyCNN()
    save_checkpoint(archive / "best.pt", model)
    shutil.copyfile(archive / "best.pt", control_dir / "best.pt")
    archived: dict[str, Any] = dict(
        status="complete",
        arm="joint",
        seed=13,
        config={"lr": 0.001},
        parameters=109128,
        feature_manifest_sha256="fixed",
        source_sha256=reviewed["archived"]["sources"],
        fit_ids=[f"beatport:{i}" for i in range(1230)] + [f"fmak:{i}" for i in range(976)],
        validation_ids=[f"beatport:v{i}" for i in range(126)] + [f"fmak:v{i}" for i in range(250)],
        history=[dict(epoch=1, loss=1.0, mirex=0.5)],
        best_epoch=1,
        early_stopped=True,
        calibration_used=False,
        final_holdout_used=False,
        domain_metrics=metrics(),
        elapsed_seconds=1,
        checkpoint_sha256=audio_digest(archive / "best.pt"),
    )
    (archive / "joint.json").write_text(json.dumps(archived))
    bp = copy.deepcopy(archived)
    bp.update(arm="control", fit_ids=archived["fit_ids"][:1230])
    (archive / "bp.json").write_text(json.dumps(bp))
    control = copy.deepcopy(archived)
    control["source_sha256"] = reviewed["reusable_control"]["sources"]
    control["loss_spec"] = {"name": "hard-cross-entropy", "smoothing": 0.0}
    (control_dir / "run.json").write_text(json.dumps(control))
    validation = control_dir / "validation.json"
    validation.write_text(json.dumps(dict(metrics=control["domain_metrics"])))
    proof = dict(
        artifact_version="k10-structured-targets-v1",
        seed=13,
        status="complete",
        kind="control",
        control_reproduced=True,
        run_sha256=audio_digest(control_dir / "run.json"),
        checkpoint_sha256=audio_digest(control_dir / "best.pt"),
        validation_artifact=validation.name,
        validation_sha256=audio_digest(validation),
    )
    (control_dir / "structured-result.json").write_text(json.dumps(proof))
    candidate = copy.deepcopy(archived)
    candidate.update(parameters=361080, source_sha256=source_manifest())
    args = Namespace(
        capacity_probe=True,
        capacity_control=False,
        seed=13,
        arm="joint",
        device="cpu",
        reference_run=archive / "joint.json",
        reference_bp=archive / "bp.json",
        reference_checkpoint=None,
        paired_control=control_dir / "run.json",
        pilot_run=None,
        output=output,
    )
    return args, candidate, control


def test_reuse_reverifies_actual_history_state_hash_and_completed_proof(tmp_path: Path) -> None:
    args, record, control = fixture_control(tmp_path)
    probe = CapacityProbe(args, record, memory_guard("cpu"))
    assert probe.seconds == 2700 and probe.width == 2
    assert probe.control_proof["exact_history_predictions_state"]
    assert (args.output / "PROTOCOL.md").read_bytes() == PROTOCOL.read_bytes()
    changed = copy.deepcopy(control)
    changed["history"][0]["loss"] += 1e-12
    args.paired_control.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="history"):
        CapacityProbe(args, record, memory_guard("cpu"))
    args.paired_control.write_text(json.dumps(control))
    proof_path = args.paired_control.with_name("structured-result.json")
    proof = json.loads(proof_path.read_bytes())
    proof["status"] = "budget_inconclusive"
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="completed exact-replay proof"):
        CapacityProbe(args, record, memory_guard("cpu"))


def test_control_checkpoint_metadata_corruption_is_rejected(tmp_path: Path) -> None:
    args, record, control = fixture_control(tmp_path)
    checkpoint = args.paired_control.with_name("best.pt")
    payload = torch.load(checkpoint, weights_only=True)
    payload["model_config"]["width"] = 2.0
    torch.save(payload, checkpoint)
    control["checkpoint_sha256"] = audio_digest(checkpoint)
    args.paired_control.write_text(json.dumps(control))
    with pytest.raises(ValueError, match="width"):
        verify_control(
            args.paired_control,
            args.reference_run,
            args.reference_run.with_name("best.pt"),
            [control["source_sha256"]],
        )


def result(seed: int, gain: float) -> dict[str, Any]:
    return dict(
        artifact_version=VERSION,
        kind="candidate",
        seed=seed,
        status="complete",
        control_reproduced=True,
        identity=dict(
            source_sha256={},
            config={},
            feature_manifest_sha256="same",
            protocol_sha256="same",
            width=2.0,
            budget_seconds=2700,
        ),
        metrics=metrics(0.5 + gain),
        control_metrics=metrics(),
        bp_metrics=metrics(),
        method_gate=compare(metrics(), metrics(0.5 + gain)),
    )


def test_capacity_gate_preserves_original_hurdle_and_never_buys_weighted_gain_with_exact() -> None:
    candidates = [result(s, 0.015) for s in (13, 37, 73)]
    summary = summarize(candidates, VERSION)
    assert (
        summary["method_gate"]["qualifies_for_confirmation"] and not summary["original_hurdle_pass"]
    )
    candidates = [result(s, 0.03) for s in (13, 37, 73)]
    assert summarize(candidates, VERSION)["qualifies_for_external_freeze"]
    for candidate in candidates:
        candidate["metrics"]["beatport"]["exact"] -= 0.001
    assert not summarize(candidates, VERSION)["qualifies_for_external_freeze"]
    candidates[0]["status"] = "budget_inconclusive"
    with pytest.raises(ValueError, match="completed"):
        summarize(candidates, VERSION)


def test_candidate_budget_failure_is_inconclusive_and_cannot_qualify(tmp_path: Path) -> None:
    args, record, _ = fixture_control(tmp_path)
    probe = CapacityProbe(args, record, memory_guard("cpu"))
    failure = probe.failure(record, time.monotonic(), "time budget", budget=True)
    assert failure["status"] == "budget_inconclusive"
    assert not failure["method_gate"]["qualifies_for_confirmation"] and not failure["promoted"]


def test_wide_selected_logits_use45minute_budget_and_preserve_rng(tmp_path: Path) -> None:
    from drone.features import window
    from experiments.k10_multidomain import domain_metrics
    from experiments.paired import score_selected
    from train import evaluate_features

    model = KeyCNN(width=2.0)
    save_checkpoint(tmp_path / "best.pt", model, width=2.0)
    rows = [dict(id=f"{d}:v", domain=d, sha256=d) for d in ("beatport", "fmak")]
    features = [window(np.ones((208, t), dtype=np.float16)).astype(np.float32) for t in (16, 24)]
    refs = np.array([0, 1])
    expected = domain_metrics(evaluate_features(model, features, "cpu", 4), refs, rows)
    rng = torch.get_rng_state().clone()
    scored = score_selected(
        tmp_path,
        dict(domain_metrics=expected),
        features,
        refs,
        rows,
        "cpu",
        time.monotonic() - 1300,
        2700,
        2.0,
    )
    assert scored["metrics"] == expected and torch.equal(torch.get_rng_state(), rng)
    artifact = json.loads((tmp_path / scored["validation_artifact"]).read_bytes())
    assert artifact["ids"] == [r["id"] for r in rows]
    assert artifact["refs"] == refs.tolist()
    assert np.asarray(artifact["logits"]).shape == (2, 24)
    with pytest.raises(ValueError, match="width metadata"):
        score_selected(
            tmp_path,
            dict(domain_metrics=expected),
            features,
            refs,
            rows,
            "cpu",
            time.monotonic(),
            2700,
            1.0,
        )


def test_oom_has_separate_inconclusive_artifact(tmp_path: Path) -> None:
    from experiments.capacity import record_oom

    record_oom(
        Namespace(output=tmp_path, capacity_probe=True, seed=13),
        torch.OutOfMemoryError("synthetic OOM"),
    )
    outcome = json.loads((tmp_path / "capacity-result.json").read_bytes())
    assert outcome["status"] == "resource_inconclusive"
    assert not outcome["method_gate"]["qualifies_for_confirmation"]


def test_evidence_helper_audit_allows_path_relocation_but_rejects_score_changes() -> None:
    source = Path("experiments/ema.py").read_text()
    approved = json.loads(REVIEWED.read_bytes())["base"]["ema_evidence_math"]
    assert ema_evidence_math(source) == approved
    relocated = source.replace(
        'PROTOCOL = Path(__file__).with_name("EMA_PROTOCOL.md")',
        'PROTOCOL = Path(__file__).parent / "studies" / "k10-ema" / "PROTOCOL.md"',
    )
    assert ema_evidence_math(relocated) == approved
    assert ema_evidence_math(source.replace("model.eval()", "model.train()")) != approved
    assert (
        ema_evidence_math(source.replace("BUDGET_SECONDS = 1200", "BUDGET_SECONDS = 2700"))
        != approved
    )
