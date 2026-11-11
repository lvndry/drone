from __future__ import annotations

import copy
import json
import shutil
import time
from argparse import Namespace
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from drone.checkpoints import load_checkpoint, save_checkpoint
from drone.features import audio_digest
from drone.model import KeyCNN
from experiments import k10_multidomain as driver
from experiments.paired import compare, score_selected, summarize, training_math
from experiments.pitch_circle_fit import PROTOCOL, REVIEWED, VERSION, PitchCircleFit
from experiments.pitch_circle_model import PitchCircleNet
from experiments.research_models import (
    ARCHITECTURE,
    ResearchModelHooks,
    load_pitch_circle,
    pitch_circle_hooks,
    save_pitch_circle,
)
from experiments.small import source_manifest


@pytest.fixture(scope="module", autouse=True)
def cpu_setup() -> Generator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(previous)


def metadata() -> dict[str, Any]:
    return dict(
        epoch=1,
        experiment="pitch-circle-joint-development-v1",
        arm="joint",
        seed=13,
        source_sha256=source_manifest(),
        feature_manifest_sha256="d" * 64,
    )


def test_research_roundtrip_is_strict_and_cannot_enter_production_dispatch(tmp_path: Path) -> None:
    model = PitchCircleNet().eval()
    state = copy.deepcopy(model.state_dict())
    path = tmp_path / "research.pt"
    rng = torch.get_rng_state().clone()
    save_pitch_circle(path, model, metadata=metadata())
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    restored, payload = load_pitch_circle(path)
    assert payload["architecture"] == ARCHITECTURE and "model_config" not in payload
    assert not restored.training
    x = torch.randn(2, 192, 17)
    with torch.inference_mode():
        assert torch.equal(restored(x), model(x))
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        load_checkpoint(path)
    production = tmp_path / "production.pt"
    save_checkpoint(production, KeyCNN())
    with pytest.raises(ValueError, match="Incompatible research"):
        load_pitch_circle(production)
    with pytest.raises(ValueError, match="own architecture"):
        save_pitch_circle(path, KeyCNN(), metadata=metadata())


@pytest.mark.parametrize(
    "field,value",
    [
        ("architecture", "KeyCNN"),
        ("format_version", 2),
        ("format_version", True),
        ("label_schema", "wrong"),
        ("feature_spec", {}),
        ("model_spec", {}),
    ],
)
def test_loader_rejects_incompatible_contract(tmp_path: Path, field: str, value: object) -> None:
    path = tmp_path / "model.pt"
    save_pitch_circle(path, PitchCircleNet(), metadata=metadata())
    payload = torch.load(path, weights_only=True)
    payload[field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Incompatible research"):
        load_pitch_circle(path)


@pytest.mark.parametrize("corruption", ["missing", "shape", "dtype", "nan", "source", "data"])
def test_loader_rejects_corrupt_state_and_provenance(tmp_path: Path, corruption: str) -> None:
    path = tmp_path / "model.pt"
    save_pitch_circle(path, PitchCircleNet(), metadata=metadata())
    payload = torch.load(path, weights_only=True)
    key = next(iter(payload["state_dict"]))
    if corruption == "missing":
        del payload["state_dict"][key]
    elif corruption == "shape":
        payload["state_dict"][key] = payload["state_dict"][key].flatten()
    elif corruption == "dtype":
        payload["state_dict"][key] = payload["state_dict"][key].double()
    elif corruption == "nan":
        payload["state_dict"][key].flatten()[0] = float("nan")
    elif corruption == "source":
        payload["metadata"]["source_sha256"]["experiments/pitch_circle_model.py"] = "0" * 64
    else:
        payload["metadata"]["feature_manifest_sha256"] = "missing"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_pitch_circle(path)


def metrics(fmak: float = 0.5) -> dict[str, Any]:
    return {
        "beatport": dict(mirex=0.5, mirex_asymmetric=0.5, exact=0.4),
        "fmak": dict(mirex=fmak, mirex_asymmetric=fmak, exact=0.4),
    }


def fixture_control(tmp_path: Path) -> tuple[Namespace, dict[str, Any], dict[str, Any]]:
    pins = json.loads(REVIEWED.read_bytes())
    archive, control_dir, output = (
        tmp_path / "archive",
        tmp_path / "control13",
        tmp_path / "candidate13",
    )
    for path in (archive, control_dir, output):
        path.mkdir()
    save_checkpoint(archive / "best.pt", KeyCNN())
    shutil.copyfile(archive / "best.pt", control_dir / "best.pt")
    archived: dict[str, Any] = dict(
        status="complete",
        arm="joint",
        seed=13,
        config={"lr": 0.001},
        parameters=109128,
        feature_manifest_sha256="d" * 64,
        source_sha256=pins["archived"]["sources"],
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
    control["source_sha256"] = pins["reusable_control"]["sources"]
    control["loss_spec"] = {"name": "hard-cross-entropy", "smoothing": 0.0}
    (control_dir / "run.json").write_text(json.dumps(control))
    validation = control_dir / "validation.json"
    validation.write_text(json.dumps(dict(metrics=control["domain_metrics"])))
    proof = dict(
        artifact_version="k10-structured-targets-v1",
        seed=13,
        kind="control",
        status="complete",
        control_reproduced=True,
        run_sha256=audio_digest(control_dir / "run.json"),
        checkpoint_sha256=audio_digest(control_dir / "best.pt"),
        validation_artifact=validation.name,
        validation_sha256=audio_digest(validation),
    )
    (control_dir / "structured-result.json").write_text(json.dumps(proof))
    candidate = copy.deepcopy(archived)
    candidate.update(parameters=73506, source_sha256=source_manifest())
    args = Namespace(
        pitch_circle_probe=True,
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


def test_control_reuse_checks_actual_states_history_and_proof_before_candidate_fit(
    tmp_path: Path,
) -> None:
    args, record, control = fixture_control(tmp_path)
    probe = PitchCircleFit(args, record, pitch_circle_hooks())
    assert probe.proof["exact_history_predictions_state"]
    assert (args.output / "PROTOCOL.md").read_bytes() == PROTOCOL.read_bytes()
    control["history"][0]["loss"] += 1e-12
    args.paired_control.write_text(json.dumps(control))
    with pytest.raises(ValueError, match="history"):
        PitchCircleFit(args, record, pitch_circle_hooks())


def test_recipe_audit_matches_all_frozen_controls_and_detects_changes() -> None:
    source = Path("experiments/k10_multidomain.py").read_text()
    pins = json.loads(REVIEWED.read_bytes())
    assert all(training_math(source) == value["training_math"] for value in pins.values())
    for before, after in (
        ("lr=0.001", "lr=0.002"),
        ("range(1, 121)", "range(1, 61)"),
        ("epoch - best_epoch >= 25", "epoch - best_epoch >= 24"),
    ):
        assert training_math(source.replace(before, after)) != training_math(source)


def test_optional_loader_scores_research_artifact_without_width_mislabel(tmp_path: Path) -> None:
    model = PitchCircleNet()
    save_pitch_circle(tmp_path / "best.pt", model, metadata=metadata())
    features = [np.ones((192, 8), dtype=np.float32), np.ones((192, 9), dtype=np.float32)]
    rows = [dict(id=f"{d}:v", domain=d, sha256=d) for d in ("beatport", "fmak")]
    refs = np.array([0, 1])
    expected = driver.domain_metrics(
        driver.evaluate_features(model, features, "cpu", 4), refs, rows
    )
    rng = torch.get_rng_state().clone()
    scored = score_selected(
        tmp_path,
        dict(domain_metrics=expected),
        features,
        refs,
        rows,
        "cpu",
        time.monotonic(),
        1200,
        loader=load_pitch_circle,
    )
    assert scored["metrics"] == expected and torch.equal(torch.get_rng_state(), rng)
    artifact = json.loads((tmp_path / "validation.json").read_bytes())
    assert artifact["ids"] == [r["id"] for r in rows] and artifact["refs"] == refs.tolist()
    assert np.asarray(artifact["logits"]).shape == (2, 24)
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        score_selected(
            tmp_path,
            dict(domain_metrics=expected),
            features,
            refs,
            rows,
            "cpu",
            time.monotonic(),
            1200,
        )


def candidate_result(seed: int, gain: float) -> dict[str, Any]:
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
            architecture=ARCHITECTURE,
            budget_seconds=1200,
        ),
        metrics=metrics(0.5 + gain),
        control_metrics=metrics(),
        bp_metrics=metrics(),
        method_gate=compare(metrics(), metrics(0.5 + gain)),
    )


def test_method_and_original_gates_never_exchange_exact_for_weighted_accuracy() -> None:
    results = [candidate_result(s, 0.015) for s in (13, 37, 73)]
    summary = summarize(results, VERSION)
    assert (
        summary["method_gate"]["qualifies_for_confirmation"] and not summary["original_hurdle_pass"]
    )
    results = [candidate_result(s, 0.03) for s in (13, 37, 73)]
    assert summarize(results, VERSION)["qualifies_for_external_freeze"]
    for result in results:
        result["metrics"]["beatport"]["exact"] -= 0.001
    assert not summarize(results, VERSION)["qualifies_for_external_freeze"]
    results[0]["status"] = "budget_inconclusive"
    with pytest.raises(ValueError, match="completed"):
        summarize(results, VERSION)


def test_default_and_identical_factory_hooks_have_exact_training_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A small synthetic crop keeps this a routing/reproducibility unit test, not a benchmark.
    monkeypatch.setattr(driver, "CROP_FRAMES", 8)
    rows = [
        dict(id=f"{domain}:{role}", domain=domain, role=role, ref=i, sha256=str(i))
        for i, (domain, role) in enumerate(
            (
                ("beatport", "train"),
                ("fmak", "train"),
                ("beatport", "validation"),
                ("fmak", "validation"),
            )
        )
    ]
    arrays = [np.random.default_rng(i).normal(size=(208, 8)).astype(np.float16) for i in range(4)]
    monkeypatch.setattr(driver, "load_data", lambda path: ({}, rows, arrays))
    monkeypatch.setattr(driver, "device_setup", lambda device: None)
    monkeypatch.setattr(driver, "score_mirex", lambda pred, refs, **kwargs: (0.5, 0.5))
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("synthetic")
    kwargs: list[dict[str, Any]] = []
    original_saver = driver.save_checkpoint

    def observe_default(path: Path, model: KeyCNN, *, width: float = 1.0, **values: object) -> None:
        kwargs.append(dict(values))
        original_saver(path, model, width=width, **values)

    monkeypatch.setattr(driver, "save_checkpoint", observe_default)
    args = Namespace(
        output=tmp_path / "default",
        data=data,
        device="cpu",
        seed=13,
        arm="joint",
        ema_probe=False,
        pitch_circle_probe=False,
    )
    driver.train(args)
    assert len(kwargs) == 1 and set(kwargs[0]) == {
        "epoch",
        "experiment",
        "arm",
        "seed",
        "source_sha256",
    }

    def same_writer(path: Path, model: nn.Module, *, metadata: dict[str, Any]) -> None:
        assert isinstance(model, KeyCNN)
        values = {k: v for k, v in metadata.items() if k != "feature_manifest_sha256"}
        original_saver(path, model, **values)

    hooks = ResearchModelHooks(
        "test-identical-k10", "k10-multidomain-development-v1", KeyCNN, same_writer, load_checkpoint
    )
    args.output = tmp_path / "hooks"
    driver.train(args, research=hooks)
    baseline, optional = [
        json.loads((tmp_path / name / "run.json").read_bytes()) for name in ("default", "hooks")
    ]
    for key in ("history", "best_epoch", "early_stopped", "domain_metrics", "config"):
        assert baseline[key] == optional[key]
    a = torch.load(tmp_path / "default" / "best.pt", weights_only=True)
    b = torch.load(tmp_path / "hooks" / "best.pt", weights_only=True)
    assert a["metadata"] == b["metadata"]
    assert all(torch.equal(value, b["state_dict"][key]) for key, value in a["state_dict"].items())


def test_shared_loop_writes_real_research_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver, "CROP_FRAMES", 8)
    rows = [
        dict(id=f"{domain}:{role}", domain=domain, role=role, ref=i, sha256=str(i))
        for i, (domain, role) in enumerate(
            (
                ("beatport", "train"),
                ("fmak", "train"),
                ("beatport", "validation"),
                ("fmak", "validation"),
            )
        )
    ]
    arrays = [np.ones((208, 8), dtype=np.float16) for _ in rows]
    monkeypatch.setattr(driver, "load_data", lambda path: ({}, rows, arrays))
    monkeypatch.setattr(driver, "device_setup", lambda device: None)
    monkeypatch.setattr(driver, "score_mirex", lambda pred, refs, **kwargs: (0.5, 0.5))
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("synthetic")
    args = Namespace(
        output=tmp_path / "research",
        data=data,
        device="cpu",
        seed=13,
        arm="joint",
        ema_probe=False,
        pitch_circle_probe=False,
    )
    driver.train(args, research=pitch_circle_hooks())
    model, checkpoint = load_pitch_circle(args.output / "best.pt")
    record = json.loads((args.output / "run.json").read_bytes())
    assert record["parameters"] == 73506 and record["best_epoch"] == 1
    assert len(record["history"]) == 26
    assert isinstance(model, PitchCircleNet)
    assert checkpoint["metadata"]["feature_manifest_sha256"] == audio_digest(data / "manifest.json")
    assert checkpoint["metadata"]["experiment"] == "pitch-circle-joint-development-v1"


def test_complete_preflight_scoring_and_paired_confusions(tmp_path: Path) -> None:
    args, record, control = fixture_control(tmp_path)
    rows = [
        dict(id=identity, domain=identity.split(":")[0], sha256=identity)
        for identity in record["validation_ids"]
    ]
    features = [np.ones((192, 8), dtype=np.float32) for _ in rows]
    refs = np.zeros(376, dtype=np.int64)
    baseline, _ = load_checkpoint(args.reference_run.with_name("best.pt"))
    control["domain_metrics"] = driver.domain_metrics(
        driver.evaluate_features(baseline, features, "cpu", 4), refs, rows
    )
    args.paired_control.write_text(json.dumps(control))
    archived = json.loads(args.reference_run.read_bytes())
    archived["domain_metrics"] = control["domain_metrics"]
    args.reference_run.write_text(json.dumps(archived))
    proof_path = args.paired_control.with_name("structured-result.json")
    proof = json.loads(proof_path.read_bytes())
    proof["run_sha256"] = audio_digest(args.paired_control)
    proof_path.write_text(json.dumps(proof))
    model = PitchCircleNet()
    record["domain_metrics"] = driver.domain_metrics(
        driver.evaluate_features(model, features, "cpu", 4), refs, rows
    )
    probe = PitchCircleFit(args, record, pitch_circle_hooks())
    (args.output / "run.json").write_text(json.dumps(record))
    save_pitch_circle(args.output / "best.pt", model, metadata=metadata())
    rng = torch.get_rng_state().clone()
    result = probe.finish(record, features, refs, rows, time.monotonic())
    assert result["status"] == "complete" and result["control_reproduced"]
    assert torch.equal(torch.get_rng_state(), rng)
    artifact = json.loads((args.output / result["validation_artifact"]).read_bytes())
    assert np.asarray(artifact["logits"]).shape == (376, 24)
    assert artifact["ids"] == record["validation_ids"] and artifact["refs"] == refs.tolist()
    for domain, count in (("beatport", 126), ("fmak", 250)):
        assert "mirex_asymmetric" in result["metrics"][domain]
        matrix = result["confusion_movements"][domain]["repo_symmetric_fifths"][
            "row_control_column_candidate"
        ]
        assert np.asarray(matrix).sum() == count
