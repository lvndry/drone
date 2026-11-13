"""Synthetic metadata and arithmetic contracts only: no model scoring or fitting."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from drone.keys import LABEL_SCHEMA
from experiments.myna_cache import (
    AUGMENTATION_METHOD,
    BASE_RESAMPLER,
    DOMAIN_MASS,
    ENCODER_SHA,
    PLAN_SCHEMA,
    SCHEMA,
    Contract,
    base_sample_count,
    load_cache,
    loss_weight,
    rate_geometry,
    validate_manifest,
)
from experiments.myna_pilot import (
    batch_proof,
    batches,
    continuation,
    make_head,
    mean_logits,
    weighted_ce,
)


def fixture(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, int],
    list[dict[str, str]],
    list[dict[str, Any]],
    dict[str, str],
]:
    pools = {
        "bp_fit": ["beatport:short", "beatport:long"],
        "expanded_fma_fit": ["fmak:fit"],
        "bp_selection_guard": ["beatport:dev"],
        "fma_inner_selection": ["fmak:dev"],
    }
    ids = [i for values in pools.values() for i in values]
    labels = {i: 23 for i in ids}
    plan = dict(pools, expected_refs=labels.copy())
    identities = {"encoder_sha256": ENCODER_SHA, "qualification_sha256": "f" * 64}
    tracks, groups, waves = [], [], []
    for identity in ids:
        role = "selection" if identity.endswith("dev") else "fit"
        domain = identity.split(":")[0]
        windows = 6 if identity.endswith("long") else 2
        n16 = windows * 196 * 512
        n24 = n16 * 3 // 2
        views = []
        for shift in range(-6, 7) if role == "fit" else [0]:
            name = identity.replace(":", "-") + f"-{shift}.npy"
            geometry = rate_geometry(n16, shift)
            data = np.full((geometry["windows"], 384), shift + 6, dtype=np.float32)
            np.save(root / name, data, allow_pickle=False)
            raw = (root / name).read_bytes()
            views.append(
                dict(
                    shift=shift,
                    **geometry,
                    dtype="float32",
                    shape=[geometry["windows"], 384],
                    window_order="all-complete-196-frame-windows-in-order",
                    raw_samples=geometry["samples_16k"],
                    trimmed_samples=0,
                    padded_samples=0,
                    path=name,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    bytes=len(raw),
                )
            )
        tracks.append(
            dict(
                id=identity,
                domain=domain,
                role=role,
                group=identity,
                target=labels[identity],
                source_sha256="a" * 64,
                pcm_sha256="b" * 64,
                samples_24k=n24,
                samples_16k=n16,
                mel_frames=windows * 196 + 1,
                complete_windows=windows,
                discarded_frames=1,
                views=views,
            )
        )
        groups.append(
            dict(id=identity, domain=domain, logical_role=role, derived_uncertainty_group=identity)
        )
        waves.append(
            dict(
                id=identity,
                role="validation" if role == "selection" else "fit",
                target=labels[identity],
                sha256="a" * 64,
                waveform_sha256="b" * 64,
                samples=n24,
            )
        )
    manifest = dict(
        schema_version=SCHEMA,
        augmentation_method=AUGMENTATION_METHOD,
        base_resampler=BASE_RESAMPLER,
        complete=True,
        label_schema=LABEL_SCHEMA,
        identities=identities,
        tracks=tracks,
    )
    return manifest, plan, labels, groups, waves, identities


def validate(
    root: Path,
    data: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, int],
        list[dict[str, str]],
        list[dict[str, Any]],
        dict[str, str],
    ],
) -> Contract:
    manifest, plan, labels, groups, waveforms, identities = data
    return validate_manifest(
        manifest, plan, labels, groups, waveforms, root, identities, require_production_counts=False
    )


def test_inverse_window_weights_equalize_tracks_and_preserve_domain_mass(tmp_path: Path) -> None:
    contract = validate(tmp_path, fixture(tmp_path))
    masses: dict[str, float] = {}
    for view in contract.views:
        if view.role == "fit":
            masses[view.identity] = (
                masses.get(view.identity, 0)
                + view.windows * loss_weight(view, contract) / contract.fit_windows
            )
    assert masses["beatport:short"] == pytest.approx(masses["beatport:long"])
    assert masses["beatport:short"] * 2 == pytest.approx(DOMAIN_MASS["beatport"])
    assert masses["fmak:fit"] == pytest.approx(DOMAIN_MASS["fmak"])
    assert sum(masses.values()) == pytest.approx(1)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "shifted_selection",
        "reserved",
        "wrong_role",
        "wrong_group",
        "wrong_pcm",
        "changed_target",
        "capped",
        "length",
        "path",
    ],
)
def test_cache_contract_rejects_semantic_corruption(tmp_path: Path, fault: str) -> None:
    data = fixture(tmp_path)
    manifest = data[0]
    row = manifest["tracks"][0]
    if fault == "missing":
        row["views"].pop()
    elif fault == "duplicate":
        row["views"][-1] = copy.deepcopy(row["views"][0])
    elif fault == "shifted_selection":
        manifest["tracks"][-1]["views"][0]["shift"] = 1
    elif fault == "reserved":
        manifest["tracks"][-1]["id"] = "fmak:reserved"
    elif fault == "wrong_role":
        row["role"] = "selection"
    elif fault == "wrong_group":
        row["group"] = "invented-group"
    elif fault == "wrong_pcm":
        row["pcm_sha256"] = "e" * 64
    elif fault == "changed_target":
        row["target"] = 0
    elif fault == "capped":
        row["views"][0]["windows"] = 19
    elif fault == "length":
        row["views"][0]["trimmed_samples"] = 1
    else:
        row["views"][0]["path"] = "../outside.npy"
    with pytest.raises(ValueError):
        validate(tmp_path, data)


def test_production_counts_cannot_accept_fixture(tmp_path: Path) -> None:
    manifest, plan, labels, groups, waves, identities = fixture(tmp_path)
    with pytest.raises(ValueError, match="approved role counts"):
        validate_manifest(manifest, plan, labels, groups, waves, tmp_path, identities)


def test_cache_loading_enforces_hash_and_memory_and_transposes_mode(tmp_path: Path) -> None:
    contract = validate(tmp_path, fixture(tmp_path))
    with pytest.raises(ValueError, match="RAM admission"):
        load_cache(contract, 1, lambda: None)
    cache = load_cache(contract, contract.embedding_bytes, lambda: None)
    assert cache.embeddings.shape == (contract.fit_windows + contract.selection_windows, 384)
    assert not cache.embeddings.flags.writeable
    assert set(cache.selection) == {"beatport:dev", "fmak:dev"}
    assert contract.views[0].target == 11  # B-minor minus six is F-minor, not major.
    first = contract.views[0]
    content = bytearray(first.path.read_bytes())
    content[-1] ^= 1
    first.path.write_bytes(content)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_cache(contract, contract.embedding_bytes, lambda: None)


def test_weighted_ce_does_not_normalize_random_batch_weights() -> None:
    logits = torch.zeros((2, 24))
    weights = torch.tensor([2.0, 4.0])
    loss = weighted_ce(logits, torch.tensor([0, 1]), weights)
    assert float(loss) == pytest.approx(3 * np.log(24))
    with pytest.raises(ValueError, match="objective input"):
        weighted_ce(logits, torch.tensor([0, 1]), torch.tensor([1.0, float("nan")]))


def test_complete_cache_passes_never_drop_tail_and_are_reproducible() -> None:
    first = list(batches(131, 2))
    assert [len(indices) for _, indices in first] == [64, 64, 3, 64, 64, 3]
    for epoch in (1, 2):
        np.testing.assert_array_equal(
            np.sort(np.concatenate([v for e, v in first if e == epoch])), np.arange(131)
        )
    for (e, a), (f, b) in zip(first, batches(131, 2), strict=True):
        assert e == f
        np.testing.assert_array_equal(a, b)


def test_logits_average_is_fp64_and_batch_partition_independent() -> None:
    rows = np.zeros((3, 24), dtype=np.float32)
    rows[:, 0] = [1e8, 1.0, -1e8]
    whole = mean_logits(iter([rows]), 3)
    split = mean_logits(iter([rows[:1], rows[1:]]), 3)
    assert whole.dtype == np.float64 and whole[0] == 1 / 3
    np.testing.assert_array_equal(whole, split)
    with pytest.raises(ValueError, match="Missing/extra"):
        mean_logits(iter([rows]), 4)


def test_head_shape_and_single_dropout_without_allocating_weights() -> None:
    head = make_head("meta")
    assert sum(p.numel() for p in head.parameters()) == 18_456_600
    drops = [m for m in head if isinstance(m, torch.nn.Dropout)]
    assert len(drops) == 1 and drops[0].p == 0.99
    assert isinstance(head[2], torch.nn.Dropout)


def test_continuation_keeps_practical_gate_and_batch_proof() -> None:
    control = {
        "fmak": {"exact_correct": 176},
        "beatport": {"exact_correct": 89, "symmetric_units10": 972, "asymmetric_units10": 947},
    }
    candidate = copy.deepcopy(control)
    candidate["fmak"]["exact_correct"] = 181
    assert continuation(candidate, control, True)
    assert not continuation(candidate, control, False)
    candidate["beatport"]["asymmetric_units10"] -= 1
    assert not continuation(candidate, control, True)
    a = {"track": [0.0] * 24}
    b = {"track": [0.0, 1e-7] + [0.0] * 22}
    assert not batch_proof(a, b)["passed"]  # tiny error still changes a tied decision


@pytest.mark.parametrize("mutation", ["float64", "nonfinite", "wrong_shape"])
def test_hash_correct_but_invalid_embedding_payload_is_rejected(
    tmp_path: Path, mutation: str
) -> None:
    data = fixture(tmp_path)
    view = data[0]["tracks"][0]["views"][0]
    values = np.zeros(tuple(view["shape"]), dtype=np.float32)
    if mutation == "float64":
        values = values.astype(np.float64)
    elif mutation == "nonfinite":
        values[0, 0] = np.nan
    else:
        values = np.zeros((view["shape"][0] + 1, 384), dtype=np.float32)
    path = tmp_path / view["path"]
    np.save(path, values, allow_pickle=False)
    raw = path.read_bytes()
    view["sha256"], view["bytes"] = hashlib.sha256(raw).hexdigest(), len(raw)
    contract = validate(tmp_path, data)
    with pytest.raises(ValueError, match="Invalid embedding"):
        load_cache(contract, contract.embedding_bytes, lambda: None)


def test_failed_pitch_qualification_cannot_admit_training() -> None:
    from experiments.myna_pilot import validate_qualification

    identities = {"encoder_sha256": ENCODER_SHA, "qualification_sha256": "a" * 64}
    proof = {
        "schema_version": "drone-myna-cache-admission-v1",
        "augmentation_method": AUGMENTATION_METHOD,
        "complete": True,
        "identities": {"encoder_sha256": ENCODER_SHA},
        "all_pitch_views_passed": False,
        "long_track_composed_parity_passed": True,
        "sustained_head_cost_measured": True,
    }
    with pytest.raises(ValueError, match="Unqualified pitch"):
        validate_qualification(proof, identities)
    proof["all_pitch_views_passed"] = True
    validate_qualification(proof, identities)
    proof["identities"] = {"encoder_sha256": "b" * 64}
    with pytest.raises(ValueError, match="identities differ"):
        validate_qualification(proof, identities)


def test_imported_key_semantics_must_come_from_the_pinned_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments import myna_pilot as pilot

    original = pilot.inspect.getsourcefile

    def substituted_origin(value: object) -> str | None:
        if value is pilot.score_mirex:
            return "/different-checkout/drone/keys.py"
        if not callable(value):
            raise TypeError("Expected a callable source")
        return original(value)

    monkeypatch.setattr(pilot.inspect, "getsourcefile", substituted_origin)
    config = {
        "schema_version": "drone-myna-head-pilot-v1",
        "passes": 10,
        "source_sha256": {name: "a" * 64 for name in pilot.SOURCES},
    }
    with pytest.raises(ValueError, match="Imported helper origin differs"):
        pilot.admit(config, lambda: None)


def test_metadata_projection_cannot_be_loaded_or_used_as_completed_cache(tmp_path: Path) -> None:
    manifest, plan, labels, groups, waves, original_identities = fixture(tmp_path)
    identities: dict[str, str | None] = dict(original_identities)
    identities["qualification_sha256"] = None
    manifest.update(schema_version=PLAN_SCHEMA, complete=False, identities=identities)
    for row in manifest["tracks"]:
        for view in row["views"]:
            for key in ("raw_samples", "trimmed_samples", "padded_samples", "sha256", "bytes"):
                view[key] = None
    projection = validate_manifest(
        manifest,
        plan,
        labels,
        groups,
        waves,
        tmp_path,
        identities,
        require_production_counts=False,
        metadata_only=True,
    )
    assert not projection.measured
    with pytest.raises(ValueError, match="metadata-only projection"):
        load_cache(projection, projection.embedding_bytes, lambda: None)
    with pytest.raises(ValueError, match="Incomplete cache/schema"):
        validate_manifest(
            manifest,
            plan,
            labels,
            groups,
            waves,
            tmp_path,
            identities,
            require_production_counts=False,
        )


def test_physical_waveform_relocation_preserves_roles_and_semantics(tmp_path: Path) -> None:
    from experiments.myna_cache import PINS, validate_waveform_lineage

    waves = fixture(tmp_path)[4]
    for row in waves:
        row.update(bytes=row["samples"] * 4, canonical_samples=row["samples"])
    canonical = {"complete": True, "rows": waves}
    physical = {
        "complete": True,
        "rows": copy.deepcopy(waves),
        "canonical_manifest_sha256": PINS["waveforms"],
        "plan_sha256": PINS["plan"],
        "labels_sha256": PINS["labels"],
        "provider": "readonly-disk-full-f32",
        "no_seek_quantization_role_or_label_changes": True,
    }
    validate_waveform_lineage(canonical, physical)
    physical["rows"][-1]["role"] = "selection"
    with pytest.raises(ValueError, match="role/target/support/hash differs"):
        validate_waveform_lineage(canonical, physical)


def test_rate_geometry_changes_length_without_changing_declared_tonic_step() -> None:
    original = 120 * 16000
    same = rate_geometry(original, 0)
    faster, slower = rate_geometry(original, 6), rate_geometry(original, -6)
    assert same["samples_16k"] == original and same["windows"] == 19
    assert faster["rate_down"] == 22627 and slower["rate_down"] == 11314
    assert faster["samples_16k"] == (original * 16000 + 22626) // 22627
    assert faster["samples_16k"] < original < slower["samples_16k"]
    assert faster["windows"] < same["windows"] < slower["windows"]
    assert all(
        abs(rate_geometry(original, shift)["pitch_cents_error"]) < 0.1 for shift in range(-6, 7)
    )


def test_each_variable_length_pitch_view_has_equal_objective_mass(tmp_path: Path) -> None:
    contract = validate(tmp_path, fixture(tmp_path))
    views = [v for v in contract.views if v.identity == "beatport:long"]
    assert len({v.windows for v in views}) > 1
    masses = [v.windows * loss_weight(v, contract) / contract.fit_windows for v in views]
    assert max(masses) == pytest.approx(min(masses))
    assert sum(masses) == pytest.approx(DOMAIN_MASS["beatport"] / 2)


def test_base_length_matches_pinned_torchaudio_float32_rounding() -> None:
    assert base_sample_count(51_723_051) == 34_482_032
    assert (51_723_051 * 2 + 2) // 3 == 34_482_034
    assert base_sample_count(2_880_000) == 1_920_000
    # The target is also limited by available convolution output.
    for n in (1, 2, 3, 1_000_001, 51_723_051):
        assert base_sample_count(n) <= 2 * (n // 3 + 1)
