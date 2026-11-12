"""Non-vacuous synthetic proofs of fixed head geometry, masks and study contracts."""

from __future__ import annotations

import copy
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from drone.features import audio_digest
from experiments.temporal_cache import (
    FrozenEncoder,
    atomic_npy,
    npy_bytes,
    preflight,
    reconstruct,
    validate_cache,
    within_tolerance,
)
from experiments.temporal_model import (
    Block,
    TemporalHead,
    better,
    gates,
    initial_heads,
    microbatches,
    permutation,
)
from experiments.temporal_study import load_head, save_head, state_digest, weighted_backward


@pytest.fixture(autouse=True)
def bounded_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def nonzero(head: TemporalHead) -> TemporalHead:
    with torch.no_grad():
        head.readout.weight.copy_(torch.linspace(-0.2, 0.2, 24 * 64).reshape(24, 64))
        head.readout.bias.copy_(torch.linspace(-0.1, 0.1, 24))
    return head


def test_parameter_counts_shared_initialization_zero_head_and_rng() -> None:
    before = torch.get_rng_state().clone()
    heads = initial_heads(13)
    assert torch.equal(before, torch.get_rng_state())
    assert {a: sum(p.numel() for p in h.parameters()) for a, h in heads.items()} == {
        "M": 108888,
        "O": 109400,
        "S": 109400,
    }
    assert state_digest(heads["O"].state_dict()) == state_digest(heads["S"].state_dict())
    for key, value in heads["M"].state_dict().items():
        reference = heads["O"].state_dict()[key]
        if value.shape != reference.shape:
            reference = reference[:, :, 2:3]
        assert torch.equal(value, reference)
    x = torch.randn(3, 64, 24, 29)
    lengths = torch.tensor([7, 20, 29])
    canonical = torch.randn(3, 24)
    for head in heads.values():
        residual = head(x, lengths)
        assert torch.equal(residual, torch.zeros_like(residual))
        assert torch.equal(canonical + residual, canonical)


def test_nonzero_mean_is_permutation_invariant_but_ordered_is_sensitive() -> None:
    heads = initial_heads(13)
    mean = nonzero(heads["M"]).double()
    ordered = nonzero(heads["O"]).double()
    x = torch.randn(1, 64, 24, 37, dtype=torch.float64)
    order = torch.from_numpy(permutation(13, "synthetic", 37, 1))
    length = torch.tensor([37])
    a, b = mean(x, length), mean(x[:, :, :, order], length)
    assert float(a.detach().abs().sum()) > 0
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)
    a, b = ordered(x, length), ordered(x[:, :, :, order], length)
    assert not torch.allclose(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kernel", [1, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_complete_masks_padded_vs_single_and_padding_gradient(
    kernel: int, dtype: torch.dtype
) -> None:
    head = nonzero(TemporalHead(kernel)).to(dtype)
    # Learned LN/depthwise/pointwise biases make missing masks observable.
    with torch.no_grad():
        for block in head.blocks:
            assert isinstance(block, Block)
            assert (
                block.norm.bias is not None
                and block.depth.bias is not None
                and block.point.bias is not None
            )
            block.norm.bias.fill_(0.7)
            block.depth.bias.fill_(0.9)
            block.point.bias.fill_(0.4)
        head.norm.bias.fill_(0.3)
    lengths = torch.tensor([11, 23, 39])
    x = torch.randn(3, 64, 24, 39, dtype=dtype, requires_grad=True)
    with torch.no_grad():
        x[0, :, :, 11:] = 1234
        x[1, :, :, 23:] = -987
    actual = head(x, lengths)
    singles = torch.cat(
        [head(x[j : j + 1, :, :, : int(n)], lengths[j : j + 1]) for j, n in enumerate(lengths)]
    )
    tol = 1e-10 if dtype == torch.float64 else 1e-5
    torch.testing.assert_close(actual, singles, rtol=tol, atol=tol)
    actual.sum().backward()
    assert x.grad is not None
    assert torch.equal(x.grad[0, :, :, 11:], torch.zeros_like(x.grad[0, :, :, 11:]))
    assert torch.equal(x.grad[1, :, :, 23:], torch.zeros_like(x.grad[1, :, :, 23:]))


def test_microbatch_gradient_weighting_and_actual_partial_batch() -> None:
    whole = nonzero(TemporalHead(5)).double()
    split = copy.deepcopy(whole)
    x = torch.randn(5, 64, 24, 31, dtype=torch.float64)
    lengths = torch.tensor([31, 19, 11, 27, 9])
    canonical = torch.randn(5, 24, dtype=torch.float64)
    refs = torch.tensor([0, 19, 7, 23, 2])
    weighted_backward(whole, x, lengths, canonical, refs, 5)
    for group in ([0, 1], [2], [3, 4]):
        weighted_backward(split, x[group], lengths[group], canonical[group], refs[group], 5)
    for a, b in zip(whole.parameters(), split.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-9, atol=1e-10)
    assert microbatches([0, 1, 2, 3], [4000, 4000, 9000, 12]) == [[0, 1], [2], [3]]


def test_independent_fixed_dev_and_fresh_epoch_permutations() -> None:
    before = torch.get_rng_state().clone()
    rng = np.random.default_rng(13)
    expected = np.random.default_rng(13).permutation(50)
    dev = permutation(13, "recording", 101, None)
    np.testing.assert_array_equal(dev, permutation(13, "recording", 101, None))
    assert not np.array_equal(dev, permutation(13, "recording", 101, 1))
    assert not np.array_equal(
        permutation(13, "recording", 101, 1), permutation(13, "recording", 101, 2)
    )
    assert sorted(dev.tolist()) == list(range(101))
    np.testing.assert_array_equal(rng.permutation(50), expected)
    assert torch.equal(before, torch.get_rng_state())


def test_exact_ce_epoch_selection_never_mirex() -> None:
    best = dict(exact=0.7, ce=0.4, epoch=5.0, mirex=0.9)
    assert better(dict(exact=0.71, ce=5.0, epoch=6.0, mirex=0.1), best)
    assert better(dict(exact=0.7, ce=0.39, epoch=6.0), best)
    assert not better(dict(exact=0.7, ce=0.4, epoch=6.0, mirex=1.0), best)
    assert better(dict(exact=0.7, ce=0.4, epoch=4.0), best)


def metric(bp_correct: int, fm_correct: int) -> dict[str, dict[str, Any]]:
    return {
        d: dict(tracks=n, exact=c / n, mirex=0.8, mirex_asymmetric=0.8)
        for d, n, c in (("beatport", 126, bp_correct), ("fmak", 250, fm_correct))
    }


def test_new_exact_candidate_and_separate_order_gates() -> None:
    canonical = metric(88, 170)
    outputs = {"O": metric(88, 175), "M": metric(88, 172), "S": metric(89, 173)}
    result = gates(canonical, outputs)
    assert result["candidate_pass"]
    assert not result["mechanism_pass"]
    outputs["S"] = metric(89, 172)
    assert gates(canonical, outputs)["mechanism_pass"]
    outputs["O"]["beatport"]["mirex"] = 0.79
    assert not gates(canonical, outputs)["candidate_pass"]
    outputs["M"] = metric(100, 200)
    assert gates(canonical, outputs)["sole_candidate"] == "O"


def row(frames: int = 17) -> dict[str, Any]:
    return dict(
        id="test:id",
        role="train",
        domain="beatport",
        ref=0,
        sha256="a" * 64,
        frames=frames,
        cache="input.npy",
        cache_sha256="b" * 64,
    )


def test_exact_cache_preflight_headers_disk_and_no_shortening(tmp_path: Path) -> None:
    manifest, size = preflight([row(17), dict(row(79), id="other")], {"identity": "fixed"}, 10**12)
    assert size["hidden_payload_bytes"] == 6144 * (2 + 9)
    for r in manifest["rows"]:
        value = np.zeros(r["hidden_shape"], dtype="<f4")
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        assert len(buffer.getvalue()) == r["hidden_npy_bytes"] == npy_bytes(tuple(value.shape))
    with pytest.raises(OSError):
        preflight([row()], {}, 1)
    with pytest.raises(MemoryError, match="4GiB"):
        preflight([row(8 * (4 * 1024**3 // 6144 + 1))], {}, 10**12)
    assert manifest["rows"][1]["hidden_shape"] == [64, 24, 9]


def test_head_artifact_roundtrip_and_source_state_corruption(tmp_path: Path) -> None:
    head = nonzero(TemporalHead(5))
    metadata = {"identity": "frozen", "arm": "O"}
    path = tmp_path / "head.pt"
    save_head(path, head, metadata)
    loaded = load_head(path, metadata)
    assert state_digest(head.state_dict()) == state_digest(loaded.state_dict())
    with pytest.raises(ValueError):
        load_head(path, {"identity": "wrong", "arm": "O"})
    payload = torch.load(path, weights_only=True)
    payload["state_dict"]["readout.bias"][0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_head(path, metadata)


def test_cache_proof_source_and_hash_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Integrity rejection does not need a2GiB real cache or any study data.
    own = {"source": "frozen"}
    monkeypatch.setattr("experiments.temporal_cache.identity", lambda: own)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(complete=True, identity=own, canonical_logits_sha256="x")))
    proof = dict(
        status="complete",
        recordings=2582,
        argmax_disagreements=0,
        all_logits_within_tolerance=True,
        manifest_sha256="wrong",
        canonical_logits_sha256="x",
    )
    (tmp_path / "reconstruction.json").write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="proof"):
        validate_cache(tmp_path, lambda: None)
    proof["manifest_sha256"] = audio_digest(path)
    (tmp_path / "reconstruction.json").write_text(json.dumps(proof))
    value = json.loads(path.read_text())
    value["identity"] = {"source": "corrupt"}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="source"):
        validate_cache(tmp_path, lambda: None)


def test_original_pool_reconstruction_and_zero_head_canonical() -> None:
    class Encoder(FrozenEncoder):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Identity(), nn.Dropout2d(0.2), nn.AdaptiveAvgPool2d((24, 1))
            )
            self.head = nn.Linear(1536, 24)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.net(x).flatten(1))

    encoder = Encoder().eval()
    hidden = torch.randn(2, 64, 24, 31)
    direct = encoder(hidden)
    rebuilt = reconstruct(encoder, hidden)
    assert torch.equal(direct, rebuilt)
    head = initial_heads(13)["O"]
    residual = head(hidden, torch.tensor([31, 31]))
    assert torch.equal(direct + residual, direct)
    assert within_tolerance(rebuilt.detach().numpy(), direct.detach().numpy())


def test_cache_success_then_hidden_and_label_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    own = {"source": "fixed-test"}
    monkeypatch.setattr("experiments.temporal_cache.identity", lambda: own)
    hidden = tmp_path / "hidden.npy"
    atomic_npy(hidden, np.zeros((64, 24, 1), dtype="<f4"))
    rows = []
    for i in range(2582):
        source = dict(
            row(8),
            id=f"track:{i}",
            hidden_shape=[64, 24, 1],
            hidden="hidden.npy",
            hidden_sha256=audio_digest(hidden),
        )
        rows.append(source)
    raw = tmp_path / "cqt-manifest.json"
    raw.write_text(json.dumps({"rows": rows}))
    canonical = tmp_path / "canonical-logits.npy"
    atomic_npy(canonical, np.zeros((2582, 24), dtype="<f4"))
    manifest: dict[str, Any] = dict(
        complete=True,
        identity=own,
        rows=copy.deepcopy(rows),
        canonical_logits_sha256=audio_digest(canonical),
        feature_manifest_sha256=audio_digest(raw),
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    proof = dict(
        status="complete",
        recordings=2582,
        argmax_disagreements=0,
        all_logits_within_tolerance=True,
        manifest_sha256=audio_digest(path),
        canonical_logits_sha256=audio_digest(canonical),
    )
    proof_path = tmp_path / "reconstruction.json"
    proof_path.write_text(json.dumps(proof))
    actual, logits = validate_cache(tmp_path, lambda: None)
    assert len(actual["rows"]) == 2582 and logits.shape == (2582, 24)
    # Even a recomputed cache-manifest proof cannot silently change original labels.
    manifest["rows"][0]["ref"] = 1
    path.write_text(json.dumps(manifest))
    proof["manifest_sha256"] = audio_digest(path)
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="label"):
        validate_cache(tmp_path, lambda: None)
    manifest["rows"][0]["ref"] = 0
    path.write_text(json.dumps(manifest))
    proof["manifest_sha256"] = audio_digest(path)
    proof_path.write_text(json.dumps(proof))
    atomic_npy(hidden, np.ones((64, 24, 1), dtype="<f4"))
    with pytest.raises(ValueError, match="hash"):
        validate_cache(tmp_path, lambda: None)


def test_confirmation_is_matched_encoder_head_cohorts(tmp_path: Path) -> None:
    from experiments.temporal_model import VERSION
    from experiments.temporal_study import confirmation_summary

    paths = []
    for seed in (13, 37, 73):
        canonical = metric(88, 170)
        o = metric(88, 175)
        m = metric(88, 172)
        s = metric(88, 172)
        payload = dict(
            version=VERSION,
            status="complete",
            identity={"frozen": "source"},
            encoder_seed=seed,
            head_seed=seed,
            canonical_metrics=canonical,
            arms={a: dict(metrics=v) for a, v in (("O", o), ("M", m), ("S", s))},
            gates={"candidate_pass": True},
            bp_control_mean=metric(85, 160),
            bp_references={str(seed): "fixed" for seed in (13, 37, 73)},
        )
        path = tmp_path / f"{seed}.json"
        path.write_text(json.dumps(payload))
        paths.append(path)
    result = confirmation_summary(paths)
    assert result["candidate_pass"] and result["mechanism_pass"]
    bad = json.loads(paths[1].read_text())
    bad["head_seed"] = 13
    paths[1].write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="cohort"):
        confirmation_summary(paths)


def test_cpu_phase_cannot_run_after_failed_accuracy(tmp_path: Path) -> None:
    import argparse

    from experiments.temporal_cpu import screen

    study = tmp_path / "study"
    study.mkdir()
    (study / "result.json").write_text(
        json.dumps(dict(status="complete", gates={"candidate_pass": False}))
    )
    output = tmp_path / "cpu"
    with pytest.raises(ValueError, match="requires passing"):
        screen(argparse.Namespace(study=study, output=output))
    assert not output.exists()


def test_synthetic_nonzero_full_export_dynamic_time(tmp_path: Path) -> None:
    from typing import cast

    ort = pytest.importorskip("onnxruntime")
    from drone.model import KeyCNN
    from experiments.temporal_cpu import FullModel, export_full

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        model = FullModel(
            cast(FrozenEncoder, KeyCNN().eval()), nonzero(initial_heads(13)["O"])
        ).eval()
    path = tmp_path / "full.onnx"
    export_full(model, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(13)
    for length in (8, 17, 39, 319, 645):
        value = torch.randn(1, 192, length, generator=generator)
        with torch.inference_mode():
            reference = model(value).numpy()
        exported = np.asarray(session.run(None, {"cqt": value.numpy()})[0])
        assert within_tolerance(exported, reference)
