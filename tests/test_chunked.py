from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from drone.checkpoints import save_checkpoint
from drone.model import KeyCNN
from experiments.chunked import ATOL, RTOL, ChunkedK10, compare_logits, core_windows
from experiments.chunked import sha256 as file_hash
from tools.evaluate_chunked import benchmark, selected_rows, source_provenance
from tools.export_chunked import HiddenPrefix, export_bundle
from tools.research.measurements import runtime_environment

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("onnx", "onnxscript", "onnxruntime")),
    reason="Install the export extra to verify chunked ONNX deployment",
)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, KeyCNN]:
    torch.set_num_threads(1)
    torch.manual_seed(702)
    model = KeyCNN().eval()
    # Nonidentity eval BN checks that artificial boundary padding is not equivalent.
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                assert layer.running_mean is not None and layer.running_var is not None
                layer.running_mean.uniform_(-0.4, 0.4)
                layer.running_var.uniform_(0.5, 1.5)
                layer.bias.uniform_(-0.2, 0.2)
    root = tmp_path_factory.mktemp("chunked")
    checkpoint = root / "synthetic.pt"
    save_checkpoint(checkpoint, model)
    destination = root / "bundle"
    export_bundle(checkpoint, destination)
    return destination, model


@pytest.mark.parametrize("core", [1, 2, 7, 64])
@pytest.mark.parametrize("remainder", range(8))
def test_aligned_coverage_and_receptive_field(core: int, remainder: int) -> None:
    frames = 8 * (core + 3) + remainder
    coverage = []
    for first_output, end_output, window_start, window_end in core_windows(frames, core):
        assert window_start % 8 == 0
        assert window_end - window_start <= 8 * core + 32
        assert end_output - window_start // 8 <= (window_end - window_start) // 8
        for output_index in range(first_output, end_output):
            assert window_start == 0 or window_start <= 8 * output_index - 14
            assert window_end == frames or window_end > 8 * output_index + 21
            coverage.append(output_index)
    assert coverage == list(range(frames // 8))


@pytest.mark.parametrize("core", [1, 2, 7, 64])
@pytest.mark.parametrize("kind", ["random", "constant", "negative", "dynamic", "impulses"])
def test_prefix_chunks_preserve_edges_and_all_remainders(
    bundle: tuple[Path, KeyCNN], core: int, kind: str
) -> None:
    path, model = bundle
    runtime = ChunkedK10(path, core)
    generator = np.random.default_rng(300)
    for remainder in range(8):
        frames = 8 * (core + 1) + remainder
        features = generator.normal(size=(1, 192, frames)).astype(np.float32)
        if kind == "constant":
            features.fill(0.7)
        elif kind == "negative":
            features = -np.abs(features)
        elif kind == "dynamic":
            features *= np.geomspace(1e-3, 1e3, frames).astype(np.float32)
        elif kind == "impulses":
            features.fill(0)
            for index in (0, frames - 1, 8 * core - 1, 8 * core, 8 * core + 1):
                features[0, :, index] = generator.normal(size=192).astype(np.float32)
        full_hidden = runtime.hidden(features)
        with torch.inference_mode():
            torch_hidden = HiddenPrefix(model.net)(torch.from_numpy(features)).numpy()
            expected_logits = model(torch.from_numpy(features)).numpy()
        # Hidden cross-runtime differences are diagnostic, not the public logit
        # promotion gate. Extreme cancellation can exceed its logit tolerance
        # at individual hidden elements even without any chunking.
        assert np.isfinite(torch_hidden).all()
        if kind != "dynamic":
            np.testing.assert_allclose(full_hidden, torch_hidden, atol=ATOL, rtol=RTOL)
        for first_output, end_output, window_start, window_end in core_windows(frames, core):
            hidden = runtime.hidden(np.ascontiguousarray(features[:, :, window_start:window_end]))
            if kind != "dynamic":
                np.testing.assert_allclose(
                    hidden[
                        :, :, :, first_output - window_start // 8 : end_output - window_start // 8
                    ],
                    full_hidden[:, :, :, first_output:end_output],
                    atol=ATOL,
                    rtol=RTOL,
                )
        # Extreme magnitudes can increase cross-runtime absolute error; compare
        # chunked to the same ORT prefix here, retaining the strict public gate.
        np.testing.assert_allclose(
            runtime.predict(features), runtime.full_prefix_logits(features), atol=ATOL, rtol=RTOL
        )
        check = compare_logits(runtime.predict(features), expected_logits)
        assert check["allclose"] and check["argmax_agrees"]


@pytest.mark.parametrize("core", [1, 2, 7, 64])
def test_float64_halo_alignment_under_extreme_cancellation(
    bundle: tuple[Path, KeyCNN], core: int
) -> None:
    # Independent same-model FP64 geometry check. This is not a claim of
    # bitwise FP32 ORT equivalence; the public promotion gate is unchanged.
    _, model = bundle
    prefix = HiddenPrefix(copy.deepcopy(model).double().net).eval()
    generator = torch.Generator().manual_seed(300)
    for remainder in range(8):
        frames = 8 * (core + 1) + remainder
        features = torch.randn(1, 192, frames, dtype=torch.float64, generator=generator)
        features *= torch.logspace(-3, 3, frames, dtype=torch.float64)
        with torch.inference_mode():
            full = prefix(features)
            for first_output, end_output, window_start, window_end in core_windows(frames, core):
                hidden = prefix(features[:, :, window_start:window_end].contiguous())
                torch.testing.assert_close(
                    hidden[
                        :, :, :, first_output - window_start // 8 : end_output - window_start // 8
                    ],
                    full[:, :, :, first_output:end_output],
                    atol=1e-10,
                    rtol=1e-10,
                )


def test_minimum_input_and_invalid_contracts(bundle: tuple[Path, KeyCNN]) -> None:
    path, _ = bundle
    runtime = ChunkedK10(path)
    features = np.zeros((1, 192, 8), dtype=np.float32)
    np.testing.assert_allclose(
        runtime.predict(features), runtime.full_prefix_logits(features), atol=ATOL, rtol=RTOL
    )
    for bad in (
        features[:, :, :7],
        features.astype(np.float64),
        np.full_like(features, np.nan),
        features[0],
    ):
        with pytest.raises(ValueError):
            runtime.predict(bad)
    with pytest.raises(ValueError):
        list(core_windows(8, 0))
    with pytest.raises(ValueError):
        ChunkedK10(path, 0)
    with pytest.raises(FileExistsError):
        export_bundle(path.parent / "synthetic.pt", path)


def test_artifact_integrity_is_checked(bundle: tuple[Path, KeyCNN], tmp_path: Path) -> None:
    path, _ = bundle
    metadata = json.loads((path / "contract.json").read_text())
    metadata["prefix_sha256"] = "changed"
    (tmp_path / "contract.json").write_text(json.dumps(metadata))
    (tmp_path / "prefix.onnx").write_bytes((path / "prefix.onnx").read_bytes())
    with pytest.raises(ValueError, match="Changed prefix"):
        ChunkedK10(tmp_path)


def test_pure_runtime_does_not_import_torch(bundle: tuple[Path, KeyCNN]) -> None:
    path, _ = bundle
    code = (
        "import sys, numpy as np; from experiments.chunked import ChunkedK10; "
        "runtime=ChunkedK10(sys.argv[1]); "
        "runtime.predict(np.zeros((1,192,8),dtype=np.float32)); "
        "assert 'torch' not in sys.modules; assert 'drone' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code, str(path)], check=True)


def test_numerical_gate_rejects_near_tie_flip() -> None:
    baseline = np.zeros((1, 24), dtype=np.float32)
    baseline[0, 0] = 1e-7
    changed = baseline.copy()
    changed[0, 1] = 2e-7
    comparison = compare_logits(changed, baseline)
    assert comparison["allclose"]
    assert not comparison["argmax_agrees"]


@pytest.mark.parametrize("source", ["runtime_source_sha256", "evaluator_source_sha256"])
def test_benchmark_rejects_code_changed_after_validation(
    bundle: tuple[Path, KeyCNN], tmp_path: Path, source: str
) -> None:
    path, _ = bundle
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    full_model = (
        path / "prefix.onnx"
    )  # Identity only; must never be executed in this rejection test.
    report = {
        "mode": "validate",
        "passed": True,
        "tracks": 376,
        "manifest_sha256": file_hash(manifest),
        "full_model_sha256": file_hash(full_model),
        "contract_sha256": file_hash(path / "contract.json"),
        "prefix_sha256": file_hash(path / "prefix.onnx"),
        "head_sha256": file_hash(path / "head.onnx"),
        "core_outputs": 64,
        "atol": ATOL,
        "rtol": RTOL,
        **source_provenance(),
        **runtime_environment(),
    }
    report[source] = "previous-code-hash"
    validation_report = tmp_path / "validation.json"
    validation_report.write_text(json.dumps(report))
    args = Namespace(
        validation_report=validation_report,
        selection=tmp_path / "unused.json",
        manifest=manifest,
        full_model=full_model,
        bundle=path,
        core_outputs=64,
    )
    with pytest.raises(ValueError, match="Validation report does not pass"):
        benchmark(args, [])


def test_existing_public_selection_uses_canonical_sha256(tmp_path: Path) -> None:
    rows = [
        {"id": str(index), "domain": "fmak", "frames": 320 + index, "sha256": f"{index:064x}"}
        for index in range(20)
    ]
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "id": row["id"],
                        "frames": row["frames"],
                        "sha256": row["sha256"],
                        "path": f"/public/fmak/{row['id']}.mp3",
                    }
                    for row in rows
                ]
            }
        )
    )
    assert selected_rows(rows, selection) == rows
    # Matching must be unique within the permitted public validation domain.
    with pytest.raises(ValueError, match="uniquely match"):
        selected_rows([*rows, dict(rows[0], id="duplicate")], selection)
    duplicated_selection = {"rows": [dict(rows[0]) for _ in range(20)]}
    selection.write_text(json.dumps(duplicated_selection))
    with pytest.raises(ValueError, match="duplicate audio"):
        selected_rows(rows, selection)
    # Do not silently accept a different identity schema.
    selection.write_text(
        json.dumps(
            {
                "rows": [
                    {"id": row["id"], "frames": row["frames"], "sha": row["sha256"]} for row in rows
                ]
            }
        )
    )
    with pytest.raises(KeyError, match="sha256"):
        selected_rows(rows, selection)
