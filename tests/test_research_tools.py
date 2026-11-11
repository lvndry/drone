from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from experiments.chunked import ATOL, RTOL, sha256
from tools.evaluate_chunked import benchmark, load_features, load_rows, selected_rows
from tools.research import evidence
from tools.research.contracts import (
    CachedTrack,
    load_cached_features,
    load_validation_tracks,
    select_performance_tracks,
)
from tools.research.evidence import ArtifactSet
from tools.research.measurements import runtime_environment, time_prediction


def prepared_inputs(tmp_path: Path) -> tuple[Path, Path]:
    cache = tmp_path / "features.npy"
    np.save(cache, np.arange(208 * 8, dtype=np.float16).reshape(208, 8))
    rows = [
        dict(
            id=f"fmak:{i:06}",
            domain="fmak",
            frames=8,
            sha256=f"{i:064x}",
            role="validation",
            cache=cache.name,
            cache_sha256=sha256(cache),
            ref=23,
        )
        for i in range(376)
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(dict(complete=True, feature_spec="cqt24-8oct-c1-pad4-hop4096-v2", rows=rows))
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(dict(rows=[dict(id=r["id"], sha256=r["sha256"], frames=8) for r in rows[:20]]))
    )
    return manifest, selection


def test_typed_inputs_preserve_legacy_feature_and_selection_outputs(tmp_path: Path) -> None:
    manifest, selection = prepared_inputs(tmp_path)
    tracks = load_validation_tracks(manifest)
    legacy = load_rows(manifest)
    assert [track.as_dict() for track in tracks] == legacy
    assert all(not hasattr(track, "ref") for track in tracks)
    chosen = select_performance_tracks(tracks, selection)
    assert [track.as_dict() for track in chosen] == selected_rows(legacy, selection)
    expected = np.load(tmp_path / "features.npy")[8:200][None].astype(np.float32)
    np.testing.assert_array_equal(load_cached_features(manifest, tracks[0]), expected)
    np.testing.assert_array_equal(load_features(manifest, legacy[0]), expected)


def test_cached_features_reject_nonfinite_values_and_directory_escape(tmp_path: Path) -> None:
    root = tmp_path / "prepared"
    root.mkdir()
    outside = tmp_path / "outside.npy"
    np.save(outside, np.zeros((208, 8), dtype=np.float16))
    track = CachedTrack("fmak:1", "fmak", 8, "audio", "../outside.npy", sha256(outside))
    with pytest.raises(ValueError, match="inside the prepared-data"):
        load_cached_features(root / "manifest.json", track)
    # A symlink must not bypass the same dataset boundary.
    (root / "link.npy").symlink_to(outside)
    linked = CachedTrack("fmak:1", "fmak", 8, "audio", "link.npy", sha256(outside))
    with pytest.raises(ValueError, match="inside the prepared-data"):
        load_cached_features(root / "manifest.json", linked)
    values = np.zeros((208, 8), dtype=np.float16)
    values[8, 0] = np.nan
    cache = root / "nan.npy"
    np.save(cache, values)
    changed = CachedTrack("fmak:1", "fmak", 8, "audio", cache.name, sha256(cache))
    with pytest.raises(ValueError, match="Nonfinite"):
        load_cached_features(root / "manifest.json", changed)


def test_support_source_change_invalidates_validation_without_entrypoint_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "tools/research"
    package.mkdir(parents=True)
    support = package / "contracts.py"
    support.write_text("version one")
    evidence_file = package / "evidence.py"
    evidence_file.write_text("fixed evidence")
    evaluator = tmp_path / "tools/evaluate_chunked.py"
    evaluator.write_text("unchanged entrypoint")
    runtime = tmp_path / "experiments/chunked.py"
    runtime.parent.mkdir()
    runtime.write_text("unchanged runtime")
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    monkeypatch.setattr(evidence, "__file__", str(evidence_file))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("contract.json", "prefix.onnx", "head.onnx"):
        (bundle / name).write_text(name)
    manifest, model = tmp_path / "manifest.json", tmp_path / "full.onnx"
    manifest.write_text("manifest")
    model.write_text("model")
    artifacts = ArtifactSet(manifest, bundle, model)
    report = dict(
        mode="validate",
        passed=True,
        tracks=376,
        atol=ATOL,
        rtol=RTOL,
        **artifacts.identity(),
        **runtime_environment(),
    )
    path = tmp_path / "validation.json"
    path.write_text(json.dumps(report))
    artifacts.require_validation(path)
    entrypoint_hash = sha256(evaluator)
    support.write_text("version two")
    assert sha256(evaluator) == entrypoint_hash
    with pytest.raises(ValueError, match="support_source_sha256"):
        artifacts.require_validation(path)


def test_shared_timing_keeps_one_warmup_and_all_repetitions() -> None:
    calls = []
    features = np.zeros((1, 192, 8), dtype=np.float32)

    def predict(value: np.ndarray) -> np.ndarray:
        calls.append(value)
        return np.full((1, 24), len(calls), dtype=np.float32)

    result = time_prediction(predict, features, repetitions=3)
    assert len(calls) == 4 and all(value is features for value in calls)
    assert len(result.seconds) == 3 and all(value >= 0 for value in result.seconds)
    np.testing.assert_array_equal(result.logits, np.full((1, 24), 4, dtype=np.float32))
    assert result.median_seconds == float(np.median(result.seconds))


def test_shared_package_stays_torch_free_in_a_fresh_process() -> None:
    code = (
        "import sys; import tools.research.contracts, tools.research.evidence, "
        "tools.research.measurements; assert 'torch' not in sys.modules; "
        "assert 'drone' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_failed_paired_evidence_is_rejected_before_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, selection = prepared_inputs(tmp_path)
    tracks = load_validation_tracks(manifest)
    chosen = select_performance_tracks(tracks, selection)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("contract.json", "prefix.onnx", "head.onnx"):
        (bundle / name).write_text(name)
    model = tmp_path / "full.onnx"
    model.write_text("full")
    artifacts = ArtifactSet(manifest, bundle, model)
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            dict(
                mode="validate",
                passed=True,
                tracks=376,
                atol=ATOL,
                rtol=RTOL,
                **artifacts.identity(),
                **runtime_environment(),
            )
        )
    )
    baseline = tmp_path / "baseline.json"
    records = [
        dict(
            id=t.id,
            sha256=t.sha256,
            cache_sha256=t.cache_sha256,
            frames=t.frames,
            median_seconds=0.1,
        )
        for t in chosen
    ]
    report = dict(
        mode="benchmark",
        bench_mode="full",
        torch_imported=False,
        tracks=20,
        repetitions=20,
        warmups_per_track=1,
        selection_sha256=sha256(selection),
        peak_process_rss_kib=1000,
        **artifacts.identity(),
        **runtime_environment(),
        records=records,
    )

    def unexpected_model_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("A model was loaded before evidence checks completed")

    monkeypatch.setattr("tools.evaluate_chunked.ChunkedK10", unexpected_model_load)
    args = Namespace(
        validation_report=validation,
        selection=selection,
        manifest=manifest,
        full_model=model,
        bundle=bundle,
        core_outputs=64,
        bench_mode="chunked",
        full_benchmark_report=baseline,
    )
    report["error"] = "failed previous benchmark"
    baseline.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="failed full benchmark"):
        benchmark(args, tracks)
    del report["error"]
    records[0]["median_seconds"] = 0.0
    baseline.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Invalid full benchmark measurement"):
        benchmark(args, tracks)


def test_native_fixture_export_preserves_fp16_core_and_is_label_blind(tmp_path: Path) -> None:
    manifest, selection = prepared_inputs(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "contract.json").write_text(json.dumps(dict(checkpoint_sha256="checkpoint")))
    for name in ("prefix.onnx", "head.onnx"):
        (bundle / name).write_text(name)
    model = tmp_path / "full.onnx"
    model.write_text("full model")
    artifacts = ArtifactSet(manifest, bundle, model)
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            dict(
                mode="validate",
                passed=True,
                tracks=376,
                atol=ATOL,
                rtol=RTOL,
                **artifacts.identity(),
                **runtime_environment(),
            )
        )
    )
    output = tmp_path / "native"
    code = """
import sys
from pathlib import Path
import numpy as np
from tools import export_native_fixtures as tool
from tools.research.evidence import ArtifactSet
class Session:
    def run(self, outputs, inputs):
        return [np.arange(24, dtype=np.float32)[None]]
tool.cpu_session = lambda _: Session()
manifest, selection, bundle, model, validation, output = map(Path, sys.argv[1:])
tool.export_fixtures(tool.ExportRequest(ArtifactSet(manifest, bundle, model), selection, validation, output))
assert 'torch' not in sys.modules
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(manifest),
            str(selection),
            str(bundle),
            str(model),
            str(validation),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((output / "manifest.json").read_bytes())
    assert report["complete"] and report["tracks"] == 376 and report["selected_tracks"] == 20
    assert not report["accuracy_scored"] and not report["torch_imported"]
    assert len(report["support_source_sha256"]) >= 4
    row = report["rows"][0]
    assert "ref" not in row and row["reference_logits"] == list(range(24))
    raw = output / row["path"]
    expected = np.load(tmp_path / "features.npy")[8:200].astype("<f4")
    assert raw.read_bytes() == expected.tobytes(order="C")
    assert sha256(raw) == row["sha256"]
    assert row["cache_sha256"] == row["source_sha256"]
    # Existing output evidence cannot be replaced by a second invocation.
    repeated = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(manifest),
            str(selection),
            str(bundle),
            str(model),
            str(validation),
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert repeated.returncode != 0 and "FileExistsError" in repeated.stderr
