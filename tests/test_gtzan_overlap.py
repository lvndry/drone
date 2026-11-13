"""Synthetic contract/flow checks; never decode audio, import vendor, or query real index."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from experiments import gtzan_overlap as worker
from tools import run_gtzan_overlap as guardian


def plan() -> dict[str, Any]:
    return dict(
        version=worker.VERSION,
        index_sha256=worker.INDEX_SHA,
        descriptor_sha256=worker.FULL_SHA,
        cache_root="/unused",
        vendor_path="/unused/vendor",
        artifact_sha256={},
        dictionary=[],
        primary_ids=[f"item{i}" for i in range(769)],
        cohort=[
            dict(id=f"item{i}", member=f"genres/mock/{i}.wav", container_sha256="a" * 64)
            for i in range(837)
        ],
        controls=[
            dict(
                id=f"control{i}",
                panel="positive" if i < 180 else "negative",
                path=f"/unused/{i}.npz",
                sha256="mock",
                expected_ref=7 if i < 180 else None,
            )
            for i in range(244)
        ],
        fresh_controls=[
            dict(
                id=f"fresh{i}",
                domain=domain,
                duration_seconds=30,
                offset_seconds=0.137,
                source_path="/unused/source",
                source_container_sha256="mock",
                expected_ref=7,
            )
            for i, domain in enumerate(("beatport", "fmak", "giantsteps"))
        ],
    )


def config() -> dict[str, Any]:
    return dict(
        version="gtzan-overlap-runtime-v1",
        mode="qualification",
        not_before_epoch=10,
        deadline_epoch=100,
        caps=dict(guardian.MAXIMUMS),
        cpus=[8, 9],
        protected_starts=[dict(pid=i, start_ticks=i) for i in range(4)],
        gpu_uuids=["GPU-test"],
        host_reserve_bytes=40 * 1024**3,
        gpu_reserve_bytes=8 * 1024**3,
        disk_reserve_bytes=1024**3,
    )


def permit() -> dict[str, Any]:
    return dict(
        authorized=True,
        config_sha256="bound",
        mode="qualification",
        allow_classifier_outputs=False,
        deadline_epoch=100,
    )


@pytest.mark.parametrize("defect", ["identity", "hash", "public_path"])
def test_portable_controls_reject_wrong_operational_source(defect: str) -> None:
    control = dict(
        expected_ref_id=7,
        expected_identity="fmak:42",
        source_container_sha256="bound",
        domain="fmak",
        offset_seconds=37.125,
    )
    source = dict(
        ref_id=7,
        id="fmak:42",
        container_sha256="bound",
        domain="fmak",
        seconds=120,
        path="/unused/source.mp3",
    )
    if defect == "identity":
        source["id"] = "fmak:43"
    elif defect == "hash":
        source["container_sha256"] = "different"
    else:
        control["source_path"] = "/unused/override.mp3"
    with pytest.raises(ValueError, match="Fresh control|local source paths"):
        guardian.resolve_fresh_controls({"controls": [control]}, [source])


def test_permit_is_exact_mode_bound_and_finite() -> None:
    guardian.check_contract(config(), permit(), "bound", 20)
    with pytest.raises(ValueError, match="permit"):
        guardian.check_contract(config(), permit(), "different", 20)
    changed = config() | {"mode": "audit"}
    with pytest.raises(ValueError, match="permit"):
        guardian.check_contract(changed, permit(), "bound", 20)
    for value in (True, float("nan"), float("inf"), 0, 3601):
        changed = config()
        changed["caps"]["wall_seconds"] = value
        with pytest.raises(ValueError):
            guardian.check_contract(changed, permit(), "bound", 20)


def test_frozen_membership_and_no_label_input() -> None:
    worker.validate_plan(plan(), "qualification")
    for change in ("duplicate", "label", "traversal", "excluded"):
        value = plan()
        if change == "duplicate":
            value["cohort"][0] = value["cohort"][1]
        elif change == "label":
            value["cohort"][0]["key"] = 4
        elif change == "traversal":
            value["cohort"][0]["member"] = "genres/../outside.wav"
        else:
            value["cohort"][0]["member"] = "genres/jazz/jazz.00054.wav"
        with pytest.raises(ValueError):
            worker.validate_plan(value, "audit")


def test_negative_reviews_do_not_fail_operational_gate() -> None:
    rows = [
        dict(
            panel="positive",
            complete=True,
            expected_retrieved=True,
            identity_flagged=True,
            review_flagged=False,
        ),
        dict(
            panel="negative",
            complete=True,
            expected_retrieved=False,
            identity_flagged=False,
            review_flagged=True,
        ),
    ]
    assert worker.control_summary(rows)["passed"]
    rows[1]["identity_flagged"] = True
    assert not worker.control_summary(rows)["passed"]


def setup_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MagicMock:
    index = MagicMock()
    monkeypatch.setattr(worker.AllOffsetIndex, "from_verified_artifacts", lambda *a, **k: index)
    monkeypatch.setattr(
        worker,
        "sha256",
        lambda p: (
            worker.FULL_SHA
            if p.name == "result.json"
            else worker.INDEX_SHA
            if p.name == "index.sqlite"
            else "mock"
        ),
    )
    monkeypatch.setattr(worker, "load", lambda p: {})
    monkeypatch.setattr(worker, "verify_dictionary", lambda *a: None)
    monkeypatch.setattr(worker, "save_features", lambda *a: dict(seconds=30, query_shifts=4))

    def crop(row: dict[str, Any], target: Path, ffmpeg: str) -> dict[str, Any]:
        target.write_bytes(b"mock crop")
        return {}

    monkeypatch.setattr(worker, "fresh_pcm", crop)
    return index


def fake_evidence(*args: object, **kwargs: object) -> dict[str, Any]:
    panel = args[8]
    return dict(
        id=args[5],
        panel=panel,
        complete=True,
        expected_retrieved=panel != "negative",
        identity_flagged=panel != "negative",
        review_flagged=panel == "negative",
        outcome="no_detected_overlap",
    )


def test_qualification_never_opens_gtzan_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    index = setup_worker(monkeypatch, tmp_path)
    monkeypatch.setattr(worker, "write_evidence", fake_evidence)
    decoded = []

    def decode(path: str, **kwargs: object) -> tuple[np.ndarray, str]:
        decoded.append((path, kwargs))
        return np.zeros(30 * 11025, dtype=np.float32), "mock"

    monkeypatch.setattr(worker, "decode", decode)
    monkeypatch.setattr(worker.tarfile, "open", lambda *a, **k: pytest.fail("GTZAN archive opened"))
    worker.run(
        plan(),
        dict(
            mode="qualification",
            deadline_epoch=10**12,
            query_seconds=120,
            qualification_binding_sha256="binding",
            ffmpeg_path="/mock/ffmpeg",
        ),
        tmp_path,
    )
    result = guardian.load(tmp_path / "result.json")
    assert result["qualification_passed"]
    assert len(result["results"]) == 247
    assert len(decoded) == 3
    assert all(path.endswith("/stage/fresh.wav") and kwargs == {} for path, kwargs in decoded)
    index.db.close.assert_called_once()


def test_failed_cached_control_prevents_fresh_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup_worker(monkeypatch, tmp_path)

    def failed(*args: object, **kwargs: object) -> dict[str, Any]:
        return fake_evidence(*args, **kwargs) | {"expected_retrieved": False}

    monkeypatch.setattr(worker, "write_evidence", failed)
    monkeypatch.setattr(worker, "decode", lambda *a, **k: pytest.fail("Audio opened before gate"))
    with pytest.raises(ValueError, match="operational"):
        worker.run(
            plan(), dict(mode="qualification", deadline_epoch=10**12, query_seconds=120), tmp_path
        )
    assert not (tmp_path / "result.json").exists()


def test_stage_rejects_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        worker.owned_directory(tmp_path)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_whole_tree_cpu_breach_terminates_only_owned_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cleanup_fails: bool
) -> None:
    value = config() | dict(output_root=str(tmp_path / "output"))
    monkeypatch.setattr(guardian, "load", lambda p: permit() if p.name == "permit" else value)
    monkeypatch.setattr(guardian, "sha256", lambda p: "bound")
    monkeypatch.setattr(guardian, "environment", lambda c: None)
    monkeypatch.setattr(guardian, "verify_bindings", lambda c: None)
    monkeypatch.setattr(guardian, "execution_plan", lambda c: {})
    monkeypatch.setattr(guardian, "reserve_check", lambda *a: None)
    monkeypatch.setattr(guardian.time, "time", lambda: 20)
    monkeypatch.setattr(guardian, "start_identity", lambda p: 11)
    monkeypatch.setattr(guardian, "process_tree", lambda p: (100, 7201))
    child = MagicMock(pid=12345)
    child.poll.return_value = None
    monkeypatch.setattr(guardian.subprocess, "Popen", lambda *a, **k: child)
    stopped = MagicMock()
    if cleanup_fails:
        stopped.side_effect = RuntimeError("cleanup identity changed")
    monkeypatch.setattr(guardian, "terminate_own", stopped)
    with pytest.raises(RuntimeError, match="owned-tree"):
        guardian.supervise(tmp_path / "config", tmp_path / "permit", tmp_path / "output")
    stopped.assert_called_once_with(child, 11)
    assert (tmp_path / "output/failure.json").exists()
    import json

    receipt = json.loads((tmp_path / "output/failure.json").read_text())
    assert "owned-tree" in receipt["error"]
    assert receipt["cleanup_error"] == (
        "RuntimeError('cleanup identity changed')" if cleanup_fails else None
    )


def test_identity_at_wrong_offset_does_not_pass_fresh_control(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    matches = [
        dict(ref_id=7, decision="identity_evidence", offset_seconds=0.0),
        dict(ref_id=8, decision="review", offset_seconds=37.125, chromaprint_distance=0.4),
    ]
    monkeypatch.setattr(
        worker, "match_query", lambda *a: (dict(complete=True, matches=matches), True)
    )
    monkeypatch.setattr(worker, "sha256", lambda p: "mock")
    result = worker.write_evidence(
        MagicMock(), tmp_path / "query.npz", tmp_path, tmp_path, 0, "fresh", 100, 7, "fresh", 37.125
    )
    assert not result["expected_retrieved"]
    assert result["identity_flagged"] and result["review_flagged"]
    evidence = guardian.load(tmp_path / result["evidence"])
    assert evidence["match"]["matches"] == matches


@pytest.mark.parametrize("scenario", ["ordinary", "vote_budget", "unknown_failure", "continuation"])
def test_streamed_union_is_queried_once_with_primary_membership(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scenario: str
) -> None:
    import hashlib
    import io
    import tarfile

    setup_worker(monkeypatch, tmp_path)
    value = plan()
    value["cohort"] = value["cohort"][:3]
    value["primary_ids"] = [value["cohort"][1]["id"]]
    archive_path = tmp_path / "mock.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for number, row in enumerate(value["cohort"]):
            raw = f"synthetic payload{number}".encode()
            row["container_sha256"] = hashlib.sha256(raw).hexdigest()
            info = tarfile.TarInfo(row["member"])
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    value.update(archive_path=str(archive_path), archive_sha256=worker.ARCHIVE_SHA)
    monkeypatch.setattr(worker, "validate_plan", lambda *a: None)

    def digest(path: Path) -> str:
        if path == archive_path:
            return worker.ARCHIVE_SHA
        return worker.FULL_SHA if path.name == "result.json" else worker.INDEX_SHA

    monkeypatch.setattr(worker, "sha256", digest)
    monkeypatch.setattr(worker, "decode", lambda *a: (np.zeros(11025, dtype=np.float32), "pcm"))
    called = []

    def matched(*args: object, **kwargs: object) -> dict[str, Any]:
        row = fake_evidence(*args, **kwargs)
        called.append(row["id"])
        if scenario in {"vote_budget", "unknown_failure"} and len(called) == 1:
            row.update(
                complete=False,
                informative=True,
                outcome="insufficient_information",
                reason="vote_memory_budget" if scenario == "vote_budget" else "unexpected_failure",
            )
        return row

    monkeypatch.setattr(worker, "write_evidence", matched)
    output = tmp_path / "owned"
    output.mkdir()
    execution: dict[str, Any] = dict(
        mode="audit",
        deadline_epoch=10**12,
        query_seconds=120,
        qualification_binding_sha256="binding",
    )
    if scenario == "continuation":
        execution["continuation"] = {"archive_order_ids": ["item0", "item1", "item2"]}
        imported = [
            dict(id="item0", primary769=False, complete=True, outcome="no_detected_overlap"),
            dict(
                id="item1",
                primary769=True,
                complete=False,
                reason="vote_memory_budget",
                outcome="insufficient_information",
            ),
        ]
        monkeypatch.setattr(worker, "prior_ledger", lambda *a: (imported, {"item2"}))
    if scenario == "unknown_failure":
        with pytest.raises(RuntimeError, match="Unknown/nonlocal"):
            worker.run(value, execution, output)
        assert called == ["item0"]
        assert not (output / "result.json").exists()
        partial = guardian.load(output / "partial.json")
        assert [r["status"] for r in partial["cohort_ledger"]] == [
            "query_incomplete",
            "unvisited",
            "unvisited",
        ]
        return
    worker.run(value, execution, output)
    result = guardian.load(output / "result.json")
    assert len(result["results"]) == 3
    assert [r["primary769"] for r in result["results"]] == [False, True, False]
    assert not list((output / "stage").iterdir())
    assert not result["qualification_passed"]
    if scenario in {"vote_budget", "continuation"}:
        assert result["status"] == "complete_with_unresolved"
        assert result["population_status"]["sensitivity837"]["unresolved"] == 1
        assert result["all_queries_attempted"] and not result["all_queries_complete"]
    assert called == (["item2"] if scenario == "continuation" else ["item0", "item1", "item2"])


def test_shared_binary_different_venv_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binary = tmp_path / "base-python"
    binary.write_bytes(b"same executable binary")
    first, second = tmp_path / "venv-one/python", tmp_path / "venv-two/python"
    for path in (first, second):
        path.parent.mkdir()
        path.symlink_to(binary)
    assert first.resolve() == second.resolve()
    monkeypatch.setattr(guardian.sys, "executable", str(first))
    monkeypatch.setattr(guardian.os, "sched_getaffinity", lambda p: {8, 9}, raising=False)
    monkeypatch.setattr(guardian.os, "getpriority", lambda *a: 19, raising=False)
    monkeypatch.setattr(guardian.os, "PRIO_PROCESS", 0, raising=False)
    monkeypatch.setattr(guardian.shutil, "which", lambda name: "/mock/ffmpeg")
    monkeypatch.chdir(guardian.ROOT)
    monkeypatch.delenv("PYTHONHOME", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in guardian.THREADS:
        monkeypatch.setenv(name, "1")
    value = config() | dict(ffmpeg_path="/mock/ffmpeg", python_path=str(first))
    guardian.environment(value)
    with pytest.raises(ValueError):
        guardian.environment(value | {"python_path": str(second)})
    monkeypatch.setenv("PYTHONPATH", "")
    with pytest.raises(ValueError):
        guardian.environment(value)
    monkeypatch.delenv("PYTHONPATH")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        guardian.environment(value)


def test_import_origin_and_loaded_package_init_must_be_pinned(tmp_path: Path) -> None:
    from types import ModuleType

    foreign = ModuleType("tools.foreign")
    path = tmp_path / "foreign.py"
    path.write_text("# foreign editable module")
    foreign.__file__ = str(path)
    with pytest.raises(ValueError, match="Foreign local import"):
        guardian.check_import_origin("tools.foreign", foreign, {})
    package = ModuleType("experiments")
    init = guardian.ROOT / "experiments/__init__.py"
    package.__file__ = str(init)
    with pytest.raises(ValueError, match="Unsealed local import"):
        guardian.check_import_origin("experiments", package, {})
    guardian.check_import_origin(
        "experiments", package, {"experiments/__init__.py": guardian.sha256(init)}
    )


def test_incomplete_qualification_stops_before_next_control(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup_worker(monkeypatch, tmp_path)
    called = []

    def incomplete(*args: object, **kwargs: object) -> dict[str, Any]:
        row = fake_evidence(*args, **kwargs)
        called.append(row["id"])
        return dict(row, complete=False, reason="vote_memory_budget")

    monkeypatch.setattr(worker, "write_evidence", incomplete)
    with pytest.raises(RuntimeError, match="Incomplete query"):
        worker.run(
            plan(), dict(mode="qualification", deadline_epoch=10**12, query_seconds=120), tmp_path
        )
    assert called == ["control0"]
    assert not (tmp_path / "result.json").exists()
