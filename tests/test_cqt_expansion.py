"""Fixed expansion contracts and resource guards, without audio or model downloads."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from tools.export_cqt_expansion import plan_identity
from tools.run_cqt_expansion import BYTE_LIMIT, check_limits, process_tree


def test_common_plan_excludes_track_dependent_state() -> None:
    stage = dict(
        index=0,
        sr=22050.0,
        n_fft=256,
        hop=4096,
        frequencies=[1.0],
        pad_mode="constant",
        basis_shape=[1, 129],
        indptr={"sha256": "a"},
        indices={"sha256": "b"},
        basis={"sha256": "c"},
        input={"shape": [1]},
        response={"sha256": "first"},
    )
    case: dict[str, Any] = dict(stages=[stage], resamplings=[], normalization={"lengths": [1.0]})
    other: dict[str, Any] = dict(
        stages=[stage | {"input": {"shape": [1000]}, "response": {"sha256": "second"}}],
        resamplings=[],
        normalization={"lengths": [2.0]},
    )
    assert plan_identity(case) == plan_identity(other)
    other["stages"][0]["basis"] = {"sha256": "changed"}
    assert plan_identity(case) != plan_identity(other)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cpu", 1200.01),
        ("wall", 1500.01),
        ("resident", BYTE_LIMIT + 1),
        ("storage", BYTE_LIMIT + 1),
        ("now", 11.0),
    ],
)
def test_each_fixed_budget_stops(field: str, value: float) -> None:
    limits: dict[str, Any] = dict(cpu=0.0, wall=0.0, resident=0, storage=0, now=0.0, deadline=10.0)
    limits[field] = value
    with pytest.raises(RuntimeError, match="Fixed budget stop"):
        check_limits(**limits)


def test_process_tree_reads_only_owned_tree() -> None:
    if not Path("/proc/self/stat").exists():
        pytest.skip("Linux-only process accounting")
    resident, cpu = process_tree(os.getpid())
    assert resident > 0
    assert cpu >= 0
    check_limits(0.0, 0.0, 1, 1, time.time(), time.time() + 60)


def test_arithmetic_arms_share_reference_and_pin_only_svml(tmp_path: Path) -> None:
    from argparse import Namespace

    from tools.run_cqt_expansion import Supervisor, sha256

    math = tmp_path / "math.so"
    math.write_bytes(b"fixed-math-artifact")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    args = Namespace(
        stage_root=tmp_path,
        budget_root=tmp_path,
        math_library=math,
        deadline="2099-01-01T00:00:00+00:00",
        native=Path("/native"),
        soxr_wrapper=Path("/soxr"),
        full_model=Path("/model"),
        ort_library=Path("/ort"),
    )
    supervisor = Supervisor(args)
    fma = supervisor.native_command(manifest, tmp_path / "fma.json", True, "fma")
    svml = supervisor.native_command(manifest, tmp_path / "svml.json", True, "fma-svml")
    assert "--numpy-magnitude" in fma and "--numpy-magnitude" in svml
    assert "--math-library" not in fma
    assert svml[svml.index("--math-library-sha256") + 1] == sha256(math)
    assert fma[fma.index("--manifest") + 1] == svml[svml.index("--manifest") + 1]
    assert (
        fma[fma.index("--expect-manifest-sha256") + 1]
        == svml[svml.index("--expect-manifest-sha256") + 1]
    )
