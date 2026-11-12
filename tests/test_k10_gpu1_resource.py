from __future__ import annotations

import copy
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from experiments import k10_gpu1_resource as probe
from experiments.k10_gpu1_resource import FEATURES, GPU0, GPU1, PLAN, POLICY, contract
from tools import run_k10_gpu1_resource as guard


def test_torch_uuid_proxy_and_nvml_prefix_compare_equal_bits() -> None:
    class _CUuuid:
        def __str__(self) -> str:
            return GPU1.removeprefix("GPU-")

    for value in (_CUuuid(), GPU1, GPU1.removeprefix("GPU-")):
        observed = probe.validate_device_identity(1, value)
        assert observed["canonical"] == GPU1.removeprefix("GPU-")


@pytest.mark.parametrize(
    "count,value",
    [
        (0, None),
        (2, GPU1),
        (1, GPU0),
        (1, "not-a-uuid"),
        (1, "GPU-GPU-" + GPU1),
    ],
)
def test_device_identity_rejects_count_wrong_or_malformed_with_observation(
    count: int, value: object
) -> None:
    with pytest.raises(ValueError) as error:
        probe.validate_device_identity(count, value)
    assert f'"count": {count}' in str(error.value)
    assert '"type":' in str(error.value) and '"canonical":' in str(error.value)


def config() -> dict[str, Any]:
    return dict(
        policy=copy.deepcopy(POLICY),
        backend_policy=probe.BACKEND_POLICY.copy(),
        gpu_uuid=GPU1,
        gpu0_uuid=GPU0,
        allow_task_fit=False,
        plan_sha256=PLAN,
        feature_manifest_sha256=FEATURES,
        selection=[dict(id=f"synthetic:{i}", frames=8 + i) for i in range(376)],
    )


def test_frozen_contract_rejects_pool_shape_precision_and_cap_rescues() -> None:
    value = config()
    contract(value)
    for field, replacement in (
        ("batch", 16),
        ("frames", 65),
        ("allocator_bytes", 3 * 1024**3),
        ("host_reserve_bytes", 32 * 1024**3),
    ):
        wrong = copy.deepcopy(value)
        wrong["policy"][field] = replacement
        with pytest.raises(ValueError, match="contract"):
            contract(wrong)
    for rows in (value["selection"][:-1], value["selection"][:-1] + [value["selection"][0]]):
        with pytest.raises(ValueError, match="contract"):
            contract({**value, "selection": rows})
    for field in ("global_deterministic_algorithms", "deterministic_warn_only", "cudnn_allow_tf32"):
        wrong = copy.deepcopy(value)
        wrong["backend_policy"][field] = True
        with pytest.raises(ValueError, match="contract"):
            contract(wrong)


def test_canonical_backend_flags_replace_global_determinism_without_cuda_initialization() -> None:
    import torch

    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )
    cuda_was_initialized = torch.cuda.is_initialized()
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        actual = probe.configure_backend()
        assert actual == probe.BACKEND_POLICY
        assert actual["global_deterministic_algorithms"] is False
        assert actual["deterministic_warn_only"] is False
        assert actual["cudnn_deterministic"] is True
        assert torch.cuda.is_initialized() is cuda_was_initialized
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.deterministic = previous[2]
        torch.backends.cudnn.benchmark = previous[3]
        torch.backends.cudnn.allow_tf32 = previous[4]
        torch.backends.cuda.matmul.allow_tf32 = previous[5]


def test_sensor_counts_only_own_pid_on_exact_gpu_and_converts_mib() -> None:
    gpus = f"{GPU0}, 97887, 22971\n{GPU1}, 32607, 9508\n"
    processes = f"{GPU1}, 3451013, 22592\n{GPU1}, 123, 1000\n"
    sample = guard.parse_sensor(gpus, processes, config(), {123})
    assert sample["owned_bytes"] == 1000 * 1024**2
    assert sample["free_bytes"] == 9508 * 1024**2
    assert guard.violation(sample, admission=True) is None
    with pytest.raises(RuntimeError, match="wrong GPU"):
        guard.parse_sensor(gpus, f"{GPU0}, 123, 1", config(), {123})
    with pytest.raises(ValueError):
        guard.parse_sensor(gpus, f"{GPU1}, 123, N/A", config(), {123})


@pytest.mark.parametrize(
    "field,bytes_,admission",
    [
        ("free_bytes", 9 * 1024**3 - 1, True),
        ("free_bytes", 13 * 1024**3 // 2 - 1, False),
        ("gpu0_free_bytes", 8 * 1024**3 - 1, False),
        ("owned_bytes", 11 * 1024**3 // 4 + 1, False),
    ],
)
def test_each_resource_boundary_stops(field: str, bytes_: int, admission: bool) -> None:
    sample = dict(free_bytes=10 * 1024**3, gpu0_free_bytes=9 * 1024**3, owned_bytes=0)
    sample[field] = bytes_
    assert guard.violation(sample, admission=admission) is not None


def test_sensor_timeout_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fail(command: list[str], *, text: bool, timeout: float) -> str:
        calls.append(timeout)
        raise subprocess.TimeoutExpired("sensor", timeout)

    monkeypatch.setattr(guard.subprocess, "check_output", fail)
    with pytest.raises(subprocess.TimeoutExpired):
        guard.sensor({"runtime": {"nvidia_smi": {"path": "/fake"}}}, set())
    assert calls == [0.075]


def test_termination_rejects_reused_pid_before_any_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    class Worker:
        pid = 123

        def poll(self) -> None:
            return None

    monkeypatch.setattr(guard, "start_identity", lambda pid: 999)
    sent = []
    monkeypatch.setattr(guard.os, "killpg", lambda *args: sent.append(args))
    with pytest.raises(RuntimeError, match="identity changed"):
        guard.terminate_own(cast(subprocess.Popen[bytes], Worker()), 111)
    assert sent == []


def test_legitimate_exit_during_signal_lookup_waits_own_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waited = []

    class Worker:
        pid = 123

        def poll(self) -> None:
            return None

        def wait(self, timeout: float) -> int:
            waited.append(timeout)
            return 0

    def exited(pid: int) -> int:
        raise ProcessLookupError

    monkeypatch.setattr(guard, "start_identity", lambda pid: 111)
    monkeypatch.setattr(guard.os, "getpgid", exited)
    guard.terminate_own(cast(subprocess.Popen[bytes], Worker()), 111)
    assert waited == [2]


def test_bound_runtime_file_mutation_is_rejected_before_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    files = []
    for name in (
        "module.py",
        "libtorch_cuda.so",
        "libcudnn.so",
        "libcublas.so",
        "libcudart.so",
        "libcuda.so.1",
        "nvidia-smi",
    ):
        path = tmp_path / name
        path.write_bytes(b"synthetic identity only")
        files.append(dict(path=str(path), sha256=probe.sha256(path)))
    monkeypatch.setattr(probe, "sources", lambda: {"synthetic-source": "a" * 64})
    value = dict(
        source_sha256=probe.sources(),
        protocol_sha256=probe.sha256(
            probe.ROOT / "experiments/studies/k10-capacity/GPU1_RESOURCE_PROTOCOL.md"
        ),
        interpreter=dict(path=sys.executable, sha256=probe.sha256(Path(sys.executable))),
        runtime=dict(
            files=files[1:-1],
            nvidia_smi=files[-1],
            modules={
                name: files[0]
                for name in ("torch", "torch._C", "numpy", "numpy._core._multiarray_umath")
            },
        ),
    )
    probe.verify_bindings(value)
    (tmp_path / "libcudnn.so").write_bytes(b"changed")
    with pytest.raises(ValueError, match="runtime file changed"):
        probe.verify_bindings(value)


def test_guard_imports_without_torch_or_cuda_initialization() -> None:
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tools.run_k10_gpu1_resource; assert 'torch' not in sys.modules",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
