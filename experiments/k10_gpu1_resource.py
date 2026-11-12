"""One prospective synthetic GPU1 resource probe; never a task fit or checkpoint."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from tools.run_cqt_expansion import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
GPU1 = "GPU-9c2004f9-0d41-d7c7-7075-863335543ee8"
GPU0 = "GPU-387d0c41-22e5-4c3b-efe8-1d8f1dd0494c"
GIB = 1024**3
PLAN = "c3e59ffb764ca5f51444447b0a18a69543228e60a9e3723d925ddd82b0fc8373"
FEATURES = "9f94f6a5c6a9e78b4c5d1013cf97b44d6d2fd69665348ebb497e2c817560d6cc"
INITIAL = {
    "baseline": "bb227d5d27d0f28c0065de14468a4ada92d79944ccbcb9da6a68c97e64daa219",
    "candidate": "4fc58182dc884d7e9e9b948e72737742628be6b6a0685472b42808b0aed715d4",
}
PREFIX = "54d0c895089289238293d49d211a4a8ce26bfe4b79e3d47419a2fc73a51d5b4a"
BACKEND_POLICY = dict(
    global_deterministic_algorithms=False,
    deterministic_warn_only=False,
    cudnn_deterministic=True,
    cudnn_benchmark=False,
    cudnn_allow_tf32=False,
    matmul_allow_tf32=False,
    default_dtype="torch.float32",
    autocast_enabled=False,
)
POLICY: dict[str, Any] = dict(
    cpus=[10, 11],
    batch=32,
    frames=320,
    updates=3,
    seed=13,
    eval_batch=4,
    torch_threads=2,
    interop_threads=1,
    nice=19,
    passive_seconds=30,
    wall_seconds=300,
    cpu_seconds=1200,
    rss_bytes=4 * GIB,
    output_bytes=64 * 1024**2,
    allocator_bytes=2 * GIB,
    owned_gpu_bytes=11 * GIB // 4,
    admission_bytes=9 * GIB,
    early_abort_bytes=13 * GIB // 2,
    hard_floor_bytes=6 * GIB,
    gpu0_reserve_bytes=8 * GIB,
    host_reserve_bytes=40 * GIB,
    disk_reserve_bytes=32 * GIB,
    poll_seconds=0.2,
    target_poll_seconds=0.15,
    query_timeout_seconds=0.075,
)


def load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw)


def validate_device_identity(count: int, actual: object) -> dict[str, Any]:
    """Compare UUID bits, accepting Torch's bare UUID and NVML's GPU- prefix."""
    raw = str(actual)
    observation = dict(
        count=count, type=type(actual).__name__, value=raw, repr=repr(actual), canonical=None
    )
    try:
        canonical = uuid.UUID(raw.removeprefix("GPU-"))
        observation["canonical"] = str(canonical)
    except ValueError:
        raise ValueError("Invalid GPU identity observation: " + json.dumps(observation)) from None
    if count != 1 or canonical != uuid.UUID(GPU1.removeprefix("GPU-")):
        raise ValueError("Wrong visible GPU identity: " + json.dumps(observation))
    return observation


def configure_backend() -> dict[str, Any]:
    """Match canonical CUDA setup; seeded, bitwise GPU training not guaranteed.

    This configures flags only. It does not create a model, tensor or CUDA context.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)
    observed = dict(
        global_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        default_dtype=str(torch.get_default_dtype()),
        autocast_enabled=torch.is_autocast_enabled(),
    )
    if observed != BACKEND_POLICY:
        raise ValueError("Effective canonical backend differs: " + json.dumps(observed))
    return observed


def sources() -> dict[str, str]:
    names = [str(p.relative_to(ROOT)) for p in (ROOT / "drone").glob("*.py")]
    names += [
        "train.py",
        "experiments/k10_resolution.py",
        "experiments/k10_gpu1_resource.py",
        "tools/run_k10_gpu1_resource.py",
        "tools/run_cqt_expansion.py",
    ]
    return {name: sha256(ROOT / name) for name in sorted(names)}


def contract(value: dict[str, Any]) -> None:
    rows = value["selection"]
    if (
        value["policy"] != POLICY
        or value["backend_policy"] != BACKEND_POLICY
        or value["gpu_uuid"] != GPU1
        or value["gpu0_uuid"] != GPU0
        or value["allow_task_fit"] is not False
        or len(rows) != 376
        or len({r["id"] for r in rows}) != 376
        or any(type(r["frames"]) is not int or r["frames"] < 8 for r in rows)
        or value["plan_sha256"] != PLAN
        or value["feature_manifest_sha256"] != FEATURES
    ):
        raise ValueError("Fixed synthetic GPU1 contract differs")


def seal(
    plan: Path, features: Path, protected: Path, runtime: Path, not_before: float, deadline: float
) -> dict[str, Any]:
    if not 0 < deadline - not_before <= POLICY["wall_seconds"]:
        raise ValueError("Explicit finite prospective UTC window required")
    if (sha256(plan), sha256(features)) != (PLAN, FEATURES):
        raise ValueError("Require admitted plan and frame-count manifest")
    p, f = load(plan), load(features)
    ordered = p["bp_selection_guard"] + p["fma_inner_selection"]
    rows = {row["id"]: row for row in f["rows"]}
    runtime_value = load(runtime)
    value = dict(
        policy=POLICY.copy(),
        backend_policy=BACKEND_POLICY.copy(),
        gpu_uuid=GPU1,
        gpu0_uuid=GPU0,
        allow_task_fit=False,
        plan_sha256=PLAN,
        feature_manifest_sha256=FEATURES,
        selection=[{"id": identity, "frames": rows[identity]["frames"]} for identity in ordered],
        source_sha256=sources(),
        protected=load(protected),
        runtime=runtime_value,
        protected_receipt_sha256=sha256(protected),
        runtime_receipt_sha256=sha256(runtime),
        interpreter={"path": sys.executable, "sha256": sha256(Path(sys.executable))},
        protocol_sha256=sha256(ROOT / "experiments/studies/k10-capacity/GPU1_RESOURCE_PROTOCOL.md"),
        not_before_epoch=not_before,
        deadline_epoch=deadline,
        launch_authorized=False,
    )
    contract(value)
    verify_bindings(value)
    return value


def verify_bindings(value: dict[str, Any]) -> None:
    if value["source_sha256"] != sources():
        raise ValueError("Bound source differs")
    if (
        sha256(ROOT / "experiments/studies/k10-capacity/GPU1_RESOURCE_PROTOCOL.md")
        != value["protocol_sha256"]
    ):
        raise ValueError("Bound protocol differs")
    if (
        str(Path(sys.executable).resolve()) != str(Path(value["interpreter"]["path"]).resolve())
        or sha256(Path(sys.executable)) != value["interpreter"]["sha256"]
    ):
        raise ValueError("Interpreter differs")
    runtime = value["runtime"]
    required = {"torch", "torch._C", "numpy", "numpy._core._multiarray_umath"}
    if not required <= runtime["modules"].keys() or not runtime["files"]:
        raise ValueError("Incomplete runtime closure")
    for record in [*runtime["modules"].values(), *runtime["files"]]:
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("Pinned runtime file changed")
    for prefix in ("libtorch_cuda", "libcudnn", "libcublas", "libcudart", "libcuda.so"):
        if not any(Path(r["path"]).name.startswith(prefix) for r in runtime["files"]):
            raise ValueError("CUDA/driver library must be bound: " + prefix)
    if sha256(Path(runtime["nvidia_smi"]["path"])) != runtime["nvidia_smi"]["sha256"]:
        raise ValueError("Sensor binary changed")


def worker(config: Path, approval: Path, output: Path, model_name: str, guardian: int) -> None:
    worker_began = time.monotonic()
    if model_name not in INITIAL:
        raise ValueError("Only baseline/candidate allowed")
    value, permit = load(config), load(approval)
    contract(value)
    verify_bindings(value)
    if (
        permit.get("authorized") is not True
        or permit.get("allow_resource_probe") is not True
        or permit.get("allow_training") is not False
        or permit.get("config_sha256") != sha256(config)
        or os.getppid() != guardian
    ):
        raise ValueError("Separate exact resource-only permit and guardian required")
    import numpy as np
    import torch
    from torch import nn

    from drone.model import KeyCNN
    from experiments.k10_resolution import SemitoneGridCNN
    from train import evaluate_features

    for name, record in value["runtime"]["modules"].items():
        module = importlib.import_module(name)
        if (
            module.__file__ is None
            or module.__file__ != record["path"]
            or sha256(Path(module.__file__)) != record["sha256"]
        ):
            raise ValueError("Runtime module shadowed")
    if str(torch.__version__) != value["runtime"]["torch_version"]:
        raise ValueError("Torch version differs")
    from tools.run_k10_gpu1_resource import environment, start_identity

    environment(value)

    def watch() -> None:
        ready = load(output.parent / "ready.json")
        if (
            ready["guardian"] != guardian
            or ready["config_sha256"] != sha256(config)
            or start_identity(guardian) != ready["start_ticks"]
            or not 0 <= time.time() - ready["epoch"] <= POLICY["poll_seconds"]
            or time.time() >= value["deadline_epoch"]
        ):
            raise RuntimeError("Stale guardian, sensor or UTC cutoff")

    watch()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    backend = configure_backend()
    count = torch.cuda.device_count()
    properties = torch.cuda.get_device_properties(0) if count == 1 else None
    device_identity = validate_device_identity(count, properties.uuid if properties else None)
    if properties is None:
        raise ValueError("Missing admitted device properties")
    total = properties.total_memory
    torch.cuda.set_per_process_memory_fraction(POLICY["allocator_bytes"] / total, 0)
    torch.manual_seed(13)
    model = (KeyCNN() if model_name == "baseline" else SemitoneGridCNN()).to("cuda:0")
    initial = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        initial.update(name.encode())
        initial.update(tensor.detach().cpu().numpy().tobytes())
    prefix = hashlib.sha256()
    for name, tensor in model.net[:14].state_dict().items():
        prefix.update(name.encode())
        prefix.update(tensor.detach().cpu().numpy().tobytes())
    if initial.hexdigest() != INITIAL[model_name] or prefix.hexdigest() != PREFIX:
        raise ValueError("Fixed seed13 initialization/prefix differs")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("FP32 master parameters required")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    setup_seconds = time.monotonic() - worker_began
    output.mkdir(exist_ok=False)
    write_json(
        output / "initial.json",
        dict(
            config_sha256=sha256(config),
            model=model_name,
            initial_state_sha256=initial.hexdigest(),
            first_two_blocks_sha256=prefix.hexdigest(),
            setup_seconds=setup_seconds,
            actual_total_device_bytes=total,
            device_identity=device_identity,
            backend_policy=backend,
            reproducibility="seeded; bitwise GPU training not guaranteed",
            allocator_bytes=POLICY["allocator_bytes"],
            task_fit=False,
        ),
    )
    times = []
    for _ in range(3):
        watch()
        torch.cuda.synchronize()
        began = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(32, 192, 320, device="cuda:0", dtype=torch.float32)
        logits = model(x)
        if logits.dtype != torch.float32 or torch.is_autocast_enabled():
            raise ValueError("FP32/no-AMP logits required")
        loss = nn.functional.cross_entropy(logits, torch.arange(32, device="cuda:0") % 24)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite synthetic loss")
        loss.backward()
        if any(
            p.grad is None
            or p.grad.dtype != torch.float32
            or not bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        ):
            raise ValueError("Nonfinite/missing gradients; no commit")
        watch()
        optimizer.step()
        if any(not bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
            raise ValueError("Nonfinite model state")
        if any(
            not bool(torch.isfinite(t).all())
            for state in optimizer.state.values()
            for t in state.values()
            if isinstance(t, torch.Tensor)
        ):
            raise ValueError("Nonfinite AdamW state")
        model.eval()
        with torch.inference_mode():
            if not bool(torch.isfinite(model(x[:1])).all()):
                raise ValueError("Nonfinite train/eval transition")
        torch.cuda.synchronize()
        times.append(time.monotonic() - began)
        write_json(
            output / "progress.json",
            dict(
                config_sha256=sha256(config),
                optimizer_commits=len(times),
                update_with_transition_seconds=times,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                task_fit=False,
            ),
        )
        del x, loss, logits
    if any(int(state["step"].item()) != 3 for state in optimizer.state.values()):
        raise ValueError("Incomplete three AdamW commits")
    optimizer.zero_grad(set_to_none=True)
    counts = {"tracks": 0, "frames": 0, "batches": 0}

    class Finite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            watch()
            out = self.base(x)
            if (
                out.shape != (x.shape[0], 24)
                or out.dtype != torch.float32
                or not bool(torch.isfinite(out).all())
            ):
                raise ValueError("Invalid synthetic evaluation")
            counts["tracks"] += x.shape[0]
            counts["frames"] += x.shape[0] * x.shape[2]
            counts["batches"] += 1
            return out

    finite = Finite()
    rng = np.random.default_rng(13)
    phases = {}
    for label, rows in (
        ("selection376", value["selection"]),
        ("synthetic_envelope4", [{"frames": t} for t in (8, 65, 320, 321)]),
    ):
        counts.update(tracks=0, frames=0, batches=0)
        torch.cuda.synchronize()
        began = time.monotonic()
        for first in range(0, len(rows), 4):
            arrays = [
                rng.standard_normal((192, r["frames"]), dtype=np.float32)
                for r in rows[first : first + 4]
            ]
            evaluate_features(finite, arrays, "cuda:0", batch=4)  # Discard transient argmax values.
            torch.cuda.synchronize()
            watch()
        if counts["tracks"] != len(rows) or counts["frames"] != sum(r["frames"] for r in rows):
            raise ValueError("Incomplete true-length coverage")
        phases[label] = {**counts, "wall_seconds": time.monotonic() - began}
    torch.cuda.synchronize()
    write_json(
        output / "result.json",
        dict(
            complete=True,
            model=model_name,
            config_sha256=sha256(config),
            initial_state_sha256=initial.hexdigest(),
            first_two_blocks_sha256=prefix.hexdigest(),
            backend_policy=backend,
            reproducibility="seeded; bitwise GPU training not guaranteed",
            setup_seconds=setup_seconds,
            update_with_transition_seconds=times,
            evaluation=phases,
            complete_worker_wall_seconds=time.monotonic() - worker_began,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            actual_total_device_bytes=total,
            allocator_bytes=POLICY["allocator_bytes"],
            task_fit=False,
            predictions_retained=False,
            weights_retained=False,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    seal_parser = sub.add_parser("seal")
    for name in ("plan", "features", "protected", "runtime", "output"):
        seal_parser.add_argument("--" + name, type=Path, required=True)
    seal_parser.add_argument("--not-before", type=float, required=True)
    seal_parser.add_argument("--deadline", type=float, required=True)
    execute = sub.add_parser("worker")
    for name in ("config", "approval", "output"):
        execute.add_argument("--" + name, type=Path, required=True)
    execute.add_argument("--model", choices=("baseline", "candidate"), required=True)
    execute.add_argument("--guardian", type=int, required=True)
    args = parser.parse_args()
    if args.mode == "seal":
        if args.output.exists():
            raise ValueError("Fresh private draft required")
        write_json(
            args.output,
            seal(
                args.plan,
                args.features,
                args.protected,
                args.runtime,
                args.not_before,
                args.deadline,
            ),
        )
    else:
        worker(args.config, args.approval, args.output, args.model, args.guardian)


if __name__ == "__main__":
    main()
