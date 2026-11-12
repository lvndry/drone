"""Offline MusicFM source inspection and explicitly reviewed synthetic resource probe."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from tools import run_cqt_expansion as guard

if TYPE_CHECKING:
    import torch

EVIDENCE = Path(__file__).parents[1] / "experiments/studies/musicfm-preflight/evidence"
WEIGHT_SHA = "218b483a0256ddef736267425fabb166fd97008983696bb9270def464b47bded"
WEIGHT_BYTES = 1_316_802_088
EXPECTED = {
    "torch": "2.15.0",
    "torchaudio": "2.11.0+cpu",
    "transformers": "4.57.6",
    "einops": "0.8.1",
}
MIB = 1024**2


def sha(path: Path) -> str:
    return guard.sha256(path)


def read_snapshot(record: dict[str, Any]) -> bytes:
    path = EVIDENCE / record.get("snapshot_path", record.get("path"))
    raw = path.read_bytes()
    if path.suffix == ".gz":
        if hashlib.sha256(raw).hexdigest() != record["compressed_sha256"]:
            raise ValueError("compressed source metadata changed")
        raw = gzip.decompress(raw)
    if hashlib.sha256(raw).hexdigest() != record["sha256"] or len(raw) != record["bytes"]:
        raise ValueError("pinned source/metadata changed")
    return raw


def verify_evidence() -> dict[str, Any]:
    source = json.loads((EVIDENCE / "source-index.json").read_bytes())
    for record in source["files"].values():
        read_snapshot(record)
    artifacts = json.loads((EVIDENCE / "artifact-index.json").read_bytes())
    for record in [*artifacts["files"].values(), artifacts["config"]]:
        read_snapshot(record)
    pointer = (EVIDENCE / "weights-lfs-pointer.txt").read_text()
    if f"oid sha256:{WEIGHT_SHA}\nsize {WEIGHT_BYTES}\n" not in pointer:
        raise ValueError("checkpoint pointer")
    dependency_source = json.loads((EVIDENCE / "dependency-source-index.json").read_bytes())
    for record in dependency_source.values():
        read_snapshot(record)
    return {"source": source, "artifacts": artifacts, "dependency_sources": dependency_source}


def shape_contract(samples: int) -> dict[str, int]:
    if samples <= 1024:
        raise ValueError("reflect padding requires more than1024 input samples")
    mel_frames = samples // 240  # center=True STFT then upstream [..., :-1]
    return {
        "samples": samples,
        "mel_frames": mel_frames,
        "mel_bands": 128,
        "encoder_frames": (mel_frames + 3) // 4,
        "encoder_channels": 1024,
        "synthetic_head_classes": 24,
    }


def state_estimate(parameters: int, active_parameters: int | None = None) -> dict[str, int | str]:
    if parameters <= 0:
        raise ValueError("positive parameter scenario required")
    active = parameters if active_parameters is None else active_parameters
    if not 0 <= active <= parameters:
        raise ValueError("active parameters must be a subset")
    return {
        "status": "analytical tensor-storage formula; excludes buffers/activations/workspaces",
        "parameter_count_input": parameters,
        "fp32_parameter_bytes": parameters * 4,
        "active_parameters": active,
        "fp32_gradient_bytes": active * 4,
        "fp32_adam_two_moment_bytes": active * 8,
        "total_bytes": parameters * 4 + active * 12,
        "bytes_per_active_parameter": 16,
    }


def prospective_cap(
    free_mib: int, total_mib: int, requested_mib: int = 12288, reserve_mib: int = 8192
) -> int:
    if min(free_mib, total_mib, requested_mib, reserve_mib) <= 0 or free_mib > total_mib:
        raise ValueError("invalid memory metadata")
    cap = min(requested_mib, math.floor((free_mib - reserve_mib) * 0.9))
    if cap < 8192:
        raise ValueError("insufficient free memory for reviewed full-model probe")
    return cap


def installed() -> dict[str, str | None]:
    result = {}
    for name in EXPECTED:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def environment_fingerprint() -> str:
    records = sorted(
        (d.metadata["Name"].lower(), d.version)
        for d in importlib.metadata.distributions()
        if d.metadata["Name"] is not None
    )
    raw = json.dumps({"python": sys.version, "distributions": records}, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def inspection() -> dict[str, Any]:
    pins = verify_evidence()
    return {
        "format": "musicfm-resource-preflight-v1",
        "execution_performed": False,
        "weights_downloaded": False,
        "source_pins": pins,
        "tool_sha256": sha(Path(__file__)),
        "guard_sha256": sha(Path(guard.__file__)),
        "protocol_sha256": sha(EVIDENCE.parent / "PROTOCOL.md"),
        "environment_fingerprint_sha256": environment_fingerprint(),
        "source_index_sha256": sha(EVIDENCE / "source-index.json"),
        "artifact_index_sha256": sha(EVIDENCE / "artifact-index.json"),
        "dependency_source_index_sha256": sha(EVIDENCE / "dependency-source-index.json"),
        "installed_packages": installed(),
        "proposed_packages": EXPECTED,
        "estimated": {
            "parameter_count_source": "nominal330M description, not counted checkpoint",
            "state": state_estimate(330_000_000),
            "shape": shape_contract(12 * 24000),
            "gpu_cap_mib_for_reported_free22971": prospective_cap(22971, 98304),
        },
        "measured_model_resources": None,
    }


def validate_config(config: dict[str, Any], now: float) -> None:
    if (
        config["tool_sha256"] != sha(Path(__file__))
        or config["source_index_sha256"] != sha(EVIDENCE / "source-index.json")
        or config["artifact_index_sha256"] != sha(EVIDENCE / "artifact-index.json")
        or config["dependency_source_index_sha256"]
        != sha(EVIDENCE / "dependency-source-index.json")
        or config["guard_sha256"] != sha(Path(guard.__file__))
        or config["protocol_sha256"] != sha(EVIDENCE.parent / "PROTOCOL.md")
        or config["environment_fingerprint_sha256"] != environment_fingerprint()
    ):
        raise ValueError("reviewed source/config identity")
    start = datetime.fromisoformat(config["start_utc"]).timestamp()
    end = datetime.fromisoformat(config["deadline_utc"]).timestamp()
    if (
        datetime.fromisoformat(config["start_utc"]).utcoffset() is None
        or datetime.fromisoformat(config["deadline_utc"]).utcoffset() is None
    ):
        raise ValueError("timezone-explicit window required")
    if not start <= now < end or end - start > 900:
        raise ValueError("explicit prospective ≤900s UTC window required")
    pids = config["protected_vllm_pids"]
    if (
        len(pids) != 2
        or len(set(pids)) != 2
        or any(not isinstance(p, int) or isinstance(p, bool) or p <= 0 for p in pids)
    ):
        raise ValueError("two distinct live vLLM process identities required")
    if (
        config["precision"] != "bf16-autocast-fp32-parameters"
        or config["attention"] != "eager"
        or config["steady_steps"] != 20
    ):
        raise ValueError("fixed resource experiment changed")
    if not 8192 <= config["requested_cap_mib"] <= 12288 or config["reserve_mib"] < 8192:
        raise ValueError("reviewed resource envelope exceeded")
    if config["gpu_uuid"].startswith("GPU-") is not True:
        raise ValueError("physical GPU UUID required")
    root = Path(config["root"])
    if not root.is_absolute() or root.exists():
        raise ValueError("new dedicated absolute output root required")


def gpu_metadata(uuid: str) -> dict[str, int | str]:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            uuid,
            "--query-gpu=uuid,index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=5,
    )
    fields = [x.strip() for x in raw.strip().split(",")]
    if len(fields) != 4 or fields[0] != uuid or int(fields[1]) != 0:
        raise ValueError("GPU identity")
    return {
        "uuid": fields[0],
        "index": int(fields[1]),
        "free_mib": int(fields[2]),
        "total_mib": int(fields[3]),
    }


def gpu_worker_mib(uuid: str, pid: int) -> int:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=5,
    )
    total = 0
    for line in raw.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 3:
            raise ValueError("NVML process metadata unavailable")
        if fields[0] == uuid and int(fields[1]) == pid:
            total += int(fields[2])
    return total


def process_identity(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def prepare_upstream(root: Path) -> None:
    records = verify_evidence()["source"]["files"]
    for name, record in records.items():
        path = root / "upstream/musicfm" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(read_snapshot(record))


def pool_key_head(head: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Resource-only adapter: reduce TIME in the official B,T,1024 sequence."""
    if hidden.ndim != 3 or hidden.shape[-1] != 1024:
        raise ValueError("upstream B,T,1024 contract")
    return head(hidden.mean(dim=1))


def worker(config_path: Path) -> None:
    config = json.loads(config_path.read_bytes())
    if (
        os.getppid() != config["supervisor_pid"]
        or process_identity(os.getppid()) != config["supervisor_identity"]
    ):
        raise ValueError("private worker requires its live bounded supervisor")
    if (
        sha(Path(__file__)) != config["tool_sha256"]
        or environment_fingerprint() != config["environment_fingerprint_sha256"]
    ):
        raise ValueError("worker source/environment changed after review")
    import torch

    Wav2Vec2ConformerConfig = getattr(
        importlib.import_module("transformers"), "Wav2Vec2ConformerConfig"
    )

    config = json.loads(config_path.read_bytes())
    root = Path(config["root"])
    if torch.__version__ != "2.15.0+cu132" or torch.version.cuda != "13.2":
        raise ValueError("reviewed CUDA Torch build changed before GPU initialization")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    prepare_upstream(root)
    sys.path.insert(0, str(root / "upstream"))
    module = importlib.import_module("musicfm.model.musicfm_25hz")
    options = json.loads((EVIDENCE / "conformer-config.json").read_bytes())
    options["_attn_implementation"] = "eager"
    cfg = Wav2Vec2ConformerConfig.from_dict(options)
    # Upstream requests an HF config in __init__. This replaces ONLY that I/O
    # with the exact pinned bytes, preventing any implicit network/model fetch.
    with patch.object(Wav2Vec2ConformerConfig, "from_pretrained", return_value=cfg):
        backbone = module.MusicFM25Hz(
            is_flash=False, stat_path=str(EVIDENCE / "msd-stats.json"), model_path=""
        )
    state = torch.load(config["weights"], map_location="cpu", weights_only=True, mmap=True)[
        "state_dict"
    ]
    if not all(key.startswith("model.") for key in state):
        raise ValueError("unexpected checkpoint prefix; no guessed mapping")
    backbone.load_state_dict({key[6:]: value for key, value in state.items()}, strict=True)
    del state
    if torch.cuda.is_initialized():
        raise ValueError("CPU dependency/checkpoint preparation unexpectedly initialized CUDA")
    if time.time() >= datetime.fromisoformat(config["deadline_utc"]).timestamp():
        raise ValueError("window closed before any CUDA initialization")
    latest = gpu_metadata(config["gpu_uuid"])
    applied_cap_mib = min(
        config["effective_cap_bytes"] // MIB,
        prospective_cap(
            int(latest["free_mib"]),
            int(latest["total_mib"]),
            config["requested_cap_mib"],
            config["reserve_mib"],
        ),
    )
    torch.cuda.set_per_process_memory_fraction(applied_cap_mib * MIB / config["gpu_total_bytes"], 0)
    backbone = backbone.float().cuda().train()
    torch.manual_seed(20261112)
    torch.cuda.manual_seed_all(20261112)
    head = torch.nn.Linear(1024, 24, device="cuda", dtype=torch.float32)
    waveform = (torch.rand(1, 288000, device="cpu") * 0.4 - 0.2).cuda()
    target = torch.zeros(1, dtype=torch.long, device="cuda")  # synthetic, no music labels
    optimizer = torch.optim.AdamW(
        [{"params": backbone.parameters(), "lr": 1e-5}, {"params": head.parameters(), "lr": 1e-4}],
        weight_decay=0.01,
        foreach=False,
        fused=False,
    )
    parameters = [*backbone.parameters(), *head.parameters()]
    report: dict[str, Any] = {
        "format": "musicfm-synthetic-resource-v1",
        "tool_sha256": sha(Path(__file__)),
        "effective_config_sha256": sha(config_path),
        "source_index_sha256": config["source_index_sha256"],
        "artifact_index_sha256": config["artifact_index_sha256"],
        "seed": 20261112,
        "shape_contract": shape_contract(288000),
        "applied_cap_mib": applied_cap_mib,
        "synthetic_only": True,
        "accuracy_scored": False,
        "checkpoint_sha256": WEIGHT_SHA,
        "packages": installed(),
        "parameter_count_source": "counted loaded model and synthetic head",
        "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
        "registered_buffer_tensor_bytes": sum(
            b.numel() * b.element_size() for b in backbone.buffers()
        ),
        "parameter_count": sum(p.numel() for p in parameters),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
        "steps": [],
        "estimated": state_estimate(sum(p.numel() for p in parameters)),
    }
    for index in range(21):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        # Original feature preprocessing is kept FP32; only encoder/head autocast.
        with torch.autocast("cuda", enabled=False):
            features = backbone.normalize(backbone.preprocessing(waveform, ["melspec_2048"]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, hidden = backbone.encoder(features["melspec_2048"])
            embedding = hidden[12]
            if tuple(embedding.shape) != (1, 300, 1024):
                raise ValueError("synthetic12s encoder shape")
            logits = pool_key_head(head, embedding)
            loss = torch.nn.functional.cross_entropy(logits.float(), target)
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite synthetic loss")
        loss.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise ValueError("nonfinite gradient")
        optimizer.step()  # FIRST step is mandatory: Adam state is lazy.
        torch.cuda.synchronize()
        report["estimated_active_state_floor"] = state_estimate(
            sum(p.numel() for p in parameters), sum(p.numel() for p in optimizer.state)
        )
        report["steps"].append(
            {
                "index": index,
                "phase": "first_optimizer_step" if index == 0 else "steady",
                "seconds": time.perf_counter() - started,
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "optimizer_active_parameter_elements": sum(p.numel() for p in optimizer.state),
                "gradient_bytes": sum(
                    p.grad.numel() * p.grad.element_size() for p in parameters if p.grad is not None
                ),
                "optimizer_tensor_bytes": sum(
                    v.numel() * v.element_size()
                    for s in optimizer.state.values()
                    for v in s.values()
                    if isinstance(v, torch.Tensor)
                ),
            }
        )
        guard.write_json(root / "worker-progress.json", report)
    report["complete"] = True
    guard.write_json(root / "worker-report.json", report)


def run(config_path: Path) -> None:
    cpu_before = guard.child_cpu()
    config = json.loads(config_path.read_bytes())
    validate_config(config, time.time())
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is None or affinity(0) != {8, 9}:
        raise ValueError("prospective Linux execution must use taskset8,9")
    verify_evidence()
    if installed() != EXPECTED:
        raise ValueError("reviewed dependencies unavailable; no provisioning performed")
    weights = Path(config["weights"])
    if weights.stat().st_size != WEIGHT_BYTES or sha(weights) != WEIGHT_SHA:
        raise ValueError("local checkpoint missing/incorrect; tool never downloads")
    protected = {pid: process_identity(pid) for pid in config["protected_vllm_pids"]}
    gpu = gpu_metadata(config["gpu_uuid"])
    cap = prospective_cap(
        int(gpu["free_mib"]),
        int(gpu["total_mib"]),
        config["requested_cap_mib"],
        config["reserve_mib"],
    )
    root = Path(config["root"])
    root.mkdir()
    effective = dict(
        config,
        supervisor_pid=os.getpid(),
        supervisor_identity=process_identity(os.getpid()),
        effective_cap_bytes=cap * MIB,
        gpu_total_bytes=int(gpu["total_mib"]) * MIB,
    )
    guard.write_json(root / "effective-config.json", effective)
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=config["gpu_uuid"],
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
    ) | {key: "1" for key in guard.THREADS}
    environment.update(
        {
            name: str(root / folder)
            for name, folder in {
                "PYTHONPYCACHEPREFIX": "pycache",
                "XDG_CACHE_HOME": "cache",
                "HF_HOME": "hf-cache",
                "TORCH_HOME": "torch-cache",
                "CUDA_CACHE_PATH": "cuda-cache",
                "TORCHINDUCTOR_CACHE_DIR": "inductor-cache",
                "TMPDIR": "tmp",
            }.items()
        }
    )
    (root / "tmp").mkdir()
    observations = []
    deadline = datetime.fromisoformat(config["deadline_utc"]).timestamp()
    started = time.monotonic()
    with (root / "worker.log").open("xb") as stream:
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tools.preflight_musicfm",
                "_worker",
                "--config",
                str(root / "effective-config.json"),
            ],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        success = False
        failure = None
        try:
            while child.poll() is None:
                snapshot = gpu_metadata(config["gpu_uuid"])
                rss, cpu = guard.process_tree(child.pid)
                own_mib = gpu_worker_mib(config["gpu_uuid"], child.pid)
                aggregate_cpu = guard.child_cpu() - cpu_before + cpu
                observations.append(
                    {
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "gpu": snapshot,
                        "worker_nvml_mib": own_mib,
                        "tree_rss_bytes": rss,
                        "live_tree_cpu_seconds": cpu,
                        "aggregate_child_cpu_seconds": aggregate_cpu,
                    }
                )
                if (
                    any(process_identity(pid) != identity for pid, identity in protected.items())
                    or int(snapshot["free_mib"]) < config["reserve_mib"]
                    or own_mib > cap + 1024
                    or rss > 8 * 1024**3
                    or aggregate_cpu > 600
                    or time.monotonic() - started > 600
                    or time.time() >= deadline
                    or guard.disk_bytes(root) > 4 * 1024**3
                ):
                    raise RuntimeError("fixed resource/protected-service/window stop")
                time.sleep(0.25)
            if child.wait() != 0:
                raise RuntimeError("synthetic worker failed; preserve log, no automatic retry")
            if any(process_identity(pid) != identity for pid, identity in protected.items()):
                raise RuntimeError("protected service identity changed before completion")
            result = json.loads((root / "worker-report.json").read_bytes())
            if result.get("complete") is not True or len(result["steps"]) != 21:
                raise RuntimeError("incomplete synthetic resource report")
            success = True
        except BaseException as error:
            failure = f"{type(error).__name__}: {error}"
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
        finally:
            guard.write_json(
                root / "supervisor.json",
                {
                    "complete": success,
                    "error": failure,
                    "wall_seconds": time.monotonic() - started,
                    "worker_pid": child.pid,
                    "exit_code": child.poll(),
                    "observations": observations,
                    "initial_gpu": gpu,
                    "protected_processes": protected,
                    "tool_sha256": sha(Path(__file__)),
                    "config_sha256": sha(config_path),
                    "effective_cap_mib": cap,
                    "aggregate_child_cpu_seconds": guard.child_cpu() - cpu_before,
                    "sampling_seconds": 0.25,
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["inspect", "run-resource", "_worker"], default="inspect", nargs="?"
    )
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps(inspection(), indent=2))
    elif args.config is None:
        parser.error("explicit reviewed config/window required")
    elif args.command == "run-resource":
        run(args.config)
    else:
        worker(args.config)


if __name__ == "__main__":
    main()
