"""Bounded CPU-only qualification/audit; requires an exact mode-specific root permit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tools.run_cqt_expansion import disk_bytes, process_tree, sha256, write_json
from tools.run_k10_gpu1_resource import start_identity, terminate_own

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "experiments/studies/v2-data-contract/data/gtzan-overlap-fresh-controls.json",
    "experiments/fma_fingerprints.py",
    "experiments/fma_alignment_matcher.py",
    "experiments/bp_gs_scope.py",
    "experiments/gtzan_overlap.py",
    "tools/audit_bp_gs.py",
    "tools/run_gtzan_overlap.py",
    "tools/run_cqt_expansion.py",
    "tools/run_k10_gpu1_resource.py",
    "experiments/k10_gpu1_resource.py",
)
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
MAXIMUMS = dict(
    wall_seconds=3600,
    cpu_seconds=7200,
    rss_bytes=2 * 1024**3,
    output_bytes=512 * 1024**2,
    query_seconds=120,
)


def load(path: Path) -> dict[str, Any]:

    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw)


def qualification_binding(config: dict[str, Any]) -> str:
    keys = (
        "source_sha256",
        "runtime_files",
        "vendor_path",
        "cache_root",
        "fresh_controls",
        "query_manifest_sha256",
        "control_witness_sha256",
        "historical_controls_sha256",
    )
    return hashlib.sha256(
        json.dumps({k: config[k] for k in keys}, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def check_contract(config: dict[str, Any], permit: dict[str, Any], digest: str, now: float) -> None:
    mode = config["mode"]
    if mode not in {"qualification", "audit"} or config["version"] != "gtzan-overlap-runtime-v1":
        raise ValueError("Unknown runtime mode/version")
    if (
        permit.get("authorized") is not True
        or permit.get("config_sha256") != digest
        or permit.get("mode") != mode
        or permit.get("allow_classifier_outputs") is not False
        or permit.get("deadline_epoch") != config["deadline_epoch"]
    ):
        raise ValueError("Exact mode-specific source/config/UTC root permit required")
    for name, maximum in MAXIMUMS.items():
        value = config["caps"][name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= maximum
        ):
            raise ValueError("Invalid finite resource cap: " + name)
    start, end = config["not_before_epoch"], config["deadline_epoch"]
    if not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        for v in (start, end)
    ):
        raise ValueError("Finite UTC window required")
    if not start <= now < end or not 0 < end - start <= config["caps"]["wall_seconds"]:
        raise ValueError("Outside sealed bounded UTC window")
    if len(config["cpus"]) != 2 or len(set(config["cpus"])) != 2:
        raise ValueError("Exactly two allocated CPUs required")
    if len(config["protected_starts"]) < 4:
        raise ValueError("Bind all four protected vLLM start identities")
    if not config["gpu_uuids"] or len(set(config["gpu_uuids"])) != len(config["gpu_uuids"]):
        raise ValueError("Bind all protected GPU identities")
    for name, minimum in (
        ("host_reserve_bytes", 40 * 1024**3),
        ("gpu_reserve_bytes", 8 * 1024**3),
        ("disk_reserve_bytes", 1024**3),
    ):
        if config[name] < minimum:
            raise ValueError("Reserve below existing safety contract: " + name)
    if mode == "audit" and not config.get("qualification_receipt_sha256"):
        raise ValueError("Audit requires root-reviewed qualification receipt")


def verify_bindings(config: dict[str, Any]) -> None:
    if config["source_sha256"] != {name: sha256(ROOT / name) for name in SOURCES}:
        raise ValueError("Worker/guardian numerical source closure differs")
    for name in ("query_manifest", "control_witness", "historical_controls"):
        if sha256(Path(config[name])) != config[name + "_sha256"]:
            raise ValueError("Bound metadata differs: " + name)
    # Runtime inventory is produced/reviewed separately, never inferred from a version string.
    if not config["runtime_files"]:
        raise ValueError("Bound installed runtime files required")
    required_paths = {
        str(Path(config[name]).resolve())
        for name in ("python_path", "ffmpeg_path", "chromaprint_path")
    }
    if not required_paths <= {str(Path(row["path"]).resolve()) for row in config["runtime_files"]}:
        raise ValueError("Bind Python, FFmpeg and Chromaprint installed bytes")
    for row in config["runtime_files"]:
        if sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError("Installed runtime file differs")
    if config.get("mode") == "audit":
        receipt = Path(config["qualification_receipt"])
        if sha256(receipt) != config["qualification_receipt_sha256"]:
            raise ValueError("Qualification receipt changed")
        result = load(receipt)
        if result.get("qualification_passed") is not True or result.get("mode") != "qualification":
            raise ValueError("Qualification did not pass")
        if result.get("qualification_binding_sha256") != qualification_binding(config):
            raise ValueError(
                "Qualification does not cover these exact sources/runtime/controls/cohort"
            )


def execution_plan(config: dict[str, Any]) -> dict[str, Any]:
    metadata = load(Path(config["query_manifest"]))
    witness = load(Path(config["control_witness"]))
    historical = load(Path(config["historical_controls"]))
    reference_path = Path(config["cache_root"]) / "references-v1/result.json"
    if sha256(reference_path) != config["descriptor_sha256"]:
        raise ValueError("Reference descriptor differs")
    descriptor = load(reference_path)
    fixed_path = (
        ROOT / "experiments/studies/v2-data-contract/data/gtzan-overlap-fresh-controls.json"
    )
    if sha256(fixed_path) != "ed91211ca3337cb835482d8665aa696a5267e9a22c8cc4e4b0864afc158dfef7":
        raise ValueError("Fixed fresh-control selection changed")
    fixed = load(fixed_path)
    expected_controls = [
        dict(row, expected_ref=row["expected_ref_id"]) for row in fixed["controls"]
    ]
    if config["fresh_controls"] != expected_controls:
        raise ValueError("Runtime must retain the six frozen fresh controls exactly")
    by_ref = {r["ref_id"]: r for r in descriptor["records"]}
    for row in config["fresh_controls"]:
        source = by_ref[row["expected_ref"]]
        if (
            row["source_path"] != source["path"]
            or row["source_container_sha256"] != source["container_sha256"]
            or row["domain"] != source["domain"]
            or not 0 < row["offset_seconds"] < source["seconds"] - 30
        ):
            raise ValueError("Fresh control must be an interior excerpt of the declared reference")
    witness_raw = Path(config["control_witness"]).read_bytes()
    witness_decoded = (
        gzip.decompress(witness_raw) if witness_raw.startswith(b"\x1f\x8b") else witness_raw
    )
    if hashlib.sha256(witness_decoded).hexdigest() != historical["plan_sha256"]:
        raise ValueError("Historical controls do not bind this exact original control witness")
    if (
        metadata["schema"] != "gtzan-overlap-query-manifest-v1"
        or metadata["query_count"] != 769
        or not historical["all244_complete"]
        or historical["expected_positive_retrievals"] != 180
    ):
        raise ValueError("Frozen cohort/control provenance differs")
    return dict(
        version="gtzan-overlap-v1",
        cache_root=config["cache_root"],
        index_sha256=metadata["reference_index_sha256"],
        descriptor_sha256=config["descriptor_sha256"],
        artifact_sha256=witness["reference_artifact_sha256"],
        dictionary=[
            {k: r[k] for k in ("ref_id", "id", "hashes")}
            for r in sorted(descriptor["records"], key=lambda r: r["ref_id"])
        ],
        controls=[
            {k: r[k] for k in ("id", "path", "sha256", "expected_ref", "panel")}
            for r in witness["controls"]
        ],
        fresh_controls=config["fresh_controls"],
        vendor_path=config["vendor_path"],
        primary_ids=[r["id"] for r in metadata["queries"]],
        cohort=[
            dict(id=r["id"], member=r["member"], container_sha256=r["wav_sha256"])
            for r in metadata["sensitivity_queries"]
        ],
        archive_path=config["archive_path"],
        archive_sha256=metadata["audio_archive"]["sha256"],
    )


def environment(config: dict[str, Any]) -> None:
    if (
        not hasattr(os, "sched_getaffinity")
        or set(os.sched_getaffinity(0)) != set(config["cpus"])
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or any(os.environ.get(name) != "1" for name in THREADS)
        or shutil.which("ffmpeg") != config["ffmpeg_path"]
        or Path(sys.executable).resolve() != Path(config["python_path"]).resolve()
    ):
        raise ValueError("Require sealed two-CPU allocation/nice19/noCUDA/native threads1")


def reserve_check(config: dict[str, Any], output: Path) -> None:
    if any(start_identity(r["pid"]) != r["start_ticks"] for r in config["protected_starts"]):
        raise RuntimeError("Protected vLLM start identity changed")
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    if int(memory["MemAvailable"].split()[0]) * 1024 < config["host_reserve_bytes"]:
        raise RuntimeError("Global host reserve")
    if shutil.disk_usage(output).free < config["disk_reserve_bytes"]:
        raise RuntimeError("Global disk reserve")
    raw = subprocess.check_output(
        [config["nvidia_smi"], "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
        text=True,
        timeout=2,
    )
    free = {
        line.split(",")[0].strip(): int(line.split(",")[1]) * 1024**2 for line in raw.splitlines()
    }
    if set(config["gpu_uuids"]) != set(free):
        raise RuntimeError("Protected GPU inventory differs")
    if any(free[gpu] < config["gpu_reserve_bytes"] for gpu in config["gpu_uuids"]):
        raise RuntimeError("Global GPU reserve")


def supervise(config_path: Path, permit_path: Path, output: Path) -> None:
    began = time.monotonic()
    config, permit = load(config_path), load(permit_path)
    digest = sha256(config_path)
    check_contract(config, permit, digest, time.time())
    if output != Path(config["output_root"]).absolute():
        raise ValueError("Output differs from sealed runtime config")
    environment(config)
    verify_bindings(config)
    output.mkdir(mode=0o700, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "permit.json", permit)
    plan = execution_plan(config)
    write_json(output / "execution-plan.json", plan)
    reserve_check(config, output)
    caps = config["caps"]
    peaks = dict(rss_bytes=0, cpu_seconds=0.0, output_bytes=0)
    child: subprocess.Popen[bytes] | None = None
    child_start: int | None = None
    try:
        with (output / "worker.log").open("xb") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "tools.run_gtzan_overlap",
                    "--private-worker",
                    "--config",
                    str(config_path),
                    "--permit",
                    str(permit_path),
                    "--guardian-pid",
                    str(os.getpid()),
                    "--guardian-start",
                    str(start_identity(os.getpid())),
                    "--plan",
                    str(output / "execution-plan.json"),
                    "--output",
                    str(output),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            child_start = start_identity(child.pid)
            if child_start is None:
                raise RuntimeError("Worker exited before identity capture")
            last_reserve = 0.0
            while child.poll() is None:
                rss, cpu = process_tree(os.getpid())
                used = disk_bytes(output)
                for key, value in (
                    ("rss_bytes", rss),
                    ("cpu_seconds", cpu),
                    ("output_bytes", used),
                ):
                    peaks[key] = max(peaks[key], value)
                if (
                    rss > caps["rss_bytes"]
                    or cpu > caps["cpu_seconds"]
                    or used > caps["output_bytes"]
                    or time.monotonic() - began > caps["wall_seconds"]
                    or time.time() >= config["deadline_epoch"]
                ):
                    raise RuntimeError("Whole owned-tree wall/CPU/RSS/output/UTC bound")
                if time.monotonic() - last_reserve >= 5:
                    reserve_check(config, output)
                    last_reserve = time.monotonic()
                time.sleep(0.1)
            child.wait()
            if child.returncode:
                raise RuntimeError(f"Worker exit{child.returncode}; remaining queries unresolved")
        reserve_check(config, output)
        rss, cpu = process_tree(os.getpid())
        used = disk_bytes(output)
        if (
            rss > caps["rss_bytes"]
            or cpu > caps["cpu_seconds"]
            or used > caps["output_bytes"]
            or time.monotonic() - began > caps["wall_seconds"]
            or time.time() >= config["deadline_epoch"]
        ):
            raise RuntimeError("Terminal owned-tree resource bound")
        for key, amount in (("rss_bytes", rss), ("cpu_seconds", cpu), ("output_bytes", used)):
            peaks[key] = max(peaks[key], amount)
        result = load(output / "result.json")
        if result.get("status") != "complete":
            raise RuntimeError("Missing complete worker result")
        write_json(
            output / "guardian-result.json",
            dict(
                status="complete",
                config_sha256=digest,
                result_sha256=sha256(output / "result.json"),
                peaks=peaks,
                wall_seconds=time.monotonic() - began,
                classifier_outputs_used=False,
                mode=config["mode"],
                qualification_passed=result.get("qualification_passed", False),
                qualification_binding_sha256=qualification_binding(config),
            ),
        )
    except BaseException as error:
        if child is not None and child_start is not None:
            terminate_own(child, child_start)
        write_json(
            output / "failure.json",
            dict(
                status="incomplete",
                error=repr(error),
                remaining_are_not_negatives=True,
                peaks=peaks,
            ),
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--permit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--guardian-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--guardian-start", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.private_worker:
        from experiments.gtzan_overlap import run

        if (
            args.plan is None
            or args.permit is None
            or os.getppid() != args.guardian_pid
            or start_identity(os.getppid()) != args.guardian_start
        ):
            raise ValueError("Private execution requires live guardian and root permit")
        config = load(args.config)
        check_contract(config, load(args.permit), sha256(args.config), time.time())
        verify_bindings(config)
        if args.output != Path(config["output_root"]).absolute() or load(
            args.plan
        ) != execution_plan(config):
            raise ValueError("Private execution plan/output changed")
        run(
            load(args.plan),
            dict(
                config,
                query_seconds=config["caps"]["query_seconds"],
                qualification_binding_sha256=qualification_binding(config),
            ),
            args.output,
        )
    else:
        if args.permit is None:
            raise ValueError("Explicit root permit required")
        supervise(args.config.resolve(), args.permit.resolve(), args.output.absolute())


if __name__ == "__main__":
    main()
