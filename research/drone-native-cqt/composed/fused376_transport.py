"""Prospective sequential PCM transport; external guardians own all resource limits."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import IO, Any

BLOCK = 512 * 1024


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            result.update(block)
    return result.hexdigest()


def write(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def exact(stream: IO[bytes], count: int) -> bytes:
    data = bytearray()
    while len(data) < count:
        block = stream.read(count - len(data))
        if not block:
            raise EOFError("Transport ended before declared boundary")
        data.extend(block)
    return bytes(data)


def send(stream: IO[bytes], value: object) -> None:
    raw = json.dumps(value, allow_nan=False).encode()
    if len(raw) > 16384:
        raise ValueError("Frame metadata exceeds bound")
    stream.write(struct.pack("!I", len(raw)))
    stream.write(raw)
    stream.flush()


def receive(stream: IO[bytes]) -> dict[str, Any]:
    count = struct.unpack("!I", exact(stream, 4))[0]
    if not 0 < count <= 16384:
        raise ValueError("Frame metadata bound")
    value = json.loads(exact(stream, count))
    if not isinstance(value, dict):
        raise ValueError("Frame object required")
    return value


def module_path(module: ModuleType) -> Path:
    origin = module.__file__
    if origin is None:
        raise ValueError("Archived support module has no file origin")
    return Path(origin)


def support(config: dict[str, Any]) -> ModuleType:
    # Fresh workers must resolve the exact archived support source, not editable worktrees.
    if "tools" in sys.modules:
        raise ValueError("Unexpected preloaded tools namespace")
    sys.path.insert(0, config["legacy_source_root"])
    transport = importlib.import_module("tools.cqt_mac_transport")
    guard = importlib.import_module("tools.run_cqt_expansion")
    for role, module in [("transport", transport), ("legacy_guard", guard)]:
        record = config["legacy_files"][role]
        if module_path(module).resolve() != Path(record["path"]).resolve():
            raise ValueError("Archived support origin mismatch")
        if sha(module_path(module)) != record["sha256"]:
            raise ValueError("Archived support bytes changed")
    return transport


def no_reference_modules() -> None:
    if any(name in sys.modules for name in ("torch", "onnxruntime", "librosa")):
        raise ValueError("Unexpected model or CQT import")


def server(config: dict[str, Any], membership: dict[str, Any]) -> None:
    root = Path(config["output_root"])
    transport = support(config)
    producer = importlib.import_module("tools.cqt_pipeline_stage")
    record = config["legacy_files"]["producer"]
    if (
        module_path(producer).resolve() != Path(record["path"]).resolve()
        or sha(module_path(producer)) != record["sha256"]
    ):
        raise ValueError("Archived producer origin/bytes changed")
    out, incoming = sys.stdout.buffer, sys.stdin.buffer
    send(out, {"kind": "ready", "membership_sha256": config["membership_sha256"], "cases": 376})
    receipts = []
    for row in membership["cases"]:
        index = row["index"]
        spool = root / "spool"
        # The original producer verifies public audio and original decoded PCM hashes.
        producer.produce(config["producer"], index, spool)
        no_reference_modules()
        pcm = spool / "pcm.f32"
        if pcm.stat().st_size != row["pcm_bytes"] or sha(pcm) != row["pcm_sha256"]:
            raise ValueError("Restored PCM differs from frozen membership")
        packed = spool / "pcm.s16"
        packed_hash, reconstructed = hashlib.sha256(), hashlib.sha256()
        with pcm.open("rb") as source, packed.open("xb") as target:
            for block in iter(lambda: source.read(BLOCK), b""):
                encoded = transport.encode(block, "pcm")
                target.write(encoded)
                packed_hash.update(encoded)
                reconstructed.update(transport.decode(encoded, "pcm"))
        if (
            packed.stat().st_size != row["pcm_bytes"] // 2
            or reconstructed.hexdigest() != row["pcm_sha256"]
        ):
            raise ValueError("Original PCM is not losslessly representable")
        header = {
            "kind": "case",
            "index": index,
            "id": row["id"],
            "packed_bytes": packed.stat().st_size,
            "packed_sha256": packed_hash.hexdigest(),
            "pcm_sha256": row["pcm_sha256"],
            "pcm_bytes": row["pcm_bytes"],
        }
        write(root / "cases" / f"{index:04}.json", {"sent": header})
        send(out, header)
        with packed.open("rb") as source:
            shutil.copyfileobj(source, out, length=BLOCK)
        out.flush()
        acknowledgment = receive(incoming)
        expected = {
            "kind": "ack",
            "index": index,
            "packed_sha256": header["packed_sha256"],
            "pcm_sha256": row["pcm_sha256"],
            "native_result_retained": True,
        }
        if acknowledgment != expected:
            raise ValueError("Missing verified receiver/native acknowledgment")
        write(root / "cases" / f"{index:04}-ack.json", acknowledgment)
        receipts.append({"index": index, "packed_sha256": header["packed_sha256"]})
        shutil.rmtree(spool)  # Only this newly created, acknowledged one-case spool.
    no_reference_modules()
    write(
        root / "result.json",
        {
            "complete": True,
            "cases": receipts,
            "membership_sha256": config["membership_sha256"],
            "no_reference_or_model": True,
        },
    )
    send(out, {"kind": "complete", "cases": 376})


def client(config: dict[str, Any], membership: dict[str, Any]) -> None:
    root = Path(config["output_root"])
    transport = support(config)
    reports = []
    first_failure_saved = False
    with (root / "remote.log").open("xb") as log:
        remote = subprocess.Popen(
            config["ssh_argv"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            env=config["environment"],
        )
        if remote.stdin is None or remote.stdout is None:
            raise RuntimeError("SSH pipes unavailable")
        incoming, outgoing = remote.stdout, remote.stdin
        try:
            if receive(incoming) != {
                "kind": "ready",
                "membership_sha256": config["membership_sha256"],
                "cases": 376,
            }:
                raise ValueError("Remote frozen membership mismatch")
            for row, inputs in zip(membership["cases"], config["local_cases"], strict=True):
                index = row["index"]
                header = receive(incoming)
                if (
                    header.get("kind") != "case"
                    or header.get("index") != index
                    or header.get("id") != row["id"]
                    or header.get("pcm_sha256") != row["pcm_sha256"]
                    or header.get("pcm_bytes") != row["pcm_bytes"]
                    or header.get("packed_bytes") != row["pcm_bytes"] // 2
                ):
                    raise ValueError("Unexpected case or transport geometry")
                spool = root / "spool"
                spool.mkdir()
                packed_hash, pcm_hash = hashlib.sha256(), hashlib.sha256()
                with (spool / "pcm.f32").open("xb") as pcm:
                    remaining = header["packed_bytes"]
                    while remaining:
                        block = exact(incoming, min(BLOCK, remaining))
                        packed_hash.update(block)
                        decoded = transport.decode(block, "pcm")
                        pcm_hash.update(decoded)
                        pcm.write(decoded)
                        remaining -= len(block)
                if (
                    packed_hash.hexdigest() != header["packed_sha256"]
                    or pcm_hash.hexdigest() != row["pcm_sha256"]
                    or (spool / "pcm.f32").stat().st_size != row["pcm_bytes"]
                ):
                    raise ValueError("Independent PCM reconstruction failed")
                for item in [inputs["manifest"], *inputs["csr"]]:
                    source = Path(item["path"])
                    if sha(source) != item["sha256"]:
                        raise ValueError("Retained Mac plan input changed")
                    target = spool / item["relative_path"]
                    if not target.resolve().is_relative_to(spool.resolve()):
                        raise ValueError("Unsafe retained artifact destination")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                output = root / "cases" / f"{index:04}"
                argv = [
                    *config["native_argv_prefix"],
                    "--index",
                    str(index),
                    "--manifest",
                    str(spool / "mac-manifest.json"),
                    "--output",
                    str(output),
                ]
                with (root / "logs" / f"{index:04}.log").open("xb") as native_log:
                    completed = subprocess.run(
                        argv,
                        env=config["environment"],
                        stdout=native_log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if completed.returncode:
                    raise RuntimeError(f"Native artifact/runtime admission failed at case {index}")
                report = json.loads((output / "result.json").read_bytes())
                if report["index"] != index or report["id"] != row["id"]:
                    raise ValueError("Native result membership changed")
                for name, key in [
                    ("separate-product-core.f32", "baseline_core_sha256"),
                    ("fused-product-core.f32", "candidate_core_sha256"),
                ]:
                    if key in report and sha(output / name) != report[key]:
                        raise ValueError("Retained native output differs from report")
                passed = (
                    report["original_control_preserved"]
                    and report.get("candidate_matches_archived_mac_core") is True
                )
                if not passed and not first_failure_saved:
                    shutil.copytree(spool, root / "first-failure")
                    first_failure_saved = True
                write(output / "transport.json", header)
                reports.append(report)
                send(
                    outgoing,
                    {
                        "kind": "ack",
                        "index": index,
                        "packed_sha256": header["packed_sha256"],
                        "pcm_sha256": row["pcm_sha256"],
                        "native_result_retained": True,
                    },
                )
                shutil.rmtree(spool)
            if receive(incoming) != {"kind": "complete", "cases": 376}:
                raise ValueError("Remote terminal membership mismatch")
            outgoing.close()
            if remote.wait() != 0:
                raise RuntimeError("Remote guardian failed terminal qualification")
        finally:
            # On failure, EOF requests remote shutdown; external guardians retire their own groups.
            outgoing.close()
            incoming.close()
            if remote.poll() is None:
                remote.terminate()
                try:
                    remote.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    remote.kill()
                    remote.wait()
    no_reference_modules()
    write(
        root / "result.json",
        {
            "complete": True,
            "cases": reports,
            "all_controls_preserved": all(r["original_control_preserved"] for r in reports),
            "all_candidate_cores_exact": all(
                r.get("candidate_matches_archived_mac_core") is True for r in reports
            ),
            "membership_sha256": config["membership_sha256"],
            "candidate_linux_exact_cases": sum(
                r.get("candidate_matches_archived_linux_core") is True for r in reports
            ),
            "linux_comparison_is_secondary": True,
            "no_reference_or_model": True,
            "no_release_promotion": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    if sha(args.config) != args.config_sha256:
        raise ValueError("Sealed worker configuration changed")
    config = json.loads(args.config.read_bytes())
    if config.get("status") != "SEALED":
        raise ValueError("Prepared configuration is not a reviewed launch packet")
    membership_path = Path(config["membership_path"])
    if sha(membership_path) != config["membership_sha256"]:
        raise ValueError("Frozen membership bytes changed")
    membership = json.loads(membership_path.read_bytes())
    if len(membership["cases"]) != 376 or [r["index"] for r in membership["cases"]] != list(
        range(376)
    ):
        raise ValueError("Fixed complete ordered membership required")
    root = Path(config["output_root"])
    for name in ["cases", "logs"]:
        (root / name).mkdir()
    function = {"server": server, "client": client}[config["role"]]
    try:
        function(config, membership)
    except BaseException as error:
        write(root / "failed.json", {"error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
