"""Provision a wheel-pinned, isolated research runtime without importing ML packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import urlopen


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def process_start(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def protected(plan: dict[str, Any]) -> None:
    for item in plan["protected_processes"]:
        if process_start(item["pid"]) != item["start_ticks"]:
            raise RuntimeError("Protected process identity changed")


def tree_usage(pgid: int) -> tuple[int, float]:
    rss = 0
    cpu = 0.0
    ticks = os.sysconf("SC_CLK_TCK")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid:
                cpu += sum(int(fields[i]) for i in (11, 12, 13, 14)) / ticks
                rss += int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
        except (FileNotFoundError, ProcessLookupError):
            continue
    return rss, cpu


def owned_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def worker(plan: dict[str, Any], manifest: dict[str, Any]) -> None:
    root = Path(plan["root"])
    run = root / "provision-v1"
    wheels = run / "wheels"
    wheels.mkdir()
    transferred = 0
    reused = 0
    records: list[dict[str, Any]] = []
    for item in manifest["wheels"]:
        protected(plan)
        parsed = urlparse(item["url"])
        if parsed.scheme != "https" or parsed.hostname not in {
            "files.pythonhosted.org",
            "download.pytorch.org",
            "download-r2.pytorch.org",
        }:
            raise ValueError("Unapproved wheel origin")
        name = unquote(Path(parsed.path).name)
        if not name.endswith(".whl") or item["bytes"] > plan["max_file_bytes"]:
            raise ValueError("Invalid wheel name or size")
        target = wheels / name
        if target.exists():
            raise FileExistsError(target)
        cached = Path(plan["reuse_wheel_directory"]) / name
        partial = target.with_suffix(".whl.partial")
        origin = "download"
        if cached.is_file() and not cached.is_symlink() and digest(cached) == item["sha256"]:
            if cached.stat().st_size != item["bytes"]:
                raise ValueError("Cached wheel size differs")
            shutil.copyfile(cached, partial)
            reused += item["bytes"]
            origin = "verified-local-wheel"
        else:
            with urlopen(item["url"], timeout=30) as response, partial.open("xb") as output:
                actual = 0
                while block := response.read(1024 * 1024):
                    actual += len(block)
                    transferred += len(block)
                    if actual > item["bytes"] or transferred > plan["max_download_bytes"]:
                        raise RuntimeError("Download budget exceeded")
                    output.write(block)
        if partial.stat().st_size != item["bytes"] or digest(partial) != item["sha256"]:
            raise ValueError("Wheel identity mismatch")
        partial.rename(target)
        records.append({"name": item["name"], "bytes": item["bytes"], "origin": origin})
        write_json(
            run / "download-progress.json", {"complete": records, "download_bytes": transferred}
        )
    requirement = run / "requirements.lock"
    requirement.write_text(
        "".join(
            f"{item['name']}=={item['version']} --hash=sha256:{item['sha256']}\n"
            for item in manifest["wheels"]
        )
    )
    env = root / ".venv"
    if env.exists():
        raise FileExistsError("Refuse to change an existing environment")
    uv = plan["uv"]
    subprocess.run([uv, "venv", "--python", plan["python"], str(env)], check=True)
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(env / "bin/python"),
            "--no-index",
            "--find-links",
            str(wheels),
            "--require-hashes",
            "--no-deps",
            "--no-build",
            "--no-cache",
            "--link-mode",
            "copy",
            "-r",
            str(requirement),
        ],
        check=True,
    )
    # This interpreter imports metadata only. No torch, audio, CUDA, or model module is imported.
    inventory_program = """
import importlib.metadata as m,json,sys
print(json.dumps({'prefix':sys.prefix,'path':sys.path,
 'distributions':sorted([{'name':d.metadata['Name'],'version':d.version} for d in m.distributions()],key=lambda d:d['name'].lower()),
 'model_modules_loaded':[n for n in sys.modules if n.split('.')[0] in {'torch','torchaudio','transformers','nnAudio'}]}))
"""
    inventory = json.loads(
        subprocess.check_output([str(env / "bin/python"), "-I", "-c", inventory_program], text=True)
    )

    def normalize(name: str) -> str:
        return name.lower().replace("_", "-").replace(".", "-")

    expected = {normalize(item["name"]): item["version"] for item in manifest["wheels"]}
    actual = {normalize(item["name"]): item["version"] for item in inventory["distributions"]}
    if actual != expected or inventory["prefix"] != str(env) or inventory["model_modules_loaded"]:
        raise ValueError("Installed distribution inventory differs")
    for entry in inventory["path"]:
        if "site-packages" in entry and not Path(entry).is_relative_to(env):
            raise ValueError("External package directory in isolated runtime")
    site = env / "lib/python3.12/site-packages"
    pth = {str(path.relative_to(env)): path.read_text() for path in site.glob("*.pth")}
    if any("/home/" in body or "sys.path" in body for body in pth.values()):
        raise ValueError("Mutable external path injection")
    inventory["pth_files"] = pth
    inventory["files"] = []
    for path in sorted(site.rglob("*")):
        if path.is_symlink():
            raise ValueError("Package symlink violates isolated inventory")
        if path.is_file():
            inventory["files"].append(
                {
                    "path": str(path.relative_to(env)),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    write_json(run / "installed-inventory.json", inventory)
    protected(plan)
    write_json(
        run / "worker-result.json",
        {
            "status": "complete",
            "download_bytes": transferred,
            "reused_bytes": reused,
            "packages": len(actual),
            "inventory_sha256": digest(run / "installed-inventory.json"),
            "model_modules_loaded": [],
            "gpu_execution": False,
        },
    )


def supervise(plan: dict[str, Any], plan_path: Path, manifest_path: Path) -> None:
    root = Path(plan["root"])
    run = root / "provision-v1"
    if root.is_symlink() or not (root / "OWNER.json").is_file():
        raise ValueError("Task ownership marker absent")
    if shutil.disk_usage(root).free < plan["min_free_disk_bytes"]:
        raise RuntimeError("Insufficient free disk")
    memory = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    }
    if memory["MemAvailable"] < plan["min_available_memory_bytes"]:
        raise RuntimeError("Insufficient available memory")
    protected(plan)
    run.mkdir()
    started = time.monotonic()
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "LD_PRELOAD", "LD_LIBRARY_PATH"):
        environment.pop(key, None)
    environment.update(
        CUDA_VISIBLE_DEVICES="",
        UV_CACHE_DIR=str(run / "uv-cache"),
        UV_PYTHON_DOWNLOADS="never",
        UV_HTTP_RETRIES="0",
        UV_NO_PROGRESS="1",
        TMPDIR=str(run / "tmp"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    (run / "tmp").mkdir()
    set_affinity = getattr(os, "sched_setaffinity", None)
    if set_affinity is None:
        raise RuntimeError("Linux CPU affinity is required")
    set_affinity(0, plan["cpus"])
    os.nice(19)
    initial_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    failure: str | None = None
    peak_rss = 0
    peak_bytes = 0
    peak_cpu = 0.0
    with (run / "worker.log").open("w") as log:
        child = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--plan",
                str(plan_path),
                "--manifest",
                str(manifest_path),
                "--worker",
            ],
            env=environment,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        write_json(
            run / "launch.json",
            {
                "worker_pid": child.pid,
                "worker_start": process_start(child.pid),
                "guardian_pid": os.getpid(),
                "guardian_start": process_start(os.getpid()),
            },
        )
        try:
            while child.poll() is None:
                protected(plan)
                rss, cpu = tree_usage(child.pid)
                size = owned_bytes(root)
                peak_rss, peak_cpu, peak_bytes = (
                    max(peak_rss, rss),
                    max(peak_cpu, cpu),
                    max(peak_bytes, size),
                )
                if (
                    time.monotonic() - started > plan["wall_seconds"]
                    or cpu > plan["cpu_seconds"]
                    or rss > plan["rss_bytes"]
                    or size > plan["owned_bytes"]
                ):
                    raise RuntimeError("Provision resource limit exceeded")
                time.sleep(0.5)
            if child.returncode:
                raise RuntimeError(f"Provision worker exited {child.returncode}")
            protected(plan)
        except BaseException as error:
            failure = f"{type(error).__name__}: {error}"
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        final_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (
        final_children.ru_utime
        + final_children.ru_stime
        - initial_children.ru_utime
        - initial_children.ru_stime
    )
    wall = time.monotonic() - started
    final_bytes = owned_bytes(root)
    if failure is None and (
        wall > plan["wall_seconds"]
        or cpu > plan["cpu_seconds"]
        or final_bytes > plan["owned_bytes"]
    ):
        failure = "Terminal resource limit exceeded"
    receipt = {
        "status": "failed" if failure else "complete",
        "error": failure,
        "worker_exit": child.returncode,
        "wall_seconds": wall,
        "child_cpu_seconds": cpu,
        "sampled_peak_rss_bytes": peak_rss,
        "sampled_peak_owned_bytes": peak_bytes,
        "final_owned_bytes": final_bytes,
        "sampled_peak_cpu_seconds": peak_cpu,
        "plan_sha256": digest(plan_path),
        "manifest_sha256": digest(manifest_path),
    }
    write_json(run / "controller.json", receipt)
    print(json.dumps(receipt))
    if failure:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    manifest = json.loads(args.manifest.read_text())
    if digest(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("Provision source differs from reviewed plan")
    if digest(args.manifest) != plan["manifest_sha256"]:
        raise ValueError("Manifest differs from reviewed plan")
    if args.worker:
        worker(plan, manifest)
    else:
        supervise(plan, args.plan.resolve(), args.manifest.resolve())


if __name__ == "__main__":
    main()
