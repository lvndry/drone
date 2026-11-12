"""Bounded task-local wheel overlay, CPU imports, then conditional pinned checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

from tools import run_cqt_expansion as guard
from tools.preflight_musicfm import WEIGHT_BYTES, WEIGHT_SHA, sha


def select_wheels(lock: Path) -> list[dict[str, Any]]:
    ranks = {tag: index for index, tag in enumerate(sys_tags())}
    result = []
    for package in tomllib.loads(lock.read_text())["packages"]:
        name = canonicalize_name(package["name"])
        if name in {"torch", "numpy"}:
            raise ValueError("Torch/NumPy must not be resolved, downloaded or shadowed")
        candidates = []
        for wheel in package["wheels"]:
            url = urlparse(wheel["url"])
            filename = unquote(Path(url.path).name)
            cpu_audio = (
                name == "torchaudio"
                and package["version"] == "2.11.0+cpu"
                and url.hostname in {"download.pytorch.org", "download-r2.pytorch.org"}
                and url.path.startswith("/whl/cpu/")
            )
            if url.scheme != "https" or not (url.hostname == "files.pythonhosted.org" or cpu_audio):
                raise ValueError("unexpected wheel origin")
            _, _, _, tags = parse_wheel_filename(filename)
            order = min((ranks[tag] for tag in tags if tag in ranks), default=None)
            if order is not None:
                candidates.append((order, filename, wheel))
        if not candidates:
            raise ValueError(f"no compatible wheel:{name}")
        _, filename, wheel = min(candidates, key=lambda item: (item[0], item[1]))
        result.append(
            {
                "name": name,
                "version": package["version"],
                "filename": filename,
                "url": wheel["url"],
                "bytes": wheel["size"],
                "sha256": wheel["hashes"]["sha256"],
            }
        )
    if len({row["name"] for row in result}) != len(result):
        raise ValueError("duplicate resolved distribution")
    return result


def provision(root: Path, base_python: Path) -> None:
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is None or affinity(0) != {8, 9}:
        raise ValueError("fixed quiet CPU pair8,9 required")
    stage = root / "provision"
    stage.mkdir()
    (stage / "logs").mkdir()
    guard.CPU_LIMIT, guard.WALL_LIMIT = 600.0, 1200.0
    setattr(guard, "BYTE_LIMIT", 3 * 1024**3)
    args = argparse.Namespace(
        stage_root=stage,
        budget_root=root,
        source_root=root / "source",
        math_library=None,
        deadline=(datetime.now(timezone.utc) + timedelta(seconds=1200)).isoformat(),
    )
    supervisor = guard.Supervisor(args)
    supervisor.environment.update(
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        UV_CACHE_DIR=str(root / "cache"),
        UV_PYTHON_DOWNLOADS="never",
        PYTHONPATH=str(root / "source"),
    )
    started = time.monotonic()
    report: dict[str, Any] = {
        "complete": False,
        "gpu_execution": False,
        "cpu_affinity": [8, 9],
        "source_sha256": sha(Path(__file__)),
        "lock_sha256": sha(root / "pylock.toml"),
        "commands": [],
    }

    def run(command: list[str], label: str) -> None:
        report["commands"].append({"label": label, "argv": command})
        supervisor.run(command, label)

    def check_file(path: Path, size: int, digest: str) -> None:
        if path.stat().st_size != size or sha(path) != digest:
            raise ValueError(f"artifact integrity:{path.name}")

    try:
        run(
            [
                str(base_python),
                "-m",
                "tools.check_musicfm_cpu",
                "baseline",
                "--root",
                str(root),
                "--output",
                str(root / "baseline.json"),
            ],
            "baseline",
        )
        wheels = select_wheels(root / "pylock.toml")
        payload = sum(row["bytes"] for row in wheels) + WEIGHT_BYTES
        if payload > 2 * 1024**3:
            raise ValueError("2GiB total wheel/checkpoint download budget exceeded")
        guard.write_json(
            root / "selected-wheels.json", {"wheels": wheels, "planned_payload_bytes": payload}
        )
        wheel_root = root / "wheels"
        wheel_root.mkdir()
        for index, wheel in enumerate(wheels):
            path = wheel_root / wheel["filename"]
            run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--connect-timeout",
                    "15",
                    "--max-time",
                    "120",
                    "--max-filesize",
                    str(wheel["bytes"]),
                    "--output",
                    str(path),
                    wheel["url"],
                ],
                f"wheel-{index:02d}",
            )
            check_file(path, wheel["bytes"], wheel["sha256"])
        overlay = root / "overlay"
        run(
            ["uv", "venv", "--python", str(base_python), "--no-python-downloads", str(overlay)],
            "venv",
        )
        site = (
            overlay / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        )
        shared = (
            base_python.parent.parent
            / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        )
        (site / "readonly_shared_torch_numpy.pth").write_text(str(shared) + "\n")
        requirements = root / "overlay-requirements.txt"
        requirements.write_text(
            "".join(
                f"{row['name']}=={row['version']} --hash=sha256:{row['sha256']}\n" for row in wheels
            )
        )
        python = overlay / "bin/python"
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--offline",
                "--no-index",
                "--find-links",
                str(wheel_root),
                "--no-deps",
                "--require-hashes",
                "--link-mode",
                "copy",
                "-r",
                str(requirements),
            ],
            "offline-install",
        )
        run(
            [
                str(python),
                "-m",
                "tools.check_musicfm_cpu",
                "imports",
                "--root",
                str(root),
                "--output",
                str(root / "imports.json"),
            ],
            "cpu-imports",
        )
        imports = json.loads((root / "imports.json").read_bytes())
        if imports.get("passed") is not True or imports["protected"] != imports["protected_after"]:
            raise ValueError("imports/protected module gate")
        # No checkpoint download can occur before the import gate above.
        inputs = root / "inputs"
        inputs.mkdir()
        checkpoint = inputs / "pretrained_msd.pt"
        url = "https://huggingface.co/minzwon/MusicFM/resolve/4513b38bc25ad1d227b1980819b9691ba97f4d87/pretrained_msd.pt"
        run(
            [
                "curl",
                "--fail",
                "--location",
                "--connect-timeout",
                "15",
                "--max-time",
                "600",
                "--max-filesize",
                str(WEIGHT_BYTES),
                "--output",
                str(checkpoint),
                url,
            ],
            "pinned-checkpoint-download",
        )
        check_file(checkpoint, WEIGHT_BYTES, WEIGHT_SHA)
        run(
            [
                str(python),
                "-m",
                "tools.check_musicfm_cpu",
                "layout",
                "--root",
                str(root),
                "--output",
                str(root / "checkpoint-layout.json"),
            ],
            "cpu-checkpoint-layout",
        )
        run(
            [
                str(base_python),
                "-m",
                "tools.check_musicfm_cpu",
                "baseline",
                "--root",
                str(root),
                "--output",
                str(root / "baseline-after.json"),
            ],
            "shared-baseline-after",
        )
        if (
            json.loads((root / "baseline.json").read_bytes())["protected"]
            != json.loads((root / "baseline-after.json").read_bytes())["protected"]
        ):
            raise ValueError("shared environment changed")
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report.update(
            resources=supervisor.resources,
            child_cpu_seconds=guard.child_cpu(),
            wall_seconds=time.monotonic() - started,
            peak_tree_rss_bytes=supervisor.peak_resident,
            storage_bytes=guard.disk_bytes(root),
        )
        guard.write_json(root / "provision-report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-python", type=Path, required=True)
    arguments = parser.parse_args()
    provision(arguments.root, arguments.base_python)
