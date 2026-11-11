"""Record immutable native-CQT source/build evidence without numerical comparison."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from experiments.chunked import sha256
from tools.research.measurements import runtime_environment


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def record(root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Preserve existing evidence")
    library = root / "soxr-build/src/libsoxr.so.0.1.2"
    loaded = ctypes.CDLL(str(library))
    loaded.soxr_version.restype = ctypes.c_char_p
    version: bytes = loaded.soxr_version()
    binaries = {}
    for module_name in ("soxr.soxr_ext", "numpy.fft._pocketfft_umath"):
        module = importlib.import_module(module_name)
        if module.__file__ is None:
            raise ValueError("Binary module has no file")
        path = Path(module.__file__)
        binaries[module_name] = dict(path=str(path), sha256=sha256(path))
    ffmpeg = Path(shutil.which("ffmpeg") or "")
    binaries["ffmpeg"] = dict(path=str(ffmpeg), sha256=sha256(ffmpeg))
    selected = [
        root / "soxr-1.1.0.tar.gz",
        root / "soxr-source.json",
        root / "soxr-build/CMakeCache.txt",
        root / "soxr-configure.log",
        root / "soxr-build.log",
        root / "libcqt_soxr_split.so",
        library,
        root / "drone-source.tar",
        root / "euphoria-source.tar",
        root / "goldens-b1c90d3/manifest.json",
        root / "owner-snapshot/cqt.rs",
        root / "owner-snapshot/halving.rs",
        root / "drone-source-b1c90d3/weights/drone-v1.pt",
    ]
    source = root / "dependencies/soxr-1.1.0"
    selected.extend(path for path in source.rglob("*") if path.is_file())
    selected.extend((root / "soxr-build").rglob("flags.make"))
    selected.extend((root / "soxr-build").rglob("link.txt"))
    selected.extend(
        root.glob("euphoria-source-*/research/drone-native-cqt/target/release/drone-native-cqt")
    )
    result: dict[str, Any] = dict(
        format="k10-native-cqt-build-evidence-v1",
        libsoxr_version=version.decode(),
        source_revision="python-soxr1.1.0 vendored libsoxr0.1.3-14-ga66f3ee",
        libsoxr_license="LGPL-2.1-or-later; research dynamic build, no product licensing claim",
        wrapper_license="python-soxr LGPL-2.1-or-later; wrapper follows split mono float32 HQ 96000-sample/flush semantics",
        upstream_source="https://pypi.org/project/soxr/1.1.0/#files",
        runtime_environment=runtime_environment(),
        compiler=command("cc", "--version"),
        cmake=command("cmake", "--version"),
        cpu=command("lscpu"),
        kernel=command("uname", "-a"),
        build_jobs=2,
        build_affinity="2,3",
        cmake_configuration=[
            "BUILD_SHARED_LIBS=ON",
            "WITH_OPENMP=OFF",
            "BUILD_TESTS=OFF",
            "BUILD_EXAMPLES=OFF",
            "WITH_LSR_BINDINGS=OFF",
            "CMAKE_BUILD_TYPE=Release",
        ],
        wrapper_compiler_flags=["-shared", "-fPIC", "-O2"],
        binary_dependencies=binaries,
        files={
            str(path.relative_to(root)): dict(bytes=path.stat().st_size, sha256=sha256(path))
            for path in sorted(set(selected))
        },
        bytes_under_root=sum(
            path.stat().st_size
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        ),
    )
    if result["bytes_under_root"] > 2 * 1024**3:
        raise RuntimeError("Experiment storage budget exceeded")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = record(args.root, args.output)
    print(
        json.dumps(
            {
                "libsoxr_version": result["libsoxr_version"],
                "bytes_under_root": result["bytes_under_root"],
            }
        )
    )


if __name__ == "__main__":
    main()
