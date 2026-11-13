"""Compile the reference wrapper against a synthetic ABI; never run a classifier."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "tools/native/keyfinder_reference.cpp"
LABELS = [
    "A major",
    "A minor",
    "A# major",
    "A# minor",
    "B major",
    "B minor",
    "C major",
    "C minor",
    "C# major",
    "C# minor",
    "D major",
    "D minor",
    "D# major",
    "D# minor",
    "E major",
    "E minor",
    "F major",
    "F minor",
    "F# major",
    "F# minor",
    "G major",
    "G minor",
    "G# major",
    "G# minor",
]
CLASSES = [18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
NAMES = [
    "A_MAJOR",
    "A_MINOR",
    "B_FLAT_MAJOR",
    "B_FLAT_MINOR",
    "B_MAJOR",
    "B_MINOR",
    "C_MAJOR",
    "C_MINOR",
    "D_FLAT_MAJOR",
    "D_FLAT_MINOR",
    "D_MAJOR",
    "D_MINOR",
    "E_FLAT_MAJOR",
    "E_FLAT_MINOR",
    "E_MAJOR",
    "E_MINOR",
    "F_MAJOR",
    "F_MINOR",
    "G_FLAT_MAJOR",
    "G_FLAT_MINOR",
    "G_MAJOR",
    "G_MINOR",
    "A_FLAT_MAJOR",
    "A_FLAT_MINOR",
    "SILENCE",
]


def compile_stub(tmp_path: Path, changed: bool = False) -> subprocess.CompletedProcess[str]:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    names = list(NAMES)
    if changed:
        names[0], names[1] = names[1], names[0]
    (tmp_path / "constants.h").write_text(
        "#pragma once\nnamespace KeyFinder { enum key_t {" + ",".join(names) + "}; }\n"
    )
    (tmp_path / "audiodata.h").write_text(
        "#pragma once\nnamespace KeyFinder { struct AudioData {"
        "void setFrameRate(unsigned) {} void setChannels(unsigned) {} "
        "void addToSampleCount(unsigned) {} void setSample(unsigned,double) {} }; }\n"
    )
    (tmp_path / "keyfinder.h").write_text(
        '#pragma once\n#include "constants.h"\n#include "audiodata.h"\n'
        "#include <stdexcept>\nnamespace KeyFinder { struct KeyFinder {"
        'key_t keyOfAudio(AudioData&) { throw std::runtime_error("STUB_CLASSIFIER"); } }; }\n'
    )
    return subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(tmp_path),
            str(SOURCE),
            "-o",
            str(tmp_path / "reference"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_exhaustive_enum_output_and_errors(tmp_path: Path) -> None:
    build = compile_stub(tmp_path)
    assert build.returncode == 0, build.stderr
    binary = str(tmp_path / "reference")
    result = subprocess.run([binary, "--self-test"], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 25
    for raw, (label, code) in enumerate(zip(LABELS, CLASSES, strict=True)):
        tonic, mode = label.split()
        assert rows[raw] == {
            "schema": "drone.keyfinder-reference.v1",
            "raw_enum": raw,
            "status": "key",
            "tonic": tonic,
            "mode": mode,
            "drone_class": code,
        }
    assert rows[24] == {
        "schema": "drone.keyfinder-reference.v1",
        "raw_enum": 24,
        "status": "silence",
        "tonic": None,
        "mode": None,
        "drone_class": None,
    }
    bad = tmp_path / "malformed.wav"
    for content in (b"", b"RIFF\x04\x00\x00\x00WAVE", b"RIFF\x10\x00\x00\x00WAVE"):
        bad.write_bytes(content)
        rejected = subprocess.run([binary, str(bad)], capture_output=True, text=True, timeout=5)
        assert rejected.returncode == 1
        assert not rejected.stdout
        assert "STUB_CLASSIFIER" not in rejected.stderr
    rejected = subprocess.run([binary, "--unknown"], capture_output=True, text=True, timeout=5)
    assert rejected.returncode == 1
    assert not rejected.stdout


def test_changed_enum_abi_fails_compile(tmp_path: Path) -> None:
    build = compile_stub(tmp_path, changed=True)
    assert build.returncode != 0
    assert "enum contract changed" in build.stderr
