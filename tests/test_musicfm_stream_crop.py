"""Synthetic bytes exercise real pipes, boundaries, integrity and termination."""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from tools.musicfm_stream_crop import stream_crop


def command(path: Path, suffix: str = "") -> list[str]:
    return [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(open(sys.argv[1],'rb').read());" + suffix,
        str(path),
    ]


@pytest.mark.parametrize("start,length", [(0, 13), (16383, 1025), (271234, 288000), (599983, 17)])
def test_streaming_exact_sample_slices_without_full_output_file(
    tmp_path: Path, start: int, length: int
) -> None:
    values = np.linspace(-0.9, 0.9, 600000, dtype="<f4")
    raw = values.tobytes()
    source = tmp_path / "input.f32"
    source.write_bytes(raw)
    calls = []
    actual = stream_crop(
        command(source),
        samples=len(values),
        waveform_sha256=hashlib.sha256(raw).hexdigest(),
        start=start,
        length=length,
        stderr=tmp_path / "stderr",
        watch=lambda: calls.append(1),
    )
    assert actual.tobytes() == values[start : start + length].tobytes()
    assert len(calls) >= 2
    assert {path.name for path in tmp_path.iterdir()} == {"input.f32", "stderr"}


def test_corruption_after_requested_crop_is_never_accepted(tmp_path: Path) -> None:
    reference = np.zeros(1000, dtype="<f4")
    changed = reference.copy()
    changed[-1] = 1
    source = tmp_path / "changed"
    source.write_bytes(changed.tobytes())
    with pytest.raises(ValueError, match="Complete decoded waveform identity"):
        stream_crop(
            command(source),
            samples=1000,
            waveform_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
            start=0,
            length=10,
            stderr=tmp_path / "stderr",
            watch=lambda: None,
        )


def test_nonzero_exit_preserves_stderr_even_with_matching_output(tmp_path: Path) -> None:
    raw = np.zeros(100, dtype="<f4").tobytes()
    source = tmp_path / "input"
    source.write_bytes(raw)
    with pytest.raises(ValueError, match="exit=7"):
        stream_crop(
            command(source, "sys.stderr.write('warning');sys.exit(7)"),
            samples=100,
            waveform_sha256=hashlib.sha256(raw).hexdigest(),
            start=0,
            length=10,
            stderr=tmp_path / "stderr",
            watch=lambda: None,
        )
    assert (tmp_path / "stderr").read_text() == "warning"


def test_watchdog_stops_and_reaps_stalled_decoder(tmp_path: Path) -> None:
    pidfile = tmp_path / "pid"
    program = "import os,sys,time;open(sys.argv[1],'w').write(str(os.getpid()));time.sleep(30)"
    began = time.monotonic()

    def watch() -> None:
        if time.monotonic() - began > 0.2:
            raise RuntimeError("watchdog stop")

    with pytest.raises(RuntimeError, match="watchdog stop"):
        stream_crop(
            [sys.executable, "-c", program, str(pidfile)],
            samples=1,
            waveform_sha256=hashlib.sha256(b"\0" * 4).hexdigest(),
            start=0,
            length=1,
            stderr=tmp_path / "stderr",
            watch=watch,
        )
    import os

    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)
    assert time.monotonic() - began < 5


def test_watchdog_remains_active_after_stdout_eof(tmp_path: Path) -> None:
    program = "import os,time;os.write(1,b'\\0'*4);os.close(1);time.sleep(30)"
    began = time.monotonic()

    def watch() -> None:
        if time.monotonic() - began > 0.2:
            raise RuntimeError("post-EOF watchdog")

    with pytest.raises(RuntimeError, match="post-EOF watchdog"):
        stream_crop(
            [sys.executable, "-c", program],
            samples=1,
            waveform_sha256=hashlib.sha256(b"\0" * 4).hexdigest(),
            start=0,
            length=1,
            stderr=tmp_path / "stderr",
            watch=watch,
        )
    assert time.monotonic() - began < 2
