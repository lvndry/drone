"""Bounded full-decode crop extraction, with full-waveform identity before return.

The caller supplies the pinned full FFmpeg command, canonical metadata, source
identity checks and a resource watchdog. This module never seeks or changes the
decoder/resampler profile, and never stores a complete waveform.
"""

from __future__ import annotations

import hashlib
import os
import select
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def stream_crop(
    command: Sequence[str],
    *,
    samples: int,
    waveform_sha256: str,
    start: int,
    length: int,
    stderr: Path,
    watch: Callable[[], None],
) -> NDArray[np.float32]:
    """Return a copied exact F32LE crop only after hashing every decoded sample.

    Watchdog calls target a 50 ms cadence while reading/waiting (sampled, not
    an instantaneous allocation limit), and forced before/after the command.
    A crop is withheld on corruption anywhere, including AFTER its own end.
    Every stderr byte is preserved. Exceptions terminate only this process group.
    """
    import numpy as np

    if (
        any(type(value) is not int for value in (samples, start, length))
        or samples <= 0
        or start < 0
        or length <= 0
        or start + length > samples
        or len(waveform_sha256) != 64
        or any(character not in "0123456789abcdef" for character in waveform_sha256)
    ):
        raise ValueError("Invalid bound full-waveform/crop contract")
    watch()
    expected_bytes = samples * 4
    first = start * 4
    stop = (start + length) * 4
    digest = hashlib.sha256()
    crop = bytearray()
    position = 0
    last_watch = time.monotonic()
    with stderr.open("xb") as errors:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=errors, start_new_session=True
        )
        assert process.stdout is not None
        try:
            while True:
                now = time.monotonic()
                if now - last_watch >= 0.05:
                    watch()
                    last_watch = time.monotonic()
                if not select.select([process.stdout], [], [], 0.05)[0]:
                    continue
                block = os.read(process.stdout.fileno(), 1024 * 1024)
                if not block:
                    break
                end = position + len(block)
                if end > expected_bytes:
                    raise ValueError("Full waveform exceeds its canonical byte count")
                digest.update(block)
                low = max(position, first)
                high = min(end, stop)
                if low < high:
                    crop.extend(block[low - position : high - position])
                position = end
            terminal_deadline = time.monotonic() + 5
            while True:
                watch()
                remaining = terminal_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Decoder did not exit after closing stdout")
                try:
                    code = process.wait(timeout=min(0.05, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            watch()
            if code != 0:
                raise ValueError(f"Full decoder exit={code}; stderr retained")
            if position != expected_bytes or digest.hexdigest() != waveform_sha256:
                raise ValueError("Complete decoded waveform identity differs; no crop returned")
            if len(crop) != length * 4:
                raise ValueError("Exact crop byte count differs")
            result = np.frombuffer(crop, dtype="<f4").copy()
            if not np.isfinite(result).all():
                raise ValueError("Canonical crop contains nonfinite samples")
            return result
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
            raise
        finally:
            process.stdout.close()
