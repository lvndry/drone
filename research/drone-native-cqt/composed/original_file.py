"""Finite, opt-in original-file CPU measurement; no extraction or inference on import."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import cast

PANEL_PATH = Path(__file__).with_name("END_TO_END_PANEL.json")
MAX_WALL_SECONDS = 1200.0
MAX_CPU_SECONDS = 600.0
MAX_RSS_BYTES = 3 * 1024**3
MAX_OUTPUT_BYTES = 256 * 1024**2
SAMPLE_SECONDS = 1.0

# Same rusage_info_v4 layout used by the qualified native-build guardian.
_USAGE_FIELDS = (
    "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size resident_size "
    "phys_footprint proc_start_abstime proc_exit_abstime child_user_time child_system_time "
    "child_pkg_idle_wkups child_interrupt_wkups child_pageins child_elapsed_abstime "
    "diskio_bytesread diskio_byteswritten cpu_time_qos_default cpu_time_qos_maintenance "
    "cpu_time_qos_background cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated "
    "cpu_time_qos_user_interactive billed_system_time serviced_system_time logical_writes "
    "lifetime_max_phys_footprint instructions cycles billed_energy serviced_energy "
    "interval_max_phys_footprint runnable_time"
).split()


class MacUsage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in _USAGE_FIELDS]


@cache
def mac_libproc() -> ctypes.CDLL:
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    library.proc_pid_rusage.restype = ctypes.c_int
    return library


def process_start(pid: int) -> int:
    if sys.platform == "linux":
        try:
            # /proc stat field 22, after the potentially space/parenthesis-containing comm.
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return int(fields[19])
        except FileNotFoundError as error:
            raise ProcessLookupError(pid) from error
    if sys.platform != "darwin":
        raise RuntimeError("Unsupported process identity platform")
    usage = MacUsage()
    if mac_libproc().proc_pid_rusage(pid, 4, ctypes.byref(usage)) != 0:
        error = ctypes.get_errno()
        if error in (errno.ESRCH, errno.ENOENT):
            raise ProcessLookupError(pid)
        raise OSError(error, os.strerror(error))
    return int(usage.proc_start_abstime)


def process_table() -> dict[int, tuple[int, int]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,pgid="], check=True, capture_output=True, text=True, timeout=5
    )
    table: dict[int, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        pid, parent, group = map(int, line.split())
        table[pid] = (parent, group)
    return table


def matching_identity(pid: int, start: int) -> bool:
    try:
        return process_start(pid) == start
    except ProcessLookupError:
        return False


def group_members(group: int) -> set[int]:
    return {pid for pid, (_, pgid) in process_table().items() if pgid == group}


def refresh_owned(group: int, identities: dict[int, int]) -> set[int]:
    """Refresh live descendants, rooted only in still-matching captured identities."""
    table = process_table()
    live = {pid for pid, start in identities.items() if matching_identity(pid, start)}
    group_anchored = any(table.get(pid, (None, None))[1] == group for pid in live)
    while True:
        before = len(live)
        for pid, (parent, pgid) in table.items():
            if pid in live or not (parent in live or (group_anchored and pgid == group)):
                continue
            try:
                start = process_start(pid)
            except ProcessLookupError:
                continue
            if pid in identities and identities[pid] != start:
                continue  # Never adopt a reused captured PID, or descendants rooted only in it.
            identities[pid] = start
            live.add(pid)
        if len(live) == before:
            return live


def accounted_cpu() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    done = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + done.ru_utime + done.ru_stime


@dataclass(frozen=True)
class InvocationBudget:
    started: float
    cpu_started: float

    @classmethod
    def start(cls) -> InvocationBudget:
        return cls(time.monotonic(), accounted_cpu())

    def usage(self, live_cpu: float = 0.0) -> tuple[float, float]:
        return time.monotonic() - self.started, accounted_cpu() - self.cpu_started + live_cpu

    def check(self, live_cpu: float = 0.0) -> None:
        wall, cpu = self.usage(live_cpu)
        if wall > MAX_WALL_SECONDS or cpu > MAX_CPU_SECONDS:
            raise RuntimeError("Whole-invocation measurement budget exceeded")


def get_affinity() -> set[int] | None:
    return os.sched_getaffinity(0) if sys.platform == "linux" else None


def set_affinity(cpus: set[int]) -> None:
    if sys.platform != "linux":
        raise RuntimeError("Linux affinity requested on another platform")
    os.sched_setaffinity(0, cpus)


def object_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError("Expected JSON object")
    return cast(dict[str, object], value)


def array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Expected JSON array")
    return cast(list[object], value)


def text(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Expected nonempty string")
    return value


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pinned(value: object) -> Path:
    spec = object_value(value)
    supplied = Path(text(spec["path"]))
    if not supplied.is_absolute():
        raise ValueError("Pinned paths must be absolute")
    path = supplied.resolve(strict=True)
    if not path.is_file() or sha256(path) != text(spec["sha256"]):
        raise ValueError(f"Pinned file changed: {path.name}")
    return path


def input_lane(config: dict[str, object]) -> str:
    if config.get("format") == "drone-original-file-measurement-v1":
        if config.get("input_lane") not in (None, "canonical"):
            raise ValueError("v1 remains canonical-only")
        return "canonical"
    if (
        config.get("format") == "drone-original-file-measurement-v2"
        and config.get("input_lane") == "decoder_fidelity"
    ):
        return "decoder_fidelity"
    raise ValueError("Explicit v2 decoder-fidelity lane required")


def validate_config(config: dict[str, object]) -> list[dict[str, object]]:
    input_lane(config)
    if type(config.get("warm_repetitions")) is not int or config["warm_repetitions"] != 3:
        raise ValueError("Exactly three warm repetitions are frozen")
    if config.get("candidate") != "fused-within-product-mac-magnitude-platform-log1p":
        raise ValueError("Explicit candidate required")
    if object_value(config["build_context"]).get("profile") != "release":
        raise ValueError("Measurement requires a pinned release build")
    for field in (
        "host_description",
        "rustc_version",
        "target_triple",
        "rustflags",
        "fma_lowering",
    ):
        text(object_value(config["build_context"])[field])
    for field in ("worker", "ffmpeg", "ort", "soxr", "bundle_contract"):
        pinned(config[field])
    panel = object_value(json.loads(PANEL_PATH.read_text()))
    rows = [object_value(row) for row in array(config["cases"])]
    expected = [object_value(row) for row in array(panel["cases"])]
    if len(rows) != len(expected):
        raise ValueError("The five-case panel is fixed")
    for row, reference in zip(rows, expected, strict=True):
        if row.get("id") != reference["id"]:
            raise ValueError("Panel membership/order changed")
        for name, key in (("audio", "audio_sha256"), ("manifest", "mac_manifest_sha256")):
            if object_value(row[name])["sha256"] != reference[key]:
                raise ValueError(f"Historical {name} identity changed")
            pinned(row[name])
    core = config.get("linux_cpu")
    if platform.system() == "Linux":
        if type(core) is not int or core not in (get_affinity() or set()):
            raise ValueError("Choose one currently permitted Linux CPU")
    elif platform.system() != "Darwin" or core is not None:
        raise ValueError("Only Linux single-CPU or macOS no-affinity measurement is supported")
    return rows


def elapsed_seconds(value: str) -> float:
    """Parse BSD/Linux ps cumulative CPU times, including day and decimal formats."""
    days = 0
    if "-" in value:
        day, value = value.split("-", 1)
        days = int(day)
    parts = [float(part) for part in value.split(":")]
    if len(parts) not in (2, 3) or any(not math.isfinite(p) or p < 0 for p in parts):
        raise ValueError("Invalid process CPU time")
    seconds = sum(part * 60**index for index, part in enumerate(reversed(parts)))
    return days * 86400 + seconds


@dataclass(frozen=True)
class Sample:
    pids: tuple[int, ...]
    rss_bytes: int
    cpu_seconds: float


def parse_group(output: str, group: int) -> Sample:
    pids: list[int] = []
    rss = 0
    cpu = 0.0
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 4:
            raise ValueError("Unexpected ps columns")
        pid, pgid, kib = map(int, fields[:3])
        if pgid == group:
            if kib < 0:
                raise ValueError("Negative RSS")
            pids.append(pid)
            rss += kib * 1024
            cpu += elapsed_seconds(fields[3])
    return Sample(tuple(pids), rss, cpu)


def sample_group(group: int) -> Sample:
    # -S includes CPU of waited children; live children remain separate rows.
    proc = subprocess.run(
        ["ps", "-S", "-axo", "pid=,pgid=,rss=,time="],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
        env={**os.environ, "LC_ALL": "C"},
    )
    return parse_group(proc.stdout, group)


def owned_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # A completed decoder may retire its WAV between enumeration and stat.
    return total


def retire(child: subprocess.Popen[bytes], identities: dict[int, int]) -> None:
    """Refresh ownership before each signal and verify each process start identity."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        refresh_owned(child.pid, identities)
        for pid, start in sorted(identities.items(), key=lambda item: item[0] == child.pid):
            try:
                if matching_identity(pid, start):
                    os.kill(pid, sig)
            except ProcessLookupError:
                pass
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            continue
        if not refresh_owned(child.pid, identities):
            if group_members(child.pid):
                raise RuntimeError("Untracked group survivors remain; ownership not proven")
            return
    child.wait(timeout=5)
    if refresh_owned(child.pid, identities) or group_members(child.pid):
        raise RuntimeError("Owned process group survived retirement")


def numerical_summary(result: dict[str, object]) -> dict[str, bool]:
    runs = array(result["runs"])
    if len(runs) != 4:
        raise ValueError("Four complete numerical results required")
    comparisons = [
        object_value(object_value(run)[name])
        for run in runs
        for name in ("comparison_mac", "comparison_linux")
    ]
    if any(
        type(comparison.get(key)) is not bool
        for comparison in comparisons
        for key in ("allclose", "argmax_agrees")
    ):
        raise ValueError("Explicit numerical and key-agreement flags required")
    numerical = all(comparison["allclose"] is True for comparison in comparisons)
    keys = all(comparison["argmax_agrees"] is True for comparison in comparisons)
    return {
        "numerical_pass": numerical,
        "unchanged_key": keys,
        "benchmark_qualified": numerical and keys,
    }


def run_case(
    config: dict[str, object], snapshot: Path, root: Path, index: int, budget: InvocationBudget
) -> dict[str, object]:
    budget.check()
    worker = pinned(config["worker"])
    output = root / f"case-{index}.json"
    argv = [
        str(worker),
        "--config",
        str(snapshot),
        "--case-index",
        str(index),
        "--output",
        str(output),
        "--execute-research-candidate",
    ]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "LC_ALL": "C",
    }
    # Affinity is inherited by the child and decoder. Restore the supervisor immediately.
    original_affinity = get_affinity()
    child: subprocess.Popen[bytes] | None = None
    start = time.monotonic()
    peak_rss = 0
    maximum_cpu = 0.0
    samples = 0
    failure: str | None = None
    identities: dict[int, int] = {}
    numerical = {"numerical_pass": False, "unchanged_key": False, "benchmark_qualified": False}
    canonical_pcm_matches = False
    with (root / f"case-{index}.log").open("xb") as log:
        try:
            if original_affinity is not None:
                set_affinity({cast(int, config["linux_cpu"])})
            child = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                env=env,
                start_new_session=True,
            )
        finally:
            if original_affinity is not None:
                set_affinity(original_affinity)
        try:
            identities[child.pid] = process_start(child.pid)
            while True:
                owned = refresh_owned(child.pid, identities)
                for pid in owned:
                    try:
                        if os.getpgid(pid) != child.pid:
                            raise RuntimeError("Owned process escaped its measurement group")
                    except ProcessLookupError:
                        pass
                sample = sample_group(child.pid)
                budget.check(sample.cpu_seconds)
                samples += 1
                peak_rss = max(peak_rss, sample.rss_bytes)
                maximum_cpu = max(maximum_cpu, sample.cpu_seconds)
                if (
                    time.monotonic() - start > MAX_WALL_SECONDS
                    or maximum_cpu > MAX_CPU_SECONDS
                    or peak_rss > MAX_RSS_BYTES
                    or owned_bytes(root) > MAX_OUTPUT_BYTES
                ):
                    raise RuntimeError("Finite measurement budget exceeded")
                if child.poll() is not None:
                    if sample_group(child.pid).pids:
                        raise RuntimeError("Worker exited with live descendants")
                    break
                time.sleep(SAMPLE_SECONDS)
            if child.returncode != 0:
                raise RuntimeError(f"Worker failed with status {child.returncode}")
            result = object_value(json.loads(output.read_text()))
            if result.get("complete") is not True or len(array(result["runs"])) != 4:
                raise ValueError("Incomplete worker result")
            numerical = numerical_summary(result)
            canonical_pcm_matches = result.get("canonical_pcm_matches") is True
            final_cpu = float(cast(float, result["process_cpu_seconds_including_waited_children"]))
            if not math.isfinite(final_cpu) or final_cpu > MAX_CPU_SECONDS:
                raise RuntimeError("Final aggregate CPU budget exceeded")
            maximum_cpu = max(maximum_cpu, final_cpu)
            if owned_bytes(root) > MAX_OUTPUT_BYTES:
                raise RuntimeError("Final storage budget exceeded")
            budget.check()
        except BaseException as error:
            failure = f"{type(error).__name__}: {error}"
            raise
        finally:
            previous = {
                sig: signal.signal(sig, signal.SIG_IGN)
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM)
            }
            remaining: list[int] | None = None
            try:
                retire(child, identities)
                remaining = sorted(refresh_owned(child.pid, identities))
            except BaseException as error:
                failure = f"{failure or ''}; retirement: {type(error).__name__}: {error}"
                try:
                    remaining = sorted(
                        refresh_owned(child.pid, identities) | group_members(child.pid)
                    )
                except (OSError, subprocess.SubprocessError):
                    remaining = None  # Unknown is not an empty cleanup certificate.
                raise
            finally:
                receipt = {
                    "index": index,
                    "complete": failure is None,
                    "timing_complete": failure is None,
                    "canonical_pcm_matches": canonical_pcm_matches,
                    "historical_reference_qualified": numerical["benchmark_qualified"]
                    and failure is None,
                    "failure": failure,
                    "process_wall_seconds": time.monotonic() - start,
                    "sampled_tree_peak_rss_bytes": peak_rss,
                    "maximum_cpu_seconds": maximum_cpu,
                    "samples": samples,
                    "sample_interval_seconds": SAMPLE_SECONDS,
                    "remaining_pids": remaining,
                    "process_start_identities": identities,
                    **numerical,
                }
                if failure is not None:
                    receipt["benchmark_qualified"] = False
                (root / f"case-{index}-resources.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
    return receipt


def main() -> None:
    budget = InvocationBudget.start()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute", action="store_true", help="Run the frozen five-case measurement"
    )
    args = parser.parse_args()
    if not args.execute:
        config = object_value(json.loads(args.config.read_text()))
        rows = validate_config(config)
        print(json.dumps({"validated": True, "cases": len(rows), "executed": False}))
        return
    args.output.mkdir(parents=False, exist_ok=False)
    snapshot = args.output / "config.json"

    def interrupted(signum: int, _frame: object) -> None:
        raise InterruptedError(f"Received signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM):
        signal.signal(sig, interrupted)
    signal.alarm(max(1, math.ceil(MAX_WALL_SECONDS - budget.usage()[0])))
    receipts: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    lane = "unadmitted"
    failure: str | None = None
    try:
        budget.check()
        config = object_value(json.loads(args.config.read_text()))
        snapshot.write_text(json.dumps(config, indent=2) + "\n")
        rows = validate_config(config)
        lane = input_lane(config)
        budget.check()
        for index in range(len(rows)):
            receipts.append(run_case(config, snapshot, args.output, index, budget))
            budget.check()
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        signal.alarm(0)
        wall, cpu = budget.usage()
        terminal_violation = failure is None and (wall > MAX_WALL_SECONDS or cpu > MAX_CPU_SECONDS)
        if terminal_violation:
            failure = "Terminal whole-invocation measurement budget exceeded"
        complete = failure is None and len(receipts) == len(rows) == 5
        numerical = complete and all(row["numerical_pass"] is True for row in receipts)
        keys = complete and all(row["unchanged_key"] is True for row in receipts)
        result = {
            "complete": complete,
            "timing_complete": complete,
            "input_lane": lane,
            "canonical_pcm_matches": complete
            and all(row["canonical_pcm_matches"] is True for row in receipts),
            "historical_reference_qualified": complete and numerical and keys,
            "failure": failure,
            "numerical_pass": numerical,
            "unchanged_key": keys,
            "benchmark_qualified": complete and numerical and keys,
            "invocation_wall_seconds": wall,
            "invocation_aggregate_cpu_seconds": cpu,
            "invocation_limits": {
                "wall_seconds": MAX_WALL_SECONDS,
                "aggregate_cpu_seconds": MAX_CPU_SECONDS,
            },
            "config_sha256": sha256(snapshot) if snapshot.is_file() else None,
            "panel_sha256": sha256(PANEL_PATH),
            "platform": platform.platform(),
            "cases": receipts,
            "accuracy_evaluated": False,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if terminal_violation:
            raise RuntimeError(failure)


if __name__ == "__main__":
    main()
