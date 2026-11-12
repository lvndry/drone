"""Stdlib-only supervisor for a frozen, label-blind 376-track CQT expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
CPU_LIMIT = 1200.0
WALL_LIMIT = 1500.0
BYTE_LIMIT = 2 * 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def child_cpu() -> float:
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def process_children(pid: int) -> dict[int, int]:
    """Return children of every thread, witnessed by parent/start identities."""
    root = Path(f"/proc/{pid}")
    try:
        before = root.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
        tasks = list(root.joinpath("task").iterdir())
    except (FileNotFoundError, ProcessLookupError):
        return {}
    candidates: set[int] = set()
    for task in tasks:
        try:
            candidates.update(int(value) for value in (task / "children").read_text().split())
        except (FileNotFoundError, ProcessLookupError):
            continue  # A decoder thread may have completed during this sample.
    children = {}
    for child in candidates:
        try:
            fields = Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) == pid:
                children[child] = int(fields[19])
        except (FileNotFoundError, ProcessLookupError):
            continue
    try:
        if root.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19] != before:
            return {}  # Never attach a reused parent's descendants to this tree.
    except (FileNotFoundError, ProcessLookupError):
        return {}
    return children


def process_tree(pid: int) -> tuple[int, float]:
    """Summed process RSS and live/reaped CPU, including threaded subprocesses."""
    resident = 0
    cpu = 0.0
    pending: list[tuple[int, int | None]] = [(pid, None)]
    seen: set[tuple[int, int]] = set()
    while pending:
        current, expected_start = pending.pop()
        try:
            stat = Path(f"/proc/{current}/stat").read_text().rsplit(")", 1)[1].split()
            identity = (current, int(stat[19]))
            if identity in seen or (expected_start is not None and identity[1] != expected_start):
                continue
            children = process_children(current)
            rss = int(Path(f"/proc/{current}/statm").read_text().split()[1]) * os.sysconf(
                "SC_PAGE_SIZE"
            )
            after = Path(f"/proc/{current}/stat").read_text().rsplit(")", 1)[1].split()
            if int(after[19]) != identity[1]:
                continue
            seen.add(identity)
            cpu += sum(float(stat[index]) for index in (11, 12, 13, 14)) / os.sysconf("SC_CLK_TCK")
            resident += rss
            pending.extend(children.items())
        except (FileNotFoundError, ProcessLookupError):
            continue
    return resident, cpu


def disk_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink()
    )


def check_limits(
    cpu: float, wall: float, resident: int, storage: int, now: float, deadline: float
) -> None:
    if (
        cpu > CPU_LIMIT
        or wall > WALL_LIMIT
        or resident > BYTE_LIMIT
        or storage > BYTE_LIMIT
        or now >= deadline
    ):
        raise RuntimeError(
            f"Fixed budget stop: CPU={cpu:.2f}s wall={wall:.2f}s RSS={resident} storage={storage} UTC={now}"
        )


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.started = time.monotonic()
        self.deadline = datetime.fromisoformat(args.deadline).timestamp()
        self.peak_resident = 0
        self.resources: list[dict[str, Any]] = []
        self.environment = dict(os.environ) | {name: "1" for name in THREADS}
        self.environment.update(
            PYTHONPYCACHEPREFIX=str(args.stage_root / "pycache"),
            NUMBA_CACHE_DIR=str(args.stage_root / "numba-cache"),
            TMPDIR=str(args.stage_root / "tmp"),
            XDG_CACHE_HOME=str(args.stage_root / "cache"),
        )
        (args.stage_root / "tmp").mkdir()
        self.math_library_sha256 = (
            sha256(args.math_library) if args.math_library is not None else None
        )
        self.budget_storage = disk_bytes(args.budget_root)
        self.last_storage_check = 0.0

    def check(self, active: subprocess.Popen[bytes] | None = None, cpu_before: float = 0.0) -> None:
        resident, _ = process_tree(os.getpid())
        cpu = child_cpu()
        if active is not None and active.poll() is None:
            _, running_cpu = process_tree(active.pid)
            cpu = max(cpu, cpu_before + running_cpu)
        self.peak_resident = max(self.peak_resident, resident)
        if time.monotonic() - self.last_storage_check > 1.0:
            self.budget_storage = disk_bytes(self.args.budget_root)
            self.last_storage_check = time.monotonic()
        check_limits(
            cpu,
            time.monotonic() - self.started,
            resident,
            self.budget_storage,
            time.time(),
            self.deadline,
        )

    def run(self, command: list[str], label: str, allow_failure: bool = False) -> int:
        self.check()
        cpu_before = child_cpu()
        started = time.monotonic()
        log = self.args.stage_root / "logs" / f"{label}.log"
        with log.open("xb") as output:
            process = subprocess.Popen(
                command,
                cwd=self.args.source_root,
                env=self.environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            print(json.dumps(dict(event="start", label=label, pid=process.pid)), flush=True)
            try:
                while process.poll() is None:
                    self.check(process, cpu_before)
                    time.sleep(0.1)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            code = process.wait()
        entry = dict(
            label=label,
            exit_code=code,
            cpu_seconds=child_cpu() - cpu_before,
            wall_seconds=time.monotonic() - started,
            command=command,
            log_sha256=sha256(log),
        )
        self.resources.append(entry)
        self.check()
        print(json.dumps(dict(event="exit", **entry)), flush=True)
        if code and not allow_failure:
            raise RuntimeError(f"{label} failed with exit{code}; see {log}")
        return code

    def native_command(
        self, manifest: Path, output: Path, single_case: bool, arm: str = "candidate"
    ) -> list[str]:
        command = [
            str(self.args.native),
            "--candidate",
            "exported",
            "--soxr-wrapper",
            str(self.args.soxr_wrapper),
            "--manifest",
            str(manifest),
            "--full-model",
            str(self.args.full_model),
            "--ort-library",
            str(self.args.ort_library),
            "--output",
            str(output),
        ]
        if arm in ("fma", "fma-svml"):
            command.append("--numpy-magnitude")
        if self.args.math_library is not None and arm != "fma":
            command.extend(
                [
                    "--math-library",
                    str(self.args.math_library),
                    "--math-library-sha256",
                    str(self.math_library_sha256),
                ]
            )
        if single_case:
            command.extend(["--single-case", "--expect-manifest-sha256", sha256(manifest)])
        return command


def execute(args: argparse.Namespace) -> None:
    if sys.platform != "linux" or os.sched_getaffinity(0) != {2, 3}:
        raise RuntimeError("Fixed expansion requires Linux affinity2,3")
    study = json.loads(args.study.read_text())
    if study.get("complete") is not True or study.get("tracks") != 376 or len(study["rows"]) != 376:
        raise ValueError("Require complete fixed376 study")
    for name in ("logs", "cases"):
        (args.stage_root / name).mkdir()
    supervisor = Supervisor(args)
    arms = ["fma", "fma-svml"] if args.arithmetic_ablation else ["candidate"]
    if args.arithmetic_ablation and args.math_library is None:
        raise ValueError("Fixed ablation requires the pinned SVML library")
    identity = dict(
        study_manifest_sha256=sha256(args.study),
        supervisor_source_sha256=sha256(Path(__file__)),
        native_sha256=sha256(args.native),
        ort_library_sha256=sha256(args.ort_library),
        soxr_wrapper_sha256=sha256(args.soxr_wrapper),
        full_model_sha256=sha256(args.full_model),
        source_root=str(args.source_root),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        thread_environment={name: "1" for name in THREADS},
        caps=dict(
            child_cpu_seconds=CPU_LIMIT,
            wall_seconds=WALL_LIMIT,
            resident_bytes=BYTE_LIMIT,
            total_experiment_storage_bytes=BYTE_LIMIT,
            deadline=args.deadline,
        ),
        math_library_sha256=supervisor.math_library_sha256,
        kernel_hardware=dict(
            uname=list(platform.uname()),
            cpu_identity=sorted(
                {
                    line.strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.split(":", 1)[0].strip()
                    in {
                        "vendor_id",
                        "cpu family",
                        "model",
                        "model name",
                        "stepping",
                        "microcode",
                        "flags",
                    }
                }
            ),
        ),
        accuracy_scored=False,
    )
    progress: dict[str, Any] = dict(
        format="k10-cqt-expansion-report-v2",
        arms=arms,
        complete=False,
        passed=False,
        identity=identity,
        cases=[],
        first_failures={arm: None for arm in arms},
    )

    def checkpoint() -> None:
        progress.update(
            resources=supervisor.resources,
            aggregate_child_cpu_seconds=child_cpu(),
            wall_seconds=time.monotonic() - supervisor.started,
            peak_owned_tree_rss_bytes=supervisor.peak_resident,
            current_experiment_bytes=disk_bytes(args.budget_root),
        )
        write_json(args.stage_root / "progress.json", progress)

    try:
        nine_passed = {}
        progress["nine_rechecks"] = {}
        for arm in arms:
            nine = args.stage_root / f"nine-recheck-{arm}.json"
            code = supervisor.run(
                supervisor.native_command(args.goldens, nine, False, arm),
                f"nine-recheck-{arm}",
                True,
            )
            recheck = json.loads(nine.read_text())
            if (
                len(recheck["cases"]) != 9
                or code not in (0, 1)
                or (code == 0) != (recheck.get("passed") is True)
            ):
                raise RuntimeError("Invalid fixed-nine recheck result")
            nine_passed[arm] = recheck.get("passed") is True
            progress["nine_rechecks"][arm] = dict(passed=nine_passed[arm], sha256=sha256(nine))
        if not nine_passed["fma-svml" if args.arithmetic_ablation else "candidate"]:
            raise RuntimeError("Primary candidate failed fixed-nine recheck")
        for index, row in enumerate(study["rows"]):
            supervisor.check()
            spool = args.stage_root / f"spool-{index:04d}"
            supervisor.run(
                [
                    str(args.python),
                    "-m",
                    "tools.export_cqt_expansion",
                    "produce",
                    "--study",
                    str(args.study),
                    "--row-index",
                    str(index),
                    "--spool",
                    str(spool),
                    "--feature-source",
                    str(args.feature_source),
                    "--full-model",
                    str(args.full_model),
                ],
                f"{index:04d}-reference",
            )
            manifest = spool / "manifest.json"
            reference = json.loads(manifest.read_text())
            if (
                reference["cases"][0]["id"] != row["id"]
                or reference["study_manifest_sha256"] != identity["study_manifest_sha256"]
            ):
                raise ValueError("Reference study identity changed")
            preserved = args.stage_root / "cases" / f"{index:04d}-reference.json"
            write_json(preserved, reference)
            arm_results = {}
            for arm in arms:
                result = args.stage_root / "cases" / f"{index:04d}-{arm}-native.json"
                code = supervisor.run(
                    supervisor.native_command(manifest, result, True, arm),
                    f"{index:04d}-{arm}-native",
                    True,
                )
                actual = json.loads(result.read_text())
                passed = actual.get("passed") is True
                if code not in (0, 1) or (code == 0) != passed:
                    raise RuntimeError("Native exit/status mismatch")
                arm_results[arm] = dict(
                    passed=passed,
                    native_report_sha256=sha256(result),
                    comparison=actual["cases"][0]["comparison"],
                    features=actual["cases"][0]["features"],
                )
            entry = dict(
                index=index,
                id=row["id"],
                domain=row["domain"],
                audio_sha256=row["sha256"],
                cache_sha256=row["cache_sha256"],
                pcm_sha256=reference["cases"][0]["pcm"]["sha256"],
                frames=row["frames"],
                reference_sha256=sha256(preserved),
                passed=all(result["passed"] for result in arm_results.values()),
                arms=arm_results,
            )
            progress["cases"].append(entry)
            new_failures = [
                arm
                for arm in arms
                if not arm_results[arm]["passed"] and progress["first_failures"][arm] is None
            ]
            if new_failures:
                retained = args.stage_root / f"first-failure-{index:04d}"
                spool.rename(retained)
                for arm in new_failures:
                    progress["first_failures"][arm] = dict(
                        index=index, id=row["id"], spool=str(retained)
                    )
            else:
                shutil.rmtree(spool)
            checkpoint()
            print(json.dumps(dict(event="case", **entry)), flush=True)
        arm_passed = {
            arm: nine_passed[arm] and all(case["arms"][arm]["passed"] for case in progress["cases"])
            for arm in arms
        }
        progress.update(complete=True, arm_passed=arm_passed, passed=all(arm_passed.values()))
        checkpoint()
        write_json(args.stage_root / "report.json", progress)
    except BaseException as error:
        progress["error"] = str(error)
        checkpoint()
        write_json(args.stage_root / "failed-stage.json", progress)
        raise
    if not progress["passed"]:
        raise RuntimeError("Fixed376 parity gate failed; all evidence retained")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "stage-root",
        "budget-root",
        "study",
        "source-root",
        "python",
        "native",
        "ort-library",
        "soxr-wrapper",
        "full-model",
        "feature-source",
        "goldens",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--math-library", type=Path)
    parser.add_argument("--arithmetic-ablation", action="store_true")
    parser.add_argument("--deadline", default="2026-11-12T06:00:00+00:00")
    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
