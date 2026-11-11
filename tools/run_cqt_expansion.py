"""Stdlib-only supervisor for a frozen, label-blind 376-track CQT expansion."""

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


def process_tree(pid: int) -> tuple[int, float]:
    """Conservative summed resident bytes and live+reaped-descendant CPU time."""
    resident = 0
    cpu = 0.0
    pending = [pid]
    while pending:
        current = pending.pop()
        try:
            stat = Path(f"/proc/{current}/stat").read_text().rsplit(")", 1)[1].split()
            cpu += sum(float(stat[index]) for index in (11, 12, 13, 14)) / os.sysconf("SC_CLK_TCK")
            resident += int(Path(f"/proc/{current}/statm").read_text().split()[1]) * os.sysconf(
                "SC_PAGE_SIZE"
            )
            children = Path(f"/proc/{current}/task/{current}/children").read_text().split()
            pending.extend(int(child) for child in children)
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

    def native_command(self, manifest: Path, output: Path, single_case: bool) -> list[str]:
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
        accuracy_scored=False,
    )
    progress: dict[str, Any] = dict(
        format="k10-cqt-expansion-report-v1",
        complete=False,
        passed=False,
        identity=identity,
        cases=[],
        first_failure=None,
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
        nine = args.stage_root / "nine-recheck.json"
        supervisor.run(supervisor.native_command(args.goldens, nine, False), "nine-recheck")
        recheck = json.loads(nine.read_text())
        if recheck.get("passed") is not True or len(recheck["cases"]) != 9:
            raise RuntimeError("New binary failed fixed-nine recheck")
        progress["nine_recheck_sha256"] = sha256(nine)
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
            result = args.stage_root / "cases" / f"{index:04d}-native.json"
            code = supervisor.run(
                supervisor.native_command(manifest, result, True), f"{index:04d}-native", True
            )
            actual = json.loads(result.read_text())
            passed = actual.get("passed") is True
            if code not in (0, 1) or (code == 0) != passed:
                raise RuntimeError("Native exit/status mismatch")
            entry = dict(
                index=index,
                id=row["id"],
                domain=row["domain"],
                audio_sha256=row["sha256"],
                cache_sha256=row["cache_sha256"],
                pcm_sha256=reference["cases"][0]["pcm"]["sha256"],
                frames=row["frames"],
                passed=passed,
                reference_sha256=sha256(preserved),
                native_report_sha256=sha256(result),
                comparison=actual["cases"][0]["comparison"],
                features=actual["cases"][0]["features"],
            )
            progress["cases"].append(entry)
            if not passed and progress["first_failure"] is None:
                retained = args.stage_root / "first-failure-spool"
                spool.rename(retained)
                progress["first_failure"] = dict(index=index, id=row["id"], spool=str(retained))
            else:
                shutil.rmtree(spool)
            checkpoint()
            print(json.dumps(dict(event="case", **entry)), flush=True)
        progress.update(complete=True, passed=all(case["passed"] for case in progress["cases"]))
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
    parser.add_argument("--deadline", default="2026-11-12T00:45:00+00:00")
    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
