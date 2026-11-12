"""Verify immutable per-case CQT evidence and emit compact, label-blind plot data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def compare_logits(actual: list[float], expected: list[float]) -> dict[str, Any]:
    if len(actual) != 24 or len(expected) != 24:
        raise ValueError("Exactly24 fixed-class logits required")
    a = [float32(value) for value in actual]
    b = [float32(value) for value in expected]
    if not all(math.isfinite(value) for value in a + b):
        raise ValueError("Nonfinite recorded logits")
    differences = [abs(x - y) for x, y in zip(a, b, strict=True)]
    ratio = max(
        difference / (1e-5 + 1e-5 * abs(reference))
        for difference, reference in zip(differences, b, strict=True)
    )
    key_a = max(range(24), key=lambda index: a[index])
    key_b = max(range(24), key=lambda index: b[index])
    return dict(
        allclose=ratio <= 1,
        argmax_agrees=key_a == key_b,
        max_abs_error=max(differences),
        max_normalized_error=ratio,
        argmax=key_a,
        reference_argmax=key_b,
    )


def summarize(root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Immutable summary exists")
    report = json.loads((root / "report.json").read_text())
    study = json.loads((root / "study-manifest.json").read_text())
    if (
        report.get("complete") is not True
        or len(report["cases"]) != 376
        or study.get("tracks") != 376
    ):
        raise ValueError("Only complete fixed376 evidence can be summarized")
    arms = report.get("arms", ["original"])
    execution: dict[str, Any] = {}
    rows = []
    for index, (row, metadata) in enumerate(zip(report["cases"], study["rows"], strict=True)):
        if (
            row["index"] != index
            or row["id"] != metadata["id"]
            or row["audio_sha256"] != metadata["sha256"]
        ):
            raise ValueError("Fixed membership/order/hash changed")
        reference_path = root / "cases" / f"{index:04d}-reference.json"
        reference = json.loads(reference_path.read_text())
        if sha256(reference_path) != row["reference_sha256"]:
            raise ValueError("Changed reference evidence")
        reference_case = reference["cases"][0]
        if (
            reference_case["id"] != row["id"]
            or reference_case["pcm"]["sha256"] != row["pcm_sha256"]
        ):
            raise ValueError("Reference PCM identity mismatch")
        entry = {
            key: row[key]
            for key in ("id", "domain", "audio_sha256", "cache_sha256", "pcm_sha256", "frames")
        }
        entry.update(reference_logits=reference_case["reference_logits"], arms={})
        for arm in arms:
            record = row["arms"][arm] if "arms" in row else row
            suffix = "native" if arm == "original" else f"{arm}-native"
            path = root / "cases" / f"{index:04d}-{suffix}.json"
            native = json.loads(path.read_text())
            if (
                sha256(path) != record["native_report_sha256"]
                or native["manifest_sha256"] != row["reference_sha256"]
            ):
                raise ValueError("Changed native/paired reference evidence")
            actual = native["cases"][0]
            comparison = compare_logits(actual["logits"], reference_case["reference_logits"])
            for key in ("allclose", "argmax_agrees", "max_abs_error"):
                if comparison[key] != actual["comparison"][key]:
                    raise ValueError(
                        f"Independent gate verification differs: {row['id']} {arm} {key}"
                    )
            identity = {
                key: native.get(key)
                for key in (
                    "executable_sha256",
                    "source_sha256",
                    "plan_source_sha256",
                    "full_model_sha256",
                    "ort_library_sha256",
                    "math_library_sha256",
                    "soxr_wrapper_sha256",
                    "numpy_magnitude",
                )
            }
            if identity["full_model_sha256"] != study["full_model_sha256"]:
                raise ValueError("Changed authoritative model")
            if arm in execution and execution[arm] != identity:
                raise ValueError("Mixed native execution identities")
            execution[arm] = identity
            entry["arms"][arm] = dict(
                comparison=comparison,
                features=actual["features"],
                logits=actual["logits"],
                native_report_sha256=sha256(path),
            )
        rows.append(entry)
    aggregate = {}
    for arm in arms:
        values = [row["arms"][arm] for row in rows]
        aggregate[arm] = dict(
            tracks=376,
            strict_failures=sum(not entry["comparison"]["allclose"] for entry in values),
            argmax_disagreements=sum(not entry["comparison"]["argmax_agrees"] for entry in values),
            max_logit_abs=max(entry["comparison"]["max_abs_error"] for entry in values),
            max_normalized_logit_error=max(
                entry["comparison"]["max_normalized_error"] for entry in values
            ),
            tracks_bit_identical_features=sum(
                entry["features"]["fp16_mismatches"] == 0 for entry in values
            ),
            total_fp16_mismatches=sum(entry["features"]["fp16_mismatches"] for entry in values),
            total_feature_values=sum(entry["features"]["values"] for entry in values),
        )
    result = dict(
        format="k10-cqt-verified-expansion-summary-v1",
        complete=True,
        tracks=376,
        accuracy_scored=False,
        raw_report_sha256=sha256(root / "report.json"),
        study_manifest_sha256=sha256(root / "study-manifest.json"),
        protocol_sha256=study["protocol_sha256"],
        summarizer_source_sha256=sha256(Path(__file__)),
        execution=execution,
        controller_identity=report["identity"],
        operational_resources={
            key: report[key]
            for key in (
                "aggregate_child_cpu_seconds",
                "wall_seconds",
                "peak_owned_tree_rss_bytes",
                "current_experiment_bytes",
            )
        },
        aggregate=aggregate,
        cases=rows,
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.stage_root, args.output)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
