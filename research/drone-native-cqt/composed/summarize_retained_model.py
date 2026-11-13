"""Verify retained native model outputs and summarize numerical agreement; no inference."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import struct
from pathlib import Path
from typing import Any


def checked(path: Path, expected: str) -> bytes:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Changed evidence: {path.name}")
    return raw


def comparison(
    actual: list[float], expected: list[float], atol: float, rtol: float
) -> dict[str, Any]:
    if (
        len(actual) != 24
        or len(expected) != 24
        or not all(math.isfinite(x) for x in actual + expected)
    ):
        raise ValueError("Expected finite 24-class vectors")
    errors = [abs(a - e) for a, e in zip(actual, expected, strict=True)]
    ratios = [error / (atol + rtol * abs(e)) for error, e in zip(errors, expected, strict=True)]
    failed = [i for i, ratio in enumerate(ratios) if ratio > 1]
    return {
        "allclose": not failed,
        "bitwise_equal": struct.pack("<24f", *actual) == struct.pack("<24f", *expected),
        "argmax_agrees": max(range(24), key=actual.__getitem__)
        == max(range(24), key=expected.__getitem__),
        "max_abs_error": max(errors),
        "max_normalized_error": max(ratios),
        "failed_elements": failed,
    }


def summarize(root: Path) -> dict[str, Any]:
    provenance = json.loads((root / "retained-model376-provenance.json").read_bytes())
    packed = checked(root / "retained-model376-results.json.gz", provenance["results_gzip_sha256"])
    with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
        raw = stream.read(16 * 1024**2 + 1)
    if (
        len(raw) > 16 * 1024**2
        or hashlib.sha256(raw).hexdigest() != provenance["decoded_exact_result_sha256"]
    ):
        raise ValueError("Changed/oversized decoded evidence")
    result = json.loads(raw)
    controller = json.loads(
        checked(root / "retained-model376-controller.json", provenance["controller_sha256"])
    )
    if (
        not controller["complete"]
        or controller["remaining_owned_pids"]
        or controller["cleanup_error"]
    ):
        raise ValueError("Resource supervision did not complete cleanly")
    membership = json.loads(
        checked(root / "FUSED_376_MEMBERSHIP.json", result["membership_sha256"])
    )
    if not result["complete"] or result["tracks"] != 376 or len(result["cases"]) != 376:
        raise ValueError("Incomplete fixed panel")
    repository = root.parents[2]
    for name, expected in provenance["executed_source_sha256"].items():
        checked(repository / name, expected)
    families: dict[str, list[dict[str, Any]]] = {
        key: [] for key in ["full_vs_mac", "chunked_vs_mac", "chunked_vs_full"]
    }
    for index, (row, member) in enumerate(zip(result["cases"], membership["cases"], strict=True)):
        if (
            row["index"] != index
            or any(row[key] != member[key] for key in ["id", "frames"])
            or row["core_sha256"] != member["archived_mac_core_sha256"]
        ):
            raise ValueError("Membership/core mismatch")
        predictions = {item["backend"]: item for item in row["predictions"]}
        if set(predictions) != {"full", "chunked"}:
            raise ValueError("Missing/extra backend")
        for backend in ["full", "chunked"]:
            item = predictions[backend]
            measured = comparison(
                item["logits"], row["reference_logits"], result["atol"], result["rtol"]
            )
            if (
                any(
                    measured[key] != item["comparison"][key]
                    for key in ["allclose", "argmax_agrees", "max_abs_error"]
                )
                or measured["failed_elements"] != item["failed_elements"]
            ):
                raise ValueError("Reported reference comparison differs")
            families[backend + "_vs_mac"].append(measured)
        measured = comparison(
            predictions["chunked"]["logits"],
            predictions["full"]["logits"],
            result["atol"],
            result["rtol"],
        )
        if any(
            measured[key] != row["chunked_vs_full"][key]
            for key in ["allclose", "argmax_agrees", "max_abs_error"]
        ):
            raise ValueError("Reported direct comparison differs")
        families["chunked_vs_full"].append(measured)
    summary = {
        name: {
            "strict_passes": sum(row["allclose"] for row in rows),
            "bitwise_equal": sum(row["bitwise_equal"] for row in rows),
            "key_agreements": sum(row["argmax_agrees"] for row in rows),
            "failed_elements": sum(len(row["failed_elements"]) for row in rows),
            "max_abs_error": max(row["max_abs_error"] for row in rows),
            "max_normalized_error": max(row["max_normalized_error"] for row in rows),
        }
        for name, rows in families.items()
    }
    passed = all(row["strict_passes"] == row["key_agreements"] == 376 for row in summary.values())
    if result["passed"] != passed:
        raise ValueError("Reported overall verdict differs")
    return {
        "tracks": 376,
        "passed": passed,
        "families": summary,
        "controller": controller,
        "native_peak_rss_bytes": result["peak_rss_bytes"],
        "not_accuracy_or_performance": True,
    }


if __name__ == "__main__":
    print(json.dumps(summarize(Path(__file__).resolve().parent), indent=2))
