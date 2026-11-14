"""Recheck frozen Mac376 vectors/receipts only; no runtime, audio, labels or models."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from summarize import load_bundle

FAMILIES = (
    "full_vs_mac",
    "chunk_vs_mac",
    "mac_vs_linux",
    "full_vs_linux",
    "chunk_vs_linux",
    "core_mac_vs_linux",
)
SOURCE = "e3055e34852ed1e2fab9e5ad6b697ea008d970bd65c9091b3eb0940f1247bd1f"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def unpack(root: Path) -> dict[str, str]:
    index = json.loads((root / "evidence/mac376-index.json").read_bytes())
    packed = (root / "evidence" / index["bundle"]).read_bytes()
    if sha(packed) != index["compressed_sha256"]:
        raise ValueError("compressed evidence hash")
    raw = gzip.decompress(packed)
    if sha(raw) != index["decoded_sha256"] or len(raw) != index["decoded_bytes"]:
        raise ValueError("decoded evidence hash")
    files = json.loads(raw)["files"]
    if set(files) != set(index["files"]):
        raise ValueError("evidence membership")
    for name, text in files.items():
        item = index["files"][name]
        if sha(text.encode()) != item["sha256"] or len(text.encode()) != item["bytes"]:
            raise ValueError("original file hash: " + name)
    return files


def comparison(actual: list[float], reference: list[float]) -> dict[str, Any]:
    if (
        len(actual) != 24
        or len(reference) != 24
        or not all(math.isfinite(value) for value in actual + reference)
    ):
        raise ValueError("24 finite logits required")
    errors = [
        abs(actual_value - reference_value)
        for actual_value, reference_value in zip(actual, reference, strict=True)
    ]
    same = max(range(24), key=actual.__getitem__) == max(range(24), key=reference.__getitem__)
    return {
        "passed": same
        and all(
            error <= 1e-5 + 1e-5 * abs(reference_value)
            for error, reference_value in zip(errors, reference, strict=True)
        ),
        "finite": True,
        "same_key": same,
        "max_absolute": max(errors),
        "logits": actual,
        "reference": reference,
    }


def certificate(row: dict[str, Any], family: str) -> dict[str, Any]:
    gate = row["gates"][family]
    actual = gate["logits"]
    reference = gate["reference"]
    winner = max(range(24), key=reference.__getitem__)
    errors = [
        abs(actual_value - reference_value)
        for actual_value, reference_value in zip(actual, reference, strict=True)
    ]
    margin = reference[winner] - max(
        value for index, value in enumerate(reference) if index != winner
    )
    slack = min(
        reference[winner] - value - errors[winner] - errors[index]
        for index, value in enumerate(reference)
        if index != winner
    )
    return {
        "id": row["id"],
        "reference_winner": winner,
        "reference_margin": margin,
        "observed_error_bound": max(errors),
        "minimum_pairwise_certificate_slack": slack,
        "certified_for_observed_errors": slack > 0,
    }


def verify(files: dict[str, str]) -> dict[str, Any]:
    def load(name: str) -> dict[str, Any]:
        return json.loads(files[name])

    raw = files["v4/combined-diagnostic.json"].encode()
    if sha(raw) != SOURCE:
        raise ValueError("original disjoint source identity")
    combined = load("v4/combined-diagnostic.json")
    prior = load("v3/qualification/incomplete.json")
    new = load("v4/qualification/report.json")
    old = load("v2/qualification/incomplete.json")
    if (
        old["complete"]
        or "KeyboardInterrupt" not in files["v2/qualification-supervisor.log"]
        or len(old["cases"]) != 11
    ):
        raise ValueError("v2 controlled stop changed")
    if (
        prior["complete"]
        or prior["error_type"] != "RuntimeError"
        or "CPU=1800." not in prior["error"]
    ):
        raise ValueError("v3 resource failure changed")
    if (
        len(prior["cases"]) != 305
        or len(new["cases"]) != 71
        or not new["complete"]
        or new["passed"]
    ):
        raise ValueError("phase completion contract")
    rows = combined["cases"]
    if (
        rows != prior["cases"] + new["cases"]
        or [row["index"] for row in rows] != list(range(376))
        or len({row["id"] for row in rows}) != 376
    ):
        raise ValueError("exact disjoint ordered376 union")
    if (
        combined["single_run_passed"]
        or combined["timing_authorized"]
        or combined["promotion_authorized"]
    ):
        raise ValueError("failed gates cannot authorize timing/promotion")
    certificates = load("root/observed-margin-certificates.json")
    independent = load("independent/observed-margin-verification.json")
    if (
        certificates["source_sha256"] != SOURCE
        or independent["source_sha256"] != SOURCE
        or independent["root_certificate_sha256"]
        != sha(files["root/observed-margin-certificates.json"].encode())
        or not independent["all_claimed_case_fields_exactly_recomputed"]
    ):
        raise ValueError("certificate/independent source binding")
    canonical_inputs = load_bundle(Path(__file__).resolve().parent, "qualification_inputs")
    for row in rows:
        expected = json.loads(
            canonical_inputs[f"qualification/cases/{row['index']:04d}-input.json"]
        )["cases"][0]
        if (
            row["id"] != expected["id"]
            or row["frames"] != expected["frames"]
            or row["pcm_sha256"] != expected["pcm"]["sha256"]
            or row["canonical_core_sha256"] != expected["core"]["sha256"]
        ):
            raise ValueError("canonical input/core identity")
        if any(
            row["gates"][name]["reference"] != expected["reference_logits"]
            for name in FAMILIES
            if name.endswith("vs_linux")
        ):
            raise ValueError("canonical Linux reference logits")
        if (
            row["plan"]["kernel_sha256_differences"]
            != [{"basis": False, "indices": False, "indptr": False}] * 9
            or row["plan"]["normalization_equal"] is not False
        ):
            raise ValueError("original nine-stage CSR/normalization observations")
    summary: dict[str, Any] = {
        "tracks": 376,
        "disjoint_phases": [305, 71],
        "strict_passed": False,
        "timing_authorized": False,
        "promotion_authorized": False,
        "families": {},
        "cases": rows,
    }
    for name in FAMILIES:
        for row in rows:
            if (
                set(row["gates"]) != set(FAMILIES)
                or comparison(row["gates"][name]["logits"], row["gates"][name]["reference"])
                != row["gates"][name]
            ):
                raise ValueError("original gate recomputation")
            case_index = row["index"]
            phase = "v3" if case_index < 305 else "v4"
            native_name = phase + f"/qualification/cases/{case_index:04d}.native.json"
            if sha(files[native_name].encode()) != row["native_report_sha256"]:
                raise ValueError("actual native report hash")
            native = load(native_name)["cases"][0]
            predictions = {
                prediction["backend"]: prediction["logits"]
                for prediction in native["iterations"][0]["predictions"]
            }
            if (
                native["id"] != row["id"]
                or native["pcm_sha256"] != row["pcm_sha256"]
                or predictions["full"] != row["gates"]["full_vs_mac"]["logits"]
                or predictions["chunked"] != row["gates"]["chunk_vs_mac"]["logits"]
            ):
                raise ValueError("actual native vectors/input")
        records = [certificate(row, name) for row in rows]
        recorded = certificates["families"][name]
        if (
            records != recorded["cases"]
            or recorded["certified_tracks"] != 376
            or recorded["minimum_slack"]
            != min(record["minimum_pairwise_certificate_slack"] for record in records)
            or recorded["minimum_reference_margin"]
            != min(record["reference_margin"] for record in records)
            or independent["families"][name]["verified_cases"] != 376
            or independent["families"][name]["minimum_slack_id"]
            != min(records, key=lambda record: record["minimum_pairwise_certificate_slack"])["id"]
            or min(record["minimum_pairwise_certificate_slack"] for record in records)
            != independent["families"][name]["minimum_slack"]
        ):
            raise ValueError("observed certificate arithmetic")
        gates = [row["gates"][name] for row in rows]
        summary["families"][name] = {
            "passed": sum(gate["passed"] for gate in gates),
            "key_disagreements": sum(not gate["same_key"] for gate in gates),
            "maximum_absolute": max(gate["max_absolute"] for gate in gates),
            "minimum_observed_slack": min(
                record["minimum_pairwise_certificate_slack"] for record in records
            ),
        }
    expected = [375, 375, 348, 350, 350, 376]
    if [summary["families"][family]["passed"] for family in FAMILIES] != expected:
        raise ValueError("published gate counts")
    witness = load("derived/csr-normalization-witnesses.json")
    if [witness_row["id"] for witness_row in witness["rows"]] != [row["id"] for row in rows] or any(
        not witness_row["all_nine_csr_arrays_equal"]
        or len(witness_row["normalization_differences"]) != 3
        or any(
            difference["ulp_distance"] != 1
            for difference in witness_row["normalization_differences"]
        )
        for witness_row in witness["rows"]
    ):
        raise ValueError("CSR/normalization witnesses")
    summary["nine_reused"] = len(new["nine"]) == 9 and new["nine"] == prior["nine"]
    summary["resources"] = load("v4/qualification/terminal-resources.json")
    return summary


def summarize(root: Path) -> dict[str, Any]:
    return verify(unpack(root))


if __name__ == "__main__":
    result = summarize(Path(__file__).resolve().parent)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in ("cases", "resources")},
            indent=2,
        )
    )
