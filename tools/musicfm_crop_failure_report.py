"""Verify frozen failed corpus metadata; never open PCM, decode, align or admit a crop."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

CONFIG = "b575ed66aab29acad8a6f421055198267554e4d36bbcc7bfe1c27b2a77d1c885"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify(files: dict[str, str]) -> dict[str, Any]:
    if digest(files["corpus/sealed-config.json"].encode()) != CONFIG:
        raise ValueError("Executed configuration changed")
    config = json.loads(files["corpus/sealed-config.json"])
    terminal = json.loads(files["corpus/terminal.json"])
    failure = json.loads(files["corpus/prepared-v1/failure.json"])
    launch = json.loads(files["corpus/launch.json"])
    approval = json.loads(files["corpus/root-approval.json"])
    if (
        terminal["exit"] != 1
        or terminal["reason"] is not None
        or approval.get("authorized") is not True
        or approval["config_sha256"] != CONFIG
        or failure["resources"]["tracks"] != 2288
        or failure["error"] != "ValueError('Fixed waveform-bit/finite/length gate failed')"
    ):
        raise ValueError("Failed fixed outcome changed")
    if digest(files["corpus/run-approved.py"].encode()) != launch["wrapper_source_sha256"]:
        raise ValueError("Outer executed source changed")
    for pid, before in launch["protected_before"].items():
        after = terminal["protected_after"][pid]
        if before["start_ticks"] != after["start_ticks"] or after["state"] in ("Z", "X"):
            raise ValueError("Protected service identity changed")
    names = sorted(n for n in files if n.startswith("corpus/prepared-v1/records/"))
    if len(names) != 2289:
        raise ValueError("Observed prefix count changed")
    records = [json.loads(files[n]) for n in names]
    for number, (row, expected) in enumerate(zip(records, config["rows"], strict=False), 1):
        if any(row[k] != expected[k] for k in ("id", "role", "target", "sha256")):
            raise ValueError("Fixed prefix identity/order changed")
        if row["crop"]["passed"] is not (number != 2289) or not row["crop"]["finite"]:
            raise ValueError("Observed numerical gate changed")
        if number < 2289 and row["crop"]["bitwise_mismatch_samples"] != 0:
            raise ValueError("Passing prefix is not bitwise equal")
        for kind in ("full_execution", "crop_execution"):
            execution = row[kind]
            raw = files["corpus/prepared-v1/" + execution["stderr"]].encode()
            if (
                execution["exit"] != 0
                or digest(raw) != execution["stderr_sha256"]
                or len(raw) != execution["stderr_bytes"]
                or "-xerror" in execution["command"]
            ):
                raise ValueError("Decoder profile/warning evidence changed")
    last = records[-1]
    if last["id"] != "fmak:046091" or last["crop"]["bitwise_mismatch_samples"] != 288000:
        raise ValueError("First failure changed")
    diagnostic = json.loads(files["diagnostic/diagnostic.json"])
    if (
        diagnostic["admission_or_repair"] is not False
        or diagnostic["row_sha256"] != digest(files[names[-1]].encode())
        or diagnostic["source_sha256"] != digest(files["diagnostic/diagnose.py"].encode())
        or diagnostic["input_file_sha256"]["reference.f32le"] != last["waveform_sha256"]
    ):
        raise ValueError("Read-only diagnostic binding changed")
    prefix = records[:-1]
    phases = {
        kind: {
            field: sum(r[kind][field] for r in prefix)
            for field in ("wall_seconds", "child_cpu_seconds")
        }
        for kind in ("full_execution", "crop_execution")
    }
    return {
        "status": "FAILED_NOT_QUALIFIED_NO_RESUME",
        "checked": 2289,
        "passed_prefix": 2288,
        "unprocessed": 5638 - 2289,
        "validation_processed": 0,
        "first_failure": {k: last[k] for k in ("id", "samples", "crop_start", "crop")},
        "prefix_audio_seconds": sum(r["samples"] / 24000 for r in prefix),
        "prefix_phase_costs": phases,
        "passed_files_with_nonempty_stderr": sum(
            bool(r["full_execution"]["stderr_bytes"] or r["crop_execution"]["stderr_bytes"])
            for r in prefix
        ),
        "terminal": terminal,
        "array_only_diagnostic": diagnostic,
        "diagnostic_terminal": json.loads(files["diagnostic/terminal.json"]),
    }


def plot(result: dict[str, Any], study: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["svg.hashsalt"] = "musicfm24k-failure-9facd95"
    figure, axis = plt.subplots(figsize=(7.7, 4.2), constrained_layout=True)
    values = [
        result["first_failure"]["crop"]["rms_error"],
        result["array_only_diagnostic"]["at_diagnostic_best_offset"]["rms_error"],
    ]
    bars = axis.bar(
        ["Requested start\nstrict gate failed", "Diagnostic −627 samples\nstill fails"],
        values,
        color=["#a3433d", "#bd7968"],
        width=0.5,
    )
    axis.set(
        yscale="log",
        ylabel="Waveform RMS difference",
        ylim=(0.001, 1),
        title="Corpus preparation stopped at fmak:046091",
    )
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value * 1.1, f"{value:.6f}", ha="center")
    axis.grid(axis="y", alpha=0.2)
    figure.text(
        0.02,
        -0.05,
        "Both positions differ at all 288,000 samples. Offset search is "
        "diagnostic only; no repaired crop was admitted.",
        fontsize=8,
    )
    figure.savefig(study / "corpus-failure.png", dpi=160, bbox_inches="tight")
    figure.savefig(study / "corpus-failure.svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    packet = args.study / "data/corpus-failure-raw.json.gz"
    index_path = args.study / "data/corpus-failure-index.json.gz"
    if args.evidence is not None:
        files = {
            p.relative_to(args.evidence).as_posix(): p.read_text()
            for p in sorted(args.evidence.rglob("*"))
            if p.is_file()
        }
        result = verify(files)
        raw = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        packet.write_bytes(gzip.compress(raw, mtime=0))
        index = {
            "compressed_sha256": digest(packet.read_bytes()),
            "decoded_sha256": digest(raw),
            "files": {name: digest(value.encode()) for name, value in files.items()},
        }
        index_path.write_bytes(
            gzip.compress((json.dumps(index, indent=2, sort_keys=True) + "\n").encode(), mtime=0)
        )
    else:
        index = json.loads(gzip.decompress(index_path.read_bytes()))
        raw = gzip.decompress(packet.read_bytes())
        if (
            digest(packet.read_bytes()) != index["compressed_sha256"]
            or digest(raw) != index["decoded_sha256"]
        ):
            raise ValueError("Raw evidence packet changed")
        files = json.loads(raw)
        if {name: digest(value.encode()) for name, value in files.items()} != index["files"]:
            raise ValueError("Original decoded file bytes changed")
        result = verify(files)
    (args.study / "data/corpus-failure-result.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(result, args.study)


if __name__ == "__main__":
    main()
