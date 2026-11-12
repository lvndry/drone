"""Verify and package the single frozen16-file waveform qualification; no decoding."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.musicfm_crop_io import crop_positions

CONFIG_SHA = "30144ca9abf7a8258983856c13cad1be544acad4ad556269769039ec2033721b"
REPORT_SHA = "9afddd0a07f3cb4ec0b97041059845ff4e01bc295a1e1da127b424c4e0e33a66"
TOOL_SHA = "3efaf109c547acaf9479a40d23055680454234560eb17ffad762151a756114ac"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(files: dict[str, str]) -> dict[str, Any]:
    config = json.loads(files["sealed-config.json"])
    report = json.loads(files["qualification-v1/report.json"])
    approval = json.loads(files["root-approval.json"])
    terminal = json.loads(files["terminal.json"])
    launch = json.loads(files["launch.json"])
    after = json.loads(files["protected-after.json"])
    if (
        digest(files["sealed-config.json"].encode()) != CONFIG_SHA
        or digest(files["qualification-v1/report.json"].encode()) != REPORT_SHA
    ):
        raise ValueError("Frozen raw config/report changed")
    if (
        config["tool_sha256"] != TOOL_SHA
        or report["tool_sha256"] != TOOL_SHA
        or report["config_sha256"] != CONFIG_SHA
        or not approval["authorized"]
        or approval["config_sha256"] != CONFIG_SHA
        or terminal["exit"] != 0
        or terminal["reason"] is not None
        or not report["complete"]
        or not report["passed"]
    ):
        raise ValueError("Incomplete/unbound execution")
    if digest(files["run_once.py"].encode()) != launch["wrapper_source_sha256"]:
        raise ValueError("Outer supervisor source changed")
    if set(approval["protected_before"]) != set(after) or any(
        row["start_ticks"] != after[pid]["start_ticks"] or after[pid]["state"] in ("Z", "X")
        for pid, row in approval["protected_before"].items()
    ):
        raise ValueError("Protected process identity changed")
    if len(report["rows"]) != 16 or len(config["rows"]) != 16:
        raise ValueError("Fixed track count changed")
    count = 0
    points = []
    for number, (expected, row) in enumerate(zip(config["rows"], report["rows"], strict=True), 1):
        if (
            row["id"] != expected["id"]
            or row["audio_sha256"] != expected["sha256"]
            or row != json.loads(files[f"qualification-v1/track-{number:02}.json"])
        ):
            raise ValueError("Track/order/source binding changed")
        if [c["start_sample"] for c in row["cases"]] != crop_positions(
            row["full_samples"], row["id"]
        ):
            raise ValueError("Crop membership changed")
        for position, case in enumerate(row["cases"], 1):
            standalone = json.loads(files[f"qualification-v1/case-{number:02}-{position:02}.json"])
            if standalone != {"id": row["id"], **case} or not case["passed"] or not case["finite"]:
                raise ValueError("Per-case result changed")
            if any(
                case[k] != 0
                for k in (
                    "max_absolute_error",
                    "rms_error",
                    "bitwise_mismatch_samples",
                    "mismatch_fraction",
                )
            ):
                raise ValueError("Bit-exact gate did not pass")
            if "-xerror" in case["execution"]["command"]:
                raise ValueError("Wrong decoder recovery profile")
            count += 1
        points.append(
            {
                "id": row["id"],
                "duration_seconds": row["full_samples"] / 24000,
                "full_wall_seconds": row["full_execution"]["wall_seconds"],
                "mean_crop_wall_seconds": sum(c["execution"]["wall_seconds"] for c in row["cases"])
                / len(row["cases"]),
            }
        )
    stderr = [value for name, value in files.items() if name.endswith(".stderr")]
    if count != 144 or len(stderr) != 160:
        raise ValueError("Case/warning evidence missing")
    return {
        "tracks": 16,
        "cases": count,
        "all_bit_exact": True,
        "points": points,
        "stderr_files": len(stderr),
        "nonempty_stderr_files": sum(bool(x) for x in stderr),
        "terminal": terminal,
        "report_sha256": REPORT_SHA,
        "config_sha256": CONFIG_SHA,
    }


def plot(result: dict[str, Any], study: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["svg.hashsalt"] = "musicfm-crop-io-6011bf7"
    points = result["points"]
    figure, axis = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    axis.scatter(
        [p["duration_seconds"] / 60 for p in points],
        [p["full_wall_seconds"] * 1000 for p in points],
        label="One full decode",
        marker="o",
        color="#295b85",
    )
    axis.scatter(
        [p["duration_seconds"] / 60 for p in points],
        [p["mean_crop_wall_seconds"] * 1000 for p in points],
        label="Mean of nine bounded crop decodes",
        marker="x",
        color="#b35430",
    )
    axis.set(
        xlabel="Actual decoded track duration (minutes)",
        ylabel="Instrumented wall time per decode (ms)",
        yscale="log",
        title=f"Identical waveforms: {result['cases']}/{result['cases']} fixed crops",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.text(
        0.01,
        -0.04,
        "Qualification only: 16 files; CPUs 8–9; threads = 1; 50ms resource polling. "
        "Not a representative throughput benchmark.",
        fontsize=8,
    )
    figure.savefig(study / "crop-qualification.png", dpi=160, bbox_inches="tight")
    figure.savefig(study / "crop-qualification.svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    data = args.study / "data"
    packet = data / "runtime-raw.json.gz"
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
        (data / "runtime-index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    else:
        index = json.loads((data / "runtime-index.json").read_bytes())
        raw = gzip.decompress(packet.read_bytes())
        if (
            digest(packet.read_bytes()) != index["compressed_sha256"]
            or digest(raw) != index["decoded_sha256"]
        ):
            raise ValueError("Raw evidence packet changed")
        files = json.loads(raw)
        if {name: digest(value.encode()) for name, value in files.items()} != index["files"]:
            raise ValueError("Original decoded raw bytes changed")
        result = verify(files)
    (data / "verified-result.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(result, args.study)


if __name__ == "__main__":
    main()
