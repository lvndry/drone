"""Verify BN v4 terminal evidence and reconstruct sealed source without running it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/studies/musicfm-key-pilot/data/bn-diagnostic/v4"
MANIFEST_SHA256 = "17914a9386d8dfc6128e167be8c96d6aae398df6fe7353ed57e16cc998e9ebe7"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_evidence(path: Path = EVIDENCE) -> dict[str, Any]:
    manifest = (path / "SHA256SUMS").read_bytes()
    require(sha(manifest) == MANIFEST_SHA256, "Evidence manifest changed")
    names: set[str] = set()
    for line in manifest.decode().splitlines():
        expected, name = line.split("  ")
        require(Path(name).name == name and name not in names, "Invalid evidence member")
        names.add(name)
        require(sha((path / name).read_bytes()) == expected, "Evidence hash mismatch: " + name)
    require(
        names == {p.name for p in path.iterdir()} - {"SHA256SUMS"},
        "Unlisted evidence member",
    )
    return {p.stem: json.loads(p.read_bytes()) for p in path.glob("*.json")}


def verify(docs: dict[str, Any]) -> dict[str, Any]:
    run, controller, result, lock = (
        docs[k] for k in ("run", "controller", "result", "source-lock")
    )
    require(
        run["config_sha256"] == controller["config_sha256"] == result["config_sha256"],
        "Controller/config/result binding mismatch",
    )
    require(
        controller["complete"] is True
        and controller["worker_started"] is True
        and controller["worker_exit"] == 0
        and controller["error"] is None
        and controller["cleanup_error"] is None,
        "Controller did not complete cleanly",
    )
    caps = run["caps"]
    require(run["wall_limit_seconds"] == caps["wall_seconds"] == 3600, "Wall limit changed")
    require(
        caps["rss_bytes"] == 8 * 1024**3
        and caps["cpu_seconds"] == 10800
        and caps["output_bytes"] == 8 * 1024**3
        and run["allocator_bytes"] == 16 * 1024**3,
        "Declared caps changed",
    )
    measurements = {
        "wall_seconds": controller["wall_seconds"],
        "rss_bytes": controller["peaks"]["rss"],
        "cpu_seconds": controller["peaks"]["cpu"],
    }
    for name, value in measurements.items():
        require(math.isfinite(value) and 0 <= value <= caps[name], "Invalid resource: " + name)
    require(
        result["elapsed_seconds"] <= controller["wall_seconds"]
        and controller["terminal_resources"]["cpu_seconds"] <= measurements["cpu_seconds"]
        and controller["terminal_resources"]["rss_bytes"] <= measurements["rss_bytes"],
        "Inconsistent terminal resources",
    )
    require(
        result["status"] == "COMPLETE_SINGLE_BN_DIAGNOSTIC_NOT_PROMOTED"
        and result["candidate_strict_state_reload"] is True
        and result["candidate_composed_batch_proof"]["argmax_disagreements"] == 0,
        "Diagnostic incomplete",
    )
    for key in (
        "promotion_authorized",
        "training_performed",
        "confirmation_used",
        "reserved_calibration_used",
    ):
        require(result[key] is False, "Unexpected diagnostic scope: " + key)
    require(
        run["promotion_authorized"] is False and run["outcome"] == "negative", "Run outcome changed"
    )
    for name, result_name in (
        ("admission", "admission"),
        ("score-admission", "score_admission"),
        ("qualification", "qualification"),
    ):
        require(docs[name] == result[result_name], "Separate receipt mismatch: " + name)
    require(
        docs["sealed-candidate"]["checkpoint_sha256"] == result["candidate_checkpoint_sha256"],
        "Candidate binding mismatch",
    )
    deltas = {}
    for domain, total in (("beatport", 126), ("fmak", 250)):
        before, after = (result[k][domain] for k in ("baseline_metrics", "candidate_metrics"))
        require(
            before["ids"] == after["ids"] and len(set(before["ids"])) == total, "Panel mismatch"
        )
        for metrics in (before, after):
            require(len(metrics["predictions"]) == total, "Prediction count mismatch")
            require(
                math.isclose(metrics["exact"], metrics["exact_correct"] / total),
                "Exact denominator mismatch",
            )
        deltas[domain] = after["exact_correct"] - before["exact_correct"]
    useful = deltas["fmak"] >= 5 and deltas["beatport"] >= 0
    require(result["useful_signal"] is useful and not useful, "Negative gate mismatch")
    require(
        run["base_commit"] == lock["base_commit"] and len(lock["source_sha256"]) == 132,
        "Source lock mismatch",
    )
    return {
        "resources": measurements,
        "exact_delta": deltas,
        "useful_signal": useful,
        "source_files": 132,
    }


def reconstruct_source(lock: dict[str, Any], patch: Path, repository: Path = ROOT) -> None:
    """Rebuild only pinned files in a disposable directory; never import or execute them."""
    require(
        sha(patch.read_bytes()) == lock["override_patch_sha256"], "Override patch hash mismatch"
    )
    with tempfile.TemporaryDirectory(prefix="drone-bn-v4-source-") as temporary:
        target = Path(temporary)
        for name in lock["source_sha256"]:
            require(isinstance(name, str), "Invalid source path")
            relative = PurePosixPath(name)
            require(
                not relative.is_absolute() and ".." not in relative.parts, "Invalid source path"
            )
            raw = subprocess.check_output(
                ["git", "show", lock["base_commit"] + ":" + name], cwd=repository
            )
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        subprocess.run(
            ["git", "apply", "--unidiff-zero", str(patch.resolve())],
            cwd=target,
            check=True,
            capture_output=True,
        )
        for name, expected in lock["source_sha256"].items():
            require(
                sha((target / name).read_bytes()) == expected,
                "Reconstructed source mismatch: " + name,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruct-source", action="store_true")
    args = parser.parse_args()
    docs = load_evidence()
    summary = verify(docs)
    for filename, key in (
        ("controller.json", "controller_sha256"),
        ("result.json", "result_sha256"),
        ("source-lock.json", "source_lock_sha256"),
    ):
        require(
            sha((EVIDENCE / filename).read_bytes()) == docs["run"][key], "Run hash binding mismatch"
        )
    require(
        sha((EVIDENCE / "source-overrides.patch").read_bytes())
        == docs["source-lock"]["override_patch_sha256"],
        "Override patch hash mismatch",
    )
    if args.reconstruct_source:
        reconstruct_source(docs["source-lock"], EVIDENCE / "source-overrides.patch")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
