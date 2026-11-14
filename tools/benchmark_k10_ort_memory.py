"""Isolate CPU ONNX memory on cached public inputs, without importing PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


def rss_mib() -> float:
    scale = 1024**2 if platform.system() == "Darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_bytes())
    manifest = json.loads((args.features / "manifest.json").read_bytes())
    if manifest["feature_spec"] != "cqt24-8oct-c1-pad4-hop4096-v2":
        raise ValueError("Different CQT contract")
    if selection["private_audio_used"] or selection["accuracy_scored"]:
        raise ValueError("Use the public performance-only selection")
    lookup = {row["id"]: row for row in manifest["rows"]}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    before = rss_mib()
    session = ort.InferenceSession(
        str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"]
    )
    loaded = rss_mib()
    for selected in selection["rows"]:
        row = lookup[selected["id"]]
        if row["sha256"] != selected["sha256"]:
            raise ValueError("Selection differs from prepared audio")
        path = args.features / row["cache"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["cache_sha256"]:
            raise ValueError("Cache changed")
        padded = np.load(path, allow_pickle=False)
        if padded.shape != (208, selected["frames"]):
            raise ValueError("Wrong CQT shape")
        core = padded[8:200].astype(np.float32)
        session.run(["logits"], {"features": core[None]})
    if "torch" in sys.modules:
        raise RuntimeError("PyTorch entered the isolated probe")
    args.output.write_text(
        json.dumps(
            {
                "tracks": len(selection["rows"]),
                "torch_imported": False,
                "cpu_affinity": sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
                "peak_before_session_mib": before,
                "peak_after_session_mib": loaded,
                "process_peak_rss_mib": rss_mib(),
                "accuracy_scored": False,
                "limitations": "Python/NumPy/ORT on cached features only; excludes audio decode/CQT and the Euphoria application. RSS values are process high-water marks.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
