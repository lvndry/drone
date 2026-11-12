"""Bounded read-only BP text/checkpoint metadata export, never audio or GS labels."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def parser_from_source(path: Path) -> Callable[[str], tuple[str, str] | None]:
    tree = ast.parse(path.read_text())
    nodes = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_key"
    ]
    if len(nodes) != 1:
        raise ValueError("Canonical BP label parser not uniquely found")
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            nodes[0],
        ],
        type_ignores=[],
    )
    namespace: dict[str, Any] = {
        "TONICS": ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["parse_key"]


def bind_sources(source: Path, loader: Path, runs: list[dict[str, Any]]) -> dict[str, Any]:
    if loader.resolve() != (source / "drone/datasets.py").resolve():
        raise ValueError("Loader must be the archived canonical source")
    sources = {
        name: dict(path=str(source / name), sha256=digest(source / name))
        for name in ("experiments/k10_multidomain.py", "train.py", "drone/datasets.py")
    }
    if any(
        run["source_sha256"].get(name) != witness["sha256"]
        for run in runs
        for name, witness in sources.items()
    ):
        raise ValueError("Archived source differs from run source map")
    return sources


def label_record(path: Path, parse: Callable[[str], tuple[str, str] | None]) -> dict[str, Any]:
    match = re.match(r"(\d+)", path.name)
    if match is None or path.stat().st_size > 65536:
        raise ValueError("Invalid/oversized label metadata file")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        reason, parseable = "invalid_utf8", False
    else:
        parseable = parse(text) is not None
        reason = "parseable" if parseable else "unparseable_text"
    branch = None
    if reason == "unparseable_text":
        parts = text.split("|")[0].strip().split()
        if len(parts) != 2:
            branch = "wrong_token_count"
        elif parts[1] not in ("major", "minor"):
            branch = "unsupported_mode"
        elif parts[0] not in (
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
            "Db",
            "Eb",
            "Gb",
            "Ab",
            "Bb",
        ):
            branch = "unsupported_tonic"
        else:
            raise ValueError("Parser rejected text outside the audited branches")
    return dict(
        id="beatport:" + str(int(match[1])),
        filename=path.name,
        sha256=hashlib.sha256(raw).hexdigest(),
        parseable=parseable,
        parse_status_reason=reason,
        unparseable_branch=branch,
    )


def collect(root: Path, data: Path, loader: Path) -> dict[str, Any]:
    began = time.monotonic()
    # Read archive source and run witnesses first; no checkpoint tensors are inspected.
    source = root / "source"
    runs = []
    for arm in ("control", "joint"):
        for seed in (13, 37, 73):
            run_path = root / f"{arm}-seed{seed}/run.json"
            run = json.loads(run_path.read_bytes())
            runs.append(
                dict(
                    model_id=f"{arm}{seed}",
                    run_path=str(run_path),
                    run_sha256=digest(run_path),
                    source_sha256=run["source_sha256"],
                    feature_manifest_sha256=run["feature_manifest_sha256"],
                    fit_ids=run["fit_ids"],
                    validation_ids=run["validation_ids"],
                    checkpoint_path=str(run_path.with_name("best.pt")),
                )
            )
    sources = bind_sources(source, loader, runs)
    # The existing prepare CLI wires --data/labels; don't search alternate directories.
    source_text = (source / "train.py").read_text()
    if 'load_beatport(args.data, args.data / "labels")' not in source_text:
        raise ValueError("Archived label-directory wiring needs manual review")
    parse = parser_from_source(loader)
    labels = []
    for path in sorted((data / "labels").iterdir()):
        if not path.is_file() or not path.name.endswith(".txt"):
            continue
        match = re.match(r"(\d+)", path.name)
        if match is None:
            continue
        labels.append(label_record(path, parse))
    import torch

    models = []
    for run in runs:
        path = Path(run["checkpoint_path"])
        if path.stat().st_size > 16 << 20:
            raise ValueError("Unexpected checkpoint size; metadata collector cap")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        raw_run = json.loads(Path(run["run_path"]).read_bytes())
        checkpoint_sha = digest(path)
        if (
            raw_run["checkpoint_sha256"] != checkpoint_sha
            or metadata["source_sha256"] != run["source_sha256"]
        ):
            raise ValueError("Checkpoint/run/source binding differs")
        if metadata["seed"] != raw_run["seed"] or metadata["arm"] != raw_run["arm"]:
            raise ValueError("Checkpoint/run configuration differs")
        models.append(
            dict(
                **run,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_metadata={
                    k: metadata[k]
                    for k in (
                        "fit_ids",
                        "train_ids",
                        "validation_ids",
                        "training_audio_sha256",
                        "source_sha256",
                        "feature_manifest_sha256",
                        "config",
                        "seed",
                        "arm",
                        "epoch",
                        "experiment",
                    )
                    if k in metadata
                },
                format_version=payload.get("format_version"),
                feature_spec=payload.get("feature_spec"),
                label_schema=payload.get("label_schema"),
            )
        )
        del payload
        if time.monotonic() - began > 120:
            raise TimeoutError("Metadata collector exceeded two-minute cap")
    return dict(
        version="bp-gs-lineage-metadata-v1",
        sources=sources,
        loader=dict(path=str(loader), sha256=digest(loader)),
        labels=labels,
        parse_reason_counts=dict(
            Counter(row["unparseable_branch"] or row["parse_status_reason"] for row in labels)
        ),
        models=models,
        audio_read=False,
        gs_labels_read=False,
        model_instantiated=False,
        elapsed_seconds=time.monotonic() - began,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--loader", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.root, args.data, args.loader), sort_keys=True))


if __name__ == "__main__":
    main()
