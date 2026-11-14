"""CPU-only import/path receipt and safe mmap checkpoint tensor-layout inspection."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tools.preflight_musicfm import (
    EVIDENCE,
    EXPECTED,
    WEIGHT_BYTES,
    WEIGHT_SHA,
    prepare_upstream,
    sha,
)
from tools.run_cqt_expansion import write_json


def protected_receipt() -> dict[str, Any]:
    import numpy
    import torch

    array_binary = importlib.import_module("numpy._core._multiarray_umath")
    torch_binary = importlib.import_module("torch._C")
    records = {}
    for name, module in (
        ("torch", torch),
        ("torch._C", torch_binary),
        ("numpy", numpy),
        ("numpy._core._multiarray_umath", array_binary),
    ):
        if module.__file__ is None:
            raise ValueError("module has no physical file")
        path = Path(module.__file__).resolve()
        records[name] = {"path": str(path), "sha256": sha(path)}
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized during CPU-only stage")
    return {
        "modules": records,
        "torch_runtime": torch.__version__,
        "torch_metadata": importlib.metadata.version("torch"),
        "numpy_version": numpy.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_initialized": False,
    }


def verify_imports(root: Path, baseline: Path) -> dict[str, Any]:
    import torch
    from packaging.requirements import Requirement

    receipt = protected_receipt()
    if receipt != json.loads(baseline.read_bytes())["protected"]:
        raise RuntimeError("shared Torch/NumPy path, bytes or version changed")
    selected = json.loads((root / "selected-wheels.json").read_bytes())["wheels"]
    packages = {}
    issues = []
    for wheel in selected:
        distribution = importlib.metadata.distribution(wheel["name"])
        packages[wheel["name"]] = {
            "version": distribution.version,
            "path": str(Path(str(distribution.locate_file(""))).resolve()),
        }
        if distribution.version != wheel["version"]:
            raise RuntimeError("resolved/installed version differs")
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                version = importlib.metadata.version(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                issues.append(f"missing:{requirement}")
                continue
            if version not in requirement.specifier:
                issues.append(f"incompatible:{requirement}:installed={version}")
    if issues:
        raise RuntimeError(f"dependency requirements: {issues}")
    prepare_upstream(root)
    sys.path.insert(0, str(root / "upstream"))
    # Fail before torch can lazily initialize any CUDA context, even if an
    # unexpected third-party import attempts it. Devices are also masked.
    with patch.object(
        torch.cuda, "_lazy_init", side_effect=RuntimeError("CUDA forbidden in CPU stage")
    ):
        torchaudio = importlib.import_module("torchaudio")
        transformers = importlib.import_module("transformers")
        einops = importlib.import_module("einops")
        upstream = importlib.import_module("musicfm.model.musicfm_25hz")
        conformer = importlib.import_module(
            "transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer"
        )
        configuration = transformers.Wav2Vec2ConformerConfig.from_dict(
            json.loads((EVIDENCE / "conformer-config.json").read_bytes())
        )
        signatures = {
            "encoder_forward": str(inspect.signature(conformer.Wav2Vec2ConformerEncoder.forward)),
            "MelSpectrogram": str(inspect.signature(torchaudio.transforms.MelSpectrogram)),
            "AmplitudeToDB": str(inspect.signature(torchaudio.transforms.AmplitudeToDB)),
            "MusicFM25Hz": str(inspect.signature(upstream.MusicFM25Hz)),
        }
        if configuration.hidden_size != 1024 or configuration.num_attention_heads != 16:
            raise ValueError("pinned configuration API changed")
    modules = {}
    for name, module in (
        ("torchaudio", torchaudio),
        ("transformers", transformers),
        ("einops", einops),
        ("musicfm", upstream),
        ("conformer", conformer),
    ):
        if module.__file__ is None:
            raise ValueError("module has no physical file")
        path = Path(module.__file__).resolve()
        modules[name] = {"path": str(path), "sha256": sha(path)}
    if {name: importlib.metadata.version(name) for name in EXPECTED} != EXPECTED:
        raise ValueError("direct package version mismatch")
    return {
        "passed": True,
        "protected": receipt,
        "protected_after": protected_receipt(),
        "packages": packages,
        "modules": modules,
        "signatures": signatures,
        "model_constructed": False,
        "gpu_execution": False,
    }


def checkpoint_layout(path: Path) -> dict[str, Any]:
    import torch

    if path.stat().st_size != WEIGHT_BYTES or sha(path) != WEIGHT_SHA:
        raise ValueError("pinned checkpoint integrity")
    with patch.object(
        torch.cuda, "_lazy_init", side_effect=RuntimeError("CUDA forbidden in CPU stage")
    ):
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = payload["state_dict"]
    rows: list[dict[str, Any]] = []
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"elements": 0, "bytes": 0, "tensors": 0}
    )
    for name, tensor in state.items():
        if (
            not isinstance(name, str)
            or not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
        ):
            raise ValueError("non-CPU or non-tensor state entry")
        dtype = str(tensor.dtype)
        count = tensor.numel()
        size = tensor.numel() * tensor.element_size()
        totals[dtype]["elements"] += count
        totals[dtype]["bytes"] += size
        totals[dtype]["tensors"] += 1
        rows.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": dtype,
                "elements": count,
                "bytes": size,
            }
        )
    prefix_matches = all(row["name"].startswith("model.") for row in rows)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized")
    return {
        "passed": prefix_matches,
        "weights_only": True,
        "mmap": True,
        "map_location": "cpu",
        "checkpoint_sha256": WEIGHT_SHA,
        "checkpoint_bytes": WEIGHT_BYTES,
        "top_level_keys": sorted(payload),
        "state_tensor_count": len(rows),
        "totals_by_dtype": dict(totals),
        "tensor_layout": rows,
        "future_loader_model_prefix_matches": prefix_matches,
        "counts_are_state_tensors_not_parameter_counts": True,
        "model_constructed": False,
        "gpu_execution": False,
        "cuda_initialized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["baseline", "imports", "layout"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
    ):
        raise RuntimeError("CPU-only visibility/no-shared-bytecode policy required")
    if args.output.exists():
        raise FileExistsError("preserve prior evidence")
    if args.phase == "baseline":
        result = {
            "protected": protected_receipt(),
            "python": sys.version,
            "distributions": sorted(
                (
                    distribution.metadata["Name"],
                    distribution.version,
                    str(Path(str(distribution.locate_file(""))).resolve()),
                )
                for distribution in importlib.metadata.distributions()
            ),
        }
    elif args.phase == "imports":
        result = verify_imports(args.root, args.root / "baseline.json")
    else:
        result = checkpoint_layout(args.root / "inputs/pretrained_msd.pt")
    result["helper_sha256"] = sha(Path(__file__))
    write_json(args.output, result)
    if result.get("passed") is False:
        raise RuntimeError("CPU compatibility gate failed; no workaround")


if __name__ == "__main__":
    main()
