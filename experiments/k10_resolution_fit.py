"""Prospective fresh GPU resolution pair. No launch budget is defined here."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from experiments.k10_context_cost import FEATURES, INITIAL, LABELS, PLAN
from experiments.k10_context_pair import train_arm
from tools.musicfm_crop_io import file_identity, load, sha

if TYPE_CHECKING:
    import torch

CANDIDATE_INITIAL = "4fc58182dc884d7e9e9b948e72737742628be6b6a0685472b42808b0aed715d4"
PREFIX = "54d0c895089289238293d49d211a4a8ce26bfe4b79e3d47419a2fc73a51d5b4a"
FORMAT = "EXPERIMENTAL_K10_RESOLUTION_GPU_STATE_DICT_V1"
STATUS = "COMPLETE_RESOLUTION_GPU_ARM_NOT_PROMOTED"
MODELS = {"baseline": INITIAL, "candidate": CANDIDATE_INITIAL}


def prefix_digest(model: torch.nn.Module) -> str:
    import torch

    net = model.net
    if not isinstance(net, torch.nn.Sequential):
        raise ValueError("Expected explicit sequential backbone")
    digest = hashlib.sha256()
    for name, value in net[:14].state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def configure_gpu() -> None:
    """Called only inside a separately approved fresh GPU worker."""
    import torch

    from experiments.k10_gpu1_resource import POLICY, validate_device_identity

    validate_device_identity(torch.cuda.device_count(), torch.cuda.get_device_properties(0).uuid)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(POLICY["allocator_bytes"] / total, 0)


def arm(
    name: str,
    plan: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    array: Callable[[str], np.ndarray],
    output: Path,
    provenance: dict[str, Any],
    watch: Callable[[], None],
) -> dict[str, Any]:
    """One fresh worker/model; outer guardian must qualify/authorize first."""
    from drone.model import KeyCNN
    from experiments.k10_resolution import SemitoneGridCNN

    if name not in MODELS:
        raise ValueError("Only the predeclared baseline and candidate exist")
    configure_gpu()
    factory = KeyCNN if name == "baseline" else SemitoneGridCNN

    first = True

    def checked_factory() -> torch.nn.Module:
        nonlocal first
        model = factory()
        if first and prefix_digest(model) != PREFIX:
            raise ValueError("Shared first-two-block initialization differs")
        first = False  # Reload state is checked strictly, not against fresh RNG.
        return model

    return train_arm(
        320,
        plan,
        rows,
        array,
        output,
        {**provenance, "model_name": name},
        watch,
        model_factory=checked_factory,
        expected_init=MODELS[name],
        device="cuda:0",
        artifact_format=FORMAT,
        result_status=STATUS,
    )


def verify_data(value: dict[str, Any], watch: Callable[[], None]) -> dict[str, dict[str, Any]]:
    """Hash all admitted existing CQT files before any model construction."""
    if (value["plan_sha256"], value["labels_sha256"], value["features_sha256"]) != (
        PLAN,
        LABELS,
        FEATURES,
    ):
        raise ValueError("Different admitted data")
    for key in ("plan", "labels", "features"):
        if sha(Path(value[key])) != value[key + "_sha256"]:
            raise ValueError("Changed data binding")
    plan = load(Path(value["plan"]))
    if load(Path(value["labels"]))["labels"] != plan["expected_refs"]:
        raise ValueError("Pinned numeric labels differ")
    rows = {r["id"]: r for r in value["rows"]}
    if len(rows) != 5638 or len(plan["bp_fit"]) != 1229 or len(plan["expanded_fma_fit"]) != 4033:
        raise ValueError("Changed fitting membership")
    expected = set(
        plan["bp_fit"]
        + plan["expanded_fma_fit"]
        + plan["bp_selection_guard"]
        + plan["fma_inner_selection"]
    )
    if (
        set(rows) != expected
        or len(plan["bp_selection_guard"]) != 126
        or len(plan["fma_inner_selection"]) != 250
    ):
        raise ValueError("Forbidden/missing inputs")
    for identity, row in rows.items():
        watch()
        path = Path(row["cache_path"])
        before = file_identity(path)
        if (
            row["ref"] != plan["expected_refs"][identity]
            or list(before) != list(row["cache_identity"])
            or sha(path) != row["cache_sha256"]
            or before != file_identity(path)
        ):
            raise ValueError("CQT identity/target changed")
    return rows


def compare(
    control: dict[str, Any], candidate: dict[str, Any], historical: dict[str, Any]
) -> dict[str, Any]:
    fields = (
        "common_draw_pitch_sha256",
        "temporal_sha256",
        "bp_augmented_sha256",
        "coverage",
        "actual_frames",
    )
    if any(control[k] != candidate[k] for k in fields):
        raise ValueError("GPU pair source/runtime/draw/exposure mismatch")
    common_provenance = []
    for name, result in (("baseline", control), ("candidate", candidate)):
        provenance = result["provenance"]
        if not isinstance(provenance, dict) or provenance.get("model_name") != name:
            raise ValueError("Checkpoint factory identity differs")
        common_provenance.append({k: v for k, v in provenance.items() if k != "model_name"})
        proof = result.get("selected_reload_proof", {})
        if (
            result.get("status") != STATUS
            or result["initial_state_sha256"] != MODELS[name]
            or result["crop_frames"] != 320
            or result["updates"] != 8280
            or result["events"] != 120
            or result.get("confirmation_used") is not False
            or result.get("calibration_used") is not False
            or not proof.get("strict_state_load")
            or not proof.get("scores_identical")
            or proof.get("tracks") != 376
            or proof.get("argmax_disagreements") != 0
        ):
            raise ValueError("Incomplete/unmatched GPU arm")
    if common_provenance[0] != common_provenance[1]:
        raise ValueError("GPU pair common provenance differs")
    keys = ("exact_correct", "symmetric_units10", "asymmetric_units10")
    a, b, old = (x["best_metrics"] for x in (control, candidate, historical))
    for domain, n in (("beatport", 126), ("fmak", 250)):
        if not (
            a[domain]["ids"] == b[domain]["ids"] == old[domain]["ids"]
            and len(a[domain]["ids"]) == n
        ):
            raise ValueError("Selection identities/order differ")
    gain = b["fmak"]["exact_correct"] - a["fmak"]["exact_correct"]
    gate = gain >= 5 and all(b["beatport"][k] >= a["beatport"][k] for k in keys)
    guard = all(a[d][k] >= old[d][k] for d in ("beatport", "fmak") for k in keys)
    return {
        "technical_pair_valid": True,
        "FMA_exact_gain": gain,
        "resolution_gate": gate,
        "historical_GPU_control_noninferiority_guard": guard,
        "continuation_earned": gate and guard,
        "promotion_authorized": False,
        "historical_gaps": {
            d: {k: a[d][k] - old[d][k] for k in keys} for d in ("beatport", "fmak")
        },
    }
