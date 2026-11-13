"""Prospective frozen-embedding Myna head pilot; execution requires an external guardian."""

from __future__ import annotations

import hashlib
import inspect
import json
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F

from drone.keys import score_mirex, transpose_key
from experiments.k10_v2_pilot import selection_key
from experiments.musicfm_pilot import metrics
from experiments.myna_cache import (
    PINS,
    LoadedCache,
    checked_bytes,
    checked_json,
    group_rows,
    load_cache,
    loss_weight,
    require,
    sha256,
    validate_manifest,
)

SEED, BATCH = 42, 64
ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "experiments/myna_cache.py",
    "experiments/myna_pilot.py",
    "experiments/k10_v2_pilot.py",
    "experiments/musicfm_pilot.py",
    "drone/keys.py",
)
IDENTITIES = {
    "encoder_sha256",
    "encoder_source_sha256",
    "environment_sha256",
    "frontend_sha256",
    "augmentation_sha256",
    "qualification_sha256",
}


def make_head(device: str = "cpu") -> nn.Sequential:
    """One dropout after the first ReLU; never instantiate the pretrained encoder."""
    return nn.Sequential(
        nn.Linear(384, 4096, device=device, dtype=torch.float32),
        nn.ReLU(),
        nn.Dropout(0.99),
        nn.Linear(4096, 4096, device=device, dtype=torch.float32),
        nn.ReLU(),
        nn.Linear(4096, 24, device=device, dtype=torch.float32),
    )


def weighted_ce(logits: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
    """Unnormalized importance weights: a random batch must not redefine the objective."""
    require(
        logits.ndim == 2
        and logits.shape[1] == 24
        and targets.shape == weights.shape == (len(logits),),
        "Invalid fitting batch",
    )
    require(
        bool(torch.isfinite(logits).all())
        and bool(torch.isfinite(weights).all())
        and bool((weights > 0).all()),
        "Nonfinite/invalid objective input",
    )
    return (F.cross_entropy(logits, targets, reduction="none") * weights).mean()


def batches(size: int, passes: int, seed: int = SEED) -> Iterator[tuple[int, NDArray[np.int64]]]:
    """Each pass visits every cached view/window once; retain the final short batch."""
    require(size > 0 and passes > 0, "Invalid exposure budget")
    rng = np.random.default_rng(seed)
    for epoch in range(1, passes + 1):
        order = rng.permutation(size)
        for start in range(0, size, BATCH):
            yield epoch, order[start : start + BATCH]


def mean_logits(
    chunks: Iterator[NDArray[np.float32]], expected_windows: int
) -> NDArray[np.float64]:
    """Average window logits, not probabilities or input embeddings, in FP64."""
    total = np.zeros(24, dtype=np.float64)
    seen = 0
    for chunk in chunks:
        require(
            chunk.dtype == np.float32
            and chunk.ndim == 2
            and chunk.shape[1] == 24
            and bool(np.isfinite(chunk).all()),
            "Invalid window logits",
        )
        # Fixed window order makes aggregation independent of chunk/batch boundaries.
        for row in chunk:
            total += row.astype(np.float64)
        seen += len(chunk)
    require(seen == expected_windows and seen > 0, "Missing/extra selection windows")
    return total / seen


def predict(
    head: nn.Module, cache: LoadedCache, batch_size: int, watch: Callable[[], None]
) -> dict[str, list[float]]:
    require(1 <= batch_size <= BATCH, "Unsupported head evaluation batch")
    device = next(head.parameters()).device
    result: dict[str, list[float]] = {}
    head.eval()
    with torch.inference_mode():
        for ids in cache.contract.selection_ids.values():
            for identity in ids:
                span = cache.selection[identity]

                def chunks() -> Iterator[NDArray[np.float32]]:
                    for offset in range(span.start, span.stop, batch_size):
                        watch()
                        values = np.array(
                            cache.embeddings[offset : min(offset + batch_size, span.stop)],
                            copy=True,
                        )
                        logits = head(torch.from_numpy(values).to(device)).cpu().numpy()
                        yield logits

                result[identity] = mean_logits(chunks(), span.stop - span.start).tolist()
    return result


def panel_metrics(logits: dict[str, list[float]], cache: LoadedCache) -> dict[str, dict[str, Any]]:
    return {
        domain: metrics(ids, [int(np.argmax(logits[i])) for i in ids], cache.contract.labels)
        for domain, ids in cache.contract.selection_ids.items()
    }


def continuation(
    candidate: dict[str, dict[str, Any]], control: dict[str, dict[str, Any]], parity: bool
) -> bool:
    return (
        parity
        and candidate["fmak"]["exact_correct"] >= control["fmak"]["exact_correct"] + 5
        and all(
            candidate["beatport"][key] >= control["beatport"][key]
            for key in ("exact_correct", "symmetric_units10", "asymmetric_units10")
        )
    )


def batch_proof(
    reference: dict[str, list[float]], candidate: dict[str, list[float]]
) -> dict[str, Any]:
    require(reference.keys() == candidate.keys(), "Different batch proof membership")
    maximum, ratio, disagreements = 0.0, 0.0, []
    for identity, values in reference.items():
        a, b = (
            np.asarray(values, dtype=np.float64),
            np.asarray(candidate[identity], dtype=np.float64),
        )
        require(
            a.shape == b.shape == (24,)
            and bool(np.isfinite(a).all())
            and bool(np.isfinite(b).all()),
            "Invalid composed head logits",
        )
        delta = np.abs(b - a)
        maximum = max(maximum, float(delta.max()))
        ratio = max(ratio, float(np.max(delta / (1e-5 + 1e-4 * np.abs(a)))))
        if int(np.argmax(a)) != int(np.argmax(b)):
            disagreements.append(identity)
    return {
        "passed": ratio <= 1 and not disagreements,
        "max_abs": maximum,
        "max_allowed_ratio": ratio,
        "key_disagreements": disagreements,
        "reference_batch": 1,
        "candidate_batch": BATCH,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def state_digest(head: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in head.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_qualification(qualification: dict[str, Any], identities: dict[str, str]) -> None:
    """A failed or differently bound numerical qualification cannot activate fitting."""
    require(
        qualification.get("schema_version") == "drone-myna-cache-admission-v1"
        and qualification.get("complete") is True,
        "Qualification incomplete",
    )
    require(
        qualification.get("identities")
        == {k: v for k, v in identities.items() if k != "qualification_sha256"},
        "Qualification numerical identities differ",
    )
    require(
        all(
            qualification.get(k) is True
            for k in (
                "all_pitch_views_passed",
                "long_track_composed_parity_passed",
                "sustained_head_cost_measured",
            )
        ),
        "Unqualified pitch/long-track/head-cost contract",
    )


def admit(config: dict[str, Any], watch: Callable[[], None]) -> tuple[LoadedCache, dict[str, Any]]:
    require(
        config["schema_version"] == "drone-myna-head-pilot-v1"
        and config["passes"] in (10, 100)
        and type(config["passes"]) is int,
        "Unsealed/unsupported horizon",
    )
    require(set(config["source_sha256"]) == set(SOURCES), "Incomplete executed-source binding")
    for function, expected in (
        (metrics, "experiments/musicfm_pilot.py"),
        (selection_key, "experiments/k10_v2_pilot.py"),
        (load_cache, "experiments/myna_cache.py"),
        (score_mirex, "drone/keys.py"),
        (transpose_key, "drone/keys.py"),
    ):
        origin = inspect.getsourcefile(function)
        require(
            origin is not None and Path(origin).resolve() == (ROOT / expected).resolve(),
            "Imported helper origin differs",
        )
    require(config["device"] in ("cpu", "cuda:0"), "Guardian must expose one selected GPU or CPU")
    for name in SOURCES:
        checked_bytes(ROOT / name, config["source_sha256"][name], watch)
    require(set(config["identities"]) == IDENTITIES, "Incomplete numerical identity binding")
    inputs = config["inputs"]
    require(set(inputs) == set(PINS) | {"cache", "qualification"}, "Different admission input set")
    documents: dict[str, Any] = {}
    for name, pin in PINS.items():
        require(inputs[name]["sha256"] == pin, "Changed approved input identity")
        raw = checked_bytes(Path(inputs[name]["path"]), pin, watch)
        documents[name] = group_rows(raw) if name == "groups" else json.loads(raw)
    for name in ("cache", "qualification"):
        documents[name] = checked_json(Path(inputs[name]["path"]), inputs[name]["sha256"], watch)
    qualification = documents["qualification"]
    require(
        inputs["qualification"]["sha256"] == config["identities"]["qualification_sha256"],
        "Qualification binding differs",
    )
    validate_qualification(qualification, config["identities"])
    contract = validate_manifest(
        documents["cache"],
        documents["plan"],
        documents["labels"]["labels"],
        documents["groups"],
        documents["waveforms"]["rows"],
        Path(inputs["cache"]["path"]).parent,
        config["identities"],
    )
    require(
        contract.embedding_bytes == config["expected_embedding_bytes"],
        "Observed extraction support differs from sealed count",
    )
    require(
        type(config["max_embedding_bytes"]) is int
        and 0 < config["max_embedding_bytes"] <= 4 * 1024**3,
        "Embedding allocation limit differs",
    )
    control = documents["control"]["best_metrics"]
    for domain, ids in contract.selection_ids.items():
        require(control[domain]["ids"] == ids, "Comparator selected different records/order")
    return load_cache(contract, config["max_embedding_bytes"], watch), control


def run(config_path: Path, config_sha256: str, watch: Callable[[], None]) -> dict[str, Any]:
    """Called only inside a separately reviewed guardian; deliberately no CLI/default watchdog."""
    config = checked_json(config_path, config_sha256, watch)
    output = Path(config["output"])
    output.mkdir(parents=False, exist_ok=False)
    committed_updates, committed_presentations = 0, 0
    try:
        cache, control = admit(config, watch)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        head = make_head()
        initial = state_digest(head)
        head.to(config["device"])
        optimizer = torch.optim.Adam(
            head.parameters(),
            lr=3e-4,
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            foreach=False,
            fused=False,
        )
        best: dict[str, dict[str, Any]] | None = None
        best_epoch, updates, epoch_draws, epoch_loss = 0, 0, 0, 0.0
        selected_file_sha = ""
        history = []
        for epoch, indices in batches(cache.contract.fit_windows, config["passes"]):
            watch()
            head.train()
            optimizer.zero_grad(set_to_none=True)
            values = torch.from_numpy(cache.embeddings[indices]).to(config["device"])
            targets = torch.from_numpy(cache.targets[indices]).to(config["device"])
            weights = torch.from_numpy(cache.weights[indices]).to(config["device"])
            loss = weighted_ce(head(values), targets, weights)
            require(bool(torch.isfinite(loss)), "Nonfinite loss; no optimizer step")
            loss.backward()
            require(
                all(
                    p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in head.parameters()
                ),
                "Missing/nonfinite gradients; no optimizer step",
            )
            optimizer.step()
            committed_updates += 1
            committed_presentations += len(indices)
            if committed_updates % 256 == 0:
                write_json(
                    output / "progress.json",
                    {
                        "pass": epoch,
                        "committed_updates": committed_updates,
                        "committed_presentations": committed_presentations,
                        "config_sha256": config_sha256,
                    },
                )
            value = float(loss.detach().cpu())
            epoch_loss += value * len(indices)
            epoch_draws += len(indices)
            updates += 1
            if epoch_draws == cache.contract.fit_windows:
                logits = predict(head, cache, BATCH, watch)
                measured = panel_metrics(logits, cache)
                event = {
                    "pass": epoch,
                    "updates": updates,
                    "draws": epoch_draws,
                    "weighted_training_ce": epoch_loss / epoch_draws,
                    "metrics": measured,
                    "logits": logits,
                }
                write_json(output / f"event-{epoch:03d}.json", event)
                history.append({k: v for k, v in event.items() if k != "logits"})
                if best is None or selection_key(measured) > selection_key(best):
                    best, best_epoch = measured, epoch
                    torch.save(
                        {
                            "state_dict": head.state_dict(),
                            "pass": epoch,
                            "config_sha256": config_sha256,
                        },
                        output / "selected.pt.pending",
                    )
                    (output / "selected.pt.pending").replace(output / "selected.pt")
                    selected_file_sha = sha256(output / "selected.pt", watch)
                epoch_draws, epoch_loss = 0, 0.0
        require(
            best is not None and len(history) == config["passes"] and epoch_draws == 0,
            "Incomplete declared horizon",
        )
        assert best is not None
        torch.save(
            {"state_dict": head.state_dict(), "config_sha256": config_sha256}, output / "last.pt"
        )
        require(
            sha256(output / "selected.pt", watch) == selected_file_sha,
            "Selected checkpoint bytes changed",
        )
        selected = torch.load(
            output / "selected.pt", weights_only=True, map_location=config["device"]
        )
        head.load_state_dict(selected["state_dict"], strict=True)
        batched, serial = predict(head, cache, BATCH, watch), predict(head, cache, 1, watch)
        require(panel_metrics(batched, cache) == best, "Selected saved checkpoint does not replay")
        proof = batch_proof(serial, batched)
        write_json(
            output / "coverage.json",
            {
                "cache_manifest_sha256": config["inputs"]["cache"]["sha256"],
                "complete_passes": config["passes"],
                "fit_views": [
                    {
                        "id": view.identity,
                        "shift": view.shift,
                        "windows": view.windows,
                        "presentations": view.windows * config["passes"],
                        "window_loss_weight": float(loss_weight(view, cache.contract)),
                    }
                    for view in cache.contract.views
                    if view.role == "fit"
                ],
                "selection_original_only": True,
            },
        )
        result = {
            "status": "COMPLETE" if proof["passed"] else "COMPLETED_FIT_BATCH_PARITY_FAILED",
            "config_sha256": config_sha256,
            "initial_state_sha256": initial,
            "selected_state_sha256": state_digest(head),
            "selected_checkpoint_sha256": selected_file_sha,
            "last_checkpoint_sha256": sha256(output / "last.pt", watch),
            "selected_pass": best_epoch,
            "updates": updates,
            "presentations": config["passes"] * cache.contract.fit_windows,
            "history": history,
            "best_metrics": best,
            "selected_batch_proof": proof,
            "continuation_earned": continuation(best, control, proof["passed"]),
            "confirmation_scored": False,
            "release_or_distillation_authorized": False,
        }
        watch()
        write_json(output / "result.json", result)
        return result
    except BaseException as error:
        write_json(
            output / "failure.json",
            {
                "status": "FAILED_INCONCLUSIVE",
                "config_sha256": config_sha256,
                "error_type": type(error).__name__,
                "error": str(error),
                "automatic_resume": False,
                "committed_updates": committed_updates,
                "committed_presentations": committed_presentations,
            },
        )
        raise
