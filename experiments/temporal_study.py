"""Standalone fixed M/O/S full-sequence fitting; exact-first prospective selection."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from drone.features import audio_digest
from drone.keys import WEIGHTS, key_relation
from experiments.ema import read_reference
from experiments.k10_multidomain import domain_metrics
from experiments.octave import device_setup
from experiments.paired import movements
from experiments.small import atomic_json
from experiments.temporal_cache import (
    REVIEWED,
    identity,
    validate_cache,
    within_tolerance,
)
from experiments.temporal_model import (
    VERSION,
    TemporalHead,
    better,
    gates,
    initial_heads,
    microbatches,
    permutation,
)


def state_digest(state: dict[str, torch.Tensor]) -> str:
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(json.dumps(list(value.shape)).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def require_pilot(path: Path | None, own: dict[str, Any]) -> None:
    if path is None:
        raise ValueError("Confirmation requires passing canonical encoder13/head13 cohort")
    pilot = json.loads(path.read_bytes())
    if (
        pilot.get("version") != VERSION
        or pilot.get("encoder_seed") != 13
        or pilot.get("head_seed") != 13
        or pilot.get("status") != "complete"
        or pilot.get("identity") != own
        or not pilot.get("gates", {}).get("candidate_pass")
    ):
        raise ValueError("Pilot candidate gate did not qualify under same frozen identity")


def batch(
    cache: Path,
    rows: list[dict[str, Any]],
    indices: list[int],
    arm: str,
    seed: int,
    epoch: int | None,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, str]]:
    lengths = [rows[i]["hidden_shape"][2] for i in indices]
    padded = np.zeros((len(indices), 64, 24, max(lengths)), dtype=np.float32)
    hashes = {}
    for j, index in enumerate(indices):
        row = rows[index]
        h = np.load(cache / row["hidden"], mmap_mode="r", allow_pickle=False)
        if arm == "S":
            order = permutation(seed, row["id"], lengths[j], epoch)
            padded[j, :, :, : lengths[j]] = h[:, :, order]
            hashes[row["id"]] = hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()
        else:
            padded[j, :, :, : lengths[j]] = h
    return (
        torch.from_numpy(padded).to(device),
        torch.tensor(lengths, device=device),
        torch.tensor([rows[i]["ref"] for i in indices], device=device),
        hashes,
    )


def weighted_backward(
    head: TemporalHead,
    inputs: torch.Tensor,
    lengths: torch.Tensor,
    canonical: torch.Tensor,
    refs: torch.Tensor,
    logical_size: int,
) -> float:
    loss = nn.functional.cross_entropy(canonical + head(inputs, lengths), refs, reduction="sum")
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite head loss")
    (loss / logical_size).backward()
    return float(loss.detach())


def score(
    head: TemporalHead,
    cache: Path,
    rows: list[dict[str, Any]],
    canonical: np.ndarray,
    indices: list[int],
    arm: str,
    seed: int,
    device: str,
    check: Callable[[], None],
) -> tuple[np.ndarray, dict[str, Any], dict[str, float], dict[str, str]]:
    head.eval()
    output = np.empty((len(indices), 24), dtype=np.float32)
    position = {index: i for i, index in enumerate(indices)}
    lengths = [r["hidden_shape"][2] for r in rows]
    hashes = {}
    for group in microbatches(indices, lengths):
        check()
        x, n, _, perms = batch(cache, rows, group, arm, seed, None, device)
        with torch.inference_mode():
            values = torch.from_numpy(canonical[group]).to(device) + head(x, n)
        output[[position[i] for i in group]] = values.cpu().numpy()
        hashes.update(perms)
    if not np.isfinite(output).all():
        raise RuntimeError("Nonfinite development head logits")
    val_rows = [rows[i] for i in indices]
    refs = np.array([r["ref"] for r in val_rows])
    metrics = domain_metrics(output.argmax(1), refs, val_rows)
    ce = nn.functional.cross_entropy(
        torch.from_numpy(output), torch.from_numpy(refs), reduction="none"
    ).numpy()
    exact = 0.5 * sum(metrics[d]["exact"] for d in ("beatport", "fmak"))
    domain_ce = 0.5 * sum(
        float(ce[[r["domain"] == d for r in val_rows]].mean()) for d in ("beatport", "fmak")
    )
    return output, metrics, dict(exact=exact, ce=domain_ce), hashes


def save_head(path: Path, head: TemporalHead, metadata: dict[str, Any]) -> None:
    torch.save(
        dict(
            version=VERSION,
            kind="frozen-K10-residual-head",
            kernel=head.kernel,
            state_dict={k: v.detach().cpu() for k, v in head.state_dict().items()},
            state_sha256=state_digest(head.state_dict()),
            metadata=metadata,
        ),
        path,
    )


def load_head(path: Path, expected: dict[str, Any], device: str = "cpu") -> TemporalHead:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("version") != VERSION
        or payload.get("kind") != "frozen-K10-residual-head"
        or payload.get("metadata") != expected
        or payload.get("kernel") != (1 if expected.get("arm") == "M" else 5)
        or any(
            value.dtype != torch.float32 or not torch.isfinite(value).all()
            for value in payload["state_dict"].values()
        )
        or payload["state_sha256"] != state_digest(payload["state_dict"])
    ):
        raise ValueError("Head artifact/version/source/state identity changed")
    with torch.random.fork_rng(devices=[]):
        head = TemporalHead(payload["kernel"])
    head.load_state_dict(payload["state_dict"], strict=True)
    return head.to(device)


def verify_padding(
    heads: dict[str, TemporalHead], cache: Path, rows: list[dict[str, Any]], device: str, seed: int
) -> dict[str, Any]:
    fitting = [
        i for i, r in enumerate(rows) if r["role"] == "train" and r["hidden_shape"][2] <= 4096
    ]
    fitting = sorted(fitting, key=lambda i: rows[i]["hidden_shape"][2])
    if len(fitting) < 2:
        raise ValueError("Need at least two fitting sequences for bounded masking preflight")
    indices = [fitting[0], fitting[-1]]
    report = {}
    for arm, initial in heads.items():
        probe = copy.deepcopy(initial).to(device).eval()
        with torch.no_grad():
            probe.readout.weight.copy_(
                torch.linspace(-0.1, 0.1, 24 * 64).reshape(24, 64).to(device)
            )
        x, n, _, _ = batch(cache, rows, indices, arm, seed, None, device)
        with torch.inference_mode():
            padded = probe(x, n).cpu().numpy()
            singles = np.concatenate(
                [
                    probe(x[j : j + 1, :, :, : int(n[j])], n[j : j + 1]).cpu().numpy()
                    for j in range(2)
                ]
            )
        if not within_tolerance(padded, singles):
            raise ValueError("Nonzero-head padded-vs-single masking integrity failure")
        report[arm] = dict(
            ids=[rows[i]["id"] for i in indices],
            max_abs_error=float(np.max(np.abs(padded - singles))),
            nonzero_readout=True,
        )
    return report


def paired_uncertainty(
    control: dict[str, Any],
    candidate: dict[str, Any],
    refs: np.ndarray,
    rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """Fixed2000 paired track bootstrap; descriptive, never a selection/gate."""
    result = {}
    for domain in ("beatport", "fmak"):
        indices = [i for i, r in enumerate(rows) if r["domain"] == domain]
        y = refs[indices]
        a = np.asarray(control[domain]["predictions"])
        b = np.asarray(candidate[domain]["predictions"])
        digest = hashlib.sha256(
            json.dumps([VERSION, "paired-uncertainty", seed, domain]).encode()
        ).digest()
        draws = np.random.default_rng(int.from_bytes(digest, "big")).integers(
            0, len(y), (2000, len(y))
        )
        statistics = {}
        for name, symmetric in (("exact", True), ("mirex", True), ("mirex_asymmetric", False)):
            if name == "exact":
                delta = (b == y).astype(float) - (a == y).astype(float)
            else:
                delta = np.array(
                    [
                        WEIGHTS[key_relation(int(new), int(ref), symmetric=symmetric)]
                        - WEIGHTS[key_relation(int(old), int(ref), symmetric=symmetric)]
                        for old, new, ref in zip(a, b, y, strict=True)
                    ]
                )
            interval = np.percentile(delta[draws].mean(1), [2.5, 97.5], method="linear")
            statistics[name] = dict(
                delta=float(delta.mean()), paired_track_bootstrap_95=interval.tolist()
            )
        result[domain] = dict(
            statistics=statistics,
            promotions=int(((b == y) & (a != y)).sum()),
            demotions=int(((a == y) & (b != y)).sum()),
            resamples=2000,
            limitation="Reused-development track bootstrap; ignores artist dependence and checkpoint-selection bias; not ensemble/cohort uncertainty",
        )
    return result


def confirmation_summary(paths: list[Path]) -> dict[str, Any]:
    results = [json.loads(path.read_bytes()) for path in paths]
    if len(results) != 3 or [r["encoder_seed"] for r in results] != [13, 37, 73]:
        raise ValueError("Need exactly matched ordered(encoder13,head13),(37,37),(73,73) cohorts")
    if not results[0]["gates"]["candidate_pass"]:
        raise ValueError("Pilot O did not qualify")
    for result in results:
        if (
            result.get("head_seed") != result["encoder_seed"]
            or result["status"] != "complete"
            or result["version"] != VERSION
            or result["identity"] != results[0]["identity"]
            or result["bp_references"] != results[0]["bp_references"]
            or result["bp_control_mean"] != results[0]["bp_control_mean"]
        ):
            raise ValueError("Confirmation source/protocol/cohort differs")
    delta = {
        d: {
            m: float(
                np.mean(
                    [
                        r["arms"]["O"]["metrics"][d][m] - r["canonical_metrics"][d][m]
                        for r in results
                    ]
                )
            )
            for m in ("exact", "mirex", "mirex_asymmetric")
        }
        for d in ("beatport", "fmak")
    }

    def net_correct(domain: str, reference: str) -> int:
        return sum(
            round(
                r["arms"]["O"]["metrics"][domain]["exact"]
                * r["arms"]["O"]["metrics"][domain]["tracks"]
            )
            - round(
                (
                    r["canonical_metrics"]
                    if reference == "canonical"
                    else r["arms"][reference]["metrics"]
                )[domain]["exact"]
                * r["arms"]["O"]["metrics"][domain]["tracks"]
            )
            for r in results
        )

    def net_credit(domain: str) -> int:
        return sum(
            round(
                r["arms"]["O"]["metrics"][domain]["mirex"]
                * r["arms"]["O"]["metrics"][domain]["tracks"]
                * 10
            )
            - round(
                r["canonical_metrics"][domain]["mirex"]
                * r["canonical_metrics"][domain]["tracks"]
                * 10
            )
            for r in results
        )

    candidate = (
        net_correct("fmak", "canonical") >= 15
        and net_correct("beatport", "canonical") >= 0
        and all(net_credit(d) >= 0 for d in delta)
    )
    mechanism = {arm: net_correct("fmak", arm) for arm in ("M", "S")}
    bpdelta = {
        d: float(np.mean([r["arms"]["O"]["metrics"][d]["mirex"] for r in results]))
        - results[0]["bp_control_mean"][d]["mirex"]
        for d in delta
    }
    return dict(
        version=VERSION,
        matched_cohorts=[[13, 13], [37, 37], [73, 73]],
        mean_paired_delta=delta,
        candidate_pass=candidate,
        mechanism_pass=all(value >= 9 for value in mechanism.values()),
        mechanism_fmak_net_exact_counts=mechanism,
        mean_ordered_control_delta={
            arm: {
                d: {
                    m: float(
                        np.mean(
                            [
                                r["arms"]["O"]["metrics"][d][m] - r["arms"][arm]["metrics"][d][m]
                                for r in results
                            ]
                        )
                    )
                    for m in ("exact", "mirex", "mirex_asymmetric")
                }
                for d in ("beatport", "fmak")
            }
            for arm in ("M", "S")
        },
        cohorts=[
            dict(
                encoder_seed=r["encoder_seed"],
                head_seed=r["head_seed"],
                canonical_metrics=r["canonical_metrics"],
                arms=r["arms"],
                gates=r["gates"],
            )
            for r in results
        ],
        bp_mean_hurdle_delta=bpdelta,
        bp_mean_hurdle_pass=bpdelta["fmak"] >= 0.02 and bpdelta["beatport"] >= 0,
        result_sha256={str(p): audio_digest(p) for p in paths},
        promoted=False,
        uncertainty="Three matched encoder/head cohorts on reused development, not independent test evidence",
    )


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise ValueError("Use fresh head study output")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    proof = json.loads((args.cache / "reconstruction.json").read_bytes())
    aggregate = proof["aggregate_started_epoch"]
    phase_start = started

    def check() -> None:
        if time.monotonic() - phase_start > 600 or time.time() - aggregate > 3600:
            raise TimeoutError("Fixed10-minute head /60-minute aggregate budget exhausted")

    manifest, canonical = validate_cache(args.cache, check)
    seed = manifest["encoder_seed"]
    own = identity()
    if seed != 13:
        require_pilot(args.pilot, own)
    rows = manifest["rows"]
    fitting = [i for i, r in enumerate(rows) if r["role"] == "train"]
    validation = [i for i, r in enumerate(rows) if r["role"] == "validation"]
    val_rows = [rows[i] for i in validation]
    refs = np.array([r["ref"] for r in val_rows])
    canonical_metrics = domain_metrics(canonical[validation].argmax(1), refs, val_rows)
    bp = []
    bp_references = {}
    approved = json.loads(REVIEWED.read_bytes())["archived"]["sources"]
    for path, expected_seed in zip(args.bp_runs, (13, 37, 73), strict=True):
        control = read_reference(path, "control", expected_seed)
        if (
            control["config"] != manifest["canonical_config"]
            or control["fit_ids"]
            != [r["id"] for r in rows if r["role"] == "train" and r["domain"] == "beatport"]
            or control["source_sha256"] != approved
            or control["feature_manifest_sha256"] != manifest["feature_manifest_sha256"]
            or control["validation_ids"] != [r["id"] for r in val_rows]
            or audio_digest(path.with_name("best.pt")) != control["checkpoint_sha256"]
        ):
            raise ValueError("Frozen BP reference provenance changed")
        bp.append(control["domain_metrics"])
        bp_references[str(expected_seed)] = dict(
            run_sha256=audio_digest(path), checkpoint_sha256=control["checkpoint_sha256"]
        )
    heads = initial_heads(seed)
    initial_hashes = {arm: state_digest(head.state_dict()) for arm, head in heads.items()}
    if initial_hashes["O"] != initial_hashes["S"]:
        raise ValueError("Ordered/shuffled initialization mismatch")
    for arm, head in heads.items():
        save_head(
            args.output / f"initial-{arm}.pt",
            head,
            dict(identity=own, arm=arm, encoder_seed=seed, initial=True),
        )
    device_setup(args.device)
    preflight = verify_padding(heads, args.cache, rows, args.device, seed)
    # All2582 zero residual cases, no labels consulted. Includes full S permutations.
    lengths = [r["hidden_shape"][2] for r in rows]
    for arm, head in heads.items():
        head.to(args.device).eval()
        for group in microbatches(range(len(rows)), lengths):
            check()
            x, n, _, _ = batch(args.cache, rows, group, arm, seed, None, args.device)
            with torch.inference_mode():
                residual = head(x, n).cpu().numpy()
            if not np.array_equal(residual, np.zeros_like(residual)) or not np.array_equal(
                canonical[group] + residual, canonical[group]
            ):
                raise ValueError("Zero-head canonical reproduction failed")
        head.cpu()
    atomic_json(
        args.output / "identity.json",
        dict(
            version=VERSION,
            identity=own,
            encoder_seed=seed,
            cache_manifest_sha256=audio_digest(args.cache / "manifest.json"),
            reconstruction_sha256=audio_digest(args.cache / "reconstruction.json"),
            initial_state_sha256=initial_hashes,
            padding_preflight=preflight,
            zero_head_recordings_per_arm=2582,
            aggregate_started_epoch=aggregate,
            config=dict(
                epochs=60,
                patience=12,
                batch=32,
                microbatch_padded_tokens=8192,
                lr=0.0003,
                min_lr=0.000003,
                weight_decay=0.001,
                clip=1.0,
                selection="balanced-domain-exact,balanced-domain-CE,earlier-epoch",
            ),
        ),
    )
    results: dict[str, Any] = {}
    order_hashes = {}
    for arm, head in heads.items():
        phase_start = time.monotonic()
        device_setup(args.device)
        free, total = torch.cuda.mem_get_info()
        torch.cuda.reset_peak_memory_stats()
        head.to(args.device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=3e-6)
        rng = np.random.default_rng(seed)
        history = []
        schedules = []
        best = None
        best_epoch = 0
        best_logits = None
        best_metrics = None
        metadata = dict(
            identity=own,
            arm=arm,
            encoder_seed=seed,
            cache_manifest_sha256=audio_digest(args.cache / "manifest.json"),
        )
        for epoch in range(1, 61):
            head.train()
            order = rng.permutation(fitting).tolist()
            loss_sum = 0.0
            perms = {}
            order_hash = hashlib.sha256(np.asarray(order, dtype="<i8").tobytes()).hexdigest()
            if epoch in order_hashes and order_hashes[epoch] != order_hash:
                raise ValueError("Arm recording orders diverged")
            order_hashes[epoch] = order_hash
            for offset in range(0, len(order), 32):
                logical = order[offset : offset + 32]
                optimizer.zero_grad()
                for group in microbatches(logical, lengths):
                    check()
                    x, n, y, hashes = batch(args.cache, rows, group, arm, seed, epoch, args.device)
                    perms.update(hashes)
                    loss_sum += weighted_backward(
                        head,
                        x,
                        n,
                        torch.from_numpy(canonical[group]).to(args.device),
                        y,
                        len(logical),
                    )
                nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            scheduler.step()
            logits, metrics, selection, dev_perms = score(
                head, args.cache, rows, canonical, validation, arm, seed, args.device, check
            )
            selection["epoch"] = float(epoch)
            selected = better(selection, best)
            history.append(
                dict(
                    epoch=epoch,
                    loss=loss_sum / len(fitting),
                    selection=selection,
                    selected=selected,
                    metrics=metrics,
                )
            )
            schedules.append(
                dict(
                    epoch=epoch,
                    recording_order_sha256=order_hash,
                    fitting_permutations=perms,
                    development_permutations=dev_perms,
                )
            )
            if selected:
                best = selection
                best_epoch = epoch
                best_logits = logits
                best_metrics = metrics
                save_head(
                    args.output / f"best-{arm}.pt",
                    head,
                    dict(metadata, selected_epoch=epoch, selection=selection),
                )
            atomic_json(args.output / f"history-{arm}.json", history)
            atomic_json(args.output / f"schedules-{arm}.json", schedules)
            check()
            if epoch - best_epoch >= 12:
                break
        restored = load_head(
            args.output / f"best-{arm}.pt",
            dict(metadata, selected_epoch=best_epoch, selection=best),
            args.device,
        )
        actual, metrics, _, _ = score(
            restored, args.cache, rows, canonical, validation, arm, seed, args.device, check
        )
        if not np.array_equal(actual, best_logits) or metrics != best_metrics:
            raise ValueError("Head checkpoint roundtrip scoring changed")
        atomic_json(
            args.output / f"validation-{arm}.json",
            dict(
                ids=[r["id"] for r in val_rows],
                audio_sha256=[r["sha256"] for r in val_rows],
                refs=refs.tolist(),
                logits=actual.tolist(),
                metrics=metrics,
            ),
        )
        check()
        results[arm] = dict(
            metrics=metrics,
            best_epoch=best_epoch,
            selected_criterion=best,
            stop_epoch=epoch,
            history_sha256=audio_digest(args.output / f"history-{arm}.json"),
            schedules_sha256=audio_digest(args.output / f"schedules-{arm}.json"),
            checkpoint_sha256=audio_digest(args.output / f"best-{arm}.pt"),
            validation_sha256=audio_digest(args.output / f"validation-{arm}.json"),
            elapsed_seconds=time.monotonic() - phase_start,
            parameters=sum(p.numel() for p in head.parameters()),
            memory=dict(
                free_bytes=free,
                total_bytes=total,
                allocator_fraction=0.04,
                actual_cap_bytes=int(total * 0.04),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            ),
        )
        head.cpu()
        del restored, optimizer, scheduler
    outputs = {arm: r["metrics"] for arm, r in results.items()}
    decision = gates(canonical_metrics, outputs)
    bpmean = {
        d: {
            m: float(np.mean([r[d][m] for r in bp])) for m in ("mirex", "mirex_asymmetric", "exact")
        }
        for d in ("beatport", "fmak")
    }
    bpdelta = {d: outputs["O"][d]["mirex"] - bpmean[d]["mirex"] for d in bpmean}
    atomic_json(
        args.output / "result.json",
        dict(
            version=VERSION,
            status="complete",
            identity=own,
            encoder_seed=seed,
            head_seed=seed,
            cache_manifest_sha256=metadata["cache_manifest_sha256"],
            canonical_metrics=canonical_metrics,
            arms=results,
            gates=decision,
            bp_control_mean=bpmean,
            bp_references=bp_references,
            bp_reference_hurdle_delta=bpdelta,
            bp_reference_hurdle_pass=bpdelta["fmak"] >= 0.02 and bpdelta["beatport"] >= 0,
            bp_reference_estimand="One encoder/head cohort vs frozen three-BP-control mean; matched-cohort means remain separate",
            confusion_movements={
                arm: movements(canonical_metrics, outputs[arm], refs, val_rows) for arm in outputs
            },
            ordered_contrasts={
                arm: dict(
                    delta={
                        d: {
                            m: outputs["O"][d][m] - outputs[arm][d][m]
                            for m in ("exact", "mirex", "mirex_asymmetric")
                        }
                        for d in ("beatport", "fmak")
                    },
                    confusion=movements(outputs[arm], outputs["O"], refs, val_rows),
                    paired_uncertainty=paired_uncertainty(
                        outputs[arm], outputs["O"], refs, val_rows, seed
                    ),
                )
                for arm in ("M", "S")
            },
            paired_uncertainty={
                arm: paired_uncertainty(canonical_metrics, outputs[arm], refs, val_rows, seed)
                for arm in outputs
            },
            elapsed_seconds=time.monotonic() - started,
            aggregate_elapsed_seconds=time.time() - aggregate,
            promoted=False,
        ),
    )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        parser = argparse.ArgumentParser("Matched-cohort temporal confirmation summary")
        parser.add_argument("results", type=Path, nargs=3)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        atomic_json(args.output, confirmation_summary(args.results))
        return
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bp-runs", type=Path, nargs=3, required=True)
    p.add_argument("--pilot", type=Path)
    p.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    args = p.parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError, TimeoutError, OSError, MemoryError) as error:
        if args.output.is_dir():
            atomic_json(
                args.output / "failure.json",
                dict(
                    version=VERSION,
                    status="integrity_or_resource_inconclusive",
                    error=str(error),
                    promoted=False,
                ),
            )
        raise


if __name__ == "__main__":
    main()
