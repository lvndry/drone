"""Metadata plan and separately authorized cached-only BP-to-GS overlap audit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import resource
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.bp_gs_scope import RULE, classify, outcome, scope
from experiments.fma_alignment_matcher import MATCHER_CONTRACT, AllOffsetIndex
from experiments.fma_fingerprints import CONTRACT, HOP, SR, chromaprint_distance

ROOT = Path(__file__).resolve().parents[1]
VERSION = "bp-gs-cached-overlap-v1"
FULL_SHA = "5b3d7296f8277d8b43470736e670d35e58c2c94230b0f95dc53172d33787ed14"
INDEX_SHA = "b2609bd2b7184291e7604c74887feae6dd399c1bd4ca696c6f616ec27d155148"
PREPARED_SHA = "39fe1842a544bd818fe48c38d9b0916d7f6c04f5531894b0396d6e6d24d1f5f5"
CACHE_ROOT = "/home/lysk-admin/drone-experiments/fma-audio-admission-20261112-v1"
PROTOCOL = "experiments/studies/bp-gs-overlap/PROTOCOL.md"
WALL_SECONDS, CPU_SECONDS, RSS_KIB, DISK_BYTES = 3600, 7200, 2 * 1024**2, 512 * 1024**2


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def atomic(path: Path, value: dict[str, Any]) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    pending.replace(path)


def sources() -> dict[str, str]:
    return {
        name: digest(ROOT / name)
        for name in (
            "experiments/fma_fingerprints.py",
            "experiments/fma_alignment_matcher.py",
            "experiments/bp_gs_scope.py",
            "tools/audit_bp_gs.py",
            "tools/collect_bp_lineage.py",
            PROTOCOL,
        )
    }


def normalize(ids: list[str]) -> set[str]:
    return {
        "beatport:" + str(int(i.split(":")[-1]))
        for i in ids
        if ":" not in i or i.startswith("beatport:")
    }


def metadata_plan(
    full: dict[str, Any],
    prepared: dict[str, Any],
    collector: dict[str, Any],
    release: dict[str, Any],
    catalog: dict[str, Any],
    witness: dict[str, Any],
    historical: dict[str, Any],
) -> dict[str, Any]:
    if (
        not historical["all244_complete"]
        or historical["expected_positive_retrievals"] != 180
        or historical["negative_identity_queries"] != 0
        or historical["historical_v2_passed"] is not False
        or historical["index_sha256_before"] != INDEX_SHA
        or historical["index_sha256_after"] != INDEX_SHA
    ):
        raise ValueError("Preserved control operational/source evidence differs")
    records = full["records"]
    if (
        full["references"] != 7579
        or not full["complete_coverage"]
        or full["contract"] != CONTRACT
        or len(records) != 7579
        or len({r["ref_id"] for r in records}) != 7579
        or any(r["status"] != "complete" for r in records)
    ):
        raise ValueError("Full frozen extraction coverage differs")
    bp = sorted((r for r in records if r["domain"] == "beatport"), key=lambda r: r["ref_id"])
    gs = sorted((r for r in records if r["domain"] == "giantsteps"), key=lambda r: r["ref_id"])
    if (
        len(bp) != 1486
        or len(gs) != 604
        or {r["id"] for r in bp} != set(catalog["ids"])
        or [r["ref_id"] for r in gs] != list(range(6975, 7579))
    ):
        raise ValueError("Official/raw/reference target membership differs")
    if (
        witness["index_sha256"] != INDEX_SHA
        or witness["extraction_contract"] != CONTRACT
        or witness["rule"] != RULE
        or len(witness["reference_artifact_sha256"]) != 7579
    ):
        raise ValueError("Original index/artifact/policy witness differs")
    prepared_bp = {r["id"]: r for r in prepared["rows"] if r["domain"] == "beatport"}
    if len(prepared_bp) != 1356 or sum(r["role"] == "train" for r in prepared_bp.values()) != 1230:
        raise ValueError("Prepared population differs")
    by_id = {r["id"]: r for r in bp}
    if any(
        identity not in by_id or by_id[identity]["container_sha256"] != row["sha256"]
        for identity, row in prepared_bp.items()
    ):
        raise ValueError("Prepared-to-reference audio metadata join differs")
    label_map: dict[str, list[dict[str, Any]]] = {}
    for row in collector["labels"]:
        label_map.setdefault(row["id"], []).append(row)
    models: list[dict[str, Any]] = [
        dict(
            model_id="release",
            fit_ids=release["metadata"]["train_ids"],
            validation_ids=release["metadata"]["validation_ids"],
            checkpoint_sha256=release["checkpoint_sha256"],
            provenance="checkpoint-declared IDs; generic audio witness reported separately",
        ),
        *collector["models"],
    ]
    if {m["model_id"] for m in models} != {
        "release",
        "control13",
        "control37",
        "control73",
        "joint13",
        "joint37",
        "joint73",
    }:
        raise ValueError("Require release plus six actual control/joint witnesses")
    exposures = {
        m["model_id"]: dict(
            fitting=normalize(m["fit_ids"]), validation=normalize(m["validation_ids"])
        )
        for m in models
    }
    if any(not (e["fitting"] | e["validation"]).issubset(by_id) for e in exposures.values()):
        raise ValueError("Model ID exposure is not covered by raw BP index")
    parseable_ids = {
        identity
        for identity, labels in label_map.items()
        if any(row["parseable"] for row in labels)
    }
    released_ids = exposures["release"]["fitting"] | exposures["release"]["validation"]
    if set(release["metadata"]["training_audio_sha256"]) != {
        by_id[i]["container_sha256"] for i in released_ids
    }:
        raise ValueError("Release prepared-audio witness and cached recording lineage differ")
    if parseable_ids != released_ids or set(label_map) != set(by_id):
        raise ValueError("Source-label eligibility and released preparation lineage differ")
    if any(
        model["feature_manifest_sha256"] != PREPARED_SHA
        or model["feature_spec"] != prepared["feature_spec"]
        or model["label_schema"] != prepared["label_schema"]
        for model in collector["models"]
    ):
        raise ValueError("Candidate feature/source witness differs")
    alias_index: dict[str, Any] = {}
    for model, roles in exposures.items():
        alias_index[model] = {}
        for role, ids in roles.items():
            alias_index[model][role] = {"container_sha256": {}, "pcm_sha256": {}}
            for identity in sorted(ids):
                for key in ("container_sha256", "pcm_sha256"):
                    alias_index[model][role][key].setdefault(by_id[identity][key], []).append(
                        identity
                    )
    rows = []
    for record in bp:
        identity = record["id"]
        exposure = {}
        for model, roles in exposures.items():
            exposure[model] = {
                role: {
                    "direct_id": identity in ids,
                    "container_aliases": alias_index[model][role]["container_sha256"].get(
                        record["container_sha256"], []
                    ),
                    "pcm_aliases": alias_index[model][role]["pcm_sha256"].get(
                        record["pcm_sha256"], []
                    ),
                }
                for role, ids in roles.items()
            }
        exact = {
            key: [g["id"] for g in gs if g[key] == record[key]]
            for key in ("container_sha256", "pcm_sha256")
        }
        rows.append(
            dict(
                record={
                    k: record[k]
                    for k in (
                        "ref_id",
                        "id",
                        "artifact",
                        "container_sha256",
                        "pcm_sha256",
                        "hashes",
                        "chromaprint_items",
                        "seconds",
                    )
                },
                fingerprint_sha256=witness["reference_artifact_sha256"][record["artifact"]],
                prepared_role=prepared_bp.get(identity, {}).get("role"),
                source_label_files=label_map.get(identity, []),
                label_eligibility="parseable"
                if any(r["parseable"] for r in label_map.get(identity, []))
                else "unparseable_or_missing",
                exact_gs_metadata=exact,
                model_exposure=exposure,
            )
        )
    release_ids = exposures["release"]["fitting"] | exposures["release"]["validation"]
    counts = dict(
        official_bp_ids=len(catalog["ids"]),
        indexed_bp_ids=len(bp),
        indexed_bp_container_hashes=len({r["container_sha256"] for r in bp}),
        indexed_bp_pcm_hashes=len({r["pcm_sha256"] for r in bp}),
        prepared_ids=len(prepared_bp),
        prepared_fitting=1230,
        prepared_validation=126,
        release_fitting_ids=len(exposures["release"]["fitting"]),
        release_validation_ids=len(exposures["release"]["validation"]),
        release_union_ids=len(release_ids),
        raw_outside_release=len(set(by_id) - release_ids),
        source_parseable_ids=len(parseable_ids),
        source_unparseable_ids=len(label_map) - len(parseable_ids),
        release_fit_container_hashes=len(
            {by_id[i]["container_sha256"] for i in exposures["release"]["fitting"]}
        ),
        release_train_validation_container_intersection=len(
            {by_id[i]["container_sha256"] for i in exposures["release"]["fitting"]}
            & {by_id[i]["container_sha256"] for i in exposures["release"]["validation"]}
        ),
        release_outside_prepared=len(release_ids - set(prepared_bp)),
        gs_target_recordings=604,
        gs_distinct_pcm_hashes=len({r["pcm_sha256"] for r in gs}),
        bp_queries=1486,
        transformed_positive_controls=180,
        target_self_controls=604,
        synthetic_negative_controls=64,
        total_planned_queries=2334,
    )
    return dict(
        version=VERSION,
        counts=counts,
        queries=rows,
        gs_records=[
            {k: r[k] for k in ("ref_id", "id", "artifact", "container_sha256", "pcm_sha256")}
            for r in gs
        ],
        model_witnesses=models,
        source_label_parse_reasons=collector["parse_reason_counts"],
        release_omitted_aliases=[
            dict(
                id=i,
                role="release_fitting"
                if i in exposures["release"]["fitting"]
                else "release_validation",
                prepared_container_aliases=[
                    j
                    for j in prepared_bp
                    if by_id[j]["container_sha256"] == by_id[i]["container_sha256"]
                ],
            )
            for i in sorted(release_ids - set(prepared_bp))
        ],
        release_audio_witness=release["metadata"].get("training_audio_sha256"),
        release_audio_witness_caveat="Field covers prepared audio, not an independently established per-gradient exposure ledger",
        dictionary=[
            dict(ref_id=r["ref_id"], id=r["id"], hashes=r["hashes"])
            for r in sorted(records, key=lambda r: r["ref_id"])
        ],
        artifact_sha256=witness["reference_artifact_sha256"],
        controls=[
            {key: row[key] for key in ("id", "panel", "expected_ref", "path", "sha256")}
            for row in witness["controls"]
        ],
        historical_control_evidence=historical,
        extraction_contract=CONTRACT,
        matcher_contract=MATCHER_CONTRACT,
        rule=RULE,
        index_sha256=INDEX_SHA,
        full_reference_result_sha256=FULL_SHA,
        cache_root=CACHE_ROOT,
        source_sha256=sources(),
        real_queries_executed=False,
        audio_opened=False,
        query_feature_origin="Cached reference landmarks: one reference shift, not re-extracted four-shift queries",
        resource_projection=dict(
            historical244_controls_seconds=303.0484818480909,
            count_scaled_seconds=303.0484818480909 / 244 * 2334,
            limitation="Conservative count scaling; scoped postings and reference/query hash counts differ. Not a measured runtime guarantee.",
            wall_cap_seconds=WALL_SECONDS,
            cpu_cap_seconds=CPU_SECONDS,
            rss_bytes=RSS_KIB * 1024,
            new_output_bytes=DISK_BYTES,
        ),
    )


def prepare(args: argparse.Namespace) -> None:
    full_raw = gzip.decompress(args.references.read_bytes())
    if hashlib.sha256(full_raw).hexdigest() != FULL_SHA or digest(args.prepared) != PREPARED_SHA:
        raise ValueError("Frozen reference/prepared metadata hashes differ")
    import torch

    release = read(args.release)
    released_path = ROOT / "weights/drone-v1.pt"
    if digest(released_path) != release["checkpoint_sha256"]:
        raise ValueError("Actual released checkpoint SHA differs from exposure export")
    payload = torch.load(released_path, map_location="cpu", weights_only=True)
    if any(payload["metadata"].get(key) != value for key, value in release["metadata"].items()):
        raise ValueError("Released checkpoint metadata differs from exposure export")
    del payload
    plan = metadata_plan(
        json.loads(full_raw),
        read(args.prepared),
        read(args.collector),
        release,
        read(args.catalog),
        read(args.witness),
        read(args.historical_controls),
    )
    witness_raw = args.witness.read_bytes()
    witness_decoded = gzip.decompress(witness_raw) if args.witness.suffix == ".gz" else witness_raw
    if hashlib.sha256(witness_decoded).hexdigest() != read(args.historical_controls)["plan_sha256"]:
        raise ValueError("Original control plan/dictionary witness hash differs")
    collector_provenance = read(args.collector_provenance)
    if (
        collector_provenance["exit_code"] != 0
        or collector_provenance["stdout_sha256"] != digest(args.collector)
        or collector_provenance["source_sha256"] != digest(ROOT / "tools/collect_bp_lineage.py")
    ):
        raise ValueError("Metadata collector source/output provenance differs")
    plan["input_artifacts"] = {
        name: dict(path=str(getattr(args, name)), sha256=digest(getattr(args, name)))
        for name in (
            "references",
            "prepared",
            "collector",
            "collector_provenance",
            "release",
            "catalog",
            "witness",
            "historical_controls",
        )
    }
    if args.output.exists():
        raise ValueError("Keep old plans; fresh output required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic(args.output, plan)


def verify_dictionary(index: AllOffsetIndex, expected: list[dict[str, Any]]) -> None:
    actual = index.db.execute("SELECT ref,id,hashes FROM references_done ORDER BY ref").fetchall()
    if actual != [(r["ref_id"], r["id"], r["hashes"]) for r in expected]:
        raise ValueError("Reference dictionary identity differs")


def match_query(
    index: AllOffsetIndex, path: Path, references: Path, deadline: float
) -> tuple[dict[str, Any], bool]:
    with np.load(path, allow_pickle=False) as query:
        anchors = np.unique(query["landmarks"][:, 0])
        informative = (
            len(anchors) >= 20
            and len(query["chromaprint"]) >= 8
            and (int(anchors.max()) - int(anchors.min())) * HOP / SR >= 8
        )
        matched = index.match(query["landmarks"], deadline=deadline)
        for match in matched["matches"]:
            cp = None
            if (
                match["count"] >= RULE["review_anchors"]
                and match["span_seconds"] >= RULE["review_span"]
            ):
                with np.load(
                    references / f"{match['ref_id']:06}.npz", allow_pickle=False
                ) as target:
                    cp = chromaprint_distance(
                        query["chromaprint"],
                        target["chromaprint"],
                        match["offset_seconds"],
                        float(query["chromaprint_step"]),
                    )
            match.update(chromaprint_distance=cp, decision=classify(match, cp))
    return matched, bool(informative)


def guard(output: Path, began: float, cpu: float, stop: threading.Event) -> None:
    while not stop.wait(0.25):
        try:
            used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if (
                used > DISK_BYTES
                or rss > RSS_KIB
                or time.monotonic() - began > WALL_SECONDS
                or time.process_time() - cpu > CPU_SECONDS
            ):
                atomic(
                    output / "budget-failure.json",
                    dict(
                        status="incomplete",
                        disk_bytes=used,
                        peak_rss_kib=rss,
                        elapsed=time.monotonic() - began,
                        remaining_queries_are_not_negatives=True,
                    ),
                )
                os._exit(70)
        except FileNotFoundError:
            continue


def run(args: argparse.Namespace) -> None:
    if digest(args.plan) != args.plan_sha256:
        raise ValueError("Reviewed plan SHA differs")
    plan = read(args.plan)
    if (
        plan["source_sha256"] != sources()
        or plan["version"] != VERSION
        or plan["rule"] != RULE
        or plan["extraction_contract"] != CONTRACT
        or plan["matcher_contract"] != MATCHER_CONTRACT
    ):
        raise ValueError("Frozen source/policy identity differs")
    if (
        not hasattr(os, "sched_getaffinity")
        or not set(os.sched_getaffinity(0)).issubset({4, 5})
        or any(
            os.environ.get(n) != "1"
            for n in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        )
    ):
        raise ValueError("Require CPU4/5 isolation and four native thread limits of one")
    if args.output.exists():
        raise ValueError("No overwrite/resume of earlier outcomes")
    args.output.mkdir(parents=True)
    began, cpu, stop = time.monotonic(), time.process_time(), threading.Event()
    monitor = threading.Thread(target=guard, args=(args.output, began, cpu, stop), daemon=True)
    monitor.start()
    index = None
    completed: list[dict[str, Any]] = []
    try:
        root = Path(plan["cache_root"])
        full = read(root / "references-v1/result.json")
        if digest(root / "references-v1/result.json") != FULL_SHA:
            raise ValueError("Reference descriptor bytes differ")
        index = AllOffsetIndex.from_verified_artifacts(
            root / "references-v1/index.sqlite",
            descriptor=full,
            query_extraction_contract=CONTRACT,
            expected_index_sha256=INDEX_SHA,
            fingerprint_root=root / "references-v1/fingerprints",
            artifact_sha256=plan["artifact_sha256"],
        )
        verify_dictionary(index, plan["dictionary"])
        for control in plan["controls"]:
            path = Path(control["path"])
            if (
                not path.resolve().is_relative_to(root.resolve())
                or path.suffix != ".npz"
                or digest(path) != control["sha256"]
            ):
                raise ValueError("Cached control query identity/path differs")
        for model in plan["model_witnesses"]:
            if model["model_id"] != "release" and (
                digest(Path(model["run_path"])) != model["run_sha256"]
                or digest(Path(model["checkpoint_path"])) != model["checkpoint_sha256"]
            ):
                raise ValueError("Model exposure witness changed")
        fp = root / "references-v1/fingerprints"
        positives = [
            dict(c, kind="transformed", targets=[c["expected_ref"]])
            for c in plan["controls"]
            if c["panel"] != "negative"
        ]
        gsids = [r["ref_id"] for r in plan["gs_records"]]
        controls = (
            positives
            + [
                dict(
                    id="self:" + r["id"],
                    path=str(fp / r["artifact"]),
                    sha256=plan["artifact_sha256"][r["artifact"]],
                    expected_ref=r["ref_id"],
                    kind="target-self",
                    targets=gsids,
                )
                for r in plan["gs_records"]
            ]
            + [
                dict(c, kind="synthetic-negative", targets=gsids)
                for c in plan["controls"]
                if c["panel"] == "negative"
            ]
        )
        audit = [
            dict(
                id=r["record"]["id"],
                path=str(fp / r["record"]["artifact"]),
                sha256=r["fingerprint_sha256"],
                kind="bp-audit",
                targets=gsids,
                exact=r["exact_gs_metadata"],
            )
            for r in plan["queries"]
        ]
        if len(controls) != 848 or len(audit) != 1486:
            raise ValueError("Fixed control/query population differs")
        for number, query in enumerate(controls + audit):
            if number == 848:
                positives_ok = all(
                    r["complete"] and r["expected_retrieved"]
                    for r in completed
                    if r["kind"] != "synthetic-negative"
                )
                negatives_ok = all(
                    r["complete"] and not r["identity_flagged"]
                    for r in completed
                    if r["kind"] == "synthetic-negative"
                )
                if not (positives_ok and negatives_ok):
                    raise ValueError("Target-scope controls failed; BP queries not executed")
            path = Path(query["path"])
            if digest(path) != query["sha256"]:
                raise ValueError("Cached query fingerprint changed")
            scope(index, root / "references-v1/index.sqlite", query["targets"])
            matched, informative = match_query(
                index, path, fp, min(began + WALL_SECONDS, time.monotonic() + 120)
            )
            raw = json.dumps(
                dict(id=query["id"], query_sha256=query["sha256"], match=matched), sort_keys=True
            ).encode()
            evidence = args.output / f"{number:06}.json.gz"
            compressed = gzip.compress(raw, mtime=0)
            used = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
            if used + len(compressed) > DISK_BYTES - 8 * 1024**2:
                raise ValueError("Evidence output cap reached; no offset pruning permitted")
            evidence.write_bytes(compressed)
            decisions = [m["decision"] for m in matched["matches"]]
            completed.append(
                dict(
                    id=query["id"],
                    kind=query["kind"],
                    complete=matched["complete"],
                    reason=matched.get("reason"),
                    expected_retrieved=any(
                        m["ref_id"] == query.get("expected_ref")
                        and m["decision"] == "identity_evidence"
                        for m in matched["matches"]
                    ),
                    identity_flagged="identity_evidence" in decisions,
                    review_flagged="review" in decisions,
                    outcome=outcome(
                        exact=any(query.get("exact", {}).values()),
                        complete=matched["complete"],
                        informative=informative,
                        decisions=decisions,
                    ),
                    alignments=len(matched["matches"]),
                    evidence=evidence.name,
                    evidence_sha256=digest(evidence),
                    decoded_sha256=hashlib.sha256(raw).hexdigest(),
                )
            )
            atomic(
                args.output / "partial.json",
                dict(results=completed, total=2334, remaining_are_not_negatives=True),
            )
        if digest(root / "references-v1/index.sqlite") != INDEX_SHA:
            raise ValueError("Immutable index bytes changed during audit")
        atomic(
            args.output / "result.json",
            dict(
                version=VERSION,
                status="complete" if all(r["complete"] for r in completed) else "incomplete",
                results=completed,
                control_summary=dict(
                    total=848,
                    complete=sum(r["complete"] for r in completed[:848]),
                    expected_positives=784,
                    retrieved=sum(
                        r["complete"] and r["expected_retrieved"] for r in completed[:784]
                    ),
                    negative_identity_flags=sum(r["identity_flagged"] for r in completed[784:848]),
                    negative_review_flags=sum(r["review_flagged"] for r in completed[784:848]),
                    historical_zero_review_admission_gate_still_failed=True,
                ),
                scientific_scope="All 1486 BP recordings against GS604 only",
                plan_sha256=args.plan_sha256,
                source_sha256=sources(),
                index_sha256_before=INDEX_SHA,
                index_sha256_after=digest(root / "references-v1/index.sqlite"),
                wall_seconds=time.monotonic() - began,
                cpu_seconds=time.process_time() - cpu,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                audio_read=False,
                labels_scored=False,
                admitted_records=[],
                training_records=[],
            ),
        )
    except Exception as error:
        atomic(
            args.output / "failure.json",
            dict(
                status="incomplete",
                error_type=type(error).__name__,
                error=str(error),
                completed=len(completed),
                remaining_are_not_negatives=True,
            ),
        )
        raise
    finally:
        if index is not None:
            index.db.close()
        stop.set()
        monitor.join(timeout=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = sub.add_parser("prepare")
    for name in (
        "references",
        "prepared",
        "collector",
        "collector-provenance",
        "release",
        "catalog",
        "witness",
        "historical-controls",
        "output",
    ):
        prepare_parser.add_argument("--" + name, type=Path, required=True)
    run_parser = sub.add_parser("run")
    for name in ("plan", "output"):
        run_parser.add_argument("--" + name, type=Path, required=True)
    run_parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    prepare(args) if args.mode == "prepare" else run(args)


if __name__ == "__main__":
    main()
