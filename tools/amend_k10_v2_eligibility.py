"""Source-bound metadata-only eligibility derivative; no resplit, labels chosen or audio read."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.k10_v2_pilot import validate_feature_manifest
from tools.prepare_k10_v2_bulk import canonical_audit

REVISION = "k10-v2-canonical-eligibility-v2"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def packed(value: object) -> bytes:
    return gzip.compress(json.dumps(value, sort_keys=True, separators=(",", ":")).encode(), mtime=0)


def read(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw)


def exclusions(
    rows: list[dict[str, Any]], plan: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    audit = canonical_audit(rows, plan)
    by = {r["id"]: r for r in rows}
    fitting = set(plan["bp_fit"] + plan["expanded_fma_fit"])
    remove = {}
    for family in audit["duplicate_families"]:
        if len(family["refs"]) > 1:
            remove.update({i: "conflicting_canonical_targets_all_members" for i in family["ids"]})
        elif len(family["roles"]) > 1:
            if any(by[i]["domain"] != "beatport" for i in family["ids"]):
                raise ValueError(
                    "New non-BP cross-role identity needs root policy;no automatic repair"
                )
            remove.update(
                {
                    i: "BP_fitting_alias_preserve_existing_selection"
                    for i in family["ids"]
                    if i in fitting
                }
            )
    return remove, audit


def uncertainty(
    rows: list[dict[str, Any]], plan: dict[str, Any], groups: dict[str, str]
) -> tuple[bytes, dict[str, Any]]:
    fitting = set(plan["bp_fit"] + plan["expanded_fma_fit"])
    base = {
        r["id"]: (
            "fma:" + groups[r["id"]]
            if r["domain"] == "fmak"
            else "bp-record-unknown-artist:" + r["id"]
        )
        for r in rows
    }
    parents = {g: g for g in base.values()}

    def root(g: str) -> str:
        while parents[g] != g:
            g = parents[g]
        return g

    audit = canonical_audit(rows, plan)
    merges = []
    for family in audit["duplicate_families"]:
        if any(i in fitting for i in family["ids"]):
            continue
        nodes = sorted({base[i] for i in family["ids"]})
        for node in nodes[1:]:
            a, b = root(nodes[0]), root(node)
            if a != b:
                parents[max(a, b)] = min(a, b)
                merges.append(
                    {"canonical_sha256": family["canonical_sha256"], "base_groups": nodes}
                )
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        ["id", "domain", "logical_role", "original_base_group", "derived_uncertainty_group"]
    )
    for row in sorted(rows, key=lambda r: r["id"]):
        writer.writerow(
            [
                row["id"],
                row["domain"],
                "fit" if row["id"] in fitting else "selection",
                base[row["id"]],
                root(base[row["id"]]),
            ]
        )
    witnesses = []
    for family in audit["duplicate_families"]:
        identities = family["ids"]
        witnesses.append(
            {
                **family,
                "domains": sorted({i.split(":")[0] for i in identities}),
                "saved_fma_components": sorted({groups[i] for i in identities if i in groups}),
            }
        )
    return gzip.compress(stream.getvalue().encode(), mtime=0), {
        "consistent_families": witnesses,
        "evaluation_uncertainty_merges": merges,
        "BP_grouping_limit": "record/observedcanonicalidentity only;artistindependence notestablished",
    }


def derive(
    parent: dict[str, Any],
    labels: dict[str, Any],
    manifest: dict[str, Any],
    groups: dict[str, str],
    provenance: dict[str, str],
    old_roles: dict[str, str] | None = None,
) -> dict[str, bytes]:
    rows = manifest["rows"]
    by = {r["id"]: r for r in rows}
    if set(by) != set(parent["train_feature_bindings"]) or len(by) != len(rows):
        raise ValueError("Parent prepared membership differs")
    for identity, row in by.items():
        if (
            row["sha256"] != parent["train_feature_bindings"][identity]
            or row["ref"] != parent["expected_refs"][identity]
            or row["ref"] != labels["labels"][identity]
        ):
            raise ValueError("Parent source/target binding differs")
    removed, before = exclusions(rows, parent)
    cache_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cache_groups[row["cache_sha256"]].append(row)
    cache_families = [v for v in cache_groups.values() if len(v) > 1]
    fitting = set(parent["bp_fit"] + parent["expanded_fma_fit"])
    extra_cache = []
    for family in cache_families:
        roles = {"fit" if r["id"] in fitting else "selection" for r in family}
        if (len(roles) > 1 or len({r["ref"] for r in family}) > 1) and len(
            {r["canonical_model_input_sha256"] for r in family}
        ) > 1:
            extra_cache.append([r["id"] for r in family])
    if extra_cache:
        raise ValueError("Additional cache-only integrity family needs root policy")
    cache_check = {
        "duplicate_cache_families": len(cache_families),
        "additional_cache_integrity_families": extra_cache,
        "canonical_families_have_bitidentical_cachefiles": all(
            len({by[i]["cache_sha256"] for i in family["ids"]}) == 1
            for family in before["duplicate_families"]
        ),
    }
    plan = copy.deepcopy(parent)
    for name in (
        "bp_fit",
        "restricted_fma_fit",
        "expanded_fma_fit",
        "bp_selection_guard",
        "fma_inner_selection",
    ):
        plan[name] = [i for i in parent[name] if i not in removed]
    for name in ("bp_selection_guard", "fma_inner_selection"):
        allowed = [
            i for i in parent[name] if removed.get(i) == "conflicting_canonical_targets_all_members"
        ]
        if plan[name] != [i for i in parent[name] if i not in allowed]:
            raise ValueError("Consistent selection IDs cannot move or be replaced")
    for name in ("train_feature_bindings", "expected_refs"):
        plan[name] = {i: v for i, v in parent[name].items() if i not in removed}
    retained = [r for r in rows if r["id"] not in removed]
    after = canonical_audit(retained, plan)
    if not after["complete"]:
        raise ValueError("Unresolved canonical conflict after general policy")
    cluster_bytes, witness = uncertainty(retained, plan, groups)
    overlay = {
        "status": "PROPOSED_ELIGIBILITY_AMENDMENT_NOT_TRAINING_AUTHORIZED",
        "revision": REVISION,
        "source_inputs": provenance,
        "policy": "allconflictingcanonicaltargetmembers excluded;consistentBPcrossrole fittingaliasesexcluded,selectionpreserved;withinroleconsistentduplicatesretained",
        "excluded": [
            {
                "id": i,
                "reason": reason,
                "proposed_role_unchanged": by[i]["role"],
                "domain": by[i]["domain"],
                "canonical_sha256": by[i]["canonical_model_input_sha256"],
                "saved_fma_component": groups.get(i),
                "old_source_role": (old_roles or {}).get(i, by[i]["role"]),
                "in_parent_restricted": i in parent["bp_fit"] or i in parent["restricted_fma_fit"],
            }
            for i, reason in sorted(removed.items())
        ],
        "cache_identity_check": cache_check,
        "before_canonical": before,
        "after_canonical": after,
        "uncertainty_witness": witness,
        "counts": {
            "bp_fit": len(plan["bp_fit"]),
            "restricted_fma_fit": len(plan["restricted_fma_fit"]),
            "expanded_fma_fit": len(plan["expanded_fma_fit"]),
            "bp_selection": len(plan["bp_selection_guard"]),
            "fma_inner": len(plan["fma_inner_selection"]),
            "prepared": len(retained),
        },
        "no_roles_moved_no_replacements_no_labels_changed": True,
        "nominal_bp_mass_unchanged": [1230, 2206],
    }
    overlay_bytes = (json.dumps(overlay, sort_keys=True, indent=2) + "\n").encode()
    overlay_sha = sha(overlay_bytes)
    targets = copy.deepcopy(labels)
    targets["labels"] = {i: v for i, v in labels["labels"].items() if i not in removed}
    targets.update(
        eligibility_revision=REVISION,
        parent_label_export_sha256=provenance["parent_labels_sha256"],
        eligibility_amendment_sha256=overlay_sha,
        derivation="subset ofdirectCSV-verifiedparent;zero relabels",
    )
    target_bytes = packed(targets)
    plan.update(
        plan_revision=REVISION,
        parent_plan_sha256=provenance["parent_plan_sha256"],
        eligibility_amendment_sha256=overlay_sha,
        label_export_sha256=sha(target_bytes),
        uncertainty_groups_sha256=sha(cluster_bytes),
        status="PROPOSED_ELIGIBILITY_REVISED_PILOT_NOT_TRAINING_AUTHORIZED",
    )
    plan_bytes = packed(plan)
    candidate = {
        **manifest,
        "parent_preparation_actions": manifest.get("actions"),
        "actions": dict(
            Counter(
                "carried_first16"
                if r.get("carried_first16")
                else "reused_v1"
                if r.get("reused_cache")
                else "new_cqt"
                for r in retained
            )
        ),
        "source_prepared_records": len(rows),
        "eligible_records": len(retained),
        "rows": retained,
        "complete": True,
        "status": "SOURCE_BOUND_ELIGIBILITY_DERIVATIVE_NOT_TRAINING_AUTHORIZED",
        "feature_manifest_revision": REVISION,
        "parent_manifest_sha256": provenance["parent_manifest_sha256"],
        "plan_sha256": sha(plan_bytes),
        "label_export_sha256": sha(target_bytes),
        "eligibility_amendment_sha256": overlay_sha,
        "derivation_source_sha256": provenance["amender_source_sha256"],
    }
    validate_feature_manifest(plan, candidate)
    return {
        "eligibility-amendment.json": overlay_bytes,
        "plan-v2.json.gz": plan_bytes,
        "numeric-labels-v2.json.gz": target_bytes,
        "feature-manifest-v2.json.gz": packed(candidate),
        "uncertainty-groups-v2.csv.gz": cluster_bytes,
        "canonical-audit-v2.json": (json.dumps(after, sort_keys=True, indent=2) + "\n").encode(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "labels", "manifest", "audit", "metadata", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    plan = read(args.plan)
    labels = read(args.labels)
    manifest = read(args.manifest)
    audit = read(args.audit)
    metadata = read(args.metadata)
    expected = {
        "parent_plan_sha256": "7e8311580ec961b4ccb0941fc484379f4eaa88bad676c1dba9fcd8eb46c7ef05",
        "parent_labels_sha256": "939d065e454ccc436e4dd53c53d1c4616dfc09c8cb248cbe1e3410b5d9c9b40d",
        "parent_manifest_sha256": "9f2db5efe4cbe2904508d8004f842acd29e90c6bc7f68612679c68993c439de0",
    }
    for key, path in [
        ("parent_plan_sha256", args.plan),
        ("parent_labels_sha256", args.labels),
        ("parent_manifest_sha256", args.manifest),
    ]:
        raw = path.read_bytes()
        actual = sha(
            gzip.decompress(raw)
            if key == "parent_manifest_sha256" and raw.startswith(b"\x1f\x8b")
            else raw
        )
        if actual != expected[key]:
            raise ValueError("Pinned original evidence changed:" + key)
    if manifest["complete"] or canonical_audit(manifest["rows"], plan) != audit:
        raise ValueError("Historical failure/audit differs")
    groups = {r["identity"]: r["component"] for r in metadata["tracks"]}
    provenance = {
        **expected,
        "parent_audit_sha256": sha(args.audit.read_bytes()),
        "saved_metadata_sha256": sha(args.metadata.read_bytes()),
        "amender_source_sha256": sha(Path(__file__).read_bytes()),
    }
    old_roles = {r["identity"]: r["old_role"] for r in metadata["tracks"]}
    artifacts = derive(plan, labels, manifest, groups, provenance, old_roles)
    counts = json.loads(artifacts["eligibility-amendment.json"])["counts"]
    if counts != {
        "bp_fit": 1229,
        "restricted_fma_fit": 976,
        "expanded_fma_fit": 4033,
        "bp_selection": 126,
        "fma_inner": 250,
        "prepared": 5638,
    }:
        raise ValueError("Reviewed actual counts differ")
    if not args.check:
        args.output.mkdir(parents=True, exist_ok=True)
    for name, raw in artifacts.items():
        if args.check:
            if (args.output / name).read_bytes() != raw:
                raise ValueError("Derivative reproduction mismatch:" + name)
        else:
            (args.output / name).write_bytes(raw)
    if not args.check:
        (args.output / "SHA256SUMS").write_text(
            "".join(f"{sha(raw)}  {name}\n" for name, raw in sorted(artifacts.items()))
        )
    print(
        json.dumps(
            {
                "revision": REVISION,
                "counts": counts,
                "plan_sha256": sha(artifacts["plan-v2.json.gz"]),
                "labels_sha256": sha(artifacts["numeric-labels-v2.json.gz"]),
                "feature_manifest_sha256": sha(artifacts["feature-manifest-v2.json.gz"]),
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
