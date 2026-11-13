"""Deterministic retrospective slices of frozen BP126/FMA250 development predictions.

No model/package initialization, audio, inference, confirmation references or new
candidate selection. Collection catalogs remain prediction/target-free.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import resource
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TypedDict

from experiments.collection_catalog import CollectionCatalog, CollectionTrack, collection_from_dict
from experiments.corpus_catalog import duration_band
from tools.beatport_tempo_metadata import BANDS as BPM_BANDS
from tools.beatport_tempo_metadata import tempo_band

ROOT = Path(__file__).resolve().parents[1]
BASE = "experiments/studies/"
K10 = BASE + "k10-v2-pilot/data/"
MFM = BASE + "musicfm-key-pilot/data/"
DATA = BASE + "v2-data-contract/data/"
DEFAULT = ROOT / DATA / "development-slices.json.gz"
MODELS = ("k10_expanded", "musicfm_pilot")
DOMAINS = {"beatport": 126, "fmak": 250}
DURATION_BANDS = ("<30s", "30s–<2m", "2–<5m", "5–10m", ">10m", "duration_unknown")
PINS = {
    "expanded": (
        K10 + "results/expanded.json.gz",
        "940e3978d1f1340f8a82414a2fbb11555acf27a0ba9a723612bdfae199a5e38e",
    ),
    "selected_development_refs": (
        K10 + "results/selected-per-id-errors.json.gz",
        "9709080c6877040b5aea8061eb0b64820dc7c10a4b17fcb6658bec988f4e46c7",
    ),
    "plan": (
        K10 + "canonical-eligibility-v2/plan-v2.json.gz",
        "c3e59ffb764ca5f51444447b0a18a69543228e60a9e3723d925ddd82b0fc8373",
    ),
    "feature_manifest": (
        K10 + "canonical-eligibility-v2/feature-manifest-v2.json.gz",
        "fcbc61081166f1845efe8af1ec2d7f807bb2d0a906dd93728b735e287c855291",
    ),
    "label_export": (
        K10 + "canonical-eligibility-v2/numeric-labels-v2.json.gz",
        "b8942b5913b8ebacdf2fc9e0ebca72a934aaa7c56d4e1a97d0f60fac84354323",
    ),
    "musicfm_terminal": (
        MFM + "terminal-raw.json.gz",
        "3c91e98b027696817eb72bb943884b9ff5f6d2afb6268c08c1ccf4974ac6f2cb",
    ),
    "beatport_catalog": (
        DATA + "beatport1355-collection.json.gz",
        "24a09353dc4e653bfc62898a0ca61ee2ff4838d62c3ea07ee261a2fe4f00b11f",
    ),
    "fmak_catalog": (
        DATA + "fmak5489-collection.json.gz",
        "a626b886b61e94a6296e6160a672be54797b62a7f8f60916c82905f2c34a73b2",
    ),
    "score_code": (
        "drone/keys.py",
        "ccf774468fb165ce27c0d9e2d3dac05bf258575665c2ff6318a1e1c5b79ea432",
    ),
}


class ModelScore(TypedDict):
    correct: int
    total: int
    exact: float | None
    mirex: float | None
    symmetric_units10: int
    asymmetric_units10: int
    n_valid: int
    n_error: int
    n_abstain: int


@dataclass(frozen=True)
class Observation:
    track_id: str
    domain: str
    reference: int
    k10_pred: int | None
    musicfm_pred: int | None
    k10_seconds: float
    musicfm_seconds: float
    metadata: CollectionTrack
    k10_status: str = "key"
    musicfm_status: str = "key"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def decode(raw: bytes) -> bytes:
    return gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw


def selected_plan_refs(text: str, ids: list[str]) -> dict[str, int]:
    """Materialize references only for the authorized selected376 IDs."""
    start = re.search(r'"expected_refs"\s*:\s*\{', text)
    if start is None:
        raise ValueError("Missing pinned reference map")
    stop = text.find("}", start.end())
    if stop < 0:
        raise ValueError("Invalid pinned reference map")
    selected = text[start.end() : stop]
    refs = {}
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated selected ID")
    for identity in ids:
        matches = re.findall(r'"' + re.escape(identity) + r'"\s*:\s*([0-9]+)\s*(?=,|$)', selected)
        if len(matches) != 1 or not 0 <= int(matches[0]) < 24:
            raise ValueError("Selected reference missing/repeated/invalid")
        refs[identity] = int(matches[0])
    return refs


def field(text: str, name: str) -> Any:  # noqa: ANN401 — selected JSON field only
    match = re.search(r'"' + re.escape(name) + r'"\s*:', text)
    if match is None:
        raise ValueError("Missing bound field: " + name)
    return json.JSONDecoder().raw_decode(text[match.end() :].lstrip())[0]


def load_score_code(root: Path) -> ModuleType:
    path, expected = PINS["score_code"]
    if digest((root / path).read_bytes()) != expected:
        raise ValueError("Canonical scorer changed")
    spec = importlib.util.spec_from_file_location("development_frozen_keys", root / path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import canonical scoring source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_score(rows: list[Observation], model: str, scorer: ModuleType) -> ModelScore:
    if model not in MODELS:
        raise ValueError("Unknown scored model")
    pred_name, status_name = (
        ("k10_pred", "k10_status") if model == MODELS[0] else ("musicfm_pred", "musicfm_status")
    )
    valid = []
    statuses = []
    for row in rows:
        prediction, status = getattr(row, pred_name), getattr(row, status_name)
        if status not in {"key", "error", "abstain"} or (prediction is None) != (status != "key"):
            raise ValueError("Prediction/status mismatch")
        statuses.append(status)
        if prediction is not None:
            if type(prediction) is not int or not 0 <= prediction < 24:
                raise ValueError("Invalid predicted class")
            valid.append((prediction, row.reference))
    n = len(rows)
    exact = sym = asym = 0.0
    if valid:
        predictions, references = [p for p, _ in valid], [r for _, r in valid]
        exact, sym = scorer.score_mirex(predictions, references)
        asym = scorer.score_mirex(predictions, references, symmetric=False)[1]
    return dict(
        correct=round(exact * len(valid)),
        total=n,
        exact=exact * len(valid) / n if n else None,
        mirex=sym * len(valid) / n if n else None,
        symmetric_units10=round(sym * len(valid) * 10),
        asymmetric_units10=round(asym * len(valid) * 10),
        n_valid=len(valid),
        n_error=statuses.count("error"),
        n_abstain=statuses.count("abstain"),
    )


def slice_summary(
    rows: list[Observation], dimension: str, label: str, scorer: ModuleType
) -> dict[str, Any]:
    scores = {model: model_score(rows, model, scorer) for model in MODELS}
    both = sum(row.k10_pred == row.reference and row.musicfm_pred == row.reference for row in rows)
    k10 = sum(row.k10_pred == row.reference and row.musicfm_pred != row.reference for row in rows)
    musicfm = sum(
        row.musicfm_pred == row.reference and row.k10_pred != row.reference for row in rows
    )
    wrong = len(rows) - both - k10 - musicfm
    return dict(
        dimension=dimension,
        label=label,
        n=len(rows),
        group_count=len({group.group_id for row in rows for group in row.metadata.frozen_groups}),
        models=scores,
        paired=dict(
            both_correct=both,
            both_wrong=wrong,
            musicfm_only_correct=musicfm,
            k10_only_correct=k10,
            musicfm_minus_k10_exact_count=musicfm - k10,
            musicfm_minus_k10_mirex=(
                (scores[MODELS[1]]["symmetric_units10"] - scores[MODELS[0]]["symmetric_units10"])
                / (10 * len(rows))
                if rows
                else None
            ),
        ),
        caveats=["Retrospective reused development data; no independent/causal claim"]
        + (["Small cell: fewer than10 recordings"] if len(rows) < 10 else [])
        + (["No recordings in this bin"] if not rows else []),
    )


def corpus_slices(
    rows: list[Observation],
    domain: str,
    scorer: ModuleType,
    genre_universe: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    result = []
    taxonomies = (
        ["beatport_workbook_native", "beatport_api_genres", "beatport_api_subgenres"]
        if domain == "beatport"
        else ["FMA-native-genre-taxonomy"]
    )
    for taxonomy in taxonomies:
        labels = sorted(
            {
                tag.name
                for row in rows
                for tag in row.metadata.genre_tags
                if tag.taxonomy == taxonomy
            }
        )
        if genre_universe is not None:
            labels = genre_universe.get("taxonomy:" + taxonomy, labels)
        for label in labels + ["unknown"]:
            selected = [
                row
                for row in rows
                if (
                    not any(tag.taxonomy == taxonomy for tag in row.metadata.genre_tags)
                    if label == "unknown"
                    else any(
                        tag.taxonomy == taxonomy and tag.name == label
                        for tag in row.metadata.genre_tags
                    )
                )
            ]
            result.append(slice_summary(selected, "taxonomy:" + taxonomy, label, scorer))
    if domain == "fmak":
        top_labels = sorted(
            {row.metadata.native_genre_top for row in rows if row.metadata.native_genre_top}
        )
        if genre_universe is not None:
            top_labels = genre_universe.get("native_genre_top", top_labels)
        for label in top_labels + ["unknown"]:
            selected = [
                row for row in rows if (row.metadata.native_genre_top or "unknown") == label
            ]
            result.append(slice_summary(selected, "native_genre_top", label, scorer))
    for label in DURATION_BANDS:
        selected = [row for row in rows if duration_band(row.k10_seconds) == label]
        result.append(slice_summary(selected, "duration_actual_evaluation", label, scorer))
    for label in BPM_BANDS:
        selected = [row for row in rows if tempo_band(row.metadata.bpm.value) == label]
        result.append(slice_summary(selected, "bpm_publisher", label, scorer))
    for label, mode in (("major", 0), ("minor", 1)):
        result.append(
            slice_summary(
                [row for row in rows if row.reference % 2 == mode], "reference_mode", label, scorer
            )
        )
    return result


def selected_collection(
    path: Path, selected_ids: set[str], expected_rows: int
) -> tuple[CollectionCatalog, dict[str, list[str]]]:
    """Read pinned canonical catalog framing, retaining only selected row dictionaries.

    This is not a general JSON streaming parser. The source hash is checked by
    the caller, and exact pretty-printed v1 catalog framing is required here.
    Unknown/malformed framing fails rather than falling back to an unbounded load.
    """
    header: list[str] = []
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    universe: dict[str, set[str]] = {"native_genre_top": set()}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for line in stream:
            if line == '  "tracks": [\n':
                break
            header.append(line)
        else:
            raise ValueError("Missing canonical catalog track array")
        block: list[str] = []
        comma_after_row: bool | None = None
        finished = False
        for line in stream:
            if not block:
                if line == "  ]\n":
                    if comma_after_row is True:
                        raise ValueError("Trailing comma in canonical catalog array")
                    finished = True
                    break
                if line != "    {\n" or comma_after_row is False:
                    raise ValueError("Invalid canonical catalog row boundary or separator")
            block.append(line)
            if line in ("    },\n", "    }\n"):
                comma_after_row = line == "    },\n"
                raw = "".join(block).rstrip().removesuffix(",")
                row = json.loads(raw)
                identity = row["track_id"]
                if not isinstance(identity, str) or identity in identities:
                    raise ValueError("Duplicate or invalid catalog identity")
                identities.add(identity)
                for tag in row["genre_tags"]:
                    taxonomy, name = tag["taxonomy"], tag["name"]
                    if not isinstance(taxonomy, str) or not isinstance(name, str):
                        raise ValueError("Invalid native genre universe")
                    universe.setdefault("taxonomy:" + taxonomy, set()).add(name)
                top = row["native_genre_top"]
                if top is not None:
                    if not isinstance(top, str):
                        raise ValueError("Invalid top genre universe")
                    universe["native_genre_top"].add(top)
                if identity in selected_ids:
                    rows.append(row)
                block = []
        if not finished or block or stream.read() != "}\n" or len(identities) != expected_rows:
            raise ValueError("Incomplete or unexpected canonical catalog population")
    data = json.loads("".join(header) + '  "tracks": []\n}\n')
    data["tracks"] = rows
    return collection_from_dict(data), {name: sorted(values) for name, values in universe.items()}


def analyze(root: Path = ROOT) -> dict[str, Any]:
    bindings = {
        "helper:" + name: dict(path=name, sha256=digest((root / name).read_bytes()))
        for name in (
            "experiments/collection_catalog.py",
            "experiments/corpus_catalog.py",
            "tools/beatport_tempo_metadata.py",
        )
    }

    def pinned(name: str, parse: bool = True) -> Any:  # noqa: ANN401 — heterogeneous frozen records
        relative, expected = PINS[name]
        packed = (root / relative).read_bytes()
        if digest(packed) != expected:
            raise ValueError("Frozen input changed: " + name)
        decoded = decode(packed)
        bindings[name] = dict(path=relative, sha256=expected, decoded_sha256=digest(decoded))
        return json.loads(decoded) if parse else decoded

    expanded, selected = pinned("expanded"), pinned("selected_development_refs")
    plan_text = pinned("plan", False).decode()
    pinned("label_export", False)  # hash only; never parse nonselected reference values
    features = pinned("feature_manifest")
    terminal = pinned("musicfm_terminal")
    archive_path = K10 + "results/archive-index.json"
    terminal_path = MFM + "terminal-index.json"
    archive, index = (
        json.loads((root / archive_path).read_bytes()),
        json.loads((root / terminal_path).read_bytes()),
    )
    bindings["k10_archive_index"] = dict(
        path=archive_path, sha256=digest((root / archive_path).read_bytes())
    )
    bindings["musicfm_terminal_index"] = dict(
        path=terminal_path, sha256=digest((root / terminal_path).read_bytes())
    )
    for name in ("expanded.json.gz", "selected-per-id-errors.json.gz"):
        entry = archive["results/" + name]
        raw = (root / K10 / "results" / name).read_bytes()
        if digest(raw) != entry["file_sha256"] or digest(decode(raw)) != entry["decoded_sha256"]:
            raise ValueError("K10 archive-index mismatch")
    if (
        index["sha256"] != PINS["musicfm_terminal"][1]
        or index["decoded_sha256"] != bindings["musicfm_terminal"]["decoded_sha256"]
    ):
        raise ValueError("MusicFM terminal-index mismatch")
    docs = {}
    for name in ("config.json", "result.json"):
        item = terminal["files"][name]
        if (
            digest(item["raw_utf8"].encode()) != item["sha256"]
            or item["sha256"] != index["files"][name]
        ):
            raise ValueError("Embedded terminal document changed")
        docs[name] = json.loads(item["raw_utf8"])
    config, musicfm = docs["config.json"], docs["result.json"]
    del terminal, docs, item  # Drop duplicate raw UTF-8 documents before loading catalogs.
    if (
        len(expanded["history"]) != 120
        or len(musicfm["history"]) != 120
        or musicfm["events"] != 120
        or expanded["updates"] != 8280
        or musicfm["updates"] != 8280
        or not any(event["update"] == expanded["best_update"] for event in expanded["history"])
        or not any(
            event["update"] == musicfm["best_update"]
            and event["metrics"] == musicfm["best_metrics"]
            for event in musicfm["history"]
        )
    ):
        raise ValueError("Completed selection history/selected event mismatch")
    for run, names in (
        (expanded, ("train.py", "experiments/k10_v2_pilot.py", "drone/features.py")),
        (
            config,
            (
                "experiments/musicfm_pilot.py",
                "experiments/musicfm_eval.py",
                "experiments/musicfm_inputs.py",
                "experiments/musicfm_teacher.py",
            ),
        ),
    ):
        for name in names:
            expected = run["source_sha256"][name]
            if digest((root / name).read_bytes()) != expected:
                raise ValueError("Bound evaluator/input source changed: " + name)
            bindings["evaluator:" + name] = dict(path=name, sha256=expected)
    if (
        config["controls"]["expanded"]["sha256"] != PINS["expanded"][1]
        or expanded["plan_sha256"] != config["plan_sha256"]
        or config["plan_sha256"] != PINS["plan"][1]
        or features["label_export_sha256"] != config["labels_sha256"]
        or config["labels_sha256"] != field(plan_text, "label_export_sha256")
        or config["labels_sha256"] != PINS["label_export"][1]
        or digest(decode((root / PINS["feature_manifest"][0]).read_bytes()))
        != expanded["feature_manifest_sha256"]
    ):
        raise ValueError("Plan/label/input/control bindings disagree")
    if (
        expanded["confirmation_used"]
        or expanded["calibration_used"]
        or musicfm["confirmation_used"]
        or musicfm["calibration_used"]
    ):
        raise ValueError("Only unconfirmed selected development results allowed")
    ids = [identity for domain in DOMAINS for identity in expanded["best_metrics"][domain]["ids"]]
    if len(ids) != 376 or len(set(ids)) != 376 or [row["id"] for row in selected] != ids:
        raise ValueError("Selected376 identity/order mismatch")
    refs = selected_plan_refs(plan_text, ids)
    by_id = {row["id"]: row for row in selected}
    features_by_id = {row["id"]: row for row in features["rows"] if row["id"] in refs}
    if len(features_by_id) != 376:
        raise ValueError("Selected feature-input witness missing")
    del features  # Retain only the authorized selected input witnesses.
    gc.collect()
    metadata = {}
    genre_universes: dict[str, dict[str, list[str]]] = {}
    for name in ("beatport_catalog", "fmak_catalog"):
        relative, expected = PINS[name]
        path = root / relative
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Frozen input changed: " + name)
        with gzip.open(path, "rb") as stream:
            decoded_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        bindings[name] = dict(path=relative, sha256=expected, decoded_sha256=decoded_sha256)
        domain = "beatport" if name == "beatport_catalog" else "fmak"
        catalog, universe = selected_collection(
            path, set(refs), 1355 if domain == "beatport" else 5489
        )
        sources = {item.source_id: item for item in catalog.external_sources}
        if not any(item.sha256 == config["cache_manifest_sha256"] for item in sources.values()):
            raise ValueError("Catalog cache witness differs from teacher input binding")
        genre_universes[domain] = universe
        metadata.update({row.track_id: row for row in catalog.tracks})
        del catalog
        gc.collect()
    proof = musicfm["selected_teacher_composed_batch_proof"]
    proof_rows = proof["per_song"]
    if [row["id"] for row in proof_rows] != ids or len(proof["reference_logits"]) != 376:
        raise ValueError("Teacher proof identity mismatch")
    song_by_id = {row["id"]: row for row in proof_rows}
    observations = []
    scorer = load_score_code(root)
    bindings["score_code"] = dict(path=PINS["score_code"][0], sha256=PINS["score_code"][1])
    for domain, count in DOMAINS.items():
        baseline, teacher = expanded["best_metrics"][domain], musicfm["best_metrics"][domain]
        expected_ids = field(
            plan_text, "bp_selection_guard" if domain == "beatport" else "fma_inner_selection"
        )
        if (
            baseline["ids"] != teacher["ids"]
            or baseline["ids"] != expected_ids
            or len(expected_ids) != count
        ):
            raise ValueError("Selected domain membership/order differs")
        for identity, a, b in zip(
            expected_ids, baseline["predictions"], teacher["predictions"], strict=True
        ):
            row, feature = metadata[identity], features_by_id[identity]
            ref = refs[identity]
            if (
                by_id[identity]["ref"] != ref
                or feature["ref"] != ref
                or by_id[identity]["expanded_pred"] != a
                or feature["sha256"] != row.source_audio.sha256
                or song_by_id[identity]["samples"] != row.analyzed_audio.samples
            ):
                raise ValueError("Selected source/ref/prediction/sample witness mismatch")
            k10_seconds, teacher_seconds = (
                feature["canonical_samples"] / 22050,
                song_by_id[identity]["samples"] / 24000,
            )
            if duration_band(k10_seconds) != duration_band(teacher_seconds):
                raise ValueError(
                    "Model-bound duration bands differ; cannot silently pool memberships"
                )
            observations.append(
                Observation(identity, domain, ref, a, b, k10_seconds, teacher_seconds, row)
            )
    for i, row in enumerate(observations):
        logits = proof["reference_logits"][i]
        candidate = proof["candidate_logits"][i]
        if (
            len(logits) != 24
            or len(candidate) != 24
            or not all(math.isfinite(x) for x in logits + candidate)
            or max(range(24), key=logits.__getitem__) != row.musicfm_pred
            or max(range(24), key=candidate.__getitem__) != row.musicfm_pred
            or max(abs(x - y) for x, y in zip(logits, candidate, strict=True)) > 1e-5
        ):
            raise ValueError("Selected teacher B1/B16 proof changed")
    corpora = {}
    for domain, count in DOMAINS.items():
        rows = [row for row in observations if row.domain == domain]
        totals = slice_summary(rows, "corpus", domain, scorer)
        for model, run in ((MODELS[0], expanded), (MODELS[1], musicfm)):
            recorded = run["best_metrics"][domain]
            observed = totals["models"][model]
            if (
                observed["correct"] != recorded["exact_correct"]
                or observed["total"] != count
                or observed["symmetric_units10"] != recorded["symmetric_units10"]
                or observed["asymmetric_units10"] != recorded["asymmetric_units10"]
            ):
                raise ValueError("Full-domain exact/MIREX numerator mismatch")
        corpora[domain] = dict(
            n=count,
            group_count=totals["group_count"],
            totals=totals,
            slices=corpus_slices(rows, domain, scorer, genre_universes[domain]),
        )
    return dict(
        schema="drone.development-slices.v1",
        generator_sha256=digest((root / "tools/analyze_development_slices.py").read_bytes()),
        scope="retrospective_reused_inner_development_376",
        bindings=bindings,
        models={
            MODELS[0]: dict(
                checkpoint_sha256=expanded["checkpoint_sha256"],
                selected_update=expanded["best_update"],
                selection_events=len(expanded["history"]),
                evaluator="full_true_length_CQT_22050Hz",
                role="expanded_data_research_control_not_released_Drone_v1",
                source_sha256=expanded["source_sha256"],
            ),
            MODELS[1]: dict(
                checkpoint_sha256=musicfm["best_checkpoint_sha256"],
                selected_update=musicfm["best_update"],
                selection_events=len(musicfm["history"]),
                evaluator="whole_recording_24kHz_12second_contexts_actual_tails_FP64_support_weighted_logits",
                source_sha256=config["source_sha256"],
            ),
        },
        corpora=corpora,
        per_track=[
            dict(
                track_id=row.track_id,
                domain=row.domain,
                reference=row.reference,
                predictions={MODELS[0]: row.k10_pred, MODELS[1]: row.musicfm_pred},
                evaluation_seconds={MODELS[0]: row.k10_seconds, MODELS[1]: row.musicfm_seconds},
                source_sha256=row.metadata.source_audio.sha256,
                k10_canonical_input_sha256=features_by_id[row.track_id][
                    "canonical_model_input_sha256"
                ],
                k10_feature_cache_sha256=features_by_id[row.track_id]["cache_sha256"],
                teacher_cache_pcm_sha256=row.metadata.analyzed_audio.pcm_identity.sha256,
            )
            for row in observations
        ],
        caveats=[
            "Both checkpoints selected from120events on these reused development IDs; retrospective/exploratory only",
            "No inference, confirmation access, independent generalization, causal effect, CI or candidate selection",
            "Native taxonomies remain separate; FMA all-tags include ancestors and cells overlap",
            "Publisher BPM is unverified and method-scoped; unknown values remain unknown",
            "FMA component units and BP record proxies remain frozen; neither is proven full recording independence",
            "Actual evaluation duration retained per model; training crop/context length is not recording duration",
        ],
    )


def analysis_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def verify_analysis(path: Path = DEFAULT, root: Path = ROOT) -> dict[str, Any]:
    expected = analyze(root)
    checked_archive(path, analysis_bytes(expected))
    return expected


def checked_archive(path: Path, canonical: bytes) -> bytes:
    """Compare canonical JSON while preserving the checked transport's actual identity."""
    archived = path.read_bytes()
    if decode(archived) != canonical:
        raise ValueError("Stored development analysis differs from verified source arithmetic")
    return archived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    stop = threading.Event()

    def guard() -> None:
        while not stop.wait(0.025):
            use = resource.getrusage(resource.RUSAGE_SELF)
            rss = use.ru_maxrss if sys.platform == "darwin" else use.ru_maxrss * 1024
            if rss > 256 * 1024**2 or time.monotonic() - started > 120:
                os.write(
                    2,
                    (
                        json.dumps(
                            {
                                "budget_failed": True,
                                "observed_peak_rss_bytes": rss,
                                "wall_seconds": time.monotonic() - started,
                            }
                        )
                        + "\n"
                    ).encode(),
                )
                os._exit(124)

    threading.Thread(target=guard, daemon=True).start()
    value = analyze()
    encoded = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=encoded, mode="wb", mtime=0) as stream:
        stream.write(analysis_bytes(value))
    archived = encoded.getvalue()
    if args.check:
        archived = checked_archive(args.output, analysis_bytes(value))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(archived)
    stop.set()
    use = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = use.ru_maxrss if sys.platform == "darwin" else use.ru_maxrss * 1024
    if (
        peak_rss > 256 * 1024**2
        or time.monotonic() - started > 120
        or use.ru_utime + use.ru_stime > 60
        or len(archived) > 16 * 1024**2
    ):
        raise RuntimeError("Resource budget exceeded at terminal check")
    print(
        json.dumps(
            dict(
                sha256=digest(archived),
                wall_seconds=time.monotonic() - started,
                cpu_seconds=use.ru_utime + use.ru_stime,
                os_peak_rss_bytes=use.ru_maxrss
                if sys.platform == "darwin"
                else use.ru_maxrss * 1024,
                output_bytes=len(archived),
                corpora={key: value["corpora"][key]["totals"] for key in DOMAINS},
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
