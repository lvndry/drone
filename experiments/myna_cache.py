"""Prospective Myna embedding-cache contract; no encoder, decoding or training calls."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from drone.keys import LABEL_SCHEMA, transpose_key

SCHEMA = "drone-myna-complete-window-cache-v1"
ENCODER_SHA = "5dde8aa186da0fc0bfdd2e5a7c1ea3e6c6af6cd4683089b65c4fed6af8335374"
PINS = {
    "plan": "c3e59ffb764ca5f51444447b0a18a69543228e60a9e3723d925ddd82b0fc8373",
    "labels": "b8942b5913b8ebacdf2fc9e0ebca72a934aaa7c56d4e1a97d0f60fac84354323",
    "groups": "7505dcae2a91bf038537cee4a2f8e24f7929011c2bb56d8fe6649f6a1df8dcd7",
    "waveforms": "7d56908dc5e84c165c7844d1e810ed678123ae7d0a93939c60a2e600b6402b4c",
    "control": "940e3978d1f1340f8a82414a2fbb11555acf27a0ba9a723612bdfae199a5e38e",
}
Domain = Literal["beatport", "fmak"]
ROLE_FIELDS = {
    "bp_fit": ("beatport", "fit", 1229),
    "expanded_fma_fit": ("fmak", "fit", 4033),
    "bp_selection_guard": ("beatport", "selection", 126),
    "fma_inner_selection": ("fmak", "selection", 250),
}
DOMAIN_MASS = {"beatport": 1230 / 2206, "fmak": 976 / 2206}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path, watch: Callable[[], None]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            watch()
            digest.update(block)
    return digest.hexdigest()


def checked_bytes(path: Path, expected: str, watch: Callable[[], None]) -> bytes:
    require(path.is_file() and not path.is_symlink(), "Missing/nonregular metadata")
    require(path.stat().st_size <= 64 * 1024**2, "Oversized metadata")
    raw = path.read_bytes()
    watch()
    require(hashlib.sha256(raw).hexdigest() == expected, "Metadata hash mismatch")
    if path.suffix == ".gz":
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            raw = stream.read(64 * 1024**2 + 1)
        require(len(raw) <= 64 * 1024**2, "Oversized decoded metadata")
    return raw


def checked_json(path: Path, expected: str, watch: Callable[[], None]) -> dict[str, Any]:
    value = json.loads(checked_bytes(path, expected, watch))
    require(isinstance(value, dict), "Expected metadata object")
    return value


def is_sha(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class View:
    identity: str
    domain: str
    role: str
    group: str
    shift: int
    target: int
    windows: int
    path: Path
    digest: str
    byte_count: int


@dataclass(frozen=True)
class Contract:
    views: tuple[View, ...]
    selection_ids: dict[str, list[str]]
    labels: dict[str, int]
    fit_windows: int
    selection_windows: int
    fit_counts: dict[str, int]

    @property
    def embedding_bytes(self) -> int:
        return (self.fit_windows + self.selection_windows) * 384 * 4


def validate_manifest(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    labels: dict[str, int],
    groups: list[dict[str, str]],
    waveforms: list[dict[str, Any]],
    root: Path,
    identities: dict[str, str],
    *,
    require_production_counts: bool = True,
) -> Contract:
    """Exact membership/lineage/13-view validation; small fixtures may relax counts only."""
    require(
        manifest.get("schema_version") == SCHEMA and manifest.get("complete") is True,
        "Incomplete cache/schema",
    )
    require(manifest.get("label_schema") == LABEL_SCHEMA, "Different label schema")
    require(manifest.get("identities") == identities, "Cache source/qualification binding differs")
    require(identities.get("encoder_sha256") == ENCODER_SHA, "Different encoder")
    require(all(is_sha(value) for value in identities.values()), "Malformed identity")
    expected: dict[str, tuple[str, str]] = {}
    selection: dict[str, list[str]] = {}
    for field, (domain, role, count) in ROLE_FIELDS.items():
        ids = plan[field]
        require(isinstance(ids, list) and len(ids) == len(set(ids)), "Duplicate role IDs")
        if require_production_counts:
            require(len(ids) == count, "Different approved role counts")
        require(all(isinstance(i, str) and i.startswith(domain + ":") for i in ids), "Wrong domain")
        require(not set(ids) & expected.keys(), "Cross-role identity")
        expected.update((identity, (domain, role)) for identity in ids)
        if role == "selection":
            selection[domain] = ids.copy()
    require(set(labels) == set(expected), "Extra/missing reference labels")
    require(all(type(y) is int and 0 <= y < 24 for y in labels.values()), "Invalid key class")
    require(plan["expected_refs"] == labels, "Plan/label export differs")
    wave = {row["id"]: row for row in waveforms}
    group = {row["id"]: row for row in groups}
    require(
        len(wave) == len(waveforms) == len(expected) and set(wave) == set(expected),
        "Waveform membership differs",
    )
    require(
        len(group) == len(groups) == len(expected) and set(group) == set(expected),
        "Group membership differs",
    )
    rows = {row["id"]: row for row in manifest["tracks"]}
    require(
        len(rows) == len(manifest["tracks"]) == len(expected) and set(rows) == set(expected),
        "Cache membership differs",
    )
    views: list[View] = []
    used_paths: set[Path] = set()
    for identity, (domain, role) in expected.items():
        row, source, family = rows[identity], wave[identity], group[identity]
        require(row["domain"] == family["domain"] == domain, "Different domain")
        require(row["role"] == family["logical_role"] == role, "Different logical role")
        require(
            source["role"] == ("fit" if role == "fit" else "validation"), "Different waveform role"
        )
        require(row["group"] == family["derived_uncertainty_group"], "Different frozen group")
        require(row["target"] == source["target"] == labels[identity], "Different original target")
        require(
            row["source_sha256"] == source["sha256"]
            and row["pcm_sha256"] == source["waveform_sha256"],
            "Different source/PCM",
        )
        require(is_sha(row["source_sha256"]) and is_sha(row["pcm_sha256"]), "Invalid source hash")
        n24 = source["samples"]
        require(type(n24) is int and n24 > 0 and row["samples_24k"] == n24, "Different PCM support")
        n16 = (n24 * 2 + 2) // 3
        frames = n16 // 512 + 1
        windows, tail = divmod(frames, 196)
        require(windows > 0, "No complete input window")
        require(
            (
                row["samples_16k"],
                row["mel_frames"],
                row["complete_windows"],
                row["discarded_frames"],
            )
            == (n16, frames, windows, tail),
            "Window geometry differs",
        )
        required_shifts = set(range(-6, 7)) if role == "fit" else {0}
        actual_shifts = [view["shift"] for view in row["views"]]
        require(
            all(type(s) is int for s in actual_shifts)
            and len(actual_shifts) == len(required_shifts)
            and set(actual_shifts) == required_shifts,
            "Missing/duplicate/extra pitch views",
        )
        for view in sorted(row["views"], key=lambda item: item["shift"]):
            require(
                view["windows"] == windows
                and view["dtype"] == "float32"
                and view["shape"] == [windows, 384],
                "Embedding geometry/dtype differs",
            )
            require(
                view["window_order"] == "all-complete-196-frame-windows-in-order",
                "Capped/reordered windows",
            )
            raw, trimmed, padded = (
                view[k] for k in ("raw_samples", "trimmed_samples", "padded_samples")
            )
            require(
                all(type(v) is int and v >= 0 for v in (raw, trimmed, padded)),
                "Invalid augmentation sample ledger",
            )
            require(
                raw > 0
                and trimmed <= raw
                and not (trimmed and padded)
                and raw - trimmed + padded == n16,
                "Augmentation length ledger differs",
            )
            if view["shift"] == 0:
                require((raw, trimmed, padded) == (n16, 0, 0), "Original view was altered")
            relative = Path(view["path"])
            require(
                not relative.is_absolute()
                and ".." not in relative.parts
                and relative.suffix == ".npy",
                "Unsafe embedding path",
            )
            path = root / relative
            require(
                path not in used_paths and path.resolve().is_relative_to(root.resolve()),
                "Repeated/escaping embedding path",
            )
            used_paths.add(path)
            require(
                is_sha(view["sha256"])
                and type(view["bytes"]) is int
                and view["bytes"] > windows * 384 * 4,
                "Invalid npy identity/size",
            )
            views.append(
                View(
                    identity,
                    domain,
                    role,
                    row["group"],
                    view["shift"],
                    transpose_key(labels[identity], view["shift"]),
                    windows,
                    path,
                    view["sha256"],
                    view["bytes"],
                )
            )
    return Contract(
        tuple(views),
        selection,
        labels,
        sum(v.windows for v in views if v.role == "fit"),
        sum(v.windows for v in views if v.role == "selection"),
        {
            domain: sum(d == domain and r == "fit" for d, r in expected.values())
            for domain in DOMAIN_MASS
        },
    )


def group_rows(raw: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(raw.decode())))


def loss_weight(view: View, contract: Contract) -> float:
    require(view.role == "fit", "Selection cannot enter fitting objective")
    return (
        contract.fit_windows
        * DOMAIN_MASS[view.domain]
        / (contract.fit_counts[view.domain] * 13 * view.windows)
    )


@dataclass
class LoadedCache:
    embeddings: NDArray[np.float32]
    targets: NDArray[np.int64]
    weights: NDArray[np.float32]
    selection: dict[str, slice]
    contract: Contract


def load_cache(
    contract: Contract, max_embedding_bytes: int, watch: Callable[[], None]
) -> LoadedCache:
    require(contract.embedding_bytes <= max_embedding_bytes, "Embedding RAM admission exceeded")
    ordered = [v for v in contract.views if v.role == "fit"] + [
        v for v in contract.views if v.role == "selection"
    ]
    embeddings = np.empty(
        (contract.fit_windows + contract.selection_windows, 384), dtype=np.float32
    )
    targets = np.empty(contract.fit_windows, dtype=np.int64)
    weights = np.empty(contract.fit_windows, dtype=np.float32)
    selection: dict[str, slice] = {}
    offset = 0
    for view in ordered:
        watch()
        before = view.path.stat()
        require(
            not view.path.is_symlink()
            and view.path.is_file()
            and before.st_size == view.byte_count,
            "Embedding blob changed",
        )
        require(sha256(view.path, watch) == view.digest, "Embedding hash mismatch")
        array = np.load(view.path, mmap_mode="r", allow_pickle=False)
        require(
            array.dtype == np.dtype("<f4")
            and array.shape == (view.windows, 384)
            and array.flags.c_contiguous
            and bool(np.isfinite(array).all()),
            "Invalid embedding array",
        )
        end = offset + view.windows
        embeddings[offset:end] = array
        after = view.path.stat()
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "Embedding changed during admission",
        )
        del array
        if view.role == "fit":
            targets[offset:end] = view.target
            weights[offset:end] = loss_weight(view, contract)
        else:
            selection[view.identity] = slice(offset, end)
        offset = end
    embeddings.flags.writeable = targets.flags.writeable = weights.flags.writeable = False
    return LoadedCache(embeddings, targets, weights, selection, contract)
