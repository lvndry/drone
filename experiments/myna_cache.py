"""Prospective Myna embedding-cache contract; no encoder, decoding or training calls."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from drone.keys import LABEL_SCHEMA, transpose_key

SCHEMA = "drone-myna-rate-window-cache-v2"
PLAN_SCHEMA = "drone-myna-rate-window-plan-v2"
AUGMENTATION_METHOD = "pitch-by-playback-rate-polyphase-v1"
BASE_RESAMPLER = {
    "method": "torchaudio-24k-to-16k-float32-target-length-v1",
    "functional_sha256": "dc462a67270b2250279c47725346b26bc6cdc9dce77dd08602d0fcce68f78711",
    "default_dtype": "float32",
    "original_rate": 24000,
    "target_rate": 16000,
}
ENCODER_SHA = "5dde8aa186da0fc0bfdd2e5a7c1ea3e6c6af6cd4683089b65c4fed6af8335374"
PINS = {
    "plan": "c3e59ffb764ca5f51444447b0a18a69543228e60a9e3723d925ddd82b0fc8373",
    "labels": "b8942b5913b8ebacdf2fc9e0ebca72a934aaa7c56d4e1a97d0f60fac84354323",
    "groups": "7505dcae2a91bf038537cee4a2f8e24f7929011c2bb56d8fe6649f6a1df8dcd7",
    "waveforms": "7d56908dc5e84c165c7844d1e810ed678123ae7d0a93939c60a2e600b6402b4c",
    "prepared_waveforms": "6de0a8b63bc62fba53d4d07163e4dd6704bdeb9fe8cf7bb827cee9077da6b7f4",
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
    payload = path.read_bytes()
    watch()
    require(hashlib.sha256(payload).hexdigest() == expected, "Metadata hash mismatch")
    if path.suffix == ".gz":
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
            payload = stream.read(64 * 1024**2 + 1)
        require(len(payload) <= 64 * 1024**2, "Oversized decoded metadata")
    return payload


def checked_json(path: Path, expected: str, watch: Callable[[], None]) -> dict[str, Any]:
    value = json.loads(checked_bytes(path, expected, watch))
    require(isinstance(value, dict), "Expected metadata object")
    return value


def is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def base_sample_count(samples_24k: int) -> int:
    """Pinned TorchAudio target-length rounding and available convolution output, no audio fix."""
    require(type(samples_24k) is int and samples_24k > 0, "Invalid original sample count")
    rounded_target = math.ceil(float(np.float32(2 * samples_24k / 3)))
    return min(rounded_target, 2 * (samples_24k // 3 + 1))


class RateGeometry(TypedDict):
    samples_16k: int
    mel_frames: int
    windows: int
    discarded_frames: int
    rate_up: int
    rate_down: int
    actual_rate: float
    pitch_cents_error: float


def rate_geometry(samples_16k: int, shift: int) -> RateGeometry:
    """Declared playback-rate geometry only; this function never resamples audio."""
    require(
        type(samples_16k) is int and samples_16k > 0 and type(shift) is int and -6 <= shift <= 6,
        "Invalid rate geometry input",
    )
    rate_up = 16000
    rate_down = round(16000 * 2 ** (shift / 12))
    output = (samples_16k * rate_up + rate_down - 1) // rate_down
    frames = output // 512 + 1
    windows, tail = divmod(frames, 196)
    return {
        "samples_16k": output,
        "mel_frames": frames,
        "windows": windows,
        "discarded_frames": tail,
        "rate_up": rate_up,
        "rate_down": rate_down,
        "actual_rate": rate_down / rate_up,
        "pitch_cents_error": 1200 * math.log2(rate_down / rate_up) - 100 * shift,
    }


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
    digest: str | None
    byte_count: int | None


@dataclass(frozen=True)
class Contract:
    views: tuple[View, ...]
    selection_ids: dict[str, list[str]]
    labels: dict[str, int]
    fit_windows: int
    selection_windows: int
    fit_counts: dict[str, int]
    measured: bool

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
    identities: Mapping[str, str | None],
    *,
    require_production_counts: bool = True,
    metadata_only: bool = False,
) -> Contract:
    """Exact membership/lineage/13-view validation; small fixtures may relax counts only."""
    require(
        manifest.get("schema_version") == (PLAN_SCHEMA if metadata_only else SCHEMA)
        and manifest.get("complete") is (False if metadata_only else True),
        "Incomplete cache/schema",
    )
    require(manifest.get("label_schema") == LABEL_SCHEMA, "Different label schema")
    require(
        manifest.get("base_resampler") == BASE_RESAMPLER,
        "Different base-resampling length contract",
    )
    require(
        manifest.get("augmentation_method") == AUGMENTATION_METHOD, "Different augmentation method"
    )
    require(manifest.get("identities") == identities, "Cache source/qualification binding differs")
    require(identities.get("encoder_sha256") == ENCODER_SHA, "Different encoder")
    require(
        all((metadata_only and value is None) or is_sha(value) for value in identities.values()),
        "Malformed identity",
    )
    if metadata_only:
        require(
            identities.get("qualification_sha256") is None,
            "A projection cannot claim qualification",
        )
    expected: dict[str, tuple[str, str]] = {}
    selection: dict[str, list[str]] = {}
    for field, (domain, role, count) in ROLE_FIELDS.items():
        ids = plan[field]
        require(isinstance(ids, list) and len(ids) == len(set(ids)), "Duplicate role IDs")
        if require_production_counts:
            require(len(ids) == count, "Different approved role counts")
        require(
            all(
                isinstance(identity, str) and identity.startswith(domain + ":") for identity in ids
            ),
            "Wrong domain",
        )
        require(not set(ids) & expected.keys(), "Cross-role identity")
        expected.update((identity, (domain, role)) for identity in ids)
        if role == "selection":
            selection[domain] = ids.copy()
    require(set(labels) == set(expected), "Extra/missing reference labels")
    require(
        all(type(label) is int and 0 <= label < 24 for label in labels.values()),
        "Invalid key class",
    )
    require(plan["expected_refs"] == labels, "Plan/label export differs")
    waveform_rows = {row["id"]: row for row in waveforms}
    group_rows_by_id = {row["id"]: row for row in groups}
    require(
        len(waveform_rows) == len(waveforms) == len(expected)
        and set(waveform_rows) == set(expected),
        "Waveform membership differs",
    )
    require(
        len(group_rows_by_id) == len(groups) == len(expected)
        and set(group_rows_by_id) == set(expected),
        "Group membership differs",
    )
    rows = {row["id"]: row for row in manifest["tracks"]}
    require(
        len(rows) == len(manifest["tracks"]) == len(expected) and set(rows) == set(expected),
        "Cache membership differs",
    )
    views: list[View] = []
    resolved_root = root.resolve()
    used_paths: set[Path] = set()
    for identity, (domain, role) in expected.items():
        row = rows[identity]
        source = waveform_rows[identity]
        family = group_rows_by_id[identity]
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
        samples_24k = source["samples"]
        require(
            type(samples_24k) is int and samples_24k > 0 and row["samples_24k"] == samples_24k,
            "Different PCM support",
        )
        samples_16k = base_sample_count(samples_24k)
        frames = samples_16k // 512 + 1
        windows, tail = divmod(frames, 196)
        require(windows > 0, "No complete input window")
        require(
            (
                row["samples_16k"],
                row["mel_frames"],
                row["complete_windows"],
                row["discarded_frames"],
            )
            == (samples_16k, frames, windows, tail),
            "Window geometry differs",
        )
        required_shifts = set(range(-6, 7)) if role == "fit" else {0}
        actual_shifts = [view["shift"] for view in row["views"]]
        require(
            all(type(shift) is int for shift in actual_shifts)
            and len(actual_shifts) == len(required_shifts)
            and set(actual_shifts) == required_shifts,
            "Missing/duplicate/extra pitch views",
        )
        for view in sorted(row["views"], key=lambda item: item["shift"]):
            geometry = rate_geometry(samples_16k, view["shift"])
            view_windows = geometry["windows"]
            require(view_windows > 0, "No complete shifted input window")
            require(
                all(
                    view[key] == geometry[key]
                    for key in (
                        "samples_16k",
                        "mel_frames",
                        "windows",
                        "discarded_frames",
                        "rate_up",
                        "rate_down",
                        "actual_rate",
                    )
                ),
                "Per-view rate/window geometry differs",
            )
            require(
                math.isclose(
                    view["pitch_cents_error"],
                    geometry["pitch_cents_error"],
                    rel_tol=0,
                    abs_tol=1e-9,
                ),
                "Different physical pitch rounding",
            )
            require(
                view["dtype"] == "float32" and view["shape"] == [view_windows, 384],
                "Embedding geometry/dtype differs",
            )
            require(
                view["window_order"] == "all-complete-196-frame-windows-in-order",
                "Capped/reordered windows",
            )
            if metadata_only:
                require(
                    all(
                        view[key] is None
                        for key in (
                            "raw_samples",
                            "trimmed_samples",
                            "padded_samples",
                            "sha256",
                            "bytes",
                        )
                    ),
                    "Projected artifacts must remain unobserved",
                )
            else:
                raw_samples, trimmed, padded = (
                    view[key] for key in ("raw_samples", "trimmed_samples", "padded_samples")
                )
                require(
                    all(type(count) is int for count in (raw_samples, trimmed, padded))
                    and (raw_samples, trimmed, padded) == (geometry["samples_16k"], 0, 0),
                    "Playback-rate views forbid padding, trimming or altered output lengths",
                )
            relative = Path(view["path"])
            require(
                not relative.is_absolute()
                and ".." not in relative.parts
                and relative.suffix == ".npy",
                "Unsafe embedding path",
            )
            path = root / relative
            require(
                path not in used_paths
                and (metadata_only or path.resolve().is_relative_to(resolved_root)),
                "Repeated/escaping embedding path",
            )
            used_paths.add(path)
            if not metadata_only:
                require(
                    is_sha(view["sha256"])
                    and type(view["bytes"]) is int
                    and view["bytes"] > view_windows * 384 * 4,
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
                    view_windows,
                    path,
                    view["sha256"],
                    view["bytes"],
                )
            )
    return Contract(
        tuple(views),
        selection,
        labels,
        sum(view.windows for view in views if view.role == "fit"),
        sum(view.windows for view in views if view.role == "selection"),
        {
            domain: sum(
                view_domain == domain and view_role == "fit"
                for view_domain, view_role in expected.values()
            )
            for domain in DOMAIN_MASS
        },
        measured=not metadata_only,
    )


def group_rows(payload: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(payload.decode())))


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
    require(contract.measured, "A metadata-only projection cannot load embeddings or train")
    require(contract.embedding_bytes <= max_embedding_bytes, "Embedding RAM admission exceeded")
    ordered = [view for view in contract.views if view.role == "fit"] + [
        view for view in contract.views if view.role == "selection"
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
        if view.digest is None or view.byte_count is None:
            raise ValueError("Unmeasured embedding identity")
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
    embeddings.flags.writeable = False
    targets.flags.writeable = False
    weights.flags.writeable = False
    return LoadedCache(embeddings, targets, weights, selection, contract)


def validate_waveform_lineage(canonical: dict[str, Any], prepared: dict[str, Any]) -> None:
    """The physical disk cache may relocate bytes, never change roles or semantic inputs."""
    require(
        canonical.get("complete") is True and prepared.get("complete") is True,
        "Incomplete waveform lineage",
    )
    require(
        prepared.get("canonical_manifest_sha256") == PINS["waveforms"]
        and prepared.get("plan_sha256") == PINS["plan"]
        and prepared.get("labels_sha256") == PINS["labels"],
        "Prepared waveform lineage differs",
    )
    require(
        prepared.get("provider") == "readonly-disk-full-f32"
        and prepared.get("no_seek_quantization_role_or_label_changes") is True,
        "Different physical cache contract",
    )
    original = {row["id"]: row for row in canonical["rows"]}
    physical = {row["id"]: row for row in prepared["rows"]}
    require(
        len(original) == len(canonical["rows"]) == len(physical) == len(prepared["rows"])
        and original.keys() == physical.keys(),
        "Physical waveform membership differs",
    )
    fields = (
        "id",
        "role",
        "target",
        "samples",
        "bytes",
        "sha256",
        "waveform_sha256",
        "canonical_samples",
    )
    for identity, row in physical.items():
        require(
            all(row[key] == original[identity][key] for key in fields),
            "Physical waveform role/target/support/hash differs",
        )
