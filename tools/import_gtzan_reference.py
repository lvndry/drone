"""Reproduce metadata-only GTZAN eligibility; never opens audio or executes detectors."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "experiments/studies/v2-data-contract/data/gtzan-reference"
GENRES = (
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
)
IDS = {f"{genre}.{i:05d}" for genre in GENRES for i in range(100)}
PINS = {
    "annotation": (
        "9ef1591f81ee2f7821c7ce41e774ea7ac7f94cbe",
        "59faf8174287d4a714f0875ee78ffb88e692533b",
        "2df9326b758c696d5097e80cb561b7baf172da392a4d42300c6dbb945baabe42",
    ),
    "faults": (
        "fedc781026cf1a20034830f8071c98d93a422016",
        "35c4f075c4bd3c8a8dfe772fb08acbc6080d80e5",
        "ea65dfdeb0e2f2944c077b402ce5b3318402dce686c9b65119c22df7a07c33f8",
    ),
}
INDEX_SHA256 = "fd9655a35aebbdd09b120ec53d6bd36056b24f4b161ed21d9571364ecdc1f5ea"
INDEX_SHA = "533ca050855f22acf2feb283d9957fe3"  # publisher MD5, not audio checksum


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_object(kind: str, raw: bytes) -> bytes:
    return hashlib.sha1(f"{kind} {len(raw)}\0".encode() + raw).digest()


def git_tree(files: dict[str, bytes]) -> str:
    """Reconstruct the complete pinned repository tree, not just selected members."""
    nodes: dict[str, bytes] = {}
    folders: dict[str, dict[str, bytes]] = {}
    for path, raw in files.items():
        head, sep, tail = path.partition("/")
        if sep:
            folders.setdefault(head, {})[tail] = raw
        else:
            nodes[head] = b"100644 " + head.encode() + b"\0" + git_object("blob", raw)
    for name, children in folders.items():
        nodes[name + "/"] = b"40000 " + name.encode() + b"\0" + bytes.fromhex(git_tree(children))
    return git_object("tree", b"".join(nodes[n] for n in sorted(nodes))).hex()


def archive(raw: bytes, tree: str) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if sum(i.file_size for i in z.infolist()) > 10 * 1024 * 1024:
            raise ValueError("metadata archive exceeds expanded bound")
        files: dict[str, bytes] = {}
        roots: set[str] = set()
        for item in z.infolist():
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe archive path")
            roots.add(path.parts[0])
            if item.is_dir():
                continue
            relative = "/".join(path.parts[1:])
            if not relative or relative in files:
                raise ValueError("duplicate/invalid member")
            files[relative] = z.read(item)
        if len(roots) != 1 or git_tree(files) != tree:
            raise ValueError("archive does not reconstruct pinned Git tree")
        return files


def map_key(value: str) -> int | None:
    if not re.fullmatch(r"-1|[0-9]|1[0-9]|2[0-3]", value):
        raise ValueError("invalid annotation integer")
    k = int(value)
    return None if k == -1 else 2 * ((k % 12 + 9) % 12) + int(k >= 12)


def filtered_ids(files: dict[str, bytes]) -> tuple[set[str], dict[str, int]]:
    union: set[str] = set()
    counts: dict[str, int] = {}
    for name, expected in [("train", 443), ("valid", 197), ("test", 290)]:
        lines = files[f"{name}_filtered.txt"].decode().splitlines()
        ids: list[str] = []
        for line in lines:
            match = re.fullmatch(r"([a-z]+)/([a-z]+\.\d{5})\.wav", line)
            if not match or match[1] != match[2].split(".")[0] or match[2] not in IDS:
                raise ValueError("unmapped fault-list ID")
            ids.append(match[2])
        selected = set(ids)
        if len(ids) != len(selected) or len(selected) != expected or union & selected:
            raise ValueError("invalid or overlapping filtered partitions")
        counts[name] = len(selected)
        union |= selected
    return union, counts


def produce(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repos: dict[str, dict[str, bytes]] = {}
    sources: dict[str, Any] = {}
    for name, (commit, tree, digest) in PINS.items():
        raw = (source / f"{name}.zip").read_bytes()
        if sha256(raw) != digest:
            raise ValueError(f"{name} archive SHA mismatch")
        witness = json.loads((source / f"{name}-commit.json").read_bytes())
        if witness["sha"] != commit or witness["tree"]["sha"] != tree:
            raise ValueError("commit/tree witness mismatch")
        repos[name] = archive(raw, tree)
        owner = (
            "audiocontentanalysis/dataset-gtzan-key" if name == "annotation" else "boblsturm/GTZAN"
        )
        sources[name] = {
            "commit": commit,
            "tree": tree,
            "archive_sha256": digest,
            "bytes": len(raw),
            "url": f"https://codeload.github.com/{owner}/zip/{commit}",
        }
    compressed = (source / "recording-index.json.gz").read_bytes()
    index_raw = gzip.decompress(compressed)
    if hashlib.md5(index_raw).hexdigest() != INDEX_SHA or sha256(index_raw) != INDEX_SHA256:
        raise ValueError("recording-index publisher MD5 mismatch")
    index = json.loads(index_raw)
    if index["version"] != "1.0" or set(index["tracks"]) != IDS:
        raise ValueError("unexpected recording-index schema/coverage")
    sources["recording_index"] = {
        "url": "https://zenodo.org/records/13993311/files/gtzan_genre_index_1.0.json?download=1",
        "publisher_md5": INDEX_SHA,
        "decoded_sha256": sha256(index_raw),
        "gzip_sha256": sha256(compressed),
        "decoded_bytes": len(index_raw),
        "version": "1.0",
    }
    annotations: dict[str, tuple[str, int | None, str]] = {}
    for path, raw in repos["annotation"].items():
        if not path.endswith(".lerch.txt"):
            continue
        match = re.fullmatch(r"gtzan_key/genres/([a-z]+)/([a-z]+\.\d{5})\.lerch\.txt", path)
        if not match or match[2] not in IDS or match[1] != match[2].split(".")[0]:
            raise ValueError("invalid annotation path")
        ident = match[2]
        if ident in annotations:
            raise ValueError("duplicate annotation ID")
        annotations[ident] = (raw.decode().strip(), map_key(raw.decode().strip()), sha256(raw))
    if set(annotations) != IDS:
        raise ValueError("annotation coverage mismatch")
    filtered, partitions = filtered_ids(repos["faults"])
    ledger: list[dict[str, Any]] = []
    for ident in sorted(IDS):
        value, ref, digest = annotations[ident]
        genre = ident.split(".")[0]
        member = f"genres/{genre}/{ident}.wav"
        audio_path, audio_md5 = index["tracks"][ident]["audio"]
        if audio_path != "gtzan_genre/" + member or not re.fullmatch("[0-9a-f]{32}", audio_md5):
            raise ValueError("recording-index audio mapping mismatch")
        reasons: list[str] = []
        if ref is None:
            reasons.append("unknown_or_modulating_key")
        if genre == "classical":
            reasons.append("classical_unannotated")
        if ident not in filtered:
            reasons.append("not_in_sturm_filtered_union")
        if ident == "jazz.00054":
            reasons.append("published_loader_corruption_flag")
        ledger.append(
            {
                "id": ident,
                "genre": genre,
                "annotation_integer": int(value),
                "drone_ref": ref,
                "annotation_sha256": digest,
                "audio_member": member,
                "expected_member_md5": audio_md5,
                "raw_single_key_reference": ref is not None,
                "in_sturm_filtered_union": ident in filtered,
                "named_filtered_intersection": ref is not None and ident in filtered,
                "metadata_eligible": not reasons,
                "exclusions": reasons,
                "audio_verified": False,
                "overlap_cleared": False,
                "admitted": False,
            }
        )

    def counts(field: str) -> dict[str, int]:
        selected = Counter(r["genre"] for r in ledger if r[field])
        return {g: selected[g] for g in GENRES}

    summary = {
        "schema": "gtzan-reference-metadata-v1",
        "status": "PROPOSED_METADATA_ONLY_NOT_AUDIO_ADMITTED",
        "sources": sources,
        "filtered_partitions": partitions,
        "counts_by_genre": {
            f: counts(f)
            for f in [
                "raw_single_key_reference",
                "in_sturm_filtered_union",
                "named_filtered_intersection",
                "metadata_eligible",
            ]
        },
        "total_ids": len(ledger),
        "admitted": 0,
        "detector_outputs": False,
        "retrieved_utc": "2026-11-12",
        "eligibility_rule": "valid_0_to_23 AND nonclassical AND sturm_union AND not jazz.00054",
        "license_scope": "MIT annotation notice only; no audio clearance inferred",
    }
    return summary, ledger


def encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary, ledger = produce(args.data / "sources")
    summary["ledger_decoded_sha256"] = sha256(encode(ledger))
    payloads = {
        "summary.json": encode(summary),
        "reference-ledger.json.gz": gzip.compress(encode(ledger), mtime=0),
        "ANNOTATION-MIT-LICENSE.txt": archive(
            (args.data / "sources/annotation.zip").read_bytes(), PINS["annotation"][1]
        )["LICENSE"],
    }
    for name, raw in payloads.items():
        path = args.data / name
        if args.check:
            if path.read_bytes() != raw:
                raise ValueError(f"output mismatch: {name}")
        else:
            path.write_bytes(raw)
    print(json.dumps(summary["counts_by_genre"], sort_keys=True))


if __name__ == "__main__":
    main()
