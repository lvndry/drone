"""Post-assignment numeric target export; no model/audio/feature arrays, no role selection."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import re
import struct
import zipfile
from pathlib import Path
from typing import Any

EXPECTED_METADATA = "94f65bb54329b239028dfffc1e8bf37f7d037f6e1e7c25f6721e73efd2c037bc"


def fma_labels(prepared: Path, annotation: Path, needed: list[str]) -> dict[str, Any]:
    with zipfile.ZipFile(prepared) as archive:
        raw = archive.read("metadata.npy")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_METADATA:
        raise ValueError("Pinned prepared target source changed")
    start = 10 if raw[6] == 1 else 12
    size = struct.unpack("<H" if start == 10 else "<I", raw[8:start])[0]
    header = ast.literal_eval(raw[start : start + size].decode())
    if header["shape"] != () or not header["descr"].startswith("<U"):
        raise ValueError("Unsupported metadata-only NPY")
    text = raw[start + size :].decode("utf-32-le").rstrip("\0")

    def field(name: str) -> list[Any]:
        match = re.search('"' + name + r'"\s*:', text)
        if match is None:
            raise ValueError("Missing metadata field")
        return json.JSONDecoder().raw_decode(text[match.end() :].lstrip())[0]

    identities = field("ids")
    wanted = set(needed)
    indices = {
        "fmak:" + identity: i
        for i, identity in enumerate(identities)
        if "fmak:" + identity in wanted
    }
    if set(indices) != wanted:
        raise ValueError("Missing selected fitting/inner ID")
    refs = field("refs")
    selected = {identity: refs[i] for identity, i in indices.items()}
    if any(type(v) is not int or not 0 <= v < 24 for v in selected.values()):
        raise ValueError("Noncanonical label encoding")
    annotation_raw = annotation.read_bytes()
    md5 = hashlib.md5(annotation_raw).hexdigest()
    if md5 != "3b2d16784ffbda850c8ddf0519478bfd":
        raise ValueError("Pinned officialFMAKv2 annotation MD5 differs")
    from drone.keys import encode_key

    csv_selected = {}
    for row in csv.DictReader(io.StringIO(annotation_raw.decode())):
        identity = "fmak:" + f"{int(row['track_id']):06}"
        if identity not in wanted:
            continue
        tonic, mode = row["key_and_mode"].split()
        csv_selected[identity] = encode_key(tonic, mode.lower())
    if csv_selected != selected:
        raise ValueError("Pinned CSV target encoding differs from prepared refs")
    return {
        "csv_selected_labels_match_prepared_refs": True,
        "labels": selected,
        "source": {
            "prepared_metadata_npy_sha256": EXPECTED_METADATA,
            "annotation_path": str(annotation),
            "annotation_md5": md5,
            "annotation_sha256": hashlib.sha256(annotation_raw).hexdigest(),
            "annotation_url": "https://zenodo.org/api/records/12759100/files/fmakv2.csv/content",
            "provenance_caveat": "FMAK expert labels;FMAKv2 changes around200 labels not verified by original FMAK authors;no relabeling here",
            "only_requested_fit_inner_ids_exported": True,
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
