"""Prepare the pinned, external FMAKv2 evaluation corpus; never train a model.

Uses official FMAK audio archives and FMAKv2 annotations. Verified downloaded
archives are removed after successful extraction unless --keep-archives is set.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from drone.download import download
from drone.keys import LABEL_SCHEMA, encode_key
from experiments.reference.datasets import Track, prepare
from experiments.reference.features import FEATURE_SPEC, audio_digest
from experiments.reference.train import prepared_path

ANNOTATION_URL = "https://zenodo.org/api/records/12759100/files/fmakv2.csv/content"
ANNOTATION_MD5 = "3b2d16784ffbda850c8ddf0519478bfd"


def prep_one(arguments):
    identity, path, cache, key = arguments
    start = time.monotonic()
    features, ref = prepare(Track(identity, path, key), cache)
    return identity, features, ref, audio_digest(path), time.monotonic() - start


def write_json(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(pending, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-metadata", type=Path, required=True)
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    args.output.mkdir(parents=True, exist_ok=True)
    downloads, audio = args.output / "downloads", args.output / "audio"
    downloads.mkdir(exist_ok=True)
    audio.mkdir(exist_ok=True)
    metadata_bytes = args.audio_metadata.read_bytes()
    audio_record = json.loads(metadata_bytes)
    if str(audio_record["id"]) != "10719860":
        raise ValueError("Expected the pinned official FMAK audio record")
    annotation = args.output / "fmakv2.csv"
    download(ANNOTATION_URL, annotation, "md5:" + ANNOTATION_MD5)
    labels = {}
    with annotation.open() as stream:
        for row in csv.DictReader(stream):
            identity = f"{int(row['track_id']):06}"
            tonic, mode = row["key_and_mode"].split()
            key = tonic, mode.lower()
            encode_key(*key)
            if identity in labels:
                raise ValueError(f"Duplicate annotation ID {identity}")
            labels[identity] = key
    if len(labels) != 5489:
        raise ValueError(f"Expected 5489 annotated tracks, got {len(labels)}")
    status_path = args.output / "preparation.json"
    record = {
        "status": "running",
        "source": "external evaluation only; no model fitting",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "audio_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "annotation_md5": ANNOTATION_MD5,
        "workers": args.workers,
        "archives": [],
    }
    all_start = time.monotonic()
    found = {}
    for item in sorted(audio_record["files"], key=lambda x: x["key"]):
        name = item["key"]
        if not name.endswith(".zip"):
            continue
        if Path(name).name != name or not item["checksum"].startswith("md5:"):
            raise ValueError("Unexpected archive identity or checksum")
        url = item["links"]["self"]
        if not url.startswith("https://zenodo.org/api/records/10719860/files/"):
            raise ValueError("Audio URL is outside the pinned official record")
        needed = 50 * 1024**3 + 2 * item["size"]
        if shutil.disk_usage(args.output).free < needed:
            raise RuntimeError("Insufficient space to preserve 50 GiB host headroom")
        archive = downloads / name
        if args.reuse and (args.reuse / name).is_file():
            archive = args.reuse / name
        start = time.monotonic()
        print(f"Downloading/verifying {name} ({item['size'] / 1e9:.2f} GB)", flush=True)
        download(url, archive, item["checksum"])
        download_seconds = time.monotonic() - start
        start = time.monotonic()
        matched = 0
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                filename = Path(member.filename).name
                if "__MACOSX" in member.filename or not filename.endswith(".mp3"):
                    continue
                try:
                    identity = f"{int(Path(filename).stem):06}"
                except ValueError:
                    continue
                if identity not in labels:
                    continue
                if identity in found:
                    raise ValueError(f"Duplicate annotated audio identity {identity}")
                target = audio / f"{identity}.mp3"
                # Always finish the ZIP read, which checks the entry CRC. Files are
                # replaced only after verified extraction from the MD5-checked archive.
                pending = target.with_suffix(".pending")
                with bundle.open(member) as source, pending.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                os.replace(pending, target)
                found[identity] = target
                matched += 1
        record["archives"].append(
            {
                "name": name,
                "md5": item["checksum"],
                "bytes": item["size"],
                "matched_tracks": matched,
                "download_verify_seconds": download_seconds,
                "extract_seconds": time.monotonic() - start,
            }
        )
        record["audio_tracks"] = len(found)
        record["elapsed_seconds"] = time.monotonic() - all_start
        write_json(status_path, record)
        print(f"Audio ready: {len(found)}/5489 tracks", flush=True)
        if not args.keep_archives and archive.parent == downloads:
            archive.unlink()
    missing = set(labels) - set(found)
    if missing:
        raise ValueError(f"Missing {len(missing)} annotated audio tracks: {sorted(missing)[:10]}")
    if shutil.disk_usage(args.output).free < 60 * 1024**3:
        raise RuntimeError("Need 60 GiB free before feature preparation")
    ids = sorted(labels)
    jobs = [
        (identity, str(found[identity]), str(args.output), labels[identity]) for identity in ids
    ]
    features, refs, tracks = [], [], []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for count, (identity, f, ref, digest, seconds) in enumerate(pool.map(prep_one, jobs), 1):
            features.append(f)
            refs.append(ref)
            tracks.append({"id": identity, "sha256": digest, "key": list(labels[identity])})
            if count % 100 == 0:
                record.update(feature_tracks=count, elapsed_seconds=time.monotonic() - all_start)
                write_json(status_path, record)
                print(f"Features ready: {count}/5489", flush=True)
    meta = {
        "feature_spec": FEATURE_SPEC,
        "label_schema": LABEL_SCHEMA,
        "frames": 300,
        "refs": refs,
        "ids": ids,
        "tracks": tracks,
        "provenance": {
            "annotation_url": ANNOTATION_URL,
            "annotation_md5": ANNOTATION_MD5,
            "audio_record": 10719860,
            "archives": record["archives"],
        },
    }
    target = prepared_path(args.output, "fmak")
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(".pending")
    with pending.open("wb") as stream:
        np.savez(stream, features=np.stack(features), metadata=json.dumps(meta))
    os.replace(pending, target)
    record.update(
        status="complete",
        feature_tracks=len(ids),
        prepared_bundle=str(target),
        feature_seconds=time.monotonic() - start,
        elapsed_seconds=time.monotonic() - all_start,
    )
    write_json(status_path, record)
    print(f"Completed external FMAKv2 bundle: {target}", flush=True)


if __name__ == "__main__":
    main()
