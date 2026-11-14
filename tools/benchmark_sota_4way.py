"""4-way musical key detection benchmark against published SOTA references.

Decodes every panel track ONCE with one identical ffmpeg command into a
canonical 44.1 kHz mono 16-bit wav, then feeds that SAME file to every
detector. No loudness/normalisation is applied (documented choice). Each
detector then runs at its own native rate internally:

* drone-v1: re-decodes the wav to 22.05 kHz mono, K10 CQT + CNN (torch CPU)
* libkeyfinder: reads the 44.1 kHz wav directly, native C++ FFT chroma + KK profiles
* madmom: all-conv key net (KEY_CNN, CC BY-NC-SA) on a 24-band log-spectrogram
* musicfm: BLOCKED (published weights missing from the public HF repo)

Scoring uses drone's exact convention (drone/keys.py):
exact 1.0, same-mode perfect fifth 0.5, relative major/minor 0.3,
parallel mode 0.2, else 0.0; symmetric fifths (interval 5 or 7).

libkeyfinder and madmom are optional. An omitted CLI or unavailable CNN is
recorded as skipped; an explicitly supplied incompatible CLI fails qualification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drone import Predictor, default_weights
from drone.keys import FLATS, TONICS, WEIGHTS, encode_key, score_breakdown, score_mirex

# ----------------------------------------------------------------------------
# label normalisation -> drone 24-class id (sharp tonic * 2 + minor)


def norm_key(text: str) -> int | None:
    """'D:min', 'C minor', 'D# minor | E major', 'Eb: maj' -> class id (first key)."""
    if text is None:
        return None
    label = str(text).strip().strip('"').split("|")[0].strip()
    if not label:
        return None
    if ":" in label:
        tonic, mode = label.split(":", 1)
    else:
        words = label.split()
        tonic = words[0] if words else ""
        mode = words[1] if len(words) > 1 else "maj"
    tonic = tonic.strip()
    mode = mode.strip().lower()
    modes = {"maj": "major", "major": "major", "min": "minor", "minor": "minor"}
    if mode not in modes:
        return None
    mode = modes[mode]
    tonic = FLATS.get(tonic, tonic)
    if tonic not in TONICS:
        return None
    return encode_key(tonic, mode)


def ffprobe_duration(path: Path) -> float:
    try:
        output = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(path),
            ],
            stderr=subprocess.DEVNULL,
        )
        return float(output.strip() or 0.0)
    except Exception:
        return 0.0


def ff_decode(source: Path, output: Path) -> None:
    """The ONE identical decode used for every detector."""
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        check=True,
    )


# ----------------------------------------------------------------------------
# genre buckets


def gs_bucket(tag: str) -> str:
    genre = (tag or "").strip().lower()
    if not genre:
        return "unknown"
    if "house" in genre:
        return "house"
    if "techno" in genre or "minimal" in genre or "minimale" in genre:
        return "techno"
    if "trance" in genre:
        return "trance"
    if "hip-hop" in genre or "hiphop" in genre or "glitch-hop" in genre:
        return "hip-hop"
    if "pop" in genre:
        return "pop"
    if "dubstep" in genre:
        return "dubstep"
    if "drum-and-bass" in genre:
        return "dubstep"
    if "break" in genre:
        return "breaks"
    if "electronica" in genre or "chill" in genre:
        return "electronica"
    if "funk" in genre:
        return "funk"
    if "hard" in genre:
        return "hard"
    return "other"


def duration_bucket(sec: float) -> str:
    if sec < 180:
        return "<3m"
    if sec < 300:
        return "3-5m"
    if sec < 480:
        return "5-8m"
    return "8m+"


# ----------------------------------------------------------------------------
# panel build


def build_panel(swd_dir: Path, gs_dir: Path, n_gs: int) -> list[dict[str, Any]]:
    panel: list[dict[str, Any]] = []
    # --- primary: SWD v2.1 (Schubert Winterreise): all labelled, classical, ~5 min
    swd_csv = swd_dir / "extracted" / "02_Annotations" / "ann_audio_globalkey.csv"
    swd_wav = swd_dir / "extracted" / "01_RawData" / "audio_wav"
    if swd_csv.is_file():
        for line in swd_csv.read_text().splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            parts = [part.strip().strip('"') for part in line.split(";")]
            if len(parts) < 3:
                continue
            work_id = parts[0]
            performance = parts[1]
            key = parts[2]
            wav = swd_wav / f"{work_id}_{performance}.wav"
            if not wav.is_file():
                continue
            class_id = norm_key(key)
            if class_id is None:
                continue
            panel.append(
                {
                    "track_id": f"swd-{work_id}_{performance}",
                    "source": "swd-v2.1",
                    "audio": str(wav),
                    "genre": "classical",
                    "bucket": "classical",
                    "ref_raw": key,
                    "ref": class_id,
                    "dev_contaminated": False,
                }
            )
    else:
        print(f"warning: {swd_csv} not found; SWD part of the panel is empty", file=sys.stderr)
    # --- secondary anchor: GiantSteps (dev-contaminated for drone), stratified sample
    gs_audio = gs_dir / "audio"
    gs_key = gs_dir / "annotations" / "key"
    gs_genre = gs_dir / "annotations" / "genre"
    rows: list[dict[str, Any]] = []
    if gs_key.is_dir():
        for key_file in sorted(gs_key.iterdir()):
            file_name = key_file.name  # e.g. 1004923.LOFI.key
            base = file_name[:-4]  # 1004923.LOFI
            mp3 = gs_audio / f"{base}.mp3"
            if not mp3.is_file():
                continue
            key_text = key_file.read_text(errors="replace").strip()
            class_id = norm_key(key_text)
            if class_id is None:
                continue
            genre_file = gs_genre / f"{base}.genre"
            tag = genre_file.read_text(errors="replace").strip() if genre_file.is_file() else ""
            rows.append(
                {
                    "track_id": f"gs-{base}",
                    "source": "giantsteps",
                    "audio": str(mp3),
                    "genre_raw": tag,
                    "bucket": gs_bucket(tag),
                    "ref_raw": key_text,
                    "ref": class_id,
                    "dev_contaminated": True,
                    "tid_sort": base,
                }
            )
    else:
        print(
            f"warning: {gs_key} not found; GiantSteps part of the panel is empty", file=sys.stderr
        )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["bucket"]].append(row)
    for bucket in groups:
        groups[bucket].sort(key=lambda track: track["tid_sort"])
    order = sorted(groups.keys())
    next_position = {bucket: 0 for bucket in groups}
    picked: list[dict[str, Any]] = []
    while len(picked) < n_gs:
        progressed = False
        for bucket in order:
            if len(picked) >= n_gs:
                break
            if next_position[bucket] < len(groups[bucket]):
                picked.append(groups[bucket][next_position[bucket]])
                next_position[bucket] += 1
                progressed = True
        if not progressed:
            break
    for track in picked:
        track.pop("tid_sort", None)
        panel.append(track)
    if not panel:
        raise SystemExit(
            "Panel is empty: pass --swd-dir and --giantsteps-dir that contain the corpora"
        )
    for track in panel:
        track["duration"] = ffprobe_duration(Path(track["audio"]))
    return panel


# ----------------------------------------------------------------------------
# detectors

KF_SCHEMA = "drone.keyfinder-reference.v1"
KF_CLASSES = (18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.absolute()), "sha256": file_sha(path)}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def model_artifacts() -> Sequence[str]:
    from madmom.models import KEY_CNN  # ty: ignore[import-not-found]

    return KEY_CNN


def keyfinder_result(raw: str) -> dict[str, Any]:
    """Reject legacy token output, unknown enums, and inconsistent JSON fields."""
    row = json.loads(raw)
    if not isinstance(row, dict) or row.get("schema") != KF_SCHEMA:
        raise ValueError("Incompatible KeyFinder JSON protocol")
    value = row.get("raw_enum")
    if type(value) is not int or not 0 <= value <= 24:
        raise ValueError("Unknown KeyFinder raw enum")
    if value == 24:
        expected = dict(status="silence", tonic=None, mode=None, drone_class=None)
    else:
        code = KF_CLASSES[value]
        expected = dict(
            status="key",
            tonic=TONICS[code // 2],
            mode="minor" if code % 2 else "major",
            drone_class=code,
        )
    if any(row.get(key) != value or key not in row for key, value in expected.items()):
        raise ValueError("Inconsistent KeyFinder enum/status/key mapping")
    if row["drone_class"] is not None and type(row["drone_class"]) is not int:
        raise ValueError("KeyFinder class must be integer")
    return row


def qualify_keyfinder(path: Path) -> dict[str, Any]:
    before = artifact(path)
    raw = subprocess.check_output([str(path), "--self-test"], text=True, timeout=30)
    rows = [keyfinder_result(line) for line in raw.splitlines()]
    if [row["raw_enum"] for row in rows] != list(range(25)) or artifact(path) != before:
        raise ValueError("KeyFinder self-test must cover all25 enums with stable binary bytes")
    source = Path(__file__).parent / "native/keyfinder_reference.cpp"
    return dict(
        status="active",
        backend="libkeyfinder JSON reference",
        binary=before,
        schema=KF_SCHEMA,
        self_test_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        source=artifact(source) if source.is_file() else None,
        source_binding="repository source witness; binary build provenance is external",
    )


def decode_contract() -> dict[str, Any]:
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise ValueError("ffmpeg unavailable")
    return dict(
        version="canonical-wav-v1",
        sample_rate=44100,
        channels=1,
        codec="pcm_s16le",
        loudness="none",
        trim="none",
        ffmpeg=artifact(Path(binary)),
    )


def validate_wav(path: Path) -> None:
    with wave.open(str(path), "rb") as stream:
        if (
            stream.getnchannels(),
            stream.getsampwidth(),
            stream.getframerate(),
            stream.getcomptype(),
        ) != (1, 2, 44100, "NONE"):
            raise ValueError("Invalid canonical WAV encoding")
        frames = stream.getnframes()
        if frames <= 0:
            raise ValueError("Empty canonical WAV")
        read = 0
        while block := stream.readframes(65536):
            read += len(block)
        if read != frames * 2:
            raise ValueError("Truncated canonical WAV")


def cached_decode(
    source: Path, directory: Path, contract: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Only content/policy-bound, complete WAVs can enter the shared decode cache."""
    source_sha = file_sha(source)
    identity = dict(source_sha256=source_sha, decode_contract=contract)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    wav = directory / f"{digest}.wav"
    receipt = directory / f"{digest}.json"
    if wav.is_file() and receipt.is_file():
        try:
            previous = json.loads(receipt.read_text())
            if previous["identity"] == identity and previous["wav_sha256"] == file_sha(wav):
                validate_wav(wav)
                return wav, dict(
                    source=artifact(source),
                    wav=artifact(wav),
                    cache_hit=True,
                    decode_identity_sha256=digest,
                )
        except (OSError, ValueError, KeyError, EOFError, wave.Error):
            pass
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as scratch:
        pending = Path(scratch) / "decode.wav"
        ff_decode(source, pending)
        validate_wav(pending)
        if file_sha(source) != source_sha:
            raise ValueError("Source audio changed during decode")
        metadata = Path(scratch) / "receipt.json"
        metadata.write_text(json.dumps(dict(identity=identity, wav_sha256=file_sha(pending))))
        os.replace(pending, wav)
        os.replace(metadata, receipt)
    return wav, dict(
        source=artifact(source), wav=artifact(wav), cache_hit=False, decode_identity_sha256=digest
    )


def aggregate_rows(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Headline scores retain the complete panel; errors and abstentions earn zero."""
    valid = [(row[f"{name}_pred"], row["ref"]) for row in rows if row[f"{name}_pred"] is not None]
    statuses = [
        row.get(
            f"{name}_status",
            "error"
            if row[f"{name}_rel"] == "error"
            else "abstain"
            if row[f"{name}_pred"] is None
            else "key",
        )
        for row in rows
    ]
    if any(status not in {"key", "error", "abstain"} for status in statuses):
        raise ValueError("Unknown detector status")
    for row, status in zip(rows, statuses, strict=True):
        if (row[f"{name}_pred"] is None) != (status != "key"):
            raise ValueError("Detector status/prediction mismatch")
    predictions = [prediction for prediction, _ in valid]
    references = [reference for _, reference in valid]
    counts = dict(score_breakdown(predictions, references)) if valid else {}
    valid_exact, valid_mirex = score_mirex(predictions, references) if valid else (0.0, 0.0)
    total = len(rows)
    return dict(
        n=total,
        n_total=total,
        n_valid=len(valid),
        n_error=statuses.count("error"),
        n_abstain=statuses.count("abstain"),
        counts=counts,
        exact=round(valid_exact * len(valid) / total, 4) if total else 0.0,
        mirex=round(valid_mirex * len(valid) / total, 4) if total else 0.0,
    )


class Detectors:
    """The detector set for one run; optional ones degrade to None."""

    def __init__(
        self,
        *,
        weights: Path | None = None,
        cache_dir: Path,
        kf_cli: Path | None = None,
        threads: int = 8,
        include_drone: bool = True,
        include_madmom: bool = True,
    ) -> None:
        torch.set_num_threads(threads)
        self.drone_pred: Predictor | None = None
        if include_drone:
            self.drone_pred = Predictor(
                weights=str(weights or default_weights()), device="cpu", workdir=str(cache_dir)
            )
        self.kf_cli = None
        self.provenance: dict[str, Any] = {
            "drone": {
                "status": "active" if include_drone else "skipped",
                "backend": "torch CPU",
                "weights": artifact(weights or default_weights()) if include_drone else None,
                "feature_cache": "content-bound CQT; may be warm on reruns",
            },
            "libkeyfinder": {"status": "skipped", "reason": "CLI omitted"},
            "madmom": {"status": "skipped", "reason": "disabled"},
        }
        if kf_cli is not None:
            kf_cli = kf_cli.absolute()
            if not kf_cli.is_file():
                raise ValueError(f"Provided KeyFinder CLI missing: {kf_cli}")
            proof = qualify_keyfinder(kf_cli)
            self.kf_cli = kf_cli
            self.provenance["libkeyfinder"] = proof
        self.madmom_mode: str | None = None
        self.mm_proc: Any = None
        self.mm_labels: Sequence[str] = ()
        if include_madmom:
            try:
                from madmom.features.key import KEY_LABELS, CNNKeyRecognitionProcessor  # ty: ignore[import-not-found]

                self.mm_proc = CNNKeyRecognitionProcessor()
                self.mm_labels = KEY_LABELS
                self.madmom_mode = "CNNKeyRecognitionProcessor (all-conv KEY_CNN)"
                self.provenance["madmom"] = {
                    "status": "active",
                    "backend": self.madmom_mode,
                    "package_version": package_version("madmom"),
                    "model_artifacts": [artifact(Path(path)) for path in model_artifacts()],
                }
            except Exception as error:
                self.mm_proc = None
                self.madmom_mode = None
                self.provenance["madmom"] = {
                    "status": "skipped",
                    "reason": f"CNN unavailable: {type(error).__name__}: {error}",
                }
                print(
                    f"warning: {self.provenance['madmom']['reason']}; skipping madmom",
                    file=sys.stderr,
                )

    def drone(self, wav: Path) -> int | None:
        assert self.drone_pred is not None
        tonic, mode, _ = self.drone_pred.predict(str(wav))
        return encode_key(tonic, mode)

    def kf(self, wav: Path) -> int | None:
        assert self.kf_cli is not None
        raw = subprocess.check_output([str(self.kf_cli), str(wav)], timeout=300, text=True)
        return keyfinder_result(raw)["drone_class"]

    def madmom(self, wav: Path) -> int | None:
        if self.mm_proc is None:
            raise RuntimeError("madmom CNN unavailable")
        probabilities = np.asarray(self.mm_proc(str(wav)))
        if probabilities.shape != (1, 24) or not np.isfinite(probabilities).all():
            raise ValueError("Expected finite madmom CNN output shape(1,24)")
        best_class = int(np.argmax(probabilities[0]))
        tonic, mode = self.mm_labels[best_class].split()
        return encode_key(FLATS.get(tonic, tonic), mode)


def musicfm_status() -> dict[str, Any]:
    """Report a dated artifact inspection, without claiming live availability checks."""
    return {
        "status": "blocked",
        "checked_revision": "6b7fc0fbd47fa28762e2f7ca26534f4e392080f6",
        "checked_date_utc": "2026-11-13",
        "reason": (
            "No fine-tuned weights in the inspected revision of "
            "jiangbing33/MusicFM_MLP_For_Music_Global_Key_Detection. "
            "No inference adapter is configured; availability is not rechecked at runtime. "
            "The locally trained pilot is a different checkpoint and recipe."
        ),
        "published_gs": "72.19% exact / 77.62% MIREX (per repo README)",
    }


# ----------------------------------------------------------------------------
def summarize(
    rows: list[dict[str, Any]],
    name: str,
    latencies_ms: Sequence[float],
) -> dict[str, Any]:
    summary = aggregate_rows(rows, name)
    summary.update(
        median_ms=round(statistics.median(latencies_ms), 1) if latencies_ms else None,
        p95_ms=round(float(np.percentile(latencies_ms, 95)), 1) if latencies_ms else None,
        total_s=round(sum(latencies_ms) / 1000, 1),
    )
    summary["by_genre"] = {
        genre: aggregate_rows([row for row in rows if row["bucket"] == genre], name)
        for genre in sorted({row["bucket"] for row in rows})
    }
    summary["by_duration"] = {
        bucket: aggregate_rows(
            [row for row in rows if duration_bucket(row["duration"]) == bucket], name
        )
        for bucket in ("<3m", "3-5m", "5-8m", "8m+")
        if any(duration_bucket(row["duration"]) == bucket for row in rows)
    }
    return summary


def track_record(track: dict[str, Any]) -> dict[str, Any]:
    return {
        **{
            key: track[key]
            for key in ("track_id", "source", "ref_raw", "ref", "duration", "dev_contaminated")
        },
        "genre": track.get("genre", track.get("genre_raw", "")),
        "bucket": track["bucket"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--swd-dir",
        type=Path,
        default=Path("~/drone-data/swd-v2.1").expanduser(),
        help="SWD v2.1 root containing extracted/",
    )
    parser.add_argument(
        "--giantsteps-dir",
        type=Path,
        default=Path("~/key-cnn/data/giantsteps").expanduser(),
        help="GiantSteps root containing audio/ and annotations/",
    )
    parser.add_argument(
        "--drone-weights",
        type=Path,
        default=None,
        help="drone checkpoint (default: drone's bundled drone-v1)",
    )
    parser.add_argument(
        "--kf-cli",
        type=Path,
        default=None,
        help="libkeyfinder benchmark CLI; omitted or missing skips libkeyfinder",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="results directory (default: <work-dir>/results)"
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("~/drone-bench").expanduser(),
        help="decode/ and cache/ live under this directory",
    )
    parser.add_argument("--n-gs", type=int, default=45, help="stratified GiantSteps sample size")
    parser.add_argument("--threads", type=int, default=8, help="torch CPU threads for drone")
    parser.add_argument("--smoke", action="store_true", help="two tracks per source only")
    parser.add_argument("--limit", type=int, default=0, help="cap the panel length")
    parser.add_argument("--skip-drone", action="store_true")
    parser.add_argument("--skip-libkeyfinder", action="store_true")
    parser.add_argument("--skip-madmom", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir or args.work_dir / "results"
    decode_dir = args.work_dir / "decode"
    cache_dir = args.work_dir / "cache"
    for directory in (out_dir, decode_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.time()
    panel = build_panel(args.swd_dir, args.giantsteps_dir, args.n_gs)
    if args.smoke:
        swd = [track for track in panel if track["source"] == "swd-v2.1"]
        giantsteps = [track for track in panel if track["source"] == "giantsteps"]
        panel = swd[:2] + giantsteps[:2]
    if args.limit:
        panel = panel[: args.limit]
    print(
        f"panel: {len(panel)} tracks "
        f"(swd={sum(track['source'] == 'swd-v2.1' for track in panel)}, "
        f"gs={sum(track['source'] == 'giantsteps' for track in panel)})",
        flush=True,
    )
    manifest: dict[str, Any] = {
        "n": len(panel),
        "source_counts": {
            source: sum(1 for track in panel if track["source"] == source)
            for source in ("swd-v2.1", "giantsteps")
        },
        "bucket_counts": {
            bucket: sum(1 for track in panel if track["bucket"] == bucket)
            for bucket in sorted({track["bucket"] for track in panel})
        },
        "duration_buckets": {
            bucket: sum(1 for track in panel if duration_bucket(track["duration"]) == bucket)
            for bucket in ("<3m", "3-5m", "5-8m", "8m+")
        },
        "tracks": [track_record(track) for track in panel],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"manifest written ({len(panel)} tracks), building panel took {time.time() - started:.1f}s",
        flush=True,
    )

    detectors = Detectors(
        weights=args.drone_weights,
        cache_dir=cache_dir,
        kf_cli=args.kf_cli if not args.skip_libkeyfinder else None,
        threads=args.threads,
        include_drone=not args.skip_drone,
        include_madmom=not args.skip_madmom,
    )
    if detectors.madmom_mode is not None:
        print("madmom mode:", detectors.madmom_mode, flush=True)
    active: list[tuple[str, Any]] = []
    if not args.skip_drone and detectors.drone_pred is not None:
        active.append(("drone", detectors.drone))
    if not args.skip_libkeyfinder and detectors.kf_cli is not None:
        active.append(("libkeyfinder", detectors.kf))
    if not args.skip_madmom and detectors.madmom_mode is not None:
        active.append(("madmom", detectors.madmom))
    if not active:
        raise SystemExit("No active detector; install one of drone/madmom or pass --kf-cli")

    rows: list[dict[str, Any]] = []
    ms_by_detector: dict[str, list[float]] = {name: [] for name, _ in active}
    errors: dict[str, list[str]] = defaultdict(list)
    decode_policy = decode_contract()
    for index, track in enumerate(panel):
        wav, decode_receipt = cached_decode(Path(track["audio"]), decode_dir, decode_policy)
        record = track_record(track)
        record["decode_artifact"] = decode_receipt
        for name, detect in active:
            detection_started = time.perf_counter()
            status = "key"
            try:
                prediction = detect(wav)
                if prediction is None:
                    status = "abstain"
            except Exception as error:
                prediction = None
                status = "error"
                errors[f"{name}:{track['track_id']}"].append(f"{type(error).__name__}: {error}")
            elapsed_ms = 1000 * (time.perf_counter() - detection_started)
            ms_by_detector[name].append(elapsed_ms)
            record[f"{name}_pred"] = prediction
            record[f"{name}_status"] = status
            record[f"{name}_ms"] = round(elapsed_ms, 1)
            if prediction is None:
                record[f"{name}_rel"] = status
                record[f"{name}_score"] = 0.0
            else:
                breakdown = score_breakdown([prediction], [track["ref"]])
                record[f"{name}_rel"] = [
                    relation for relation, matched in breakdown.items() if matched
                ][0]
                record[f"{name}_score"] = score_mirex([prediction], [track["ref"]])[1]
        record["musicfm_pred"] = None
        record["musicfm_score"] = None
        rows.append(record)
        if (index + 1) % 5 == 0 or args.smoke:
            print(
                f"[{index + 1}/{len(panel)}] {track['track_id']} ref={track['ref']} "
                + " ".join(f"{name}={record[f'{name}_pred']}" for name, _ in active)
                + f"  ({time.time() - started:.0f}s elapsed)",
                flush=True,
            )

    summary: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "panel": {
            "n": len(rows),
            "source_counts": manifest["source_counts"],
            "bucket_counts": manifest["bucket_counts"],
            "duration_buckets": manifest["duration_buckets"],
            "median_duration_s": round(statistics.median([row["duration"] for row in rows]), 1),
        },
        "decode": {
            "command": "ffmpeg -ac 1 -ar 44100 -c:a pcm_s16le (once per track, shared by all detectors)",
            "loudness": "none (no normalisation)",
            "contract": decode_policy,
        },
        "scoring": {
            "convention": "drone/keys.py",
            "weights": dict(WEIGHTS),
            "fifths": "symmetric (interval 5 or 7)",
        },
        "label_normalization": {
            "target": "drone 24-class: class_id = sharp_tonic_index*2 + (minor?1:0)",
            "swd": "'Tonic:maj|min' -> split on ':', maj->major/min->minor, flats->sharps, first '|' key",
            "giantsteps": "'Tonic major|minor' via drone parse_key, first '|' key for two-key",
            "musicfm_vocabulary": "sharps only (C..B#)",
        },
        "detectors": {},
        "provenance": {"harness": artifact(Path(__file__)), "torch_version": torch.__version__},
        "track_artifacts": [
            {"track_id": row["track_id"], **row["decode_artifact"]} for row in rows
        ],
        "detectors_extra": {
            "madmom_mode": detectors.madmom_mode,
            "provenance": detectors.provenance,
            "per_detector_errors": dict(errors) if errors else "none",
            "wall_clock_total_s": round(time.time() - started, 1),
        },
    }
    summary["detectors"]["drone-v1"] = (
        summarize(rows, "drone", ms_by_detector["drone"])
        if "drone" in ms_by_detector
        else {"status": "skipped"}
    )
    summary["detectors"]["libkeyfinder-2.2.8"] = (
        summarize(rows, "libkeyfinder", ms_by_detector["libkeyfinder"])
        if "libkeyfinder" in ms_by_detector
        else {"status": "skipped"}
    )
    summary["detectors"]["madmom"] = (
        summarize(rows, "madmom", ms_by_detector["madmom"])
        if "madmom" in ms_by_detector
        else {"status": "skipped"}
    )
    summary["detectors"]["musicfm-mlp"] = musicfm_status()
    manifest.update(
        provenance=summary["provenance"],
        detector_provenance=detectors.provenance,
        decode_contract=decode_policy,
        track_artifacts=summary["track_artifacts"],
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "bench.json").write_text(json.dumps(summary, indent=2))

    columns = [
        "track_id",
        "source",
        "genre",
        "bucket",
        "ref_raw",
        "ref",
        "duration",
        "dev_contaminated",
        "drone_status",
        "libkeyfinder_status",
        "madmom_status",
        "drone_pred",
        "drone_rel",
        "drone_score",
        "drone_ms",
        "libkeyfinder_pred",
        "libkeyfinder_rel",
        "libkeyfinder_score",
        "libkeyfinder_ms",
        "madmom_pred",
        "madmom_rel",
        "madmom_score",
        "madmom_ms",
        "musicfm_pred",
        "musicfm_score",
    ]
    with open(out_dir / "bench.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    print(f"DONE in {time.time() - started:.0f}s. results -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
