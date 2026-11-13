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
from drone.keys import FLATS, TONICS, encode_key, score_breakdown

# ----------------------------------------------------------------------------
# label normalisation -> drone 24-class id (sharp tonic * 2 + minor)


def norm_key(text: str) -> int | None:
    """'D:min', 'C minor', 'D# minor | E major', 'Eb: maj' -> class id (first key)."""
    if text is None:
        return None
    t = str(text).strip().strip('"').split("|")[0].strip()
    if not t:
        return None
    if ":" in t:
        tonic, mode = t.split(":", 1)
    else:
        parts0 = t.split()
        tonic = parts0[0] if parts0 else ""
        mode = parts0[1] if len(parts0) > 1 else "maj"
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
        out = subprocess.check_output(
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
        return float(out.strip() or 0.0)
    except Exception:
        return 0.0


def ff_decode(src: Path, out: Path) -> None:
    """The ONE identical decode used for every detector."""
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "44100",
            "-c:a",
            "pcm_s16le",
            str(out),
        ],
        check=True,
    )


# ----------------------------------------------------------------------------
# genre buckets


def gs_bucket(tag: str) -> str:
    g = (tag or "").strip().lower()
    if not g:
        return "unknown"
    if "house" in g:
        return "house"
    if "techno" in g or "minimal" in g or "minimale" in g:
        return "techno"
    if "trance" in g:
        return "trance"
    if "hip-hop" in g or "hiphop" in g or "glitch-hop" in g:
        return "hip-hop"
    if "pop" in g:
        return "pop"
    if "dubstep" in g:
        return "dubstep"
    if "drum-and-bass" in g:
        return "dubstep"
    if "break" in g:
        return "breaks"
    if "electronica" in g or "chill" in g:
        return "electronica"
    if "funk" in g:
        return "funk"
    if "hard" in g:
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
            parts = [p.strip().strip('"') for p in line.split(";")]
            if len(parts) < 3:
                continue
            workid, perf, key = parts[0], parts[1], parts[2]
            wav = swd_wav / f"{workid}_{perf}.wav"
            if not wav.is_file():
                continue
            cid = norm_key(key)
            if cid is None:
                continue
            panel.append(
                {
                    "track_id": f"swd-{workid}_{perf}",
                    "source": "swd-v2.1",
                    "audio": str(wav),
                    "genre": "classical",
                    "bucket": "classical",
                    "ref_raw": key,
                    "ref": cid,
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
        for keyf in sorted(gs_key.iterdir()):
            tid = keyf.name  # e.g. 1004923.LOFI.key
            base = tid[:-4]  # 1004923.LOFI
            mp3 = gs_audio / f"{base}.mp3"
            if not mp3.is_file():
                continue
            keytext = keyf.read_text(errors="replace").strip()
            cid = norm_key(keytext)
            if cid is None:
                continue
            gfile = gs_genre / f"{base}.genre"
            tag = gfile.read_text(errors="replace").strip() if gfile.is_file() else ""
            rows.append(
                {
                    "track_id": f"gs-{base}",
                    "source": "giantsteps",
                    "audio": str(mp3),
                    "genre_raw": tag,
                    "bucket": gs_bucket(tag),
                    "ref_raw": keytext,
                    "ref": cid,
                    "dev_contaminated": True,
                    "tid_sort": base,
                }
            )
    else:
        print(
            f"warning: {gs_key} not found; GiantSteps part of the panel is empty", file=sys.stderr
        )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["bucket"]].append(r)
    for g in groups:
        groups[g].sort(key=lambda x: x["tid_sort"])
    order = sorted(groups.keys())
    idx = {g: 0 for g in groups}
    picked: list[dict[str, Any]] = []
    while len(picked) < n_gs:
        progressed = False
        for g in order:
            if len(picked) >= n_gs:
                break
            if idx[g] < len(groups[g]):
                picked.append(groups[g][idx[g]])
                idx[g] += 1
                progressed = True
        if not progressed:
            break
    for r in picked:
        r.pop("tid_sort", None)
        panel.append(r)
    if not panel:
        raise SystemExit(
            "Panel is empty: pass --swd-dir and --giantsteps-dir that contain the corpora"
        )
    for r in panel:
        r["duration"] = ffprobe_duration(Path(r["audio"]))
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
    if [r["raw_enum"] for r in rows] != list(range(25)) or artifact(path) != before:
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
    src: Path, directory: Path, contract: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Only content/policy-bound, complete WAVs can enter the shared decode cache."""
    source_sha = file_sha(src)
    identity = dict(source_sha256=source_sha, decode_contract=contract)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    wav, receipt = directory / f"{digest}.wav", directory / f"{digest}.json"
    if wav.is_file() and receipt.is_file():
        try:
            previous = json.loads(receipt.read_text())
            if previous["identity"] == identity and previous["wav_sha256"] == file_sha(wav):
                validate_wav(wav)
                return wav, dict(
                    source=artifact(src),
                    wav=artifact(wav),
                    cache_hit=True,
                    decode_identity_sha256=digest,
                )
        except (OSError, ValueError, KeyError, EOFError, wave.Error):
            pass
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as scratch:
        pending = Path(scratch) / "decode.wav"
        ff_decode(src, pending)
        validate_wav(pending)
        if file_sha(src) != source_sha:
            raise ValueError("Source audio changed during decode")
        metadata = Path(scratch) / "receipt.json"
        metadata.write_text(json.dumps(dict(identity=identity, wav_sha256=file_sha(pending))))
        os.replace(pending, wav)
        os.replace(metadata, receipt)
    return wav, dict(
        source=artifact(src), wav=artifact(wav), cache_hit=False, decode_identity_sha256=digest
    )


def aggregate_rows(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Headline scores retain the complete panel; errors and abstentions earn zero."""
    valid = [(r[f"{name}_pred"], r["ref"]) for r in rows if r[f"{name}_pred"] is not None]
    statuses = [
        r.get(
            f"{name}_status",
            "error"
            if r[f"{name}_rel"] == "error"
            else "abstain"
            if r[f"{name}_pred"] is None
            else "key",
        )
        for r in rows
    ]
    if any(status not in {"key", "error", "abstain"} for status in statuses):
        raise ValueError("Unknown detector status")
    for row, status in zip(rows, statuses, strict=True):
        if (row[f"{name}_pred"] is None) != (status != "key"):
            raise ValueError("Detector status/prediction mismatch")
    counts = dict(score_breakdown([p for p, _ in valid], [q for _, q in valid])) if valid else {}
    weighted = sum(
        counts.get(k, 0) * w
        for k, w in (("exact", 1), ("fifth", 0.5), ("relative", 0.3), ("parallel", 0.2))
    )
    n = len(rows)
    return dict(
        n=n,
        n_total=n,
        n_valid=len(valid),
        n_error=statuses.count("error"),
        n_abstain=statuses.count("abstain"),
        counts=counts,
        exact=round(counts.get("exact", 0) / n, 4) if n else 0.0,
        mirex=round(weighted / n, 4) if n else 0.0,
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
        probs = np.asarray(self.mm_proc(str(wav)))
        if probs.shape != (1, 24) or not np.isfinite(probs).all():
            raise ValueError("Expected finite madmom CNN output shape(1,24)")
        idx = int(np.argmax(probs[0]))
        tonic, mode = self.mm_labels[idx].split()
        return encode_key(FLATS.get(tonic, tonic), mode)


def musicfm_status() -> dict[str, Any]:
    """MusicFM-MLP: reported blocked, not run.

    The MIREX 2026 submission repo (Hugging Face
    jiangbing33/MusicFM_MLP_For_Music_Global_Key_Detection) is public: every
    file of code (predict.py, model.py, musicfm/ package, requirements)
    fetches without a token. But the repo contains no weight file at all:
    the recursive tree listing has zero .pt/.bin/.safetensors entries, and

    ```
    curl -I https://huggingface.co/jiangbing33/MusicFM_MLP_For_Music_Global_Key_Detection/resolve/main/weights/best_model.pt
    -> HTTP 404 "Entry not found"
    ```

    predict.py requires weights/best_model.pt (the complete fine-tuned state
    dict; it does not need the base pretrained_msd.pt at inference), so
    inference cannot run. A locally trained pilot checkpoint uses a different
    architecture (mel-spectrogram backbone) and is not the published
    submission, so it is excluded to avoid conflating models.
    """
    return {
        "status": "blocked",
        "reason": (
            "HF repo jiangbing33/MusicFM_MLP_For_Music_Global_Key_Detection is "
            "public (code fetches, no 401), but the required weights file "
            "weights/best_model.pt is ABSENT: resolve -> HTTP 404 'Entry not found'; "
            "recursive tree/main lists no .pt/.bin/.safetensors. Cannot run inference."
        ),
        "published_gs": "72.19% exact / 77.62% MIREX (per repo README)",
    }


# ----------------------------------------------------------------------------
def summarize(
    rows: list[dict[str, Any]],
    name: str,
    ms: Sequence[float],
) -> dict[str, Any]:
    out = aggregate_rows(rows, name)
    out.update(
        median_ms=round(statistics.median(ms), 1) if ms else None,
        p95_ms=round(float(np.percentile(ms, 95)), 1) if ms else None,
        total_s=round(sum(ms) / 1000, 1),
    )
    out["by_genre"] = {
        g: aggregate_rows([r for r in rows if r["bucket"] == g], name)
        for g in sorted({r["bucket"] for r in rows})
    }
    out["by_duration"] = {
        b: aggregate_rows([r for r in rows if duration_bucket(r["duration"]) == b], name)
        for b in ("<3m", "3-5m", "5-8m", "8m+")
        if any(duration_bucket(r["duration"]) == b for r in rows)
    }
    return out


def track_record(t: dict[str, Any]) -> dict[str, Any]:
    return {
        **{
            k: t[k]
            for k in ("track_id", "source", "ref_raw", "ref", "duration", "dev_contaminated")
        },
        "genre": t.get("genre", t.get("genre_raw", "")),
        "bucket": t["bucket"],
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
    for d in (out_dir, decode_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    panel = build_panel(args.swd_dir, args.giantsteps_dir, args.n_gs)
    if args.smoke:
        swd = [t for t in panel if t["source"] == "swd-v2.1"]
        gs = [t for t in panel if t["source"] == "giantsteps"]
        panel = swd[:2] + gs[:2]
    if args.limit:
        panel = panel[: args.limit]
    print(
        f"panel: {len(panel)} tracks "
        f"(swd={sum(t['source'] == 'swd-v2.1' for t in panel)}, "
        f"gs={sum(t['source'] == 'giantsteps' for t in panel)})",
        flush=True,
    )
    manifest: dict[str, Any] = {
        "n": len(panel),
        "source_counts": {
            s: sum(1 for t in panel if t["source"] == s) for s in ("swd-v2.1", "giantsteps")
        },
        "bucket_counts": {
            b: sum(1 for t in panel if t["bucket"] == b)
            for b in sorted({t["bucket"] for t in panel})
        },
        "duration_buckets": {
            b: sum(1 for t in panel if duration_bucket(t["duration"]) == b)
            for b in ("<3m", "3-5m", "5-8m", "8m+")
        },
        "tracks": [track_record(t) for t in panel],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"manifest written ({len(panel)} tracks), building panel took {time.time() - t0:.1f}s",
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
    for i, t in enumerate(panel):
        wav, decode_receipt = cached_decode(Path(t["audio"]), decode_dir, decode_policy)
        rec = track_record(t)
        rec["decode_artifact"] = decode_receipt
        for name, fn in active:
            t_s = time.perf_counter()
            status = "key"
            try:
                pred = fn(wav)
                if pred is None:
                    status = "abstain"
            except Exception as e:
                pred = None
                status = "error"
                errors[f"{name}:{t['track_id']}"].append(f"{type(e).__name__}: {e}")
            ms = 1000 * (time.perf_counter() - t_s)
            ms_by_detector[name].append(ms)
            rec[f"{name}_pred"] = pred
            rec[f"{name}_status"] = status
            rec[f"{name}_ms"] = round(ms, 1)
            if pred is None:
                rec[f"{name}_rel"] = status
                rec[f"{name}_score"] = 0.0
            else:
                bd = score_breakdown([pred], [t["ref"]])
                rec[f"{name}_rel"] = [k for k, v in bd.items() if v][0]
                rec[f"{name}_score"] = (
                    bd["exact"] * 1.0
                    + bd["fifth"] * 0.5
                    + bd["relative"] * 0.3
                    + bd["parallel"] * 0.2
                )
        rec["musicfm_pred"] = None
        rec["musicfm_score"] = None
        rows.append(rec)
        if (i + 1) % 5 == 0 or args.smoke:
            print(
                f"[{i + 1}/{len(panel)}] {t['track_id']} ref={t['ref']} "
                + " ".join(f"{name}={rec[f'{name}_pred']}" for name, _ in active)
                + f"  ({time.time() - t0:.0f}s elapsed)",
                flush=True,
            )

    summary: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "panel": {
            "n": len(rows),
            "source_counts": manifest["source_counts"],
            "bucket_counts": manifest["bucket_counts"],
            "duration_buckets": manifest["duration_buckets"],
            "median_duration_s": round(statistics.median([r["duration"] for r in rows]), 1),
        },
        "decode": {
            "command": "ffmpeg -ac 1 -ar 44100 -c:a pcm_s16le (once per track, shared by all detectors)",
            "loudness": "none (no normalisation)",
            "contract": decode_policy,
        },
        "scoring": {
            "convention": "drone/keys.py",
            "weights": {"exact": 1.0, "fifth": 0.5, "relative": 0.3, "parallel": 0.2, "other": 0.0},
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
        "track_artifacts": [{"track_id": r["track_id"], **r["decode_artifact"]} for r in rows],
        "detectors_extra": {
            "madmom_mode": detectors.madmom_mode,
            "provenance": detectors.provenance,
            "per_detector_errors": dict(errors) if errors else "none",
            "wall_clock_total_s": round(time.time() - t0, 1),
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

    cols = [
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
    with open(out_dir / "bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"DONE in {time.time() - t0:.0f}s. results -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
