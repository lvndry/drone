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

libkeyfinder and madmom are optional plugins: a missing CLI binary or
package skips that detector with a warning instead of aborting the run.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drone import Predictor, default_weights
from drone.keys import FLATS, TONICS, encode_key, score_breakdown, score_mirex

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
    mode = "major" if mode.startswith("maj") else "minor"
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
        self.kf_cli = kf_cli if kf_cli is not None and kf_cli.is_file() else None
        if kf_cli is not None and self.kf_cli is None:
            print(f"warning: {kf_cli} not found; skipping libkeyfinder", file=sys.stderr)
        # madmom: prefer the all-conv key net (KEY_CNN, CC BY-NC-SA); fall back to DSP chroma + KK.
        self.madmom_mode: str | None = None
        self.mm_proc: Any = None
        self.mm_labels: Sequence[str] = ()
        if include_madmom and importlib.util.find_spec("madmom") is None:
            print("warning: madmom is not installed; skipping madmom", file=sys.stderr)
        elif include_madmom:
            try:
                from madmom.features.key import KEY_LABELS, CNNKeyRecognitionProcessor  # ty: ignore[import-not-found]

                self.mm_proc = CNNKeyRecognitionProcessor()
                self.mm_labels = KEY_LABELS
                self.madmom_mode = "CNNKeyRecognitionProcessor (all-conv KEY_CNN)"
            except Exception as e:
                from madmom.audio.chroma import HarmonicPitchClassProfile  # ty: ignore[import-not-found]
                from madmom.audio.signal import Signal  # ty: ignore[import-not-found]

                self.mm_proc = None
                self.madmom_mode = f"HarmonicPitchClassProfile+KK (fallback: {type(e).__name__})"
                self.mm_signal = Signal
                self.mm_chroma = HarmonicPitchClassProfile()
                self.mm_major = np.array(
                    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
                )
                self.mm_minor = np.array(
                    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
                )

    def drone(self, wav: Path) -> int | None:
        assert self.drone_pred is not None
        tonic, mode, _ = self.drone_pred.predict(str(wav))
        return encode_key(tonic, mode)

    def kf(self, wav: Path) -> int | None:
        assert self.kf_cli is not None
        out = (
            subprocess.check_output([str(self.kf_cli), str(wav)], stderr=subprocess.DEVNULL)
            .decode()
            .split()
        )
        try:
            return int(out[2])
        except (IndexError, ValueError):
            return None

    def madmom(self, wav: Path) -> int | None:
        if self.mm_proc is not None:
            probs = self.mm_proc(str(wav))
            idx = int(np.argmax(probs[0]))
            tonic, mode = self.mm_labels[idx].split()
            return encode_key(FLATS.get(tonic, tonic), mode)
        sig = self.mm_signal(str(wav), sr=44100)
        ch = np.asarray(self.mm_chroma(sig), dtype=np.float64).sum(0)
        ch = ch - ch.mean()

        def best(profile: np.ndarray) -> tuple[float, int]:
            bc, br = -2.0, 0
            for rot in range(12):
                r = np.roll(ch, rot)
                denom = np.linalg.norm(r) * np.linalg.norm(profile)
                c = (float(r @ profile) / denom) if denom > 0 else 0.0
                if c > bc:
                    bc, br = c, rot
            return bc, br

        cm, rm = best(self.mm_major)
        cn, rn = best(self.mm_minor)
        if cm >= cn:
            pc, minor = rm, 0
        else:
            pc, minor = rn, 1
        return pc * 2 + minor


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
    pred_key, rel_key = f"{name}_pred", f"{name}_rel"
    pred = [r[pred_key] for r in rows]
    ref = [r["ref"] for r in rows]
    valid = [(p, q) for p, q in zip(pred, ref) if p is not None]
    if valid:
        exact, mirex = score_mirex([p for p, _ in valid], [q for _, q in valid])
        counts = dict(score_breakdown([p for p, _ in valid], [q for _, q in valid]))
    else:
        exact = mirex = 0.0
        counts = {}
    n_err = sum(1 for r in rows if r[rel_key] == "error")
    out: dict[str, Any] = {
        "n_valid": len(valid),
        "n_error": n_err,
        "exact": round(exact, 4),
        "mirex": round(mirex, 4),
        "counts": counts,
        "median_ms": round(statistics.median(ms), 1) if ms else None,
        "p95_ms": round(float(np.percentile(ms, 95)), 1) if ms else None,
        "total_s": round(sum(ms) / 1000, 1),
    }
    # by genre bucket
    out["by_genre"] = {}
    for g in sorted({r["bucket"] for r in rows}):
        sub = [r for r in rows if r["bucket"] == g]
        pv = [(r[pred_key], r["ref"]) for r in sub if r[pred_key] is not None]
        if pv:
            e, m = score_mirex([p for p, _ in pv], [q for _, q in pv])
            out["by_genre"][g] = {"n": len(sub), "exact": round(e, 3), "mirex": round(m, 3)}
    # by duration bucket
    out["by_duration"] = {}
    for b in ("<3m", "3-5m", "5-8m", "8m+"):
        sub = [r for r in rows if duration_bucket(r["duration"]) == b]
        pv = [(r[pred_key], r["ref"]) for r in sub if r[pred_key] is not None]
        if pv:
            e, m = score_mirex([p for p, _ in pv], [q for _, q in pv])
            out["by_duration"][b] = {"n": len(sub), "exact": round(e, 3), "mirex": round(m, 3)}
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
    for i, t in enumerate(panel):
        wav = decode_dir / f"{t['track_id']}.wav"
        if not wav.is_file():
            ff_decode(Path(t["audio"]), wav)
        rec = track_record(t)
        for name, fn in active:
            t_s = time.perf_counter()
            try:
                pred = fn(wav)
            except Exception as e:
                pred = None
                errors[f"{name}:{t['track_id']}"].append(f"{type(e).__name__}: {e}")
            ms = 1000 * (time.perf_counter() - t_s)
            ms_by_detector[name].append(ms)
            rec[f"{name}_pred"] = pred
            rec[f"{name}_ms"] = round(ms, 1)
            if pred is None:
                rec[f"{name}_rel"] = "error"
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
        "detectors_extra": {
            "madmom_mode": detectors.madmom_mode,
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
