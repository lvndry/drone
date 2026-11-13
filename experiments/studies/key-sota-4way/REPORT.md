# 4-way key detection benchmark against published SOTA — 2026-11-12

**Decision: the reference (madmom) leads the drone-v1 model on this panel; the
gap is genre-specific, not uniform. No weights or architecture decision changes
from this benchmark; it is measurement evidence, not a promotion gate.**

| Detector | MIREX | Exact | Notes |
|---|---:|---:|---|
| drone-v1 (this repo) | 75.7% | 67.7% | 109,128 parameters, K10 architecture |
| madmom (all-conv KEY_CNN) | **82.9%** | **77.4%** | reference; CC BY-NC-SA benchmark use |
| libkeyfinder 2.2.8 | 47.6% | 37.6% | native C++ DSP reference |
| MusicFM + MLP (MIREX 2026) | — | — | blocked: published weights absent (below) |

Panel: 93 tracks, CPU-only, one shared ffmpeg decode per track. Scoring is
drone's own convention imported directly (`drone.keys.score_mirex` /
`score_breakdown`): exact 1.0, same-mode perfect fifth 0.5 (symmetric, interval
5 or 7), relative major/minor 0.3, parallel mode 0.2, else 0.0.

## Panel

- **Primary — Swiss Army (SWD) v2.1**: all 48 tracks carrying global-key labels
  (Schubert *Winterreise*, performances HU33 and SC06;
  `02_Annotations/ann_audio_globalkey.csv`, `WorkId;PerfId;Tonic:maj|min`).
  Full-length, roughly 5 minutes each. Genre: classical.
- **Secondary anchor — GiantSteps Key 604**: deterministic round-robin
  stratified sample of 45 tracks (one per genre tag in rotation, 12 tags,
  sorted by track id). These are marked `dev_contaminated: true` in the
  manifest: this is drone's published development benchmark, so drone's
  numbers on them are not an independent estimate.

Duration coverage is limited: 71 tracks under 3 minutes, 20 in 3–5 minutes,
2 in 5–8 minutes, 0 longer. The panel has **no 5–10 minute DJ-length tracks**;
the classical set covers ~5 minutes only.

## Detectors

| # | Detector | Implementation | Status |
|---|---|---|---|
| 1 | drone-v1 | this repository, `drone.Predictor`, `weights/drone-v1.pt`, torch CPU | ran |
| 2 | libkeyfinder 2.2.8 | source sha256-checked; prebuilt `libkeyfinder.so`; bench CLI `kf_cli` linked against it | ran |
| 3 | madmom | `madmom.features.key.CNNKeyRecognitionProcessor` (all-conv KEY_CNN, weights CC BY-NC-SA — benchmark use) | ran |
| 4 | MusicFM + MLP | HF `jiangbing33/MusicFM_MLP_For_Music_Global_Key_Detection` | **blocked** |

Architecture context: drone-v1 uses the K10 architecture — "K10: A Compact
Convolutional Network for Global Key Estimation" (John Hurliman, MIREX 2026) —
trained on 1,363 Beatport previews. madmom's KEY_CNN is a much larger
all-convolutional network trained on a multi-genre corpus, which the panel
results reflect.

### MusicFM block (exact evidence)

The HF repository is **public**: all code (`predict.py`, `model.py`, the
`musicfm/` package, requirements) fetches without a token. But it contains
**no weight file at all**: the recursive tree listing has zero
`.pt/.bin/.safetensors` entries, and

```
curl -I https://huggingface.co/jiangbing33/MusicFM_MLP_For_Music_Global_Key_Detection/resolve/main/weights/best_model.pt
→ HTTP 404 "Entry not found"
```

`predict.py` requires `weights/best_model.pt` (the complete fine-tuned state
dict; the base `pretrained_msd.pt` is not needed at inference), so inference
cannot run. A locally trained pilot checkpoint uses a different architecture
(mel-spectrogram backbone) and is not the published submission; it was
excluded to avoid conflating models. Published GiantSteps numbers per the repo
README: 72.19% exact / 77.62% MIREX.

## Decode and label normalisation

One ffmpeg decode per track, once, shared by every detector:

```
ffmpeg -ac 1 -ar 44100 -c:a pcm_s16le <track>
```

No loudness normalisation or trim (documented choice). Drone re-resamples to
22.05 kHz internally; libkeyfinder consumes 44.1 kHz directly; madmom uses its
own internal resampling of the same file.

All references normalised to drone's 24-class scheme
`class_id = sharp_tonic_index * 2 + (minor ? 1 : 0)` with
`TONICS = C C# D D# E F F# G G# A A# B`:

- SWD `D:min` / `C:maj` → split on `:`, `maj`→major, `min`→minor,
  flats→sharps (Db→C#, Eb→D#, Gb→F#, Ab→G#, Bb→A#). First key of a
  `|`-separated list.
- GiantSteps `D minor` / `F# major` parsed via drone's `parse_key`; two-key
  labels (`A major | C# minor`) use the **first listed key**.
- libkeyfinder's flat-named output and madmom `KEY_LABELS` (flats for
  Db/Eb/Ab/Bb) mapped to sharps with the same substitution.

## Results

Overall:

| Detector | Exact | MIREX | Error-free? | Median ms/track |
|---|---:|---:|---|---:|
| drone-v1 | 67.7% (63/93) | 75.7% | yes | 111.7 |
| madmom | 77.4% (72/93) | 82.9% | yes | 1981.6 |
| libkeyfinder 2.2.8 | 37.6% (35/93) | 47.6% | yes | 1878.1 |

drone-v1 breakdown: 63 exact, 10 fifth, 4 relative, 6 parallel, 10 other.
libkeyfinder: 35 exact, 11 fifth, 2 relative, 16 parallel, 29 other — a large
share of its misses are parallel-mode or unrelated keys.

### By genre

| Genre (n) | drone-v1 MIREX | madmom MIREX | libkeyfinder MIREX |
|---|---:|---:|---:|
| classical (48) | 72.5% | **88.5%** | 44.6% |
| breaks (5) | 90.0% | 90.0% | 54.0% |
| dubstep (5) | 80.0% | 80.0% | 60.0% |
| electronica (4) | 57.5% | 57.5% | 12.5% |
| funk (3) | 100.0% | 100.0% | 73.3% |
| hard (4) | 100.0% | 100.0% | 100.0% |
| hip-hop (4) | 80.0% | 80.0% | 75.0% |
| house (4) | 100.0% | 87.5% | 87.5% |
| other (4) | 62.5% | 87.5% | 5.0% |
| pop (4) | 100.0% | 100.0% | 50.0% |
| techno (4) | 75.0% | 25.0% | 12.5% |
| trance (4) | 27.5% | 40.0% | 32.5% |

Per-genre sample sizes of 3–5 are not statistically meaningful; the classical
row (n=48) is the only one worth reading closely. The honest summary: drone
is strong on its EDM training domain (house, funk, hard, dubstep, breaks, pop)
and weak on **trance and electronica** (0/4 exact on trance for both drone and
libkeyfinder, and madmom also weak there at 0/4 exact — 40.0% MIREX). On
classical, madmom leads drone-v1 by 16 MIREX points (88.5% vs 72.5%);
drone-v1's 0/4 exact on trance is its most visible weakness on the panel.

### By duration

| Duration (n) | drone-v1 MIREX | madmom MIREX | libkeyfinder MIREX |
|---|---:|---:|---:|
| <3m (71) | 77.6% | 77.6% | 48.3% |
| 3–5m (20) | 66.5% | 100.0% | 40.0% |
| 5–8m (2) | 100.0% | 100.0% | 100.0% |

madmom's 100.0% on the 20-track 3–5m bucket is the classical rows; the
3–5m bucket is classical-dominated (20 SWD tracks, all ≥180 s). Drone-v1
drops 11 points from short to medium duration, consistent with its
~2-minute training clips. The 5–8m bucket has 2 tracks; it is not a
measurement.

## How to rerun

```bash
uv sync --locked --all-extras        # torch is a core dependency
pip install madmom                    # optional; missing -> detector skipped
uv run python -m tools.benchmark_sota_4way \
  --swd-dir /path/to/swd-v2.1 \
  --giantsteps-dir /path/to/giantsteps \
  --kf-cli /path/to/kf_cli \
  --work-dir /path/to/work \
  --out-dir /path/to/results
```

Options: `--n-gs 45` (stratified sample size), `--drone-weights` (default:
drone's bundled `drone-v1.pt`), `--threads 8`, `--smoke` (2 tracks per source),
`--limit N`, and `--skip-drone` / `--skip-libkeyfinder` / `--skip-madmom`.
Requires `ffmpeg`/`ffprobe` on PATH. libkeyfinder needs the sha256-checked
2.2.8 release source and the bench CLI linked against the prebuilt
`libkeyfinder.so`; a missing
or skipped detector degrades the run with a warning.

The committed run artifacts are in [data/](data/): `manifest.json`
(93-track panel: id, source, genre, duration, raw and normalised reference),
`bench.json` (per-detector summary, by-genre, by-duration, timing) and
`bench.csv` (per-track predictions, relations, scores, timings).
`drone-v1` weights: the checkpoint in `weights/drone-v1.pt`.

## Limitations

- **No 5–10 minute DJ-length tracks** on the panel. The README already flags
  that reliable accuracy on 5–10 minute house/melodic-house is unestablished;
  this benchmark cannot speak to it.
- **SWD key labels are classical-only** (vocal art songs), so the "classical"
  number is one narrow style, not classical in general.
- The 45 GiantSteps tracks are **dev-contaminated for drone-v1** (its
  published development set). drone-v1's GiantSteps numbers are optimistic;
  madmom and libkeyfinder have no such exposure.
- Per-genre buckets below n=10 do not support genre-specific claims beyond
  description.
- libkeyfinder numbers depend on the particular 2.2.8 build and bench CLI;
  they are a native-DSP reference point, not a tuned configuration.
- madmom KEY_CNN weights are CC BY-NC-SA: benchmark use only, not
  redistributable with this repository.
