# Training documentation: drone

Audit update after release: the original K10 fitting/validation split contained
two identical-audio crossings (seven duplicate groups overall). Future training
uses the shared content-deduplicated split. The old variable-length evaluation
also padded shorter tracks to their batch peers; corrected evaluation batches
only equal-length inputs and preserves each track's actual length. Historical
scores below describe the original runs and have not been silently recomputed.
See [the controlled multi-domain protocol](../experiments/K10_MULTIDOMAIN_PROTOCOL.md)
for a fresh comparison of fitting data under both corrected contracts. These
findings do not establish contamination of the GiantSteps test set.

The inference audit also corrected `Predictor`'s old feature transpose: K10 expects
`(batch, 192 bins, time)`. ONNX now has dynamic time as well as batch dimensions,
with CPU parity checked against the unchanged PyTorch model at multiple lengths.
The original PyTorch weights remain the release candidate; the controlled
multi-domain comparison and its gate decision are recorded in
[the results](../experiments/K10_MULTIDOMAIN_RESULTS.md).

Current research status: the release recipe's three-seed GiantSteps mean is
73.58% MIREX / 66.8% exact. The corrected multi-domain comparison missed its
predeclared FMAK gate; the later [EMA pilot](../experiments/K10_EMA_RESULTS.md)
also failed. The [chunked CPU prototype](../experiments/K10_CHUNKED_RESULTS.md)
passed observed decision parity and its memory/latency development gates, but
native Euphoria integration is not measured yet. Release weights are unchanged.
The historical narrative below preserves original runs and code-line references;
these linked reports supersede its prospective experiment status. SWD audio is
now prepared but has not been scored by these candidates.

How the drone key classifier is trained, the strategies behind each choice, and what
we learned getting here. Audience: a competent engineer who has never seen this repo.
Every code reference below is `file:line` against the `k10-arch` branch.

The current model is a **109k-parameter CNN** that predicts 24 musical keys
(12 tonics × major/minor) from audio, trained on 1,363 EDM track previews and
scored on 604 held-out GiantSteps tracks. Its architecture follows **K10**:
"K10: A Compact Convolutional Network for Global Key Estimation" by John
Hurliman, MIREX 2026 Audio Key Detection, from the open-source jams project
(github.com/jhurliman/jams, MIT; `train_key_cnn.py`). The architecture and
training recipe below follow that reference, with independent training and the
checkpoint-selection differences described in §5. K10's published figure on
GiantSteps is **78.0% exact / 83.2% MIREX**; our port scores **68.5% exact /
74.8% MIREX** in its single reference run (seed 13). These are different training
and selection protocols, so the score gap is not an architecture-only comparison.

## Why K10

We did not start with K10. The first architecture was ours: a global-pooling
CNN on a 33-band log-STFT feature (`round(log2(f/55))` binning). Its first
real-data run scored **7.1% exact / 9.6% MIREX**. The diagnosis found an
octave-bucket feature bug and global frequency pooling that discarded useful
absolute-pitch location information.

We then ran a focused screen of eight preregistered architecture recipes on the
cleaned split. The frequency-preserving candidate reached 61.13% versus 38.81%
MIREX for the global-pooling control on the one GiantSteps exposure, and 61.75%
versus 40.45% across three validation seeds. That comparison also changed the
learning rate, so its full gain cannot be attributed to pooling alone.
The 61.13% result was still 8+
points below the 69.35% native libkeyfinder bar, so the documented gate
promoted nothing.

K10 offered a published implementation aligned with this evidence. We adopted
it because:

1. **It has a published GiantSteps reference result** — 78.0% exact / 83.2%
   MIREX under its own protocol. This is useful prior evidence, not a state-of-
   the-art claim for our trained weights.
2. **It is MIT-licensed with the full training script published**
   (github.com/jhurliman/jams), so the port is auditable line-by-line and the
   recipe is reproducible from the source, not from a paper figure.
3. **It is the scale we wanted for CPU inference** — 109k parameters, one
   forward pass, no GPU in the serving path.
4. **Its feature (24 bins/octave CQT) makes pitch-shift augmentation an exact
   re-slice** of the padded spectrogram — no resampling, no approximation.
5. **Its time-only pooling preserves frequency position**, consistent with the
   direction supported by our compact-model screen.

The product criterion is measured accuracy and CPU cost. Architecture novelty
does not earn promotion. The full timeline, including the tranche numbers and
gate decisions, is in §7 and the current reports linked above.

## 1. The problem and the baseline

Key detection: given a track, output one of 24 classes. Scoring follows the
MIREX convention implemented in `drone/keys.py:9`:

```
WEIGHTS = {"exact": 1.0, "fifth": 0.5, "relative": 0.3, "parallel": 0.2, "other": 0.0}
```

`score_mirex` (`drone/keys.py:64`) returns both the exact-match fraction and the
weighted MIREX fraction. The symmetric convention counts the fifth in both
directions (ascending and descending), matching our benchmark; evaluation also
reports the asymmetric variant.

The bar to beat is **libkeyfinder**, the DSP detector already shipping in the host
application. Rerun unmodified on the same 604 GiantSteps files (Euphoria commit
`cde6db9`, libkeyfinder 2.2.8 / FFTW 3.3.10, zero missing predictions) it scores
**69.35% symmetric MIREX / 60.60% exact**. That rerun, not any historical table,
is the reference number every candidate is compared against under the identical
held-out protocol.

Why not stay with template matching? Key profile fitting (chroma against 24
Krumhansl-style templates, plus the DSP evidence around it) plateaus: on this
data libkeyfinder sits at 69.35% and its error mass is concentrated in
tonic-adjacent and relative-mode confusions that template weighting can no longer
resolve. A learned model that sees absolute frequency position and time structure
has room above that; template matching, given the feature it extracts, has not.
The first 347k-parameter CNN we trained from scratch stayed below 69.35%
(§7, tranche), so the bar was not beaten by adding capacity alone — it was
beaten by porting the representation and architecture that a working published
system actually uses.

## 2. The data

Two datasets, both two-minute preview-length audio, both checksum-verified at
fetch time by `drone/download.py`:

| Dataset | Role | Size | Provenance and verification |
|---|---|---|---|
| Beatport EDM Key Dataset | Training | 1,363 usable tracks (933 minor / 430 major) | Zenodo record 1101082 (`drone/download.py:117-128`); archives verified against Zenodo-published checksums; 1,486 total entries, unparseable labels skipped (first option of `X \| Y` labels) |
| GiantSteps Key Dataset | Held-out test, scored once per candidate | 604 tracks | annotations pinned to revision `6bcd492c…`, archive SHA-256 pinned, each preview verified against its upstream MD5 (`drone/download.py:109-146`); count asserted to be exactly 604 |

The training set is **entirely EDM previews** — that is the biggest domain
constraint on the model and is treated as a limitation, not hidden (§9).

Disjointness is enforced in code, not by trust:

- `assert_disjoint` (`train.py:109`) is called on track IDs at prepare
  (`train.py:342`), at the start of every training run (`train.py:159-162`), and
  again before evaluation — against both the training *and* validation IDs and
  the exact audio SHA-256 lists (`train.py:282-288`). Zero ID overlap and zero
  audio-hash overlap between the 1,363 training tracks and the 604 test tracks.
- `DATA.md` states what these checks do *not* establish: artist-, album-, remix-
  or near-duplicate independence. Broader-corpus evaluation must audit those.

**The cleaned 1,230 / 126 split.** A split audit (see PLAN.md, "First tranche")
found 7 identical-audio pairs inside the 1,363-row training cache, two of which
crossed the original training/validation boundary. Their training copies were
quarantined and remaining training audio deduplicated, yielding **1,230 training
recordings**, with the original 126 validation IDs kept unchanged. That cleaned
split was used for the experiment tranches in §7; it is not used by the current
`train.py`, which trains on the full manifest. There were no conflicting labels
among the duplicate pairs.

**Content-addressed CQT caching.** Every audio file's features are cached at
`data/<dir>/cqt/<FEATURE_ID>/<sha256-of-audio>.npy` (`drone/features.py:96-97,
148-166`). The cache key is the SHA-256 of the decoded *content*, so no track can
ever consume another track's cache under a shared ID (this was a real bug class
before `01bbc88`). The `FEATURE_ID` (`drone/features.py:36-49`) is a hash of the
feature spec, so changing any feature parameter puts caches in a fresh namespace
instead of silently reusing stale arrays. Cached arrays are stored float16,
loaded as float32.

Each `prepare` run also writes a manifest (`train.py:56-91`) recording, per
track: ID, path, audio SHA-256, parsed key, and class reference. The manifest is
validated before training (`train.py:94-106`) — a feature-spec or label-schema
mismatch forces re-preparation.

## 3. The feature: K10's CQT

`drone/features.py` defines the input:

- `librosa.cqt` at `SR = 22050`, `hop = 4096` (~5.4 fps), **24 bins per octave**,
  `N_TOTAL = 208` bins starting at `FMIN_HZ = C1 × 2^(-4/12) ≈ 27.4 Hz`
  (`features.py:23-31, 58, 61-81`). The core used by the network is 192 bins
  (8 octaves of C1 upward); the extra 8 bins are a **±4-semitone pad**.
- Amplitude: `log1p(|coeff|)` (`features.py:78-81`).
- Layout: **`(bins, frames)`** — frequency on axis 0, time on axis 1.

Why each choice:

**24 bins per octave (2 bins per semitone).** Two jobs at once. First, it makes
pitch-shift augmentation an *exact re-slice*: shifting a track by `s` semitones
moves its content exactly `2s` bins down the frequency axis, so `window(cqt, s)`
(`features.py:84-93`) returns a pixel-identical view of the audio shifted by
`s` semitones — no resampling, no octave wrap, no interpolation error. The
augmentation in `sample_features` (`datasets.py:108-124`) is therefore a slice,
and the cost of "more augmented data" is a memory copy, not a re-decode. Second,
2 bins per semitone is enough resolution that a note lands in its bin and the
network can read the exact tonic from absolute bin position. (One bin per
semitone would work for the slice trick but is coarser; 12 bins/octave — pure
chroma — is what fails, see the pooling section.)

**The ±4-semitone pad.** Augmentation samples shifts uniformly from −4…+4
(`datasets.py:115`). Without pad bins, a +4 shift would need 8 bins above the
top core bin and a −4 shift 8 bins below the bottom one. The 208-bin spectrogram
carries them, so the re-slice is valid for the full augmentation range
(`features.py:90` raises outside ±4).

**`(bins, frames)` layout.** This is not cosmetics. The network's head pools
over the *time* axis (`model.py:38`) and keeps the *frequency* axis alive at the
readout. The layout choice and the pooling choice are one decision: absolute
frequency position is what encodes the tonic, and it must survive to the output
layer. The old feature, with time as a fixed 300-column axis and bands in
frequency as rows, forced the old net to pool both axes to produce a fixed-size
vector — which destroyed exactly the information the task needs (§4).

**Why a CQT at all, at 24/octave:** log-frequency resolution matches how pitch
works (equal semitone spacing, constant relative bandwidth per bin), keeps the
tonic's absolute position in a stable coordinate system, and — combined with the
pad — gives the one property that made the whole augmentation strategy cheap and
exact.

**The failure this design replaces.** The first feature was 33 "semitone" bands
from an STFT, with band index computed as `round(log2(freq / 55))` — **missing
the ×12 factor**, i.e. `log2` of the ratio instead of semitones. `log2(f/55)`
counts *octaves*, not semitones, so the 33 columns were octave buckets: only
9 of 33 columns ever had energy, each bucket spanning 12 semitones of
indistinguishable pitch content. The feature literally could not express "this
is C, not G, not F". A model on that input is at best guessing the tonic from
timbre: the first real-data run scored **7.1% exact / 9.6% MIREX** — random
guessing on 24 classes is ~4.2% exact — with validation plateauing around
10–12%. Commit `01bbc88` replaced it; the test suite pins tone placement and
octave specificity so this bug cannot return.

## 4. The architecture and the key insight

`KeyCNN` (`drone/model.py:27-45`):

```
input (B, 192, T)
  → 3 blocks, each: Conv3×3 → BN → ELU → Conv3×3 → BN → ELU → MaxPool2(2)
      widths 16 / 32 / 64 (scaled by `width`, default 1.0)
  → Dropout2d(0.2)
  → AdaptiveAvgPool2d((24, 1))      # 192 bins → 24 after 3 halvings
  → Linear(64 × 24, 24)
```

109k parameters at width 1.0. `width` is a capacity knob (medium/large research
profiles in the README); the default is the one used for the reported numbers.

**The insight is the pooling shape.** The head pools over **time only**
(`AdaptiveAvgPool2d((24, 1))`, `model.py:38`): each time frame gets collapsed
away, but 24 frequency positions (one per octave cell) survive to the linear
readout. The three MaxPools halve the frequency axis 192 → 96 → 48 → 24, so the
readout sees frequency *structure*: where in the absolute spectrum the energy
sits, which is where the tonic is.

Our first net used `AdaptiveAvgPool2d(1)` — global pooling over **both** axes.
That makes the network **transposition-invariant**: shifting every bin by 24
(octave) or even 12 (semitone) leaves the pooled vector unchanged, so C-major
and G-major produce the same input summary. The result: training pinned at
chance level, with **54% of all errors falling in the "other" (wrong tonic)
bucket** — the signature of a model that had learned mode and texture but had
been architecturally forbidden from reading pitch. The K10 paper states this
failure explicitly: an earlier variant with global pooling over both axes failed
for the same reason. Pooling over time only is the fix; pooling over frequency is
a bug for this task, and `model.py:31-34` carries the comment.

The 24×1 readout also aligns with the class space: 12 tonics × 2 modes, encoded
as `class_id = tonic * 2 + mode` (`drone/keys.py:12-16`).

## 5. The training protocol

Ported from K10's `train_key_cnn.py`, implemented in `train.py`:

| Parameter | Value | Location |
|---|---|---|
| Crop | 320 frames (~60 s at hop 4096) | `features.py:33`, `train.py:226` |
| Augmentation | uniform ±4-semitone shift = exact re-slice; label transposed by −shift | `datasets.py:108-124`, `keys.py:25-27` |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 | `train.py:31, 173` |
| Schedule | cosine annealing, 120 epochs | `train.py:29, 174` |
| Batch | 32 | `train.py:30` |
| Loss | plain cross-entropy (label smoothing supported, default 0.0) | `train.py:175, 329` |
| Checkpoint selection | best validation MIREX on a fixed seeded stratified 10% split | `train.py:115-129, 249-264` |
| Seed | 13 (reference run) | `train.py:28` |

Details that matter:

- **Cropping.** Training samples are random 320-frame crops; short tracks are
  zero-padded (`datasets.py:118-123`). Inference uses the *full* track with no
  padding (`train.py:34-49`) — K10 does not pad at inference either.
- **The shift/label contract.** A positive shift moves content *down* the
  frequency axis (the track sounds lower), so the label transposes by **−shift**
  (`datasets.py:112, 124`, `features.py:85-93`). `transpose_key`
  (`keys.py:25-27`) wraps the tonic mod 12 and preserves the mode bit. This
  contract is pinned by tests; the old pipeline got it wrong in two independent
  places (see Run 1).
- **Validation split.** Stratified 10% per key class, shuffled with a fixed seed
  (`SEED = 13`), computed identically for every training seed (`train.py:115-129`).
  Changing the training seed does *not* change the split. Augmentation is applied
  only to the training portion. Best-checkpoint = best validation MIREX; the
  checkpoint is saved at that moment with full provenance (`train.py:249-264`).
- **Label smoothing.** Available (`--label-smoothing`), off in the reference run,
  matching K10's published setup.
- **Nonfinite guard.** A nonfinite training loss raises immediately
  (`train.py:238-239`) instead of quietly producing a dead checkpoint.

**Why our 74.8% is below K10's paper figure of 83.2% on the same data.**
Protocol difference, documented and expected: K10 selects the epoch using
**5-fold cross-validation** and then trains the final model on **all the data**
with the median CV epoch budget. We use a single 10% validation split for
checkpoint selection and train on 90% of the data. Fewer training rows and a
noisier epoch estimate both cost points on a 1,363-track dataset. (K10's MIREX
2026 task entry reports 74.57 weighted / 67.88 exact under the task's scoring
convention — a different protocol again. All three numbers describe the same
model trained under three different evaluation setups; compare within one setup.)

## 6. Provenance: run manifests and checkpoints

Every run writes `run.json` alongside the checkpoint (`train.py:177-273`):
SHA-256 of every source file used, the `uv.lock` hash, git revision, torch and
numpy versions, the full training manifest (per-track audio SHA-256, key, ID),
the exact train/validation/held-out ID lists, config, per-epoch loss and
validation metrics, elapsed time, and finally the checkpoint's own SHA-256.

Checkpoints are `format_version: 2` artifacts (`drone/checkpoints.py:12-45`):
state dict plus `feature_spec`, `label_schema`, crop length, model config, and
metadata. Loading rejects anything not version 2 with a mismatched feature spec
or label schema (`checkpoints.py:35-41`) — a v1 (33-band) checkpoint cannot be
silently loaded into a 192-bin network. This versioning is what made the
33-band → K10 port a clean cut: old weights fail loudly with an actionable
message instead of scoring garbage.

**The ordinary training scoring-once rule.** GiantSteps is scored **exactly once per candidate**
by the explicit `evaluate` command (`train.py:278-314`); `train` never touches
it. Repeated held-out evaluation would let seed/recipe selection drift toward
the test set, which is precisely what turned GiantSteps from a pristine test
into an "exposed" one across the program (§9).

## 7. The learnings timeline

This is the honest history, from PLAN.md and the commit log.

### Run 1 — the buggy baseline: 7.1% exact / 9.6% MIREX (random)

Three independent bugs stacked, all in the "small model" phase:

1. **Octave-bucket CQT** — the missing ×12 in `round(log2(freq/55))` (§3): the
   feature had 9 live columns of 12-semitone-wide buckets.
2. **Split personality in class encoding** — labels were encoded
   `tonic * 2 + mode` (24 classes) while the scorer decoded predictions as
   `class % 12 // 12` (12 pitch classes). A model that predicted "class 17"
   (correct: F# minor) was scored against a different class space than the one
   it was trained on.
3. **Wrong augmentation indexing** — the pitch shift was sampled per track but
   the feature roll was applied by *dataset row number* instead of the sampled
   shift, so features and labels were decorrelated even when each was right.

The dangerous part was the coupling: model, scorer, and inference API all shared
the *same wrong decode*, so everything agreed with itself. It was a closed
ecosystem of misinformation — no component individually looked broken, and the
pipeline was internally consistent at the wrong answer. Random-level scores
(7.1% exact on 24 classes) were the only honest signal, and it took an external
review of the run to unpack all three. Fixes: `e61b323` (normalize guard) and
`01bbc88` (semitones, single label schema, augmentation, cache, manifests).

**Lesson: when a trained model scores at chance, suspect the feature and the
label contract before the model, and verify feature, scorer, and API decode the
same space.**

### Run 2 — bugs fixed, feature still weak: 29.5% exact / 35.9% MIREX

With the correct semitone mapping, single label schema, and correct
augmentation, the same 33-band net and global-pooling head trained to
29.5% exact / 35.9% MIREX on the validation split. Better than chance by a wide
margin, nowhere near the 69.35% bar. The residual gap was the representation
and the pooling: octave-limited bands, and a head that pooled away frequency
position.

### The tranche: confirming the pooling insight on our own data

Eight preregistered recipes were screened on the cleaned 1,230/126 split (seed
13, 120 epochs, CPU 16 threads, batch 64), followed by three-seed confirmation
(seeds 13/37/73) for the challengers — 17 runs total, all completed, repeated
seed-13 runs reproducing exactly. Results:

| Model | Params | 3-seed val MIREX | Seed-13 GiantSteps (scored once) |
|---|---:|---|---:|
| Control (global pooling, 33-band successor) | 331,704 | 40.45% ± 1.40 | **38.81%** |
| Frequency-preserving head (time-only pool), lr 3e-4 | 347,064 | 61.75% ± 1.73 | **61.13%** |

The +22 validation / +22 held-out points for the pooling change confirmed, on
our data, the insight the K10 paper had documented: **pooling over time only is
worth ~20 MIREX points on this task, and capacity is not the lever**. But
61.13% was still below the verified native 69.35%, so per the documented gate
**the native detector was kept** and no model was promoted. This is the most
important negative result in the program: a confirmed architectural insight that
still was not enough, which is why the next move was to port the full proven
recipe instead of extending our own.

### The K10 port (this branch): 68.5% exact / 74.8% MIREX, seed 13

Commit `a7be2c0` replaces the 33-band STFT feature and global-pooling net with
K10's pipeline exactly: librosa.cqt 24 bins/octave (208 bins from C1−4st),
109k-parameter CNN with time-only pooling, 320-frame crops, ±4-semitone
re-slice augmentation with label transpose, AdamW 1e-3, cosine 120 epochs,
plain CE, batch 32. 77 tests pin the CQT (tone placement, octave specificity),
the window/shift contract, the manifest flow, and end-to-end train/eval.

Single reference run, seed 13, one held-out GiantSteps 604 evaluation:
**68.5% exact / 74.8% MIREX**, error breakdown 414 exact / 40 fifth / 44
relative / 24 parallel / 82 other. Best validation MIREX 75.8%. First candidate
to pass the native bar: **+5.45 MIREX over 69.35%** and +13.7 over the tranche
winner. Below the paper's 83.2% for the protocol reasons in §5.

Promotion gate (decided with the user, confirm-then-merge): seeds 37 and 73 are
trained on CPU, each receives **exactly one** held-out evaluation, and the branch
merges and the model promotes only if the **three-seed mean stays at or above
native + 0.02 MIREX** (≥ 71.35%). One lucky seed has already been shown
elsewhere in the program not to be enough.

### The bug gallery (small fixes that mattered)

- **`normalize()` constant-input precision** (`e61b323`): an absolute 1e-6
  epsilon on the std was *below float32 rounding of the mean*, so a constant
  (or shifted-to-constant) spectrogram normalized to −0.0147 instead of 0.
  Fixed with a relative guard: σ below 1e-8 × max(1, |μ|) returns exactly
  zeros, pinned by a test.
- **Hand-rolled CQT off by 1–2 bins at high frequencies**: our first DFT-table
  implementation drifted at the top of the range. We stopped fighting it and
  called `librosa.cqt` directly, as K10 does — the reference implementation is
  the ground truth for a port, and the tests pin where a pure tone lands.
- **`_note_to_hz` one-octave error**: the semitone offset from the A reference
  was wrong (the −9 factor), so "C1" came out at 38.89 Hz instead of
  32.703199 Hz. The current implementation is `16.351597933828117 ×
  2^(semitones-from-C0 / 12)` (`features.py:52-55`), with the C1 value
  load-bearing for the whole 208-bin layout and FMIN.
- **Frame-gather overrun in the centered-window CQT**: the first extraction
  read past the available frame range; fixed by padding before the gather.
- **Shared-ID cache collision**: inference could return another track's cached
  features under a shared track ID; the content-addressed cache (§2) makes the
  class of bug impossible.

## 8. Reproduction

Host used for all reported runs: `ssh lysk-admin@lysk-server`,
**AMD Ryzen Threadripper PRO 9955WX, 16 cores, CPU-only training** — the GPUs on
that box (RTX PRO 6000, RTX 5090) are reserved for other workloads and were not
used; a 109k-parameter model trains in well under a minute on 16 CPU threads, so
a GPU buys nothing measurable here.

```bash
# one-time, on the training host
git checkout k10-arch
uv sync --locked
uv run pytest                      # 77 tests

bash fetch.sh                      # Beatport (audio + labels) + GiantSteps, checksum-verified
# or: bash fetch.sh --dataset train | --dataset test ; DRONE_DATA / --data to relocate

uv run train.py prepare --data <DATA> --threads 16
uv run train.py train   --data <DATA> --device cpu --threads 16
uv run train.py evaluate <DATA>/runs/<run-dir>/drone.pt --data <DATA> \
    --device cpu --threads 16 --output <DATA>/results.json
```

Wall-time expectations (measured, not estimates): a 5-epoch timing sample at 16
CPU threads took 3.29 s of training loop / 5.43 s process wall
(`PLAN.md` run register); the full 120-epoch, 1,230-track clean-split runs in
the tranche took **~63–79 s each** (347k-parameter net; the 109k K10 net is
cheaper per step). A full reference training run is therefore a few minutes of
wall time including per-epoch validation on 126 tracks; the dominant costs are
the one-time `fetch.sh` download (~2.1 GB training archive + test audio) and
first-run feature preparation (CQT over ~1,967 files, cached afterwards by
content hash). The reference K10 run's artifacts live at
`~/key-cnn/data/runs/w1.0-seed 13-20261111-185528-lr7fvv4k/` (checkpoint) with
the evaluation JSON at `~/drone-experiments/k10-arch-giantsteps.json`; `run.json`
in the run directory carries the source hashes, git revision, lock hash,
manifests, split IDs, and per-epoch history for that run.

Reproduction rules:

1. Verify source and lock hashes in `run.json` against the reviewed revision
   before trusting any run.
2. Changing any training seed keeps the validation split fixed; changing the
   feature spec or label schema invalidates caches, manifests, and checkpoints
   (by design, loudly).
3. Score GiantSteps exactly once per candidate. Do not use repeated test
   evaluations to select architectures, hyperparameters, or seeds.

## 9. Known limitations and open work

- **EDM preview bias.** All 1,363 training tracks are two-minute EDM previews.
  The generalization test is **FMAKv2** (5,489 songs, 17 genres): prepared and
  frozen into a 4,489 development / 1,000 final-holdout partition with zero
  exact Beatport audio overlap. An adaptation pilot (artist/album/audio-
  quarantined, 976 FMAK fit rows joined with the 1,230 clean Beatport rows,
  fixed recipe, three seeds) **passed its development gate at 62.64% ± 1.15
  FMAK dev MIREX** (frozen reference 56.00% on the same 250 rows) and 61.64%
  ± 0.97 Beatport validation — but **the final 1,000-track FMAK holdout remains
  unscored** for the K10 model. Until it is scored, "generalizes beyond EDM"
  is a hypothesis with pilot evidence, not a result.
- **The 5-fold-protocol gap.** We train on 90% of the data with single-split
  checkpoint selection; K10's 83.2% uses 5-fold CV plus a full-data final model.
  Closing that gap is the most direct documented path to their number and has
  not yet been run.
- **Promotion is gated.** The K10 model enters the host application only after
  the 3-seed confirmation: seeds 37 and 73 each get one held-out evaluation,
  and the merge requires the three-seed mean ≥ native 69.35% + 0.02 MIREX.
- **Exposed test sets.** GiantSteps and the FMAK 1,000 have been scored by
  earlier candidates in the program; they are no longer pristine at the program
  level. A fresh external protocol (GTZAN key annotations, SWD v2.1 are
  identified but not yet downloaded) is needed before the next "final quality"
  claim.
- **Label ambiguity.** `X | Y` labels take the first option
  (`keys.py:30-38`); no soft-label experiment has been run.
- **Scoring convention.** Symmetric-fifth MIREX is the headline convention
  (matches the host benchmark); both symmetric and asymmetric are reported per
  evaluation, and cross-paper comparisons must state which they use.

## 10. What would change this analysis

Evidence that would overturn or materially revise the conclusions above:

- **A three-seed mean below 71.35% MIREX** (native + 0.02): the port's single
  74.8% was seed 13's draw; if 37/73 land low, the "+5.45 over native" claim
  evaporates and the native detector stays, exactly as the gate intends.
- **K10's 83.2% reproduced by us under their 5-fold/full-data protocol**: would
  confirm the protocol-gap explanation of §5. If we *fail* to reproduce it on
  the same 1,363 tracks with their protocol, the gap is not the protocol and the
  port has a real defect to find.
- **A score at chance (~4% exact / ~7% MIREX) after re-running the fixed
  pipeline** on a fresh track subset: would indicate a regression in the
  feature/label contract of the kind that produced Run 1.
- **FMAK final holdout scoring near the 56% frozen reference** rather than the
  62.6% pilot band: would show the adaptation did not carry and the model is
  an EDM specialist, changing the promotion calculus for a multi-genre
  deployment.
- **Duplicate/artist overlap found between Beatport and GiantSteps or FMAK**
  beyond the ID/audio-sha checks the code asserts: the disjointness guarantees
  are ID- and content-level only; artist-level leakage would inflate every
  held-out number in this document.
- **A DSP or template baseline rerun on the same 604 files beating 74.8%**:
  would show the learned model's advantage is in its protocol and data, not its
  architecture.


## 11. Optional Euphoria correction fine-tuning

This optional workflow adapts an existing compatible K10 checkpoint to corrected
keys, mixed with base fitting audio. It is separate from the confirmed release
recipe above. No real correction dataset has established an accuracy gain or
GiantSteps nonregression; synthetic tests exercise correctness only. No tuned
checkpoint is automatically promoted or copied into `weights/`.

1. Prepare the ordinary Beatport train and GiantSteps test manifests with
   `uv run train.py prepare`. Fine-tuning requires these plus both correction
   manifests; `--data` can select another prepared directory.
2. Save JSONL from `euphoria corrections export`. Run
   `uv run train.py prepare-corrections --corrections corrections.jsonl --audio <folder>`.
   Schema 1 key corrections are matched recursively by full-file BLAKE3, with
   tempo-only corrections skipped and counted. Renaming does not break a match.
   Matching corrected GiantSteps audio is refused. Unmatched counts are printed;
   both correction fit/held-out sets must contain at least one matching file.
3. Run `uv run train.py finetune <starting.pt> --device cpu --threads 4`.
   `--epochs`, `--lr`, `--batch-size`, `--corrections-share`, `--seed` and
   `--weight-decay` configure this explicit run. Defaults are 10 epochs, 1e-4,
   batch 32, share .25, seed 13, weight decay 1e-4. Architecture width is inherited
   from the starting checkpoint. Hard CE, existing feature crops and label-
   preserving pitch shifts apply. Each epoch visits the filtered base set once;
   shuffled correction passes cycle to supply `max(1, round(batch*share))` examples
   per step. The final partial base batch can have a larger correction fraction.
4. Inspect the new run directory's `finetune.json` before using `drone.pt`.
   The report contains configuration, per-epoch loss, checkpoint hash, inherited
   witness availability and both sets' starting/tuned exact and symmetric MIREX
   scores, predictions and improved/worsened/unchanged MIREX-credit counts.
   It is not an accuracy guarantee. No release artifact is overwritten.

The **last epoch** is always retained. GiantSteps and corrections-held-out are
scored before and after, never each epoch and never for checkpoint selection.
This differs from the ordinary train/evaluate scoring-once workflow. A pre/post
report is a fixed experiment, not permission to search settings using these sets.

Correction split membership is determined by the first 64 bits of the file's
BLAKE3 hash modulo 5. It stays fixed as additional unchanged-file corrections arrive.
Tags are part of the hash: metadata rewrites, including writing a corrected key,
can change identity and split even if decoded audio is unchanged. There is no
recording/artist/album or decoded-audio deduplication guarantee. Preserve an
independently audited audio split when that level of independence is required.

Before scoring or optimization, fine-tuning checks both evaluation sets against
starting-checkpoint `train_ids`, `validation_ids` and `training_audio_sha256`
witnesses, plus IDs/hashes of the actual new base+correction fitting tracks.
Corrected fit and held-out audio are removed from new base fitting; this cannot
undo exposure in the starting weights. If a correction-held-out sample was already
used by that checkpoint, the command fails and names the evaluation set. Supply
a genuinely suitable starting checkpoint or a separately prepared evaluation
protocol; the command never silently excludes or reassigns samples.

The existing evaluator's incomplete-metadata policy remains: absent witnesses do
not themselves cause rejection. The report lists present/missing inherited fields;
missing means **unknown exposure**, not clean. Preserve checkpoint provenance when
fine-tuning repeatedly. Saved provenance keeps inherited fitting/validation IDs
and training hashes and adds only actual filtered new base and correction-fit
exposure, so excluded-but-unseen held-out files are not falsely marked trained.
