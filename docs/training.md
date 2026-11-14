# Training the released Drone model

This document explains how the released `drone-v1` model is trained, why the
recipe looks the way it does, and how to reproduce it. It is the reference for
the current release. The earlier compact-model research that led to it is kept
as a historical record in [RESEARCH_TRAINING.md](RESEARCH_TRAINING.md); method
directions considered for future work are in
[CLASSIFIER_RESEARCH.md](CLASSIFIER_RESEARCH.md); every later experiment is
indexed in [experiments/README.md](../experiments/README.md).

## Summary

`drone-v1` is a 109,128-parameter CNN that predicts 24 musical keys (12 tonics,
major or minor) from a constant-Q spectrogram. It is our own training of
**K10** ("K10: A Compact Convolutional Network for Global Key Estimation",
John Hurliman, MIREX 2026 Audio Key Detection; open-source
[jams project](https://github.com/jhurliman/jams), MIT, `train_key_cnn.py`).
It was trained on 1,363 Beatport EDM previews (1,237 fitting, 126 validation)
and scored on the 604 GiantSteps previews.

| GiantSteps 604, symmetric MIREX / exact | Score |
|---|---:|
| Shipped checkpoint (seed 13) | 74.8% / 68.5% |
| Same recipe, mean of seeds 13, 37, 73 | 73.58% / 66.8% |
| libkeyfinder 2.2.8 as run inside Euphoria | 69.35% / 60.60% |
| K10, official MIREX 2026 result | 74.57% / 67.88% |

GiantSteps was used during development of this and earlier candidates, so it is
not an untouched test set. Our training and the official K10 result are
independent systems and protocols; the table is not a paired comparison or
evidence of statistical parity. The K10 author's
[project documentation](https://github.com/jhurliman/jams#key-detection-our-cnn)
lists 78.0% exact / 83.2% MIREX as a legacy result on a different 567-file
evaluation with development reuse; it is not a comparable target.

## Corrections found after release

A later audit found two problems. Neither changes the released weights or the
historical scores reported here, which have not been recomputed.

- The release's fitting/validation split contained seven groups of identical
  audio, two of which crossed the fitting/validation boundary. Later training
  uses a content-deduplicated split (1,230 fitting / 126 validation).
- The old variable-length evaluation padded shorter tracks to their batch peers.
  Corrected evaluation batches only equal-length inputs and keeps each track's
  actual length. `Predictor`'s old feature transpose was also fixed: K10 expects
  `(batch, 192 bins, time)`. ONNX now has dynamic batch and time dimensions, with
  CPU parity checked against the unchanged PyTorch model at several lengths.

These findings do not show contamination of the GiantSteps test set.

## Why K10

We did not start with K10. Three stages led to it.

**A first model that scored at chance.** The first architecture was ours: a
global-pooling CNN on a 33-band log-STFT feature. Its first real-data run scored
**7.1% exact / 9.6% MIREX** (random guessing on 24 classes is about 4.2% exact).
Three bugs stacked:

1. The feature computed band index as `round(log2(freq / 55))`, missing the
   factor of 12. It counted octaves, not semitones: only 9 of 33 columns ever had
   energy, so the input could not distinguish C from G.
2. Labels were encoded as `tonic * 2 + mode` (24 classes) while the scorer
   decoded predictions as 12 pitch classes.
3. Pitch-shift augmentation rolled features by dataset row number instead of the
   sampled shift, so features and labels were decorrelated.

Model, scorer and inference API shared the same wrong decoding, so every
component agreed with every other and none looked broken. Chance-level accuracy
was the only signal. The lesson: when a trained model scores at chance, check the
feature and the label contract before the model, and verify that feature,
scorer and API decode the same class space.

**Fixing the bugs.** With a correct semitone mapping, one label schema and
correct augmentation, the same 33-band global-pooling network reached
29.5% exact / 35.9% MIREX on validation: far above chance, far below
libkeyfinder's 69.35%.

**A controlled screen.** Eight recipes fixed in advance were screened on the
deduplicated 1,230/126 split at seed 13, then the control and the best
challengers were confirmed at seeds 13, 37 and 73 (17 runs).

| Model | Parameters | Three-seed validation MIREX | Seed-13 GiantSteps MIREX |
|---|---:|---:|---:|
| Global pooling (control) | 331,704 | 40.45% ± 1.40 | 38.81% |
| Time-only pooling, lr 3e-4 | 347,064 | 61.75% ± 1.73 | 61.13% |

Pooling over time only was worth about 22 MIREX points, and capacity was not the
lever. But 61.13% was still about 8 points below libkeyfinder, so no model was
promoted. Full details are in the
[compact-model report](../experiments/studies/compact-models/REPORT.md).

At that point we chose to adopt the published system that already used the
insight we had confirmed, rather than keep iterating our own design:

1. K10 is a concrete, auditable key-estimation reference with an official
   604-file result. The author's
   [claim audit](https://github.com/jhurliman/jams/blob/main/paper/REVIEW.md#claim-audit)
   documents development reuse and unsupported ranking claims, which we respect.
2. It is MIT-licensed with its full training script published, so the port can
   be checked line by line.
3. It is the scale we want for CPU inference: about 109k parameters, one forward
   pass, no GPU needed.
4. Its 24-bins-per-octave CQT allows exact integer-semitone re-slicing for
   augmentation (see below). Re-slicing cached CQT features is a surrogate for
   waveform pitch shifting, not a claim of numerical equivalence to resynthesized
   audio.
5. Its central design choice, time-only pooling, is what our own screen
   confirmed.

**The K10 port.** The first single-seed run (seed 13) scored **68.5% exact /
74.8% MIREX** on GiantSteps (414 exact, 40 fifth, 44 relative, 24 parallel, 82
other errors), with best validation MIREX 75.8%: 5.45 points above libkeyfinder.
Before training more seeds we fixed a promotion rule: seeds 37 and 73 each get
exactly one GiantSteps evaluation, and the model is released only if the
three-seed mean is at least libkeyfinder + 2.00 points (71.35%). The mean was
73.58%, so the seed-13 checkpoint was released.

## 1. The task and the baseline

Key detection: given a track, output one of 24 classes. Scoring follows the
MIREX convention in `drone/keys.py`:

```
WEIGHTS = {"exact": 1.0, "fifth": 0.5, "relative": 0.3, "parallel": 0.2, "other": 0.0}
```

`score_mirex` returns both the exact-match fraction and the weighted MIREX
fraction. The symmetric convention counts the fifth in both directions, matching
Euphoria's benchmark; evaluation also reports the asymmetric variant.

The baseline is **libkeyfinder**, the DSP detector shipping in Euphoria. Rerun
unmodified on the same 604 GiantSteps files (libkeyfinder 2.2.8, FFTW 3.3.10,
zero missing predictions) it scores **69.35% symmetric MIREX / 60.60% exact**.
That rerun is the reference for every candidate.

Template matching (chroma against Krumhansl-style key profiles) plateaus on this
data: libkeyfinder's errors concentrate in tonic-adjacent and relative-mode
confusions that profile weighting cannot resolve. A learned model that sees
absolute frequency position and time structure has room to do better.

## 2. The data

Both datasets are two-minute previews, checksum-verified at download time by
`drone/download.py`:

| Dataset | Role | Size | Provenance |
|---|---|---|---|
| Beatport EDM Key Dataset | Training | 1,363 usable tracks (933 minor / 430 major) of 1,486 entries | Zenodo record 1101082; archives verified against Zenodo checksums; unparseable labels skipped; `X \| Y` labels take the first option |
| GiantSteps Key Dataset | Evaluation | 604 tracks | Annotations pinned to a repository revision, archive hash pinned, each preview checked against its upstream MD5; count asserted to be exactly 604 |

The training set is entirely EDM previews. That is the largest domain limitation
of the model (see Limitations).

Disjointness is enforced in code. `assert_disjoint` in `train.py` checks track
IDs at prepare time, at the start of every training run, and before evaluation,
against both training and validation IDs and their audio SHA-256 lists. There is
zero ID overlap and zero audio-hash overlap between the 1,363 training tracks and
the 604 test tracks. These checks do not establish artist, album, remix or
near-duplicate independence; the
[overlap audit](../experiments/studies/bp-gs-overlap/REPORT.md) leaves 539
Beatport IDs unresolved.

**Content-addressed feature cache.** Features are cached at
`data/<dir>/cqt/<FEATURE_ID>/<sha256-of-audio>.npy`. The key is the hash of the
audio content, so no track can read another track's cache through a shared ID
(a real bug in an earlier version). `FEATURE_ID` is a hash of the feature
specification, so changing any feature parameter uses a fresh namespace. Arrays
are stored as float16 and loaded as float32. Each `prepare` run writes a manifest
(ID, path, audio SHA-256, parsed key, class) that is validated before training; a
feature-spec or label-schema mismatch forces re-preparation.

## 3. The feature: K10's CQT

`drone/features.py` defines the input:

- `librosa.cqt` at 22,050 Hz, hop 4096 (about 5.4 frames per second),
  **24 bins per octave**, 208 bins starting at C1 x 2^(-4/12) (about 27.4 Hz).
  The network uses the 192 core bins (8 octaves from C1); the 16 extra bins are a
  +/-4 semitone pad.
- Amplitude: `log1p(|coefficient|)`.
- Layout: `(bins, frames)`, frequency on axis 0 and time on axis 1.

**Why 24 bins per octave.** Two bins per semitone make pitch-shift augmentation
an exact re-slice: shifting by `s` semitones moves content exactly `2s` bins, so
`window(cqt, s)` returns the shifted view with no resampling, octave wrap or
interpolation error. Augmentation is therefore a memory copy, not a re-decode.
The resolution also lets a note land in its bin so the network can read the
tonic from absolute bin position.

**Why the pad.** Augmentation samples shifts uniformly from -4 to +4. The 208-bin
spectrogram carries the 8 extra bins needed at each end, so the re-slice is valid
for the full range (`window` raises outside +/-4).

**Why the layout matters.** The network pools over time and keeps the frequency
axis to the readout. Absolute frequency position encodes the tonic and must
survive to the output layer. The old feature forced a pool over both axes, which
destroyed exactly that information.

## 4. The architecture

`KeyCNN` in `drone/model.py`:

```
input (B, 192, T)
  -> 3 blocks, each: Conv3x3 -> BN -> ELU -> Conv3x3 -> BN -> ELU -> MaxPool2(2)
      widths 16 / 32 / 64 (scaled by `width`, default 1.0)
  -> Dropout2d(0.2)
  -> AdaptiveAvgPool2d((24, 1))      # 192 bins -> 24 after 3 halvings
  -> Linear(64 x 24, 24)
```

109,128 parameters at width 1.0. `width` is a capacity setting for research
profiles; the default is the released model.

**The pooling shape is the key choice.** The head pools over **time only**: time
is averaged away, but 24 frequency positions survive to the linear readout. The
three max-pools halve frequency 192 -> 96 -> 48 -> 24, so the readout sees where
in the absolute spectrum the energy sits.

Our first network used `AdaptiveAvgPool2d(1)`, a global pool over both axes. That
makes the network transposition-invariant: shifting every bin leaves the pooled
vector unchanged, so C major and G major look the same. Training stayed at chance,
with 54% of errors in the "other" (wrong tonic) bucket, the signature of a model
that learned mode and texture but could not read pitch. The K10 paper reports the
same failure for an earlier global-pooling variant.

The 24-class output uses `class_id = tonic * 2 + mode` (`drone/keys.py`).

## 5. The training protocol

Ported from K10's `train_key_cnn.py`, implemented in `train.py`:

| Parameter | Value |
|---|---|
| Crop | 320 frames (about 60 s at hop 4096) |
| Augmentation | Uniform +/-4 semitone shift as an exact re-slice; label transposed by minus the shift |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| Schedule | Cosine annealing, 120 epochs |
| Batch | 32 |
| Loss | Plain cross-entropy (label smoothing supported, default 0.0) |
| Checkpoint selection | Best validation MIREX on a fixed seeded stratified 10% split |
| Seed | 13 (released checkpoint) |

- **Cropping.** Training samples random 320-frame crops; short tracks are
  zero-padded. Inference uses the full track without padding, as K10 does.
- **Shift and label.** A positive shift moves content down the frequency axis
  (the track sounds lower), so the label transposes by minus the shift.
  `transpose_key` wraps the tonic modulo 12 and keeps the mode. Tests pin this
  contract.
- **Validation split.** Stratified 10% per key class with a fixed seed (13),
  identical for every training seed. Augmentation applies only to fitting data.
  The best-validation checkpoint is saved with full provenance.
- **Non-finite loss.** A non-finite training loss stops the run immediately.

**Differences from the K10 recipe.** K10 chooses its epoch count with five-fold
cross-validation and refits on the full fitting corpus; our port uses one 10%
validation split. The effect of that difference has not been measured. It does
not explain a gap to the legacy 83.2% figure, which came from a different
567-file evaluation with development reuse.

## 6. Run records and checkpoints

Every run writes `run.json` next to the checkpoint: SHA-256 of every source file
used, the `uv.lock` hash, Git revision, torch and NumPy versions, the full
training manifest, the exact train/validation/test ID lists, configuration,
per-epoch loss and validation metrics, elapsed time, and the checkpoint's own
SHA-256.

Checkpoints are `format_version: 2` files containing the state dict plus feature
spec, label schema, crop length, model configuration and metadata. Loading
rejects other versions or a mismatched feature spec or label schema, so an old
33-band checkpoint cannot be loaded into the 192-bin network by mistake.

**Score the test set once per candidate.** GiantSteps is scored only by the
explicit `evaluate` command; `train` never reads it. Repeated test evaluation
lets seed or recipe choices drift toward the test set, which is how GiantSteps
became development-exposed over the program.

## 7. Smaller bugs that mattered

- **`normalize()` on constant input.** An absolute 1e-6 epsilon was below
  float32 rounding of the mean, so a constant spectrogram normalized to -0.0147
  instead of 0. A relative guard (std below 1e-8 x max(1, |mean|) returns zeros)
  fixed it; a test pins it.
- **Hand-written CQT off by 1 to 2 bins at high frequencies.** We switched to
  `librosa.cqt`, as K10 does; tests pin where a pure tone lands.
- **`_note_to_hz` one octave off.** C1 came out at 38.89 Hz instead of
  32.703199 Hz. The current formula is `16.351597933828117 x 2^(semitones from C0 / 12)`.
- **Frame-gather overrun** in the centered-window CQT, fixed by padding before
  the gather.
- **Shared-ID cache collision**, made impossible by the content-addressed cache.

## 8. Reproduction

```bash
uv sync --locked
uv run pytest

bash fetch.sh                      # Beatport (audio + labels) + GiantSteps, checksum-verified
# or: bash fetch.sh --dataset train | --dataset test ; DRONE_DATA / --data to relocate

uv run train.py prepare --data <DATA> --threads 16
uv run train.py train   --data <DATA> --device cpu --threads 16
uv run train.py evaluate <DATA>/runs/<run-dir>/drone.pt --data <DATA> \
    --device cpu --threads 16 --output <DATA>/results.json
```

All released runs trained on CPU (AMD Threadripper PRO 9955WX, 16 threads). The
seed-13 run took 298.47 seconds, with best epoch 39 of 120. A GPU gives no
measurable benefit at this model size. The main costs are the one-time download
(about 2.1 GB of training audio plus test audio) and first-run feature
preparation (CQT over about 1,967 files, cached by content hash afterwards).
`run.json` in each run directory records the source hashes, Git revision, lock
hash, manifests, split IDs and per-epoch history.

Rules:

1. Check the source and lock hashes in `run.json` before trusting a run.
2. Changing the training seed keeps the validation split fixed. Changing the
   feature spec or label schema invalidates caches, manifests and checkpoints,
   with an error.
3. Score GiantSteps once per candidate. Do not use repeated test evaluations to
   choose architectures, hyperparameters or seeds.

## 9. Limitations

- **EDM preview bias.** All training tracks are two-minute EDM previews. Accuracy
  on full-length club tracks and other genres is not established. Adding FMAK
  fitting data raised FMAK validation MIREX by 1.60 points in a
  [controlled comparison](../experiments/studies/k10-multidomain/REPORT.md),
  short of its +2.00 threshold, and the release weights were kept. The reserved
  1,000-track FMAK final set has not been scored for K10.
- **Exposed test sets.** GiantSteps and the FMAK final 1,000 were scored by
  earlier candidates and are no longer untouched. The Schubert Winterreise
  Dataset is now used in a descriptive
  [reference comparison](../experiments/studies/key-sota-4way/REPORT.md); GTZAN
  is catalogued but not yet used for scoring. An independent evaluation set is
  needed before the next accuracy claim.
- **Checkpoint selection.** The effect of single-split selection versus grouped
  cross-validation with a full refit is unmeasured.
- **Label ambiguity.** `X | Y` labels take the first option; no soft-label
  experiment has been run.
- **Scoring convention.** Symmetric-fifth MIREX is the headline convention; both
  symmetric and asymmetric scores are reported, and cross-paper comparisons must
  state which they use.

## 10. What would change these conclusions

- **A paired grouped-CV/full-refit comparison on common recordings** would
  measure the effect of the selection recipe. Failing to match the legacy
  567-file result would not by itself show a porting bug.
- **A score near chance (about 4% exact / 7% MIREX)** after rerunning the
  pipeline on a fresh subset would indicate a regression in the feature or label
  contract.
- **FMAK final-set accuracy near libkeyfinder's** would show the model is an EDM
  specialist.
- **Artist or near-duplicate overlap between Beatport and GiantSteps or FMAK**
  beyond the ID and audio-hash checks would inflate every test number here.
- **A DSP or template baseline beating 74.8% on the same 604 files** would show
  the advantage comes from protocol and data rather than architecture.

<a id="11-optional-euphoria-correction-fine-tuning"></a>

## 11. Optional annotation fine-tuning

This optional workflow adapts an existing compatible K10 checkpoint to explicit
human key annotations, mixed with base fitting audio. Drone is a reusable public
classifier; Euphoria is its main consumer and its JSONL export is one adapter for
those annotations, not a required data source. Any producer may implement the
portable schema documented in [DATA.md](../DATA.md#personal-corrections).
Nothing is collected, uploaded or added to the public model automatically. It is
separate from the release recipe above. No real correction dataset has shown an
accuracy gain or GiantSteps nonregression; synthetic tests check correctness only.
No tuned checkpoint is automatically promoted or copied into `weights/`.

1. Prepare the ordinary Beatport train and GiantSteps test manifests with
   `uv run train.py prepare`. Fine-tuning requires these plus both correction
   manifests; `--data` can select another prepared directory.
2. Save JSONL from `euphoria corrections export`. Run
   `uv run --extra corrections train.py prepare-corrections --corrections corrections.jsonl --audio <folder>`.
   Schema 1 key corrections are matched recursively by full-file BLAKE3, with
   tempo-only corrections skipped and counted. Renaming does not break a match.
   Matching corrected GiantSteps audio is refused. Unmatched counts are printed;
   both correction fit and held-out sets must contain at least one matching file.
3. Run `uv run train.py finetune <starting.pt> --device cpu --threads 4`.
   `--epochs`, `--lr`, `--batch-size`, `--corrections-share`, `--seed` and
   `--weight-decay` configure the run. Defaults are 10 epochs, 1e-4, batch 32,
   share 0.25, seed 13, weight decay 1e-4. Architecture width comes from the
   starting checkpoint. Hard cross-entropy, existing feature crops and
   label-preserving pitch shifts apply. Each epoch visits the filtered base set
   once; shuffled correction passes cycle to supply `max(1, round(batch*share))`
   examples per step. The final partial base batch can have a larger correction
   fraction.
4. Inspect the new run directory's `finetune.json` before using `drone.pt`. It
   contains configuration, per-epoch loss, checkpoint hash, inherited exposure
   records and both sets' starting/tuned exact and symmetric MIREX scores,
   predictions and improved/worsened/unchanged MIREX-credit counts. It is not an
   accuracy guarantee. No release artifact is overwritten.

The **last epoch** is always kept. GiantSteps and the correction held-out set are
scored before and after, never each epoch and never for checkpoint selection.
This differs from the ordinary score-once workflow. A before/after report is a
fixed experiment, not permission to search settings using these sets.

Correction split membership is the first 64 bits of the file's BLAKE3 hash
modulo 5. It stays fixed as more unchanged-file corrections arrive. Tags are part
of the hash: rewriting metadata, including writing a corrected key, can change
identity and split even if decoded audio is unchanged. There is no recording,
artist, album or decoded-audio deduplication. Use an independently audited audio
split when that level of independence is required.

Before scoring or optimization, fine-tuning checks both evaluation sets against
the starting checkpoint's `train_ids`, `validation_ids` and
`training_audio_sha256` records, plus the IDs and hashes of the new base and
correction fitting tracks. Corrected fit and held-out audio are removed from new
base fitting; this cannot undo exposure already in the starting weights. If a
correction held-out sample was used by that checkpoint, the command fails and
names the evaluation set. It never silently excludes or reassigns samples.

Absent exposure records do not by themselves cause rejection. The report lists
present and missing inherited fields; missing means **unknown exposure**, not
clean. Saved provenance keeps inherited fitting/validation IDs and training
hashes and adds only the filtered new base and correction-fit data actually used,
so excluded held-out files are not marked as trained.

The [recorded integration test](../tests/evidence/corrections-integration-smoke.json)
uses only the Beatport fitting pool, an internal hash-based split, a fresh
three-epoch baseline and 48 synthetic corrections, followed by the fixed
ten-epoch fine-tuner. It verifies the complete data, model and report path and
the exposure checks without using reserved benchmarks or private audio. Its
legacy `giantsteps` report field denotes the fitting-only inner held-out set in
this test, not an actual GiantSteps result. This is software validation, not
evidence that personal corrections improve accuracy.

The optional `corrections` extra supplies BLAKE3 for file matching and synthetic
annotation generation. Base inference, ordinary training, model export and schema
parsing work without it; already prepared correction manifests do not need BLAKE3
for fine-tuning. Default development installs include it so the tests remain
usable. A local fine-tuning report does not establish a gain for Drone's public
model: publishing new weights requires separately declared provenance, rights and
evaluation.
