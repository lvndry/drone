# Compact key models before K10

A 47,284-parameter octave-aware CQT model scored slightly higher than
libkeyfinder on both final test sets (68.99% versus 67.82% MIREX on 1,000 FMAK
tracks, 70.40% versus 69.35% on 604 GiantSteps tracks), but both paired 95%
intervals include zero. We therefore kept libkeyfinder (the detector shipped in
Euphoria) and continued accuracy research, which later led to the K10 release.
Study date: 11 November 2026.

## Question

Can a small, corrected CNN beat libkeyfinder's key accuracy, and which choices
(pooling, training data, pitch representation) matter? Rust optimization and
integration into Euphoria were deferred until a model showed a clear accuracy
gain.

## Method

The work ran in four stages, each with a protocol fixed before results were seen.
The pipeline (data splits, labels, augmentation, model and calibration details)
is explained in [RESEARCH_TRAINING.md](../../../docs/RESEARCH_TRAINING.md).

1. **Controlled screen** ([protocol](PROTOCOL.md)). Seven identical-audio pairs
   were found in the 1,363-row Beatport cache, two crossing the old split. We kept
   the 126 validation IDs, removed matching training copies and deduplicated the
   rest: 1,230 training recordings. Eight recipes were screened at seed 13; the
   control and two qualifying candidates were confirmed at seeds 13, 37 and 73
   (17 runs, CPU, 16 threads, 120 epochs, batch 64). Selection used three-seed
   mean symmetric MIREX with exact accuracy as tie-breaker; seed 13 was fixed in
   advance as the canonical artifact. Both architectures overfit a 48-recording
   training subset to 100% in 20 steps (a plumbing check). No test data entered
   recipe selection. The earlier 42.9% validation score came from the leaky old
   split and is not the control.
2. **FMAK diagnosis and partition.** FMAKv2 (5,489 recordings, 5,442 unique audio
   hashes, zero exact Beatport overlaps) was split before scoring into 1,000
   final rows by audio group and 4,489 development rows. FMA metadata (106,574
   rows) gave primary artist and album IDs for every FMAK recording. Connected
   artist/album/audio components quarantined 3,104 development rows linked to the
   final set, leaving 1,385 for fitting (976), validation (250) and calibration
   (159).
3. **Joint-data adaptation** ([protocol](ADAPTATION_PROTOCOL.md)). The winning
   recipe, unchanged, trained on Beatport plus FMAK fitting rows. Pre-set
   threshold: at least +2 FMAK validation points and no more than 2 Beatport
   points lost.
4. **Octave model** ([protocol](OCTAVE_PROTOCOL.md)) and **frozen final
   evaluation** ([protocol](FINAL_PROTOCOL.md)). Same 2,206 fitting / 376
   validation identities, trained from scratch with no pretrained weights. The
   model changes frontend (gamma-zero CQT, 99 bins), sampling (two 32-second
   segments, 84-bin transposition crops with matching target shifts),
   architecture (temporal-only downsampling, octave pooling) and outputs
   (structured tonic/mode logits) together. After selection, a temperature was
   fitted on the 159 calibration rows, and the final sets were scored once.

libkeyfinder references come from Euphoria's unmodified detector (libkeyfinder
2.2.8, FFTW 3.3.10) on audio with matching hashes. On GiantSteps it had zero
missing predictions and reproduces the historical 69.4% figure.

## Results

### Controlled screen

| Recipe | Parameters | Validation MIREX, mean ± sample SD | Validation exact, mean |
|---|---:|---:|---:|
| Frequency head, lr 3e-4 | 347,064 | 61.75% ± 1.73 points | 53.44% |
| Frequency head, dropout 0.1 | 347,064 | 59.68% ± 2.83 points | 50.79% |
| Global-pooling control | 331,704 | 40.45% ± 1.40 points | 33.07% |

![Three-seed validation confirmation](figures/small-confirmation.png)

*Figure 1. Validation MIREX for each confirmed recipe; dots are seeds 13, 37 and
73, bars are mean ± sample SD. Look at the gap between the two frequency-pooling
recipes and the global-pooling control.*

The same 126-track validation set selected recipes and checkpoints, so these are
development results. The SD describes training-seed variation, not uncertainty on
new data. Repeated seed-13 histories and predictions reproduced exactly.

### GiantSteps after selection (seed 13, each model scored once)

| Model | Exact | Symmetric MIREX | Asymmetric MIREX |
|---|---:|---:|---:|
| Global-pooling control | 29.30% | 38.81% | 34.83% |
| Frequency head, lr 3e-4 | 53.31% | 61.13% | 58.39% |
| libkeyfinder | 60.60% | 69.35% | 67.70% |

The selected model gains 22.32 MIREX points over the control but stays 8.23
points below libkeyfinder (unrounded; the earlier 8.27 used the rounded 69.4%).
After this, GiantSteps was no longer independent of later recipe changes.

### FMAK development diagnosis (4,489 rows)

| Model | Exact | Symmetric MIREX | Mode accuracy |
|---|---:|---:|---:|
| Global-pooling control | 22.61% | 33.27% | 54.44% |
| Frequency head, lr 3e-4 | 41.30% | 52.68% | 64.13% |
| libkeyfinder | 56.76% | 66.87% | 74.98% |

The CNN trails libkeyfinder by 14.19 MIREX points (paired artist/album-cluster
bootstrap 95% CI [-15.56, -12.65], 5,000 draws, seed 13, 1,076 components);
libkeyfinder's three missing predictions count as zero. A common-decoded
comparison agrees. The CNN predicts minor for 3,012 of 4,489 tracks although the
references are 2,313 major / 2,176 minor. These are results on our partition,
not comparable to published all-FMAK scores.

### Joint-data adaptation

The seed-13 pilot passed its threshold (FMAK 56.00% to 63.96%, Beatport 60.56% to
61.43%) and all three confirmation seeds passed.

| Same validation rows | Beatport-only seed 13 MIREX | Adapted mean ± sample SD | Adapted exact, mean | libkeyfinder MIREX / exact |
|---|---:|---:|---:|---:|
| Beatport, 126 | 60.56% | 61.64% ± 0.97 points | 52.65% | 65.16% / 53.97% |
| FMAK, 250 | 56.00% | 62.64% ± 1.15 points | 51.87% | 66.76% / 55.20% |

The reference column is one seed-13 model, not a paired three-seed control.
Adaptation helped but still trailed libkeyfinder. No adapted-model GiantSteps
score was computed.

### Octave model

The pilot stopped at epoch 66, keeping epoch 41, and passed the same threshold.

| Validation subset | Octave three-seed MIREX mean ± sample SD | libkeyfinder |
|---|---:|---:|
| Beatport, 126 | 72.57% ± 0.64 points | 65.16% |
| FMAK, 250 | 68.36% ± 0.47 points | 66.76% |

The repeated seed 13 matched the pilot's best epoch and predictions, but CUDA
training histories were not bitwise identical (maximum per-epoch loss difference
9.66e-8; 21 per-epoch score differences). A three-seed probability ensemble,
declared in advance, scored 70.37% combined validation versus 70.16% for seed 13;
the 0.21-point gain missed the 1-point requirement, so seed 13 was kept.

### Final evaluation

The calibration temperature 0.7371346086 was fitted on the 159 calibration rows
by NLL only and does not change any predicted key (calibration-row NLL 1.615 to
1.531, a fitting result). Everything was frozen before final predictions.

| Final set | System | Exact | Symmetric MIREX | Asymmetric MIREX | Mode |
|---|---|---:|---:|---:|---:|
| FMAK, 1,000 | Octave model | **60.30%** | **68.99%** | 66.49% | 78.00% |
| FMAK, 1,000 | libkeyfinder | 58.40% | 67.82% | 66.22% | 75.00% |
| GiantSteps, 604 | Octave model | **62.75%** | **70.40%** | 68.58% | 80.63% |
| GiantSteps, 604 | libkeyfinder | 60.60% | 69.35% | 67.70% | **81.79%** |

| Octave model minus libkeyfinder | MIREX difference, 95% CI | Exact difference, 95% CI |
|---|---:|---:|
| FMAK final | +1.17 points [-1.19, +3.44] | +1.90 points [-1.03, +4.77] |
| GiantSteps | +1.04 points [-1.75, +3.76] | +2.15 points [-1.32, +5.63] |

![Final paired accuracy differences and uncertainty](figures/final-comparison.png)

*Figure 2. Octave model's symmetric MIREX minus libkeyfinder's, in percentage
points, with 95% paired bootstrap intervals. Both intervals cross the dashed zero
line, so neither advantage is established.*

Intervals use 5,000 paired draws, seed 13. FMAK resamples 460
artist/album/audio components; GiantSteps resamples track hashes because artist
metadata was unavailable. Octave-model NLL / Brier / ECE (15 equal-width bins):
1.419 / 0.553 / 3.64% on FMAK and 1.393 / 0.517 / 7.51% on GiantSteps.
libkeyfinder had one missing FMAK prediction, counted as zero; on the 999
commonly decoded rows the MIREX gain is still 1.17 points. On GiantSteps the
octave model's mode accuracy is slightly lower (80.63% versus 81.79%) despite
higher exact accuracy.

The first calibration attempt stopped on MP3 files whose reported duration
(about 192 s) did not match decoded audio (about 79 s). Evaluation now falls back
to full decoding after a fast-path failure; two calibration and two final FMAK
files used it. No weights, splits or selection rule changed.

### CPU and size

These measurements are for the 347,064-parameter frequency-head model, not the
octave model.

- 1,388,256 bytes of FP32 parameters; checkpoint 1,512,199 bytes; ONNX 1,402,048
  bytes, parity verified for batches 1, 2 and 3.
- Threadripper PRO 9955WX, PyTorch CPU, one thread, batch 1: forward p50 0.448 ms,
  p95 0.525 ms over 100 synthetic inputs.
- Apple Silicon Mac, one thread: PyTorch p95 0.776 ms, ONNX p95 0.750 ms.
- One two-minute Beatport preview: cold decode plus features plus forward 228 ms;
  warm cache p95 2.39 ms.
- 100 FMAK files (median 232 s, p95 360 s, max 453 s): Python/PyTorch full path
  p50 / p95 0.523 / 1.026 s, peak parent RSS 796 MiB. Streaming the FFT
  (bitwise-identical features, also checked on a 7.5-minute file) gave p95 0.623 s
  and 671 MiB. A NumPy/ONNX Runtime path without PyTorch gave p50 / p95
  0.403 / 0.824 s, 65 MiB before audio and 197 MiB peak.

These are single runs with uncontrolled file caches, not randomized backend
comparisons. Parent RSS excludes the decoder process. The octave model's CPU cost
was not measured. It has 189,136 FP32 parameter bytes and a 436,217-byte
checkpoint; fewer parameters do not establish lower latency.

### Compute

Screen and confirmation took 20.79 minutes of wall time (typical 120-epoch run 63
to 79 seconds); the adaptation pilot 2.44 minutes and its confirmation 8.45
minutes, for 31.68 minutes of CPU fitting in total (16 threads, wall time).
Octave features took 136.6 seconds (2,582 tracks, about 2.82 GB), the pilot
432.39 seconds and the three confirmation runs 19.46 minutes on one GPU. FMAK
download and preparation took 54.73 minutes.

## Decision

Keep libkeyfinder. The octave model's final gains (+1.17 and +1.04 MIREX points)
have intervals that include zero, so they do not show a reliable advantage or a
state-of-the-art result. Development diagnostics showed largely complementary
errors: an oracle choosing the better of libkeyfinder and the octave model per
track would score 82.14% Beatport and 77.24% FMAK, versus 72.94% / 68.76% for the
octave model. That upper bound motivated one DSP-guided residual pilot, which
failed (see [RESEARCH_TRAINING.md](../../../docs/RESEARCH_TRAINING.md#what-came-next)).
Later work moved to K10, which became the release.

## Limitations

- GiantSteps was exposed by earlier comparisons, and both final sets are now
  exposed; repeated improvements on them are not fresh confirmation.
- The 159 calibration rows were included in the earlier whole-development
  diagnosis, though not in fitting or selection of the adapted and octave models.
- Near-duplicate, remix and cover independence is unaudited; GiantSteps artist
  independence is unknown.
- The octave model changes several components at once; no ablation separates them.
- This FMAK partition is not comparable with published all-FMAK results.
- The octave checkpoint format is experimental and rejected by Drone's predictor.
  The private DJ library was reserved for final review and not used here.

## Reproduce

Commands for every stage are in
[RESEARCH_TRAINING.md](../../../docs/RESEARCH_TRAINING.md#reproducibility). Run
records (histories, confusion matrices, probabilities, source/data/split
fingerprints and the frozen configuration) are kept outside the repository with
the original experiment outputs; no audio or checkpoints are committed.
