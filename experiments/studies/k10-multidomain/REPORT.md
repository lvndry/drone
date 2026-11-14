# K10 multi-domain training: gains below the confirmation threshold

Adding 976 FMAK recordings to K10's Beatport training data raised mean symmetric
MIREX over three seeds in both domains: +2.38 points on Beatport and +1.60 points on
FMAK. The FMAK gain missed the pre-set +2.00 threshold, so these weights were not
promoted and the shipped Beatport K10 checkpoint is unchanged. With three seeds, a
gain this close to the threshold is inconclusive rather than evidence of no benefit.
The study ran on 11 November 2026.

All scores here use symmetric-fifth MIREX unless marked exact. They are measured on
validation data used for checkpoint selection, so they are not independent test
estimates or comparable to published state-of-the-art numbers.

## Question

The release recipe trains on Beatport EDM previews. Does adding the approved FMAK
fitting subset improve key accuracy across domains while keeping Beatport
performance, with the architecture and training recipe unchanged?

## Method

The [protocol](PROTOCOL.md) was fixed before training. Both arms share the K10 model,
input features, initialization seeds, optimizer, augmentation, early-stopping rule,
376 validation recordings and the combined per-track checkpoint-selection metric.
Only the fitting data differs: 1,230 clean Beatport recordings, versus the same plus
976 FMAK recordings. Validation is 126 Beatport and 250 FMAK recordings; FMAK
artist, album and audio-duplicate separation comes from the earlier audit.

Pre-set success threshold: a seed-13 pilot first had to show an FMAK gain of at least
2 points with a Beatport loss of no more than 2 points. Passing it triggers seeds 37
and 73, after which the three-seed mean must again show an FMAK gain of at least 2
points and a Beatport loss of no more than 2 points.

## Results

| Seed | Beatport-only: BP | Joint: BP | BP change | Beatport-only: FMAK | Joint: FMAK | FMAK change |
|---|---:|---:|---:|---:|---:|---:|
| 13 | 73.57% | 76.11% | +2.54 | 71.96% | 75.48% | +3.52 |
| 37 | 70.79% | 75.08% | +4.29 | 73.68% | 73.80% | +0.12 |
| 73 | 73.97% | 74.29% | +0.32 | 72.48% | 73.64% | +1.16 |
| Mean | 72.78% | 75.16% | +2.38 | 72.71% | 74.31% | +1.60 |

![Paired multi-domain mean scores and seed variation](figures/k10-multidomain.png)

*Figure 1. Mean scores for Beatport-only and joint training in each domain. Joint
training is higher in both domains, but the FMAK gain falls short of the 2-point
threshold. Error bars are standard deviations over three training seeds, not
confidence intervals for new recordings.*

Changes are in percentage points. Seed standard deviations were 1.73 (control) and
0.92 (joint) points on Beatport, and 0.88 and 1.02 points on FMAK. Mean exact
accuracy also rose: Beatport from 65.34% to 67.46%, FMAK from 63.87% to 66.80%.

The seed-13 pilot passed, so both arms were run with seeds 37 and 73. The Beatport
condition held across the three seeds, but the mean FMAK gain of 1.60 points did not
reach 2.00. The summary tool reports that the result does not qualify and names no
selected seed. The seed-13 gain (+3.52) was larger than the confirmed mean.

### Data and inference issues found along the way

The original K10 manifest had 1,363 rows but only 1,356 unique audio files: seven
duplicate groups, two of which crossed between fitting and validation. The data
splitter now keeps the validation copy, removes training copies and other fitting
duplicates, and rejects conflicting labels. This affected checkpoint selection; it is
not a demonstrated overlap with GiantSteps. Earlier test scores were not recomputed.

The old evaluation padded short tracks to the length of other tracks in the batch.
With batches of four, this changed 9 of 376 validation predictions of the shipped
model: FMAK MIREX was 71.52% with padding and 71.96% at true length; Beatport stayed at
75.79%. These shipped-model checks are not the fresh control above: the shipped model was
trained with the duplicate crossings and a different selection rule.

The production predictor also applied a leftover transpose although K10's features
are already laid out as (192 bins, time). The corrected path now matches a direct
call to the model. This bug affected inference only, not the training and evaluation
used here.

The previous ONNX export had a variable batch size but a fixed 320-frame length.
Making length variable exposed an exporter issue with adaptive pooling, so the export
now computes the equivalent mean over time directly, keeping all 24 frequency
positions. CPU outputs match the original PyTorch model at 8, 56, 319, 320 and 645
frames. The packaged ONNX was regenerated from the same shipped weights; the trained
model did not change.

### Cost

Preparing features for 2,582 tracks took 81.72 seconds (1,356 existing caches reused,
1,226 FMAK caches extracted). The six training runs took 634.09 seconds (10.57 minutes)
in total; individual runs took 58.54 to 134.68 seconds and all stopped early.

| Seed | Control best epoch / seconds | Joint best epoch / seconds |
|---|---:|---:|
| 13 | 41 / 134.68 | 37 / 134.60 |
| 37 | 38 / 83.49 | 23 / 103.91 |
| 73 | 19 / 58.54 | 30 / 118.88 |

Software: PyTorch 2.15.0+cu132, CUDA 13.2, NumPy 2.5.4, librosa 1.0.0, SciPy 1.18.1,
numba 0.68.1, soxr 1.1.0.

## Decision

Keep the pre-set threshold rather than lowering it after seeing the results. No extra
seeds, ensemble, larger model or new final-test score followed. The direction (more
diverse labelled data) remains promising, but the seed-13 pilot overstated the gain.
The lasting improvements from this study are the corrected inference path, ONNX export
for any length, and the data and evaluation integrity fixes. Further training needs a
new hypothesis and protocol.

## Limitations

- Three seeds, evaluated on validation data used for checkpoint selection. The FMAK
  gain (+1.60) is close to the threshold, and with seed standard deviations near 1
  point this design cannot tell whether the true gain is above or below 2 points.
- Seed spread measures training variation only, not uncertainty over new recordings.
- No calibration, final-test or private recordings were scored.

## Reproduce

The [summary](data/summary.json) holds the per-seed and mean results. The original
training code is preserved in the archived run directory, which is kept outside the
repository and contains:

- `source/`, `data/manifest.json`, `split-audit.json`, `environment.json`;
- `control-seed{13,37,73}/run.json` and `joint-seed{13,37,73}/run.json`, with source
  and feature checksums, fitting and validation IDs, histories and checkpoint checksums;
- `shipped-diagnostic/diagnostic.json`, `pilot-result.json`, paired comparisons, and
  `confirmation-summary.json` with all six runs and the threshold outcome.

Run `python -m tools.summarize_k10_multidomain --root <run-directory>` to recompute
the decision; it refuses incomplete arms or mismatched sources, configurations, seeds
or validation recordings. Regenerate the figure with
`uv run --extra plots python -m tools.plot_research`.
