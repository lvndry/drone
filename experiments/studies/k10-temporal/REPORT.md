# Temporal heads on frozen K10 features did not improve key recognition

A small temporal head trained on frozen K10 features made key recognition worse.
The ordered head lost 5.60 points of FMAK exact accuracy and 2.38 points on Beatport
against the original K10 classifier, and it did not beat the same head with
shuffled frame order. Both pre-set thresholds failed, so this recipe was stopped.
The study used seed 13 and ran in November 2026.

## Question

Does modeling how frozen K10 features change over a complete recording improve
exact key recognition, and if so, is temporal order the reason? A negative answer
rejects this specific frozen-feature head. It does not show that temporal modeling,
end-to-end temporal training or other representations cannot help.

## Method

The [protocol](PROTOCOL.md) was fixed before feature extraction and training. The
encoder was the standard joint K10 model (seed 13), kept frozen. Training used 1,230
Beatport and 976 FMAK recordings; evaluation used the existing 126 Beatport and 250
FMAK development recordings. FMAK is split by audited artist and album groups.
The Beatport split is stratified by label with duplicate-ID and exact-audio checks,
but is not known to be artist-disjoint. Both development sets have been used in
earlier studies, so they are not untouched test sets. No private, calibration or
final-test recordings were used.

All 2,582 recordings were represented by the encoder's 1,536-dimensional hidden
frames (tokens), stored in full FP32 with no time sampling or truncation. The
original classifier's logits were kept as a frozen residual baseline. Three heads
were trained:

- M: nonlinear, order-independent token processing followed by uniform pooling.
- O: an ordered two-block depthwise temporal residual head.
- S: the same head as O, with each recording's token order shuffled.

O and S started from the same parameters. All heads had zero-initialized output
layers, so they initially reproduced the original logits exactly. Padding tokens
were masked out of convolutions, normalization and pooling. O and S project 1,536
to 64 dimensions, apply kernel-5 depthwise blocks with dilations 1 and 4, and read
out 64 to 24 classes as a residual. M has 108,888 trainable parameters and O and S
have 109,400 each. The receptive field is 196 CQT frames, about 36 seconds.

Training: AdamW, learning rate 0.0003, weight decay 0.001, 60-epoch cosine
schedule, gradient clipping at 1, batch size 32 (split into micro-batches of at
most 8,192 padded tokens so long tracks were not dropped), early-stopping patience
12. Selection used the mean exact accuracy of the two domains, then the
equal-weighted cross-entropy, then the earlier epoch.

Pre-set thresholds:

- Accuracy threshold: O must gain at least 2 FMAK exact points (5 recordings) over
  the original classifier, with no Beatport exact loss and no symmetric MIREX
  decline in either domain.
- Mechanism threshold: O must beat both M and S by at least 1 FMAK exact point
  (3 recordings).

## Results

![Selected full-track exact accuracy and symmetric-fifth MIREX](figures/selected-results.png)

*Figure 1. Exact accuracy and symmetric MIREX for the original classifier and the
three heads. Every head scores below the original classifier on FMAK exact.*

| Model | Beatport exact | Beatport symmetric / asymmetric MIREX | FMAK exact | FMAK symmetric / asymmetric MIREX | Selected / stop epoch |
|---|---:|---:|---:|---:|---:|
| Original K10, seed 13 | 68.254 | 76.111 / 73.333 | 68.400 | 75.480 / 74.480 | Archived control |
| M | 62.698 | 71.349 / 68.571 | 65.600 | 73.520 / 72.320 | 1 / 13 |
| O | 65.873 | 73.413 / 70.635 | 62.800 | 71.200 / 70.000 | 11 / 23 |
| S | 66.667 | 73.571 / 71.190 | 63.200 | 71.360 / 69.960 | 12 / 24 |

Values are percentages. Symmetric MIREX gives fifth credit in both directions;
asymmetric MIREX gives credit only for the ascending fifth.

O lost 14 FMAK and 3 Beatport exact predictions, so the accuracy threshold failed.
O minus M was -2.80 FMAK points and O minus S was -0.40, so the mechanism threshold
failed too.

O was also compared against the mean of three earlier Beatport-only control seeds,
which required a 2-point FMAK gain. O gained 0.635 Beatport symmetric MIREX points
but lost 1.507 FMAK points, so this comparison failed as well. It compares one head
with a three-seed mean, not two three-seed estimates.

![Fitting and development cross entropy for all three heads](figures/learning-curves.png)

*Figure 2. Training and development cross-entropy per epoch, with dashed lines at
the selected epochs. Training loss falls steadily while development loss rises for
O and S.*

O's training cross-entropy fell from 1.013 to 0.147 while its development
cross-entropy (domains weighted equally) rose from 1.163 to 2.113. S followed a
similar path (1.013 to 0.116 training, 1.163 to 2.161 development). M changed less
(1.017 to 0.653 training, 1.162 to 1.266 development). The heads fit the training
labels, but the gain did not carry over to development recordings. These curves do
not identify a single cause: limited data, domain shift, the frozen representation
and head flexibility all remain possible.

![FMAK exact-accuracy contrasts and paired track bootstrap intervals](figures/paired-exact.png)

*Figure 3. FMAK exact-accuracy differences with 95% paired bootstrap intervals (2,000
resamples over tracks). O and S are clearly below the original classifier, and O is
indistinguishable from S.*

| Contrast | Beatport exact change, points [95% interval] | FMAK exact change, points [95% interval] |
|---|---:|---:|
| M minus original | -5.56 [-10.32, -1.59] | -2.80 [-6.00, 0.00] |
| O minus original | -2.38 [-9.52, 3.99] | -5.60 [-10.40, -1.20] |
| S minus original | -1.59 [-7.94, 4.76] | -5.20 [-10.00, -0.80] |
| O minus M | 3.17 [-3.17, 9.52] | -2.80 [-7.20, 1.20] |
| O minus S | -0.79 [-4.76, 2.38] | -0.40 [-3.60, 2.40] |

The intervals ignore artist dependence and checkpoint-selection bias, and do not
reflect seed variation.

Against the original classifier, O fixed 8 Beatport errors and broke 11 correct
predictions; on FMAK it fixed 9 and broke 23. The 23 broken FMAK predictions became
8 fifth, 4 relative, 4 parallel and 7 other errors. The 9 fixed ones came from 3
fifth, 1 relative, 1 parallel and 4 other errors. The weighted MIREX scores
therefore do not hide an exact-accuracy gain. O versus S fixed 7 FMAK errors and
broke 8; on Beatport it fixed 2 and broke 3, which gives no support for temporal
order mattering. Full error-transition matrices are in the
[result file](data/result13.json).

Before training, all 2,582 cached features reproduced the original classifier's
logits exactly (zero maximum error, zero prediction disagreements). The cached
features took 1,856,372,736 bytes, below the 4 GiB limit set in advance, and
extraction took 16.066 seconds. Training and scoring took 37.138 seconds for M,
62.605 for O and 83.955 for S; the total including extraction was 220.621 seconds,
within the 60-minute limit. Peak GPU memory was 257,705,472 bytes for extraction and
at most 237,454,336 bytes for the heads.

## Decision

Stop this frozen-head approach. No confirmation seeds, CPU measurement, checkpoint
promotion or tuning followed, and no learning-rate, regularization, feature or
architecture variant was tried after the results. The evidence points toward
studying better tonal representations and data efficiency before adding another
head; that needs its own protocol and clean evaluation.

## Limitations

- One seed, evaluated on development sets already used in earlier studies.
- The encoder is frozen, so its representation cannot adapt to a temporal objective.
- Beatport development data is not artist-disjoint, and the bootstrap intervals
  ignore artist grouping.

## Reproduce

The executed source is preserved at the git tag `research/k10-temporal-v1`; use
that tag for a full replay, since later changes on the main branch differ. The
[provenance file](data/provenance.json) lists the size and checksum of every
evidence file. Validation logits, histories and schedules are stored compressed;
no checkpoints, audio or feature cache are committed.
[Reconstruction](data/reconstruction.json) and [preflight](data/preflight.json)
record the feature-cache checks. Figure inputs are listed in the
[shared registry](../figure-inputs.json). Regenerate the figures with
`uv run --extra plots python tools/plot_research.py`.
