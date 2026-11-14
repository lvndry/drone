# K10 v2 pilot: more FMA fitting data at a fixed number of updates

Training K10 on 4,033 instead of 976 FMA recordings, with the number of updates held
fixed, raised FMA exact accuracy from 172/250 to 176/250 (+1.60 points). That is
below the pre-set threshold of +2.00 points, so the pilot did not pass. Beatport
exact and both MIREX scores also improved. The 95% interval for the FMA gain (-1.94
to +5.08 points) contains both zero and the threshold, so at this sample size the
result is inconclusive rather than evidence of no effect. The study ran in November
2026 with one seed.

## Question

Can more FMA fitting data improve this small classifier without changing its
architecture? The [v2 data contract](../v2-data-contract/REPORT.md) reassigned
recordings to new roles using whole metadata components, a rule that does not look
at labels, and one fixed seed. The restricted fitting pool keeps the old 976 FMA
recordings; the expanded pool adds eligible recordings, and both are evaluated on
the same new validation panels. Recordings from the old final test set can enter
the new fitting pool. That exposure is documented, and the old benchmarks do not
become untouched again through repartitioning.

## Method

The [protocol](PROTOCOL.md) and [execution amendment](EXECUTION.md) fixed the method
before the run.

Data: after the input-integrity fix described below, 976 versus 4,033 unique FMA
fitting recordings, with the same 1,229 Beatport fitting recordings in both arms.
Both arms are evaluated on the same 126 Beatport and 250 FMA validation recordings.
A 998-recording confirmation panel (997 unique audio files) and 159 calibration
recordings are held out. FMA is grouped by audited artist and album components.
The Beatport split is stratified by label with exact-audio checks but is not
artist-disjoint: official metadata gives the artist for only 618 of the 1,230
original Beatport fitting recordings, and the other 612 are unknown.

Training: both arms are fresh K10 models (109,128 parameters) with the same seed-13
initial weights, hard cross-entropy, AdamW (learning rate 0.001, weight decay
0.0001), CQT input, 320-frame crops and pitch-shift augmentation by shifting CQT
bins. Evaluation uses full-length features without truncation. Each arm runs 120
blocks of 69 updates with batch size 32, so 8,280 updates, the same cosine schedule
and no early stopping. Beatport sampling weight is 1230/2206. Separate random
streams per domain give both arms identical Beatport samples and augmentations.
This is a new matched recipe, not a rerun of the old full-pass recipe (69 × 32 =
2,208 draws per block).

Selection: highest mean exact accuracy over the two domains, then symmetric MIREX,
then the earliest evaluation.

Pre-set success threshold: at least +2.00 FMA exact points (5 of 250 recordings),
with no decline in Beatport exact or either Beatport MIREX score. FMA MIREX changes
are reported but not part of the threshold.

![Training loss and exact-first inner selection](figures/learning-curves.png)

*Figure 1. Training loss and validation exact accuracy for both arms over the same
8,280 updates. Dashed lines mark the selected updates, 3,864 (restricted) and 5,796
(expanded). The validation curves use the same data that selects checkpoints.*

## Input-integrity issue found during preparation

A first preparation batch of 16 recordings completed normally (12 new, 4 reused
caches). The full preparation run then processed all 5,641 proposed inputs and
stopped, as designed, because some recordings decoded to identical model inputs.
There were 25 such duplicate families, and their cached features were also
identical. 23 families had consistent labels and stayed in the fitting pool, within
the same FMA components.

Three inputs were changed without resplitting, replacing recordings or choosing
labels:

| Input | Original role | Action |
|---|---|---|
| BP 2997186 | Fitting | Removed from fitting in both arms as a duplicate of validation recording BP 1926377 |
| FMA 051779 | Expanded fitting; previously quarantined | Quarantined: same audio as FMA 096100 with a conflicting label |
| FMA 096100 | Expanded fitting; previously in the old final test | Quarantined with FMA 051779; neither was in the restricted 976 |

The final manifest has 5,638 eligible recordings: 1,229 Beatport and 976 or 4,033
FMA for fitting, plus the unchanged 126 + 250 for validation. All existing feature
caches were reused.

An earlier audio-duplicate check had covered FMA families across roles and Beatport
fitting against FMA evaluation, but not Beatport fitting against Beatport
validation, or conflicting labels within the fitting pool. Older audio hashes
already showed the two problematic pairs as identical, so the miss was a gap in the
check's scope, not decoder drift. Decoded durations agree with all 5,641 earlier
fingerprints to within one sample at 22,050 Hz. Publisher duration mismatches and
decoder warnings are recorded in the diagnostic files.

## Results

| Validation panel / metric | Restricted | Expanded | Change, points | Paired 95% interval, points |
|---|---:|---:|---:|---:|
| BP exact, N=126 | 68.25% | 70.63% | +2.38 | -3.97 to +9.52 |
| BP symmetric MIREX | 76.43% | 77.14% | +0.71 | -4.52 to +6.03 |
| BP asymmetric MIREX | 72.86% | 75.16% | +2.30 | -3.02 to +7.94 |
| FMA exact, N=250 in 100 groups | 68.80% | 70.40% | +1.60 | -1.94 to +5.08 |
| FMA symmetric MIREX | 74.00% | 75.84% | +1.84 | -0.94 to +4.57 |
| FMA asymmetric MIREX | 73.60% | 75.04% | +1.44 | -1.44 to +4.23 |

![Paired changes with descriptive uncertainty](figures/paired-scores.png)

*Figure 2. Expanded minus restricted change for each metric with its 95% interval.
Every interval includes zero, and the FMA exact interval also includes the +2.00
threshold.*

Intervals use 10,000 paired group bootstrap draws (seed 20261112), resampling FMA
metadata components and individual Beatport recordings. They are computed after
checkpoint selection on validation data that has been used during development, so
they do not represent fresh-test coverage or seed variation. The FMA exact interval
(-1.94 to +5.08) spans the +2.00 threshold, so this sample size cannot distinguish a
gain of that size from no gain.

Exact-accuracy movement: on Beatport 11 recordings became correct and 8 became
wrong; on FMA 12 and 8. The net gain of four FMA recordings is one short of the five
required. Full per-recording predictions and error-transition matrices are in the
compressed result files.

In the expanded arm each FMA recording is seen only 29 to 30 times, versus 120 to 121
times in the restricted arm. Every recording is visited, but each gets fewer views.
This does not show underfitting or that longer training would help. Both arms see
the same 147,733 Beatport draws and 117,227 FMA draws. The pilot measures the value
of more data at this fixed update count, not the best accuracy a larger dataset
could reach with more training.

### Accuracy by track duration

![Actual-duration coverage and FMA exact scores](figures/duration-coverage.png)

*Figure 3. FMA validation recordings by actual duration, with exact accuracy per
bin for both arms. Very few recordings are longer than 5 minutes.*

| FMA duration | Recordings / groups | Restricted correct | Expanded correct | Top-level genre |
|---|---:|---:|---:|---|
| Under 2 minutes | 32 / 18 | 19 | 21 | 15 missing; 1 Electronic |
| 2 to under 5 minutes | 179 / 80 | 124 | 126 | 104 missing; 14 Electronic |
| 5 to 10 minutes | 37 / 27 | 27 | 27 | 23 missing, 13 Electronic, 1 Rock |
| Over 10 minutes | 2 / 2 | 2 | 2 | 1 missing, 1 Folk |

Durations come from actual sample counts, not publisher metadata. Groups can span
bins and should not be summed. The 5-to-10-minute bin shows no change, with a 95%
interval of -12.50 to +13.51 points. Beatport has 5 short recordings and 121 previews
of about 120 seconds, and no recordings of 5 to 10 minutes.

Only five FMA validation recordings of 5 to 10 minutes are tagged House (genre 182),
in three components: 3 of 5 correct for restricted and 5 of 5 for expanded. This is
far too few to judge (the bootstrap interval spans 0 to +100 points). Broad
Electronic is not verified house, and there are no deep or melodic house labels or
8-to-10-minute house tracks. Top-level genre is missing for 143 of the 250 FMA
validation recordings. The study cannot measure accuracy on long house tracks, the
main intended use.

### Resources

Both arms ran one after the other on the same GPU with four PyTorch threads, TF32 off
and the same software (Python 3.12.3, PyTorch 2.15.0+cu132, NumPy 2.5.4). Wall time
was 378.93 seconds (restricted) and 374.40 seconds (expanded), and CPU time 390.01 and
386.40 seconds. Peak memory was about 2.18 GB; the expanded arm's figure is an upper
bound that includes a child process. These are training costs, not inference speed.

## Decision

The pilot did not pass, so no confirmation, calibration, GiantSteps or private DJ
evaluation was run, and the release weights are unchanged. No extra seed, longer
training or threshold change was tried. The new input-integrity checks are kept.
Any further long-house evaluation or follow-up experiment needs its own protocol.

## Limitations

- One seed, on validation panels that are used for checkpoint selection and have
  been consulted during development. This is not an independent test.
- The FMA gain is near the threshold and its interval spans both zero and +2.00, so
  the pilot is inconclusive about a gain of that size.
- Expanded fitting gives each FMA recording about four times fewer views at the fixed
  update count.
- Beatport is not artist-disjoint, and there is no adequate long house-music panel.
- FMAK v2 labels ([source](https://zenodo.org/records/12759100)) include about 200
  modifications to the original expert FMAK labels that the original authors have not
  verified, so they are not adjudicated ground truth.

## Reproduce

The [archive index](data/results/archive-index.json) records the code, eligibility
checks and launch configuration as they ran, and the
[run record](data/results/terminal.json) records the runtime. No weights or audio are
published. Checksums for all evidence files are in [SHA256SUMS](SHA256SUMS).

Check the report arithmetic (no model or audio needed) and regenerate the figures:

```bash
python -m tools.check_k10_v2_report
python -m tools.plot_research
python -m pytest tests/test_k10_v2_pilot.py tests/test_k10_v2_bulk.py tests/test_k10_v2_amendment.py tests/test_k10_v2_report.py
```
