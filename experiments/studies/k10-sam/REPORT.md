# K10 with sharpness-aware minimization (SAM): lower accuracy

Training K10 with a fixed sharpness-aware minimization (SAM) step lowered accuracy in
both domains: symmetric MIREX fell 1.51 points on Beatport and 1.16 points on FMAK,
and exact accuracy fell 1.59 and 1.20 points. The pilot failed its pre-set threshold,
so no confirmation seeds, tuning or release change followed. The pilot used seed 13
and ran in November 2026.

## Question

Does a sharpness-aware update help K10 learn more useful key evidence than ordinary
AdamW? This tests one optimization recipe transferred from the literature; the
representation, model size and inference are unchanged.

## Method

The [protocol](PROTOCOL.md) was fixed before training. SAM uses a Euclidean radius of
0.05 over all trainable gradients, including biases and BN scale and shift, wrapped
around the existing AdamW step. Each batch gets two forward and backward passes with
hard cross-entropy. Both passes use the same dropout randomness and batch statistics,
but only the first pass updates the stored BN statistics. Parameters are restored
exactly before the single AdamW step, and the random-number state advances as it
would after one ordinary pass. Augmentation is computed once per batch. There is no
EMA, BN recalibration, gradient clipping, label smoothing or optimizer sweep.

Both arms use the same 109,128-parameter K10, the same cached CQT features, 2,206
fitting recordings (1,230 Beatport and 976 FMAK) and 376 validation
recordings (126 Beatport and 250 FMAK). Shared settings: 320-frame crops,
pitch-shift augmentation by shifting CQT bins, batch size 32, AdamW with learning
rate 1e-3 and weight decay 1e-4, cosine schedule up to 120 epochs, early-stopping
patience 25, and selection of the earliest epoch with the best combined symmetric
MIREX. Each arm selects its own checkpoint by this rule. Validation uses complete
tracks.

Before SAM was run, a fresh control with SAM disabled reproduced the archived
ordinary K10 run exactly: history, predictions, selected and stopping epochs, and
every checkpoint tensor. File checksums differ because the saved metadata differs.
The code audit of the ordinary training step is in [base-audit.json](data/base-audit.json).

Pre-set success threshold: at least +1.00 FMAK symmetric MIREX point, no Beatport
symmetric MIREX loss and no exact-accuracy loss in either domain. Only a passing
pilot would earn seeds 37 and 73, whose means would then also need +2.00 FMAK points
over the earlier Beatport-only controls, with no Beatport MIREX loss and no regression
against ordinary joint training.

## Results

| Arm | BP symmetric / asymmetric MIREX | BP exact | FMAK symmetric / asymmetric MIREX | FMAK exact |
|---|---:|---:|---:|---:|
| Ordinary K10 (reproduced) | 76.11 / 73.33% | 68.25% | 75.48 / 74.48% | 68.40% |
| SAM + AdamW | 74.60 / 72.22% | 66.67% | 74.32 / 73.52% | 67.20% |

Symmetric MIREX (fifth credit in both directions) is the primary weighted score;
asymmetric MIREX (ascending fifth only) is reported alongside, and exact accuracy is
a separate condition.

![SAM versus its verified ordinary control](figures/accuracy.png)

*Figure 1. SAM scores lower than ordinary training on every metric in both domains.*

Symmetric MIREX changed by -1.51 points on Beatport and -1.16 on FMAK; asymmetric by
-1.11 and -0.96; exact by -1.59 and -1.20. The threshold failed.

| Paired exact movement | Beatport, n=126 | FMAK, n=250 |
|---|---:|---:|
| Wrong to correct | 8 | 10 |
| Correct to wrong | 10 | 13 |
| Net change | -2 | -3 |

![Paired exact-error promotions and demotions](figures/paired-exact-movements.png)

*Figure 2. Tracks that SAM fixed and broke relative to ordinary training. More correct
predictions are lost than gained in both domains. Full error-category matrices are in
the [candidate result](data/candidate-result.json).*

![Observed training and checkpoint-selection traces](figures/training-history.png)

*Figure 3. Training loss (from the first, unperturbed pass) and validation score per
epoch. Ordinary AdamW selected epoch 37 and stopped at 62; SAM selected 49 and stopped
at 74. The curves describe this recipe but do not explain why it failed.*

Training took 138.86 seconds for the control and 282.49 seconds for SAM (138.62 and
282.25 seconds for fitting alone), within the 1,200 and 2,400 second limits set in
advance. The arms ran different numbers of epochs, so the ratio is not a matched
estimate of SAM's overhead. Peak GPU memory was 1,529,839,104 and 1,530,291,200 bytes
allocated, and 1,916,796,928 bytes reserved in both arms. Software: PyTorch
2.15.0+cu132, NumPy 2.5.4, four CPU threads, TF32 off, deterministic settings as in
the control. SAM changes only training, not inference.

## Decision

Stop this optimizer route. No seeds 37 and 73, radius changes, BN alternatives,
teacher use, export or student training followed. Current weights are unchanged.

## Limitations

- One seed. The 376 validation recordings selected checkpoints and were used in
  earlier development, so this is not an independent estimate of generalization.
- Beatport validation is split by label with exact-audio checks but is not
  artist-disjoint; FMAK uses audited artist and album groups.
- No calibration, final-test or private recordings were used.
- This rejects one radius, BN and randomness handling with AdamW, not all SAM variants.

## Reproduce

The executed source is preserved at the git tag `research/k10-sam-v1`. The
[execution provenance](data/execution-provenance.json) lists checksums for the
launcher, runs, validation outputs and results. The [control result](data/control-result.json),
[candidate result](data/candidate-result.json), compressed
[control run](data/control-run.json.gz) and [candidate run](data/candidate-run.json.gz),
and [histories](data/histories.json) are included; checkpoints and audio are kept
outside the repository.

Check the report numbers with `uv run python -m tools.check_optimizer_reports` and
regenerate the figures with `uv run --extra plots python -m tools.plot_research`.
Neither command trains or runs a model. Figure inputs are listed in the
[shared registry](../figure-inputs.json).
