# K10 probability ensemble: no stronger teacher

Averaging the probabilities of three independently trained K10 models improved
Beatport but lowered FMAK compared with the standard seed-13 model (symmetric MIREX
+1.75 and -1.48 points; exact +1.59 and -2.40 points). It therefore failed the
pre-set threshold for a stronger teacher, and also failed a separate comparison
against Beatport-only controls. No distillation or release change followed. The
study ran in November 2026.

## Question

Can an ensemble of independently trained K10 models serve as a stronger teacher,
trained only on project data, before distillation is considered?

## Method

The [protocol](PROTOCOL.md) was fixed before any predictions were extracted. The
ensemble uses the joint K10 models with seeds 13, 37 and 73, equal weights of 1/3 and
temperature 1. Each model's logits go through a numerically stable float64 softmax
and the three probability vectors are averaged. This is neither a softmax of averaged
logits nor a majority vote. Exact ties go to the lowest class index. No weight or
temperature tuning, member replacement or best-seed selection was allowed.

Each member is a 109,128-parameter K10 checkpoint selected by its original joint
training protocol. The models were rebuilt with the same code that trained them,
and each member's predictions and metrics matched its recorded run exactly before
the ensemble was scored. Predictions were extracted one model at a time on full-length
tracks in batches of up to four, with BN frozen and no augmentation or chunking.

Only the 376 validation recordings (126 Beatport and 250 FMAK) were scored, from the
same cached CQT features; no training took place. Beatport validation is split by
label with exact-audio checks but is not artist-disjoint; FMAK uses audited artist and
album groups. No calibration, final-test or private recordings were used.

Pre-set success thresholds:

- Teacher threshold: at least +1.00 FMAK symmetric MIREX point over the standard
  seed-13 model, with no Beatport symmetric MIREX loss and no exact loss in either
  domain.
- Separately, at least +2.00 FMAK points over the mean of the three earlier
  Beatport-only controls, with no Beatport MIREX loss.

Passing either would not by itself trigger promotion or distillation.

## Results

| Model | BP symmetric / asymmetric MIREX | BP exact | FMAK symmetric / asymmetric MIREX | FMAK exact |
|---|---:|---:|---:|---:|
| Joint seed 13 (standard) | 76.11 / 73.33% | 68.25% | 75.48 / 74.48% | 68.40% |
| Joint seed 37 | 75.08 / 72.30% | 67.46% | 73.80 / 72.40% | 66.40% |
| Joint seed 73 | 74.29 / 72.30% | 66.67% | 73.64 / 72.44% | 65.60% |
| Equal-weight ensemble | 77.86 / 74.68% | 69.84% | 74.00 / 73.00% | 66.00% |

![Fixed members and their single probability ensemble](figures/accuracy.png)

*Figure 1. Accuracy of the three members and their ensemble; dotted lines mark the
seed-13 model. The ensemble beats seed 13 on Beatport but not on FMAK. There is one
ensemble, so no error bar.*

Against seed 13, symmetric MIREX changed by +1.75 points on Beatport and -1.48 on
FMAK, and exact accuracy by +1.59 and -2.40. The teacher threshold failed. The member
means are 75.16% (Beatport) and 74.31% (FMAK) symmetric MIREX, and 67.46% and 66.80%
exact; the spread across members is not an uncertainty estimate for the ensemble.

| Ensemble versus member | BP change symmetric / exact, points | BP exact fixed / broken | FMAK change symmetric / exact, points | FMAK exact fixed / broken |
|---|---:|---:|---:|---:|
| Seed 13 (standard) | +1.75 / +1.59 | 4 / 2 | -1.48 / -2.40 | 6 / 12 |
| Seed 37 | +2.78 / +2.38 | 4 / 1 | +0.20 / -0.40 | 6 / 7 |
| Seed 73 | +3.57 / +3.17 | 7 / 3 | +0.36 / +0.40 | 9 / 8 |

![Paired exact-error changes against every fixed member](figures/paired-exact-movements.png)

*Figure 2. Tracks the ensemble fixed and broke relative to each member. Beatport gains
come with FMAK exact losses against seeds 13 and 37; comparing only with seed 73 would
hide the failure against the standard model. Full error-category matrices are in the
[result file](data/result.json).*

Against the mean of the three Beatport-only controls, symmetric MIREX rose by 5.08
points on Beatport and 1.29 points on FMAK, below the required +2.00 on FMAK. This
compares one ensemble with a mean of three runs, not three ensembles.

The whole run took 3.75 seconds, within its 20-minute limit. Peak GPU memory was
229,993,984 bytes allocated and 788,529,152 bytes reserved. Software: PyTorch
2.15.0+cu132, NumPy 2.5.4, four CPU threads, TF32 off. This is not an inference
latency figure: an ensemble runs three models per track, and the CQT frontend cost
was not measured.

## Decision

Stop. No distillation, alternative weights, temperatures or member sets, product tier
or export followed. A stronger teacher was not found, and current weights are
unchanged.

## Limitations

- One fixed set of three members and one combination rule, so there is no standard
  error for the ensemble.
- The 376 validation recordings selected each member's checkpoint and were used in
  earlier development, so they are not independent evidence of generalization.
- The result does not show that every ensemble or distillation method fails.

## Reproduce

The executed source is preserved at the git tag `research/k10-ensemble-v1`. The
[result file](data/result.json) records the member identities, checksums for source,
protocol, features, runs, checkpoints and logits, the prediction reproduction check,
both metrics, all paired movements and the threshold outcomes. The
[execution provenance](data/execution-provenance.json) records the run's output
checksums. Full logits and probabilities are kept outside the repository.

Check the report numbers with `uv run python -m tools.check_optimizer_reports`, then
regenerate the figures with `uv run --extra plots python -m tools.plot_research`.
Neither command trains or runs a model. Figure inputs are listed in the
[shared registry](../figure-inputs.json).
