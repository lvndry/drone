# First small-model tranche

This is a validation-only experiment, not a release or architecture promotion.
The reviewed control is the 331,704-parameter CNN. The frequency-preserving head
has 347,064 parameters, still below the 2 MiB FP32 weight budget. Global pooling
was the historical control for this tranche. Production now uses the K10 model;
these older controls live in `experiments/reference/`, with their original feature
and checkpoint contracts. See [the research guide](../../../docs/RESEARCH_TRAINING.md).

## Fixed protocol

- Same corrected 1,363-row Beatport cache, split seed 13. An audit found seven
  identical-audio pairs, two crossing the old split. Preserve validation identities,
  quarantine their training copies, and deduplicate remaining training recordings:
  1,230 training and 126 validation recordings. All recipes, including control,
  use this cleaned split. Earlier baseline scores are diagnostic, not comparable
  control results. The runner never opens test features or test labels.
- Primary development criterion: symmetric MIREX; exact accuracy is a tie-breaker.
- Eight registered recipes in `small.py`, seed 13, 120 epochs, batch 64, CPU with
  16 intra-op threads. The control uses smoothing 0.1, dropout 0.3, AdamW LR 1e-3,
  weight decay 1e-4, and bounded pitch augmentation.
- Compare global vs frequency-preserving pooling, augmentation on/off, frequency
  pooling without smoothing, reduced dropout, reduced LR, and a global-pooling
  control with inverse-frequency mode weights. This is a focused screen rather
  than a factorial estimate of all interactions.
- Promote the control plus at most two challengers whose screening MIREX exceeds
  control by at least 0.02. Confirm at seeds 13, 37 and 73 on the same fixed split.
  Re-running seed 13 verifies reproducibility; it is not a fourth independent seed.
- Per-run training wall cap 300 seconds; one-hour cap per screening/confirmation
  command. At most eight screening runs and nine confirmation runs. No GPU service
  needs to be stopped. Do not broaden the search based on held-out results.

## Commands

```bash
python -m experiments.small diagnose --data <data> --output <diagnostic-dir> \
  --baseline <reviewed-reference-checkpoint>
python -m experiments.small screen --data <data> --output <screen-dir>
python -m experiments.small confirm --data <data> --output <confirm-dir> --screen <screen-dir>
```

Each output snapshots source files and records data/source/split hashes, explicit
thread count, configuration, histories and confusion matrices. `last.pt` stores
model/optimizer/scheduler/RNG state so restarting the same command resumes after
the last complete epoch. Mismatching source, configuration or data is rejected.
An independent test checks resumed training against uninterrupted training exactly.
The `.pt` files are local experiment artifacts until promoted and evaluated.

Run statistics on 126 validation tracks have considerable uncertainty. Multi-seed
means measure training variability on this split, not uncertainty over new datasets.
Compare FMAK generalization and CPU end-to-end cost before release claims. Do not
use the 69.4% external libkeyfinder reference as a validation score to optimize.
