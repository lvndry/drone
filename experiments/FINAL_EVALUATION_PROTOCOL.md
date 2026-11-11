# Frozen octave-candidate evaluation

Declared before observing any calibration scores for the octave candidate or any
final-holdout model scores. It runs only if the fixed confirmation/selection gate
in OCTAVE_PROTOCOL passes. It does not authorize a new recipe selected on tests.

1. Audit completed seeds 13/37/73, identical source/data/splits except seed,
   checkpoint identities and repeated seed-13 predictions. Report floating-point
   history differences rather than claiming bitwise CUDA training reproducibility.
2. Select canonical seed 13 or the single predeclared equal-probability ensemble.
   Freeze checkpoint SHA-256s, input contract and selection result before
   calibration. No subset, mixing-weight or frontend search.
3. Fit one positive scalar temperature using NLL on the 159 assigned calibration
   rows; search 401 log-spaced values in [0.25,4]. Apply to log of the mean model/
   segment probability distribution. Verify class argmax remains unchanged.
   Save an immutable freeze packet before final evaluation.
4. Evaluate the original sealed 1,000-row FMAK final holdout once. No fitting,
   checkpoint or recipe choice uses these rows. Compare unmodified native Euphoria
   on these exact files, with decode/no-prediction failures counted as zero and
   coverage reported. Save raw predictions and paired artist/album/audio-cluster
   bootstrap intervals (5,000 draws, seed 13) for MIREX and exact differences.
5. Evaluate the same frozen candidate once on all 604 GiantSteps files. Disclose
   the program's earlier exposure from the initial v2 comparison. Do not call
   GiantSteps a pristine program-level holdout. The native reference is its
   previously completed same-audio 69.35% MIREX / 60.60% exact run.

Report exact, symmetric/asymmetric MIREX, mode accuracy, per-class support/confusion,
NLL, multiclass Brier and 15-equal-width-bin ECE. Distinguish seed SD, paired sample
uncertainty and calibration fitting metrics. The calibration rows appeared in the
original frozen v2 model's whole-development diagnosis before their later roles
were assigned; they have not fitted or selected the adapted/octave models. The
final 1,000 rows were reserved before that diagnosis and remain unscored at this
protocol declaration.

This is our artist-purged partition, not a direct replication of papers' all-FMAK
or different 1,000-track results. Exact hashes/primary artist and album IDs do not
establish cover/remix independence; Beatport lacks matched artist metadata.

The chosen candidate should meet native MIREX and exact accuracy on both final
datasets before progressing to Rust integration/default-detector consideration.
Report uncertainty and do not claim a significant gain when intervals include
zero. A failed final gate retains the current detector; do not tune on failure
and repeatedly score this holdout. No state-of-the-art claim follows from adopting
a published architecture or beating one local reference.

Budget: frozen neural calibration/evaluation stage capped at 20 minutes; native
1,000-track reference capped at 15 minutes and may run concurrently after the
freeze. CPU/GPU performance optimization and the private DJ-library final runtime/
manual review follow accuracy validation. Preserve vLLM throughout.
