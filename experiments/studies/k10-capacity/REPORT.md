# K10 capacity and spatial resolution

The completed width-2 pilot below remains negative. A separate
[resolution follow-up](#follow-up-a-finer-grid-at-the-same-parameter-budget) has
synthetic resource evidence only; its musical-key accuracy is not measured.

## Problem

Does more supervised capacity produce a stronger own-key teacher before investing
in distillation? The small release candidate has 109,128 parameters; this study
tests exactly one wider model with 361,080. It does not commit to product tiers
or assume parameter count is the current accuracy bottleneck.

## Method

The [protocol](PROTOCOL.md) fixes width 2.0 versus width 1.0, with the same clean
2,206 fitting / 376 validation identities, CQT caches, augmentation, hard CE,
batch size, AdamW, cosine horizon and selection/early-stopping rule. No width,
learning-rate or regularization sweep. Different parameter shapes necessarily
consume different initialization/Dropout RNG draws; shared channel initialization
is not claimed. The NumPy data/crop schedule remains unchanged.

The already reproduced seed-13 control from the [structured-target study](../k10-structured-targets/REPORT.md)
was reused only after rechecking actual checkpoint tensors, history, predictions
and artifact hashes against the archived joint run. Explicit source/math audits
complement that proof. The candidate needed at least one FMAK MIREX point, no
Beatport MIREX loss and no exact loss in either domain to earn confirmation.

An upfront 45-minute candidate budget allowed the full 120-epoch horizon; controls
retain 20 minutes. Only this candidate could raise the allocator cap to 8% after
requiring 12 GiB free. OOM/timeout would be resource/budget-inconclusive rather
than an accuracy-negative result. Neither occurred; vLLM stayed running.

## Results

| Arm | BP symmetric / asymmetric MIREX | BP exact | FMAK symmetric / asymmetric MIREX | FMAK exact |
|---|---:|---:|---:|---:|
| Verified width-1 control | 76.11 / 73.33% | 68.25% | 75.48 / 74.48% | 68.40% |
| Width-2 candidate | 73.97 / 71.98% | 67.46% | 74.40 / 72.80% | 66.80% |

![Paired width-1 and width-2 validation scores](figures/accuracy.png)

*Figure 1. The wider candidate misses the method gate. Symmetric-fifth MIREX
is the fixed selection convention; asymmetric MIREX and exact accuracy expose
different error consequences.*

Symmetric-MIREX changes were **−2.14 Beatport / −1.08 FMAK points**. Exact accuracy
fell 0.79 / 1.60 points. FMAK had 12 exact promotions and 16 exact demotions;
Beatport had eight promotions and nine demotions. Full category movement matrices
under both conventions are retained in the raw result.

![Fitting loss and combined validation selection histories](figures/training-history.png)

*Figure 2. Both arms use hard CE. Markers identify selected epochs. The candidate
selected epoch 20 and stopped at 45; the control selected 37 and stopped at 62.
These descriptive traces do not isolate a general capacity/optimization effect.*

At matched epoch 20, fitting loss was 1.331 for control and 1.367 for width 2.
The data do not establish a simple “larger model fits better but overfits” story.
Candidate wall time was 274.26 seconds; differing stop epochs and system load
prevent a direct training-throughput claim. Peak Torch allocations were 3.16 GB,
with 4.57 GB reserved. These exclude some CUDA/runtime memory; they are not an
inference or application RSS measurement.

## Discussion and decision

The fixed wider recipe did not earn a stronger teacher, confirmation seeds or a
larger product tier. It does not prove that every larger model is worse: model
scale and optimizer compatibility interact, and only one unchanged recipe was
tested. Do not start an opportunistic hyperparameter sweep from this result.

Keep release weights unchanged. A more targeted inductive-bias or data hypothesis
is justified before assuming capacity alone will close the accuracy gap. One seed
on reused validation cannot establish generalization or a route to 90% exact.
No calibration/final/SWD/private predictions or distillation followed.

## Reproduction and evidence

The actual fitting source is preserved by the [`research/k10-capacity-v1`](https://github.com/lvndry/drone/tree/research/k10-capacity-v1) tag.

Actual fit source: `679b7fdb021af64cb2816c8b61d571a5d3ac773c`, based on
`a1abac9ddad927830b0409f80d6908502eeab2cd`. Replay fitting from the frozen source,
not an evolving combined runner. [Candidate result](data/candidate-result.json)
contains the independently reverified control proof, source/math audits, logits
artifact hashes, memory policy and gate. [History extracts](data/histories.json)
retain original run hashes; full runs/checkpoints remain under
`/home/lysk-admin/drone-experiments/k10-capacity-20261111-v1` and the referenced
structured-control directory. No archived source/protocol bytes were changed.
Regenerate figures with `uv run --extra plots python -m tools.plot_research`.


## Follow-up: a finer grid at the same parameter budget

### Problem and method

Could retaining finer frequency positions improve key recognition without making
Drone a larger product? The [prospective protocol](RESOLUTION_PROTOCOL.md) fixes
one candidate: six canonical convolutions with channels 16/32/32 and pooling
(2,2), (1,2), (1,2), followed by time averaging and an absolute-frequency head.
It uses the existing 192-bin CQT. No input octave folding, circular sharing,
teacher features or optimizer change is introduced.

The final grid has one-semitone spacing. Ordinary K10's coarser grid does not prove
hard pitch quantization: channels can retain within-cell information. This is a
joint capacity-allocation/resolution hypothesis. The [InceptionKeyNet paper](https://archives.ismir.net/ismir2021/paper/000004.pdf)
motivates preserving frequency localization; it does not establish this compact
candidate's accuracy or reproduce its architecture.

### Measured resource screen, no accuracy result

![Synthetic capacity allocation and operation counts](figures/resolution-screen.png)

*Figure 3. Parameter counts and analytical convolution-plus-linear MACs from the
preserved synthetic screen. These exclude BN, activations, pooling, reductions
and backend effects. They are not accuracy results or deployment latency.*

| Quantity | Ordinary K10 | Finer-grid candidate |
|---|---:|---:|
| Trainable parameters | 109,128 | 108,936 |
| Backbone / head parameters | 72,240 / 36,888 | 35,184 / 73,752 |
| Last hidden shape before time mean | 64×24×floor(T/8) | 32×96×floor(T/8) |
| Conv+linear MACs at T=320 | 575,115,264 | 504,373,248 |
| Local frequency receptive field | 36 input bins | 22 input bins |
| Local temporal receptive field | 36 frames | 36 frames |

The candidate has about 12.3% fewer counted MACs. Its largest individual FP32
activation in this T=320 screen is unchanged at 3,932,160 bytes, although some
intermediates double. This is not a total inference or training memory bound.

A tiny CPU-only timing screen used Torch 2.15 on this arm64 Mac, two intra-op
threads, one inter-op thread, inference mode, one warmup and three alternating
samples per model/length. Median K10→candidate times were 4.466→3.865 ms at T=65,
21.812→17.899 ms at T=320, and 74.945→74.537 ms at T=1615. The long case is
essentially tied. These samples do not establish a reliable speedup, ordinary-PC
performance, complete pipeline cost or per-model RSS. The timing process held
both models, so its lifetime memory peak cannot be assigned to either one.

### Decision and limitations

Prepare this one candidate for an exact-source server resource qualification,
then a bounded matched comparison only if measured cost fits the budget. The
control is the fixed CPU320 arm of the ongoing context study, not whichever crop
arm happens to win. Cohort, crop/augmentation streams, labels, update horizon,
selection and full-song evaluator remain fixed. The protocol requires at least
five additional FMA exact cases with no BP exact or weighted-score decline;
confirmation and release remain separate decisions.

The principal architectural risk is less nonlinear frequency context: 22 input
bins span only 11 semitones. A dense head can combine distant features linearly,
but cannot restore arbitrary lost nonlinear octave interactions. A negative
outcome must retain that limitation and cannot be recast as evidence that every
high-resolution architecture fails. No task training, accuracy evaluation,
confirmation or release promotion has occurred for this candidate.

[Raw synthetic screen](data/resolution-shape-audit.json) and its
[provenance](data/resolution-shape-audit-provenance.json) preserve the original
standalone prototype source byte-for-byte. That prototype, rather than the later
imported model module, produced these timings. The implementation's nine synthetic
tests establish shape, initialization and finite-update contracts, not accuracy.
Regenerate this resource figure with `uv run --extra plots python -m tools.plot_k10_resolution`.
