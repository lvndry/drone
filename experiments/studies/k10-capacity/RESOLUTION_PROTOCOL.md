# Prospective K10 spatial-resolution comparison

Status: model preparation and local synthetic shape/cost evidence only. No accuracy
result, task-fit launch, release change or automatic confirmation follows this file.
The earlier width-2 experiment and its fixed negative result remain unchanged.

## Problem and selected hypothesis

K10 reduces 192 CQT frequency bins to 24 spatial positions through three frequency
pooling stages. Its feature channels may retain within-cell pitch information;
this is not literal four-semitone pitch quantization or a proven accuracy bug.
We will test one alternative allocation of almost the same parameter budget:
retain 96 frequency positions while reducing the final block's channel count.

The candidate uses the same six 3×3 convolution / BN / ELU operations, with channels
16/32/32 rather than 16/32/64. Pooling shapes are (2,2), (1,2), (1,2), with frequency
first. Time is still reduced eightfold. Time-only averaging produces 32×96 features
for a linear 24-class head. There are 108,936 trainable parameters, versus 109,128
for ordinary K10. The input CQT, labels and decoding contract remain unchanged.
This is a capacity-allocation AND frequency-resolution comparison, not a pure
causal frequency-resolution ablation. No wider variant or alternative head is
part of this pilot.

The motivation is supported by harmonic-classification work that preserves
frequency localization, including [InceptionKeyNet, ISMIR 2021](https://archives.ismir.net/ismir2021/paper/000004.pdf).
This proposed compact model is not InceptionKeyNet and inherits none of its scores.
The [shift-sensitivity work, ICML 2019](https://proceedings.mlr.press/v97/zhang19a.html)
also motivates examining pooling, without proving an advantage for this recipe.

## Known tradeoffs before fitting

The candidate has 35,184 backbone and 73,752 head parameters, compared with
72,240 and 36,888. Its final frequency receptive field shrinks from 36 input bins
(18 semitones) to 22 bins (11 semitones); its temporal receptive field stays 36
frames. The dense readout can combine distant positions linearly, but cannot
restore arbitrary lost nonlinear octave interactions. This is a substantial
accuracy risk, not something to omit if the candidate fails.

At T=320, convolution-plus-linear MACs are 504,373,248 versus 575,115,264, about
12.3% fewer. Counts exclude BN, activations, pooling, reductions and backend costs.
The largest individual FP32 activation is unchanged, but some intermediates double;
this does not bound training RSS or prove a runtime improvement. The local synthetic
screen is only resource evidence. Long-case timings are essentially tied and do
not establish an ordinary-PC performance gain.

## Matched accuracy comparison

The canonical control is the fresh **CPU320 arm** of the fixed context experiment,
source `90a318e069ed63a3bc1f3af639dcdf7c3255b93f`, not whichever context arm wins.
Its results must be complete, source-bound and pass their selected-checkpoint
reload proof. Do not replace it with the historical GPU control or a partial epoch.
If reuse cannot be proven numerically equivalent for the unchanged baseline path,
prepare a fresh matched control rather than weakening the comparison.

The candidate must use the same admitted 1,229 Beatport plus 4,033 FMA fitting IDs,
320-frame crops, keyed track/augmentation draws, ±4-semitone CQT reslicing with
negative-shift target transposition, padding rule, batch 32, hard CE, AdamW defaults
(lr .001, weight decay .0001), and cosine schedule. Freeze 8,280 updates and 120
selection events, one every 69 updates; no independent early stop or seed search.
Use seed 13. The first two blocks' initial tensors must match ordinary K10 under
that seed. Different-shaped later blocks/readout consume different random draws;
identical complete initialization or dropout masks are not claimed.

Selection remains equal-domain exact first, symmetric MIREX second, earliest tie.
Evaluate the same full-length BP126/FMA250 inputs at true lengths with batch 4.
Require finite outputs and exact 376-prediction identity after strict selected-file
reload. Retain final and selected checkpoints separately. No calibration, reserved
confirmation, historical final or private-library inputs enter the fit or selection.

Continuation requires at least **five additional FMA exact cases out of 250**,
with no BP exact or symmetric/asymmetric MIREX decline versus the fixed CPU320
control. Require the control's historical score guard as well. Report both domains'
paired repairs/new errors and all weighted metrics, even where not gated.
Predeclare 10,000 paired metadata-cluster bootstrap replicates, seed 20261112,
and duration strata; these are descriptive adaptive-development intervals for one
training seed, not fresh-test or seed-variation uncertainty. BP artist-group gaps
remain explicit. No automatic release, confirmation, seed expansion or distillation.

## Qualification and bounded execution

Before any task fit, qualify this exact architecture and source on the server with
three disposable synthetic-target batch-32 updates and a finite-only full selection
pass. Verify source/runtime/data provenance, initialization and resources without
retaining task predictions, scores or disposable weights. Proposed qualification
cap: ten minutes wall, 1,200 aggregate CPU-seconds, 4 GiB whole-tree RSS.

Use a reviewed pair of distinct physical CPU cores separate from the current
teacher and context workers; two intra-op threads, one inter-op, numeric-library
environment threads one, nice 19 and CUDA masked. Preserve both vLLM instances,
existing runs, shared environments and read-only caches. Do not start the fit unless
observed cost plus at least 25% contingency fits a separately sealed budget.
The maximum candidate-fit proposal is six hours wall, 40,000 aggregate CPU-seconds,
8 GiB whole-tree RSS and 512 MiB owned output. Actual caps/window and start identities
must be reviewed and recorded before launch; this paragraph is not a launch permit.

Resource failure is inconclusive for accuracy. Preserve partial evidence and stop
only owned processes, without retry, rescue, horizon extension or batch changes.
The running context comparison and MusicFM pilot must retain their original source,
configuration, gates and deadlines. Local Mac timings do not qualify server training,
Windows/x86 deployment or complete decoding/CQT/inference memory.

## Parallel alternatives considered

A 64-channel high-resolution head would raise parameter count to 219,720; it is not
an additional candidate. Input octave folding/circular sharing, width 2, SAM,
structured targets and frozen temporal heads already have negative studies.
A train-time multi-branch CNN that fuses into ordinary inference convolutions is a
possible later hypothesis: [Rep-CTFA, DCASE 2025](https://dcase.community/documents/challenge2025/technical_reports/DCASE2023_Chang_81_t1.pdf)
provides adjacent-domain evidence. Its attention, distillation and other recipe
changes do not isolate reparameterization or establish a key-estimation benefit.
No such second architecture is implemented or scheduled here.
