# Matched K10 crop-length/audio-exposure pair (prospective)

**Preparation only. No execution or promotion is authorized by this protocol.**
This single study consolidates the completed disposable CPU resource qualification
(source `804b21c`) and the prospective task pair. Its frozen resource protocol is
[retained here](RESOURCE_PROTOCOL.md); the old archive and all control results remain.

The question is whether 65 versus 320 cached CQT frames changes K10 development
accuracy on the admitted expanded cohort. These are nominally 12.07 versus 59.44
seconds of CQT, not identical waveform support to MusicFM's 12-second mel frontend.
The change also affects repeated audio exposure (about 4.9 times), temporal pooling,
BN statistics and gradient noise. It is not a pure context or teacher-architecture
comparison, an architecture/optimizer search, or a deployment tier. Full-song
validation stays unchanged. The sparse five-song long-House development panel
cannot establish full-length house/deep-house/melodic-house acceptance.

## Fixed methods and selection

Run fresh CPU **320 control, then 65 candidate**, both seed 13 and initialized tensor
SHA `bb227d5d...`. Same admitted BP 1,229 + FMA 4,033 fitting IDs, 32-example keyed
sampler with nominal BP mass 1230/2206, pitch-bin reslice, per-draw seed, uniform legal
start and zero padding. Temporal starts depend on crop length; compare common
ID/seed/pitch traces separately from temporal traces. No global feature-spec or
`CROP_FRAMES` mutation. Retain actual-frame/padding exposure and sample coverage.
The 320 BP augmentation/start trace must reproduce the historical source replay.

Both arms complete 8,280 AdamW updates, lr 1e-3 / weight decay 1e-4 and the original
Torch defaults, cosine 120 events, no independent early stop. Every 69 updates use
the original true-length evaluator with batch 4 on the same BP126/FMA250 panel.
Finite wrappers validate outputs/coverage without changing logits. The selector
is equal-domain exact first, equal-domain symmetric MIREX second, earliest tie.
Report both MIREX conventions. No model-score-guided role, seed, horizon or gate
changes. Calibration, reserved confirmation, historical final and private audio
stay unopened; old development exposure remains disclosed.

Best/final checkpoints use explicitly **EXPERIMENTAL_K10_CONTEXT_STATE_DICT_V1**
payloads. The canonical saver hardcodes crop320 and is not used. After the fixed
horizon, reload the selected on-disk artifact into a fresh model with strict state
loading and repeat all 376 predictions. Require exact prediction/score identity;
write a completed arm record only after this proof. The final checkpoint is saved
before restoring the selected one. A second-arm failure retains the first completed
arm and checkpoints as partial evidence; no restart or incomplete comparison.

## Predeclared interpretation and gates

The technical pair fails on mismatched source/runtime/data/init/common draw traces,
incomplete horizon, missing reload proof, wrong panel order or crop320 replay.
Fresh CPU320 must be measured: the historical CUDA result is not its replacement.
Both backend and Torch-thread differences are disclosed. As a conservative failed-
control guard, flag any CPU320 decline against the historical expanded CUDA control
in either domain's exact or symmetric/asymmetric MIREX; a failed guard blocks
continuation, while the within-CPU diagnostic effect and uncertainty are still
reported. This is a score guard, not a claim of CUDA/CPU trajectory reproduction.

The canonical candidate is always **65**, never whichever arm wins afterward.
Its context/exposure continuation gate is at least **five additional FMA exact
cases out of 250**, with BP exact and both BP MIREX conventions nondeclining versus
fresh CPU320. FMA weighted metrics are reported, not an additional candidate gate.
Continuation requires the technical checks, candidate gate and historical-control
guard. Nothing automatically promotes weights, opens confirmation or adds seeds.

Report selected paired repair/new-error counts and duration strata (<2, 2–<5,
5–10, >10 minutes), with actual canonical samples and group counts. Predeclare
10,000 paired cluster-bootstrap replicates, seed 20261112, using the frozen derived
uncertainty-group CSV. CIs are descriptive for adaptive development selection and
one training seed. BP unknown artist identities and proxy grouping remain gaps;
these are not independent final intervals or estimates of training-seed variance.

## Shared resource and teacher protection

The measured disposable qualification took 34.83 seconds, 56.44 aggregate CPU-
seconds and 2.46 GB whole-tree peak RSS. Its pipeline/validation extrapolation is
about 3.85 hours before verification/checkpoint/guard/I/O uncertainty; it is not a
completion promise. Root proposed a single combined **six-hour wall / 40,000
aggregate CPU-second / eight-GiB whole-tree RSS / 512-MiB owned-output cap**. Both
arms share this budget, including guardian/query/reaped-child cost. CPU14–15,
Torch2/interop1/env1, nice19, CUDA masked. Preserve global RAM40GiB/disk32GiB and
GPU-free8GiB reserves; both vLLM services remain unrestricted over other work.

Four vLLM PID/start identities are continuously protected. Teacher1294/1307 starts
are protected while active. If either exits, allow at most ten seconds for normal
receipt publication (pending is never accepted as completion). Require the pinned
teacher configuration, controller exit0/complete state, held-success/result hashes,
all8,280 updates/120 events and source/data/init provenance, with no reserved input
use. Accuracy/continuation outcome is irrelevant: a negative completed teacher is
legitimate. Missing, failed, partial or unbound terminal evidence stops only the
owned CPU pair. Never signal, restart, alter or extend the teacher or services.
