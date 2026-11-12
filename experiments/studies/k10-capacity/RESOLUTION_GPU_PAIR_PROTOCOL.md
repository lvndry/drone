# Prospective matched GPU resolution pair

Status: SOURCE PREPARATION ONLY. This amendment supersedes the CPU320-control and
CPU-placement paragraphs of RESOLUTION_PROTOCOL for this prospective comparison.
All earlier results and the live CPU context pair remain unchanged. GPU1 resource
qualification must pass before a source/config-bound execution can be authorized.
No task-fit resource budget or execution window is selected here.

The v2 resource qualification failed before its first optimizer commit because
global deterministic algorithms reject AdaptiveAvgPool2d CUDA backward. The
prospective correction uses the historical CUDA recipe for BOTH fresh arms:
`torch.use_deterministic_algorithms(False)`, not warn-only or a changed operator.
All other numerical/data/memory/gate settings remain fixed. Actual backend flags
are recorded in both checkpoint/result provenance. Seeded CUDA training is not
guaranteed bitwise reproducible. Hardware feasibility remains unproven until the
separate corrected resource qualification completes; no v2 timing is extrapolated.

Use two fresh worker processes, baseline first then SemitoneGridCNN, both on the
same GPU1 backend. Both use seed 13, FP32, global deterministic algorithms OFF,
cuDNN deterministic ON, TF32 off,
benchmark false, two Torch threads, one interop thread, numeric environment one,
nice 19, and the reviewed isolated physical CPU placement. Preserve the GPU0
8 GiB free floor, GPU1 2 GiB allocator/2.75 GiB owned-NVML ceiling and 6.5 GiB early
abort/6 GiB hard floor. Preserve all four vLLM start identities and active teacher
and CPU context identities; only independently verified legitimate completion may
retire an active-job guard. No process belonging to another task may be signaled.
The probe's immutable pure sensor, violation, environment, process-tree and owned
termination helpers will be source-bound, not modified. Actual wall/CPU/RSS/output
caps require the completed corrected resource qualification and root review.

The common data contract is plan c3e59ffb..., labels b8942b..., features 9f94f6a5...:
1,229 BP + 4,033 FMA fitting IDs and BP126/FMA250 selection, no other inputs.
All existing feature files and pinned labels must be verified before construction.
Both arms use the original keyed draws, pitch reslicing/negative-shift label rule,
320-frame crops, batch 32 hard CE, AdamW .001/.0001, cosine schedule, 8,280 updates
and 120 events every 69 updates. The selector remains equal-domain exact, symmetric
MIREX tie-break, earliest tie. No early stop, tuning, seed selection or rescue.

Baseline initialization bb227d5d... and candidate 4fc58182... are distinct and
pinned. Both must match first-two-block prefix 54d0c895.... Full initialization and
prefix hashes use ordered state-dict names and CPU tensor bytes. Different later
shapes consume different RNG draws: matching later initialization/dropout masks
are not claimed. The comparison changes frequency resolution AND channel/head
capacity allocation; it does not identify pure frequency-resolution causality.

Every validation consumes full true-length features with original batch 4. Save
experimental selected and final state dictionaries with model/provenance identity;
strictly reload the selected file into the same factory and reproduce all 376
selected predictions before an arm can complete. Both full draw/pitch/temporal
streams, exposure counts, selection ordering and shared provenance must match.
A fresh GPU baseline is mandatory; CPU320 is not substituted as a backend control.

Continuation requires candidate at least five more FMA exact cases out of 250,
with BP exact and both symmetric/asymmetric MIREX nondecline against fresh GPU
baseline. The baseline must also meet the historical expanded GPU control in both
domains' exact and both MIREX conventions. A failed control guard is reported as
failed comparability/continuation, never tuned away. No automatic promotion,
confirmation scoring, seed expansion or release change follows any outcome.

Retain paired repairs/new errors, both weighted metrics, 10,000 paired metadata-
cluster bootstrap replicates (seed 20261112) and fixed duration strata. These are
one-seed, adaptive-development diagnostics; BP identity gaps and five inner
5–10 minute House examples in three groups prevent primary long-house acceptance.
No reserved calibration/confirmation, historical final or private DJ input is used.

## Qualified execution amendment

The corrected v3 resource qualification (source 57e0fac2b63a242e2d96d86f6fcc0be4d8831bc0,
config 132978eae5395599d7e9ed0da0a12ef62ad7fd5f13646bb06d9147076577728c)
completed three disposable optimizer commits and full 376-song finite-only selection
plus four envelope inputs for each model. It establishes bounded resource
feasibility, not accuracy. Earlier failed qualifications remain failed.

The prospective whole pair has fixed caps of 1,800 wall seconds, 3,600 aggregate
CPU seconds, 8 GiB whole-tree RSS and 512 MiB owned output. Warm update and selection
costs project about 12 minutes for both arms before real CQT sampling, transfers,
checkpointing and monitoring; the cap allows data/I/O overhead and contingency,
not a completion promise. GPU thresholds and the 0.15-second parent sampling target
remain the qualified policy. Worker heartbeat age is at most 0.6 seconds; its large
config SHA is computed once. All qualified shared source bytes, runtime closure,
protected identities, model initialization, historical control and uncertainty
groups are checked before model construction. No launch follows this document:
a separate root-reviewed source/config/UTC-bound permit is required.

The guardian uses a new process for each model and verifies owned GPU allocation
release between them. It conservatively stops if any protected active-job start
disappears; it never signals those jobs or automatically retires their guards.
If the second arm fails, retain the completed baseline and partial candidate without
retry or continuation. Compute metadata-cluster intervals, paired movements and
actual-duration strata only after both GPU workers have exited. These diagnostic
analyses consume no additional model evaluations or reserved examples.
