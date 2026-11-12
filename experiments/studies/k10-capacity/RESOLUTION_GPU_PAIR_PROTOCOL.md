# Prospective matched GPU resolution pair

Status: SOURCE PREPARATION ONLY. This amendment supersedes the CPU320-control and
CPU-placement paragraphs of RESOLUTION_PROTOCOL for this prospective comparison.
All earlier results and the live CPU context pair remain unchanged. GPU1 resource
qualification must pass before a source/config-bound execution can be authorized.
No task-fit resource budget or execution window is selected here.

Use two fresh worker processes, baseline first then SemitoneGridCNN, both on the
same GPU1 backend. Both use seed 13, FP32, deterministic algorithms/cuDNN, TF32 off,
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
