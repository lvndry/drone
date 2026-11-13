# Fixed native prediction parity on retained Mac cores

This protocol records the scope sealed before the single model pass. Execution
packets and binary/source identities are bound in
[technical provenance](retained-model376-provenance.json). The completed result
belongs in the [consolidated report](REPORT.md#retained-core-native-prediction-parity--november-13).

## Question

Does the corrected native frontend remove the historical single strict-logit
failure when its retained outputs pass through the unchanged native full and
chunked model adapters? This is implementation agreement, not musical accuracy.

## Fixed inputs and implementation

Use all 376 entries of `FUSED_376_MEMBERSHIP.json`, in their original order,
with the previously retained candidate core whose bytes equal the archived Mac
reference. Bind each original Mac manifest and retain its reference logits.
The panel contains 126 Beatport and 250 FMAK recordings; no private recordings,
new labels, exclusions, truncation or new references enter this pass.

The existing native cached library supplies `Full`, `Chunked`, and `compare`.
Its source remains identical to the historical comparison. Use the same full
ONNX model, prefix/head bundle, CPU runtime library and original session settings.
A minimal Rust entrypoint reads one core at a time, verifies its identity and
finite shape, computes one full and one chunked output, and reuses those vectors
for the direct comparison. Sessions are reused across cases. No decode, CQT,
Python inference, alternate backend, model training or GPU execution occurs.

An offline, frozen-lock, single-job build in a private target/cache has separate
limits: 600 wall seconds, 300 aggregate CPU seconds, 2 GiB sampled process-tree
RSS and 1 GiB owned disk. Building does not authorize model execution. The actual
binary and runtime loader identities are sealed before the separately reviewed
model launch. No shared environment or cache is modified.

## Fixed numerical decisions

For each case, require finite 24-element outputs and check three families:

1. Native full versus archived Mac logits.
2. Native chunked versus archived Mac logits.
3. Native chunked versus native full.

Every element must satisfy `abs(actual-reference) <= 1e-5 + 1e-5*abs(reference)`
and argmax must agree. The existing native binary32 tolerance constants are
unchanged. Strict numeric passes, failed element counts and argmax agreement
are reported separately. Retain all vectors and failures; a numerical failure
does not drop a case or stop the remaining fixed sequence. Overall numerical
qualification requires every case in every family to pass.

## Bounded run and interpretation

One pass, without automatic retry, uses 900 wall seconds, 300 aggregate CPU
seconds, 2 GiB sampled process-tree RSS and 16 MiB owned outputs. The memory
allowance accounts for the historical full-path peak near 1.34 GB; it is not a
claim that the full model uses negligible memory. Numerical thread settings are
one. An external guardian samples the process tree every 0.1 seconds, checks
terminal limits, and retires only owned processes with matching identities.

Per-case JSONL survives interruption. A complete scientific failure is distinct
from a runtime/resource failure with incomplete coverage. Neither permits a
threshold adjustment or retry within this protocol. Preserve the original
failed diagnostic and separately differing Linux references.

Resource counters describe this diagnostic, including retained-core reads,
verification, both model paths and reporting. Reused sessions and fixed order
make this unsuitable as an independent performance comparison. Passing does not
authorize production arithmetic changes, model replacement or accuracy claims.
