# Original-file CPU measurement — prospective

This protocol asks whether the prepared native frontend's measured savings survive
decoding and session setup on complete recordings. It is a finite research
measurement, not a shipping API, accuracy study, architecture search, or default
promotion. Results and figures belong in the existing [composed report](REPORT.md).
No recording has been processed under this protocol yet.

## Fixed inputs and execution

[The five-case panel](END_TO_END_PANEL.json) fixes original audio, decoded PCM,
historical per-case Mac plans, Mac reference cores, and Linux comparison cores.
Cases were selected only for duration from the existing 376-case panel:

| Case | Decoded duration | Role in this measurement |
|---|---:|---|
| Beatport 1059355 | 120.000 s | Preview-sized reference |
| FMAK 083351 | 299.573 s | Approximately five minutes |
| FMAK 101615 | 444.604 s | Approximately seven minutes |
| FMAK 088812 | 612.214 s | Approximately ten minutes |
| FMAK 087373 | 1,682.260 s | Long-input stress case |

Genre and BPM are not inferred from the source or duration. This panel does not
establish accuracy on house, techno, hip-hop, or other genre/length groups.

Run one fresh worker process per case, in the fixed order. Construct one plan and
one chunked ONNX session. Process the original file once, then three more times in
the same process/session, decoding again each time. Retain all four results; do
not select the fastest. “Cold” means a fresh process/session, **not** a flushed
filesystem or CPU cache. Warm measurements reuse the session, not predictions,
PCM, or CQT features. No automatic rerun or extension follows a failure.

The worker calls `research_fused_product_core` explicitly. Ordinary `compute`
behavior is unchanged. The frozen chunked bundle/checkpoint, FP16 feature
roundtrip and FP64 hidden aggregation are reused through `drone-native-cached`.
The exact copied historical per-case plans are sufficient here; a portable
general-purpose kernel bundle is outside this change.

## Decoder and CPU policy

The historical decoder used FFmpeg's default decoder/filter thread policy. This
new measurement declares the following policy prospectively:

```text
ffmpeg -loglevel error -y -threads 1 -filter_threads 1 -i INPUT -ac 1 -ar 22050 OUTPUT.wav
```

The decoder thread option precedes the input. The sample format remains the
historical mono 22,050 Hz S16 WAV contract, converted to float32 by division by
32768. Every repetition must match the retained PCM length and SHA256 exactly.
A decoder/output difference stops the run; it does not relax admission or
regenerate a reference. The supplied FFmpeg executable is pinned. Its linked
libraries/build and the ORT/Soxr dependencies must also be recorded in the sealed
host packet; an executable hash alone does not identify all dynamic dependencies.

ORT uses CPU only, sequential execution, one intra/inter thread, and no spinning.
Rust CQT has no parallel worker pool. On Linux the supervisor temporarily sets
single-CPU affinity while spawning each worker, which the worker and decoder
inherit, then restores its own affinity. macOS has no such affinity guarantee:
thread settings are explicit, but the report must not claim one hard-bound CPU
or a universal maximum thread count. No service or other workload is stopped.

## Measurements and finite limits

Record setup, decode/read/convert, PCM identity verification, CQT, and ONNX wall
times. Stage CPU times include the worker and waited decoder children. Record
per-pass totals and whole-worker totals separately. The first-pass wall cost plus
setup is the instrumented cold cost, including PCM verification. The sum of
decode, CQT and ONNX stages (plus setup for the first pass) separately estimates
the compute path without diagnostic hashing. The supervisor's launch-to-exit time additionally
includes executable startup, admission hashes, JSON and diagnostic work. Do not
call prepared CQT/ORT timing audio-to-key latency.

The supervisor samples aggregate process-group RSS and cumulative CPU once a
second, including live FFmpeg children. Final worker CPU includes waited children.
One shared invocation budget begins before input admission and accumulates the
controller's CPU, already-reaped children and currently live worker/decoder CPU.
It is never reset between cases; monitoring subprocesses count toward this budget.
The execution alarm is installed before configuration reading and admission hashes;
an admission failure retains an incomplete invocation receipt. Metadata-only
validation is a separate, non-executing mode.
Worker-only OS high-water RSS is separate from sampled aggregate RSS; neither is
an exact simultaneous process-tree high-water measurement. Admission and
diagnostic work remains in whole-worker resource totals, with stage boundaries
explicitly documented. Hashing PCM does not allocate another complete PCM copy.

The entire five-case invocation is limited to 1,200 wall seconds and 600 aggregate
CPU seconds, including the controller. The same per-worker limits remain a
backstop, alongside 3 GiB sampled aggregate RSS and 256 MiB owned output. These are research
termination ceilings, not product performance promises. A five-case invocation
cannot silently expand the panel or repetitions. On a budget, signal, decoding,
identity, or worker failure, stop and retire owned processes. Cleanup captures
actual process start identities, refreshes descendant ownership before each signal,
and rechecks each identity before signaling an individual PID. It never signals
a whole possibly reused process group. A reused PID and descendants rooted only
in that unrelated process are not adopted. Retain
untracked group survivors as a cleanup failure; an empty captured identity set
alone is not evidence that the group disappeared. Retain
failure logs/receipts; do not process later cases. Successful passes delete their
temporary WAV only after PCM length and hash admission. Its retirement belongs to
the identity-verification stage, not decode timing. Failure WAVs remain only in the finite owned
output directory for diagnosis.

## Interpretation and portability

Retain every prediction, both reference comparisons, maximum logit error, reference
top-two margin, and Mac/Linux feature-hash comparisons. Numerical disagreements
are results, not reasons to omit a case. Existing allclose and key-agreement
criteria are reported unchanged; this measurement creates no new promotion rule.
`complete` means all expected work finished. Separately, `numerical_pass` requires
the existing allclose checks against both references on every pass, and
`unchanged_key` requires every existing argmax comparison to agree.
`benchmark_qualified` requires completion and both aggregates. A completed
numerical disagreement is retained, with positive benchmark qualification withheld.
A Mac feature match does not imply an x86 bitwise match. Cross-platform arithmetic
differences require prediction-level assessment, not automatic default promotion
or universal bitwise equality. No labels are scored here.

The report must name CPU, OS, compiler, target triple, Rust flags, FMA lowering,
runtime versions and thread policy. On generic x86, `f32::mul_add` may use a
software/libm path; inspect the actual binary's lowering before interpreting its
throughput. Do not silently substitute `target-cpu=native` and call that a generic
shipping build. The Threadripper server is a reference host, not evidence about
an ordinary PC. A commodity x86 host is still required for that claim; Windows
support is not implied by Linux/macOS results. Per-track tables and stage plots
are appropriate; five cases do not justify a population p95 or genre conclusions.

## Configuration and opt-in

`original_file.py --config CONFIG --output NEW_DIRECTORY` validates metadata and
hashes only. Add `--execute` only after review of a sealed host-specific packet.
The worker separately requires `--execute-research-candidate`. Neither import nor
default CLI operation constructs a model or decodes audio.

The JSON configuration requires:

- `format`: `drone-original-file-measurement-v1`;
- `candidate`: `fused-within-product-mac-magnitude-platform-log1p`;
- `warm_repetitions`: `3`;
- `worker`, `ffmpeg`, `ort`, `soxr`, `bundle_contract`: objects containing an
  absolute `path` and exact `sha256`; the contract must be named `contract.json`;
- `cases`: the five ordered `id` values, each with pinned `audio` and `manifest`
  objects. Manifest-relative CSR paths remain those in the historical plan;
- `linux_cpu`: one currently permitted CPU number on Linux, otherwise `null`;
- `build_context`: `profile` set to `release`, and nonempty `host_description`, `rustc_version`, `target_triple`,
  `rustflags`, and `fma_lowering` strings. Explicitly record unknown lowering at
  preparation time; resolve it before an x86 throughput claim.

No real paths, host-specific binaries, audio, credentials, or result artifacts
are bundled with this source-only preparation.
