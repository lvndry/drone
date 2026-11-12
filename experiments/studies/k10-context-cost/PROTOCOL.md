# CPU context-length resource qualification (prospective)

No task fit or execution is authorized by this protocol. This is one disposable
resource qualification for a possible later matched K10 crop-length control.
The active MusicFM fit, its source, selection, observer and lease remain unchanged.

Use the existing admitted 5,638-row CQT manifest `9f94f6a5...`, plan `c3e59ffb...`
and numeric export `b8942b59...`. Verify every existing cache file's exact content
hash and stable stat before model work. No audio, CQT recomputation, calibration,
confirmation or private inputs are accessed. Original labels only validate identity.

Run serial crop sizes **320 then 65**, with separate fresh KeyCNN initialization,
CPU seed 13 and identical initial tensor digest `bb227d5d...`. Each receives the
actual first three batch-32 draws from the unchanged expanded fitting sampler
(1,229 BP + 4,033 FMA). Replay the original per-draw pitch-bin reslice and uniform
legal start rule with that arm's explicit crop size, preserving zero padding.
The global feature spec and `CROP_FRAMES=320` never change. ID/seed/pitch witnesses
must agree between arms; temporal starts differ by crop size. Synthetic targets
are global draw ordinal modulo 24, never the recording key or its transposition.

Commit exactly three FP32 AdamW updates per arm (lr 1e-3, weight decay 1e-4, unchanged
Torch optimizer defaults). Check finite loss/gradients/parameters. These are the
first three updates before the original 69-update scheduler event: no scheduler
step or task checkpoint selection occurs. Discard weights and optimizer states.

After each shape, run one full 376-song pass at actual lengths through the original
`evaluate_features(..., batch=4)` grouping. A composition wrapper checks finite
24-class outputs and counts each input/frame; it changes no logits. The evaluator's
transient argmax return is discarded. Retain no predictions, labels, task scores or
weights. Retain context IDs/lengths, draw/padding witnesses, update and phase timings,
finite flags, runtime/source identities and whole-tree resource receipts. Per-arm
`ru_maxrss` is an explicitly cumulative Linux lifetime upper bound; the second
arm's value is not an independent peak. First-call/shape warmup and cache effects
remain visible; this is resource forecasting, not a product-speed comparison.

One fresh root approval and finite UTC window are required. Draft caps: 900 wall
seconds, 1,200 aggregate CPU-seconds, 8 GiB whole-tree RSS and 1 GiB owned output.
CUDA is masked; CPU affinity is exactly 14–15 (distinct physical cores), nice 19,
Torch intra-op 2 / inter-op 1, numeric-library environment threads 1. The guardian
uses the already tested thread-aware process-tree/live+reaped CPU and start-bound
cleanup helpers. It protects all four vLLM start identities AND active teacher
1294/1307 starts, global MemAvailable >=40 GiB, disk >=32 GiB, tmpfs >=1 GiB and GPU
free >=8 GiB. A bounded guardian heartbeat remains valid for at most ten seconds;
its GPU query has a five-second timeout. Stop only owned cost processes on failure.
No retry, rescue, serving-affinity change, dataset drop or numerical source change.

CPU14–15 are disjoint from teacher8–9 but vLLM remains unrestricted over0–31;
placement is not an exclusive-serving guarantee. This qualification cannot approve
any full paired fit. The earlier eight-hour/60,000-CPU-second pair envelope was a
rough inference-derived estimate, not permission; a later budget needs measured
cost and separate review. The CPU pair would measure context plus audio exposure,
pooling and BN changes, not isolate the teacher architecture or match its frontend.
