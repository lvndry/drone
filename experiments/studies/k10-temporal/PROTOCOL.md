# Proposed frozen-K10 temporal mechanism probe

Status: approved fixed mechanism specification; frozen implementation protocol before any real extraction or fitting. No result exists yet. No release weights are changed. CPU export/performance work is conditional on accuracy qualification. No retrospective changes to gates are permitted.

## Question and scope

Test whether frozen K10 features contain useful ordered local context discarded by the existing global mean and linear classifier. Use a two-block depthwise temporal-convolution residual head. Keylime's primary ablation motivates retaining frame-level evidence (https://arxiv.org/html/2611.02408); choosing a linear-time convolution here is an engineering hypothesis, not evidence that TCNs outperform Transformers.

Freeze the approved 2,206 fitting and 376 development identities, unchanged labels/quarantine, canonical joint seed-13 checkpoint, source hashes, feature manifest, and existing CQT contract. No new audio, private audio, final/SWD/calibration evaluation, encoder fine-tuning, teacher, architecture search, or release-weight change. Three fitting arms only: M (nonlinear mean control), O (ordered temporal head), S (shuffled temporal head). Canonical K10 is a frozen reference, not a fourth fitting arm. O is the sole predeclared candidate; M or S cannot replace it after results.

## Cache preflight and extraction

Read actual immutable manifest frame counts before allocating/saving bulk arrays. For each recording with T CQT frames, N=floor(T/8). Cache the full pre-Dropout2d/pre-adaptive-pool activation H as contiguous little-endian FP32[64,24,N]. Each temporal token has 1,536 features. Do not substitute 24 logits, truncate, sample, quantize, or temporally resample.

Exact hidden payload bytes = 6,144 * sum_i floor(T_i/8). Add separately the actual NPY headers, manifest, logits and temporary-write overhead; reserve disk for the full committed total before extraction. Atomic per-file writes need at most one additional largest-track temporary array if implemented serially; do not silently duplicate the whole cache.

Known development counts from the local v3 validation records:

| Domain | Recordings | Hidden tokens | Payload |
|---|---:|---:|---:|
| Beatport | 126 | 10,080 | 59.0625 MiB |
| FMAK | 250 | 43,471 | 254.712890625 MiB |
| Total | 376 | 53,551 | 313.775390625 MiB |

Domain-stratified extrapolation to fitting 1,230 BP + 976 FMAK predicts approximately 1.84 GiB total for all 2,582 recordings. This is not the actual fitting allocation: calculate that from the manifest. Single-encoder hidden-cache ceiling: 4 GiB. If actual payload exceeds it, stop for a new prospective resource protocol; never silently shorten tracks. Three encoders would be approximately 5.52 GiB by the same estimate.

Use existing FP16 CQT caches, their original FP32 conversion, and the full unshifted 192-bin core. Encoder BN/dropout remain in eval mode. No AMP, new quantization, or changed TF32/runtime policy. Run each recording at its true length under frozen canonical execution settings.

Store one uncompressed NPY per recording and FP32[24] canonical logits. Manifest records ID, role, domain, original CQT shape/hash, encoder/checkpoint/source hashes, hidden shape/hash, environment, and logits identity. Preserve channel-major/frequency-major/time-last contiguous layout for reconstruction.

Before fitting, reload every cached H and apply unchanged canonical pooling and classifier. Against direct full K10, require all 2,582 cases finite, shapes identical, zero argmax disagreements, and every logit within abs(delta)<=1e-5+1e-5*abs(reference). Report bit identity separately. Use the original pooling operator/layout because mathematically equivalent reductions can round differently. Failure is a cache/inference integrity failure, not an accuracy result.

## Fixed head architecture

Final logits = frozen canonical logits + learned residual logits. Original encoder and classifier stay frozen.

For each token x_t in R^1536:

1. Linear 1536->64, with bias.
2. LayerNorm over channels, epsilon 1e-5; GELU with exact/non-tanh approximation.
3. Two residual blocks, each: LayerNorm(64); depthwise temporal convolution with 64 groups and bias; GELU; pointwise linear 64->64 with bias; add block input.
4. Uniform arithmetic mean over valid tokens.
5. Linear 64->24, with bias.

No attention pooling, positional embedding, head dropout, or additional normalization.

| Arm | Temporal operation | Order |
|---|---|---|
| M | Kernel 1 in both depthwise blocks | Original |
| O | Kernel 5, dilations 1 and 4, symmetric zero padding 2 and 8 | Original |
| S | Identical architecture to O | Permuted before head |

M is permutation-invariant nonlinear mean aggregation. O and S are matched in architecture and parameter count. O/S add 109,400 parameters; full frozen-K10-plus-head total 218,528. M adds 108,888: the 512-parameter difference is the omitted off-centre taps. O>M alone does not establish order dependence; O>S is the matched-capacity comparison.

Head receptive field is 21 hidden tokens. K10 local receptive field is 36 CQT frames, stride 8; combined receptive field is 196 CQT frames, approximately 36 seconds on the frame grid, plus CQT acoustic support. This tests phrase-scale ordered local context, not arbitrary song-wide chronology.

## Initialization and independent shuffle RNG

Pilot seed 13. Initialize linear/convolution weights as independent zero-mean normals with std=1/sqrt(fan_in), biases zero. LayerNorm scale one/bias zero. Set final residual classifier weight and bias to zero so all arms initially reproduce canonical K10.

Construct O once, clone its full initial state into S. Copy shared weights into M; its kernel-1 coefficient is O's corresponding centre coefficient without rescaling. Archive initial state hashes.

Use a separate deterministic permutation RNG stream, not the model or recording-order RNG. Stable keys include study seed, recording ID, and epoch for fitting; use an explicitly specified stable hash/seed encoding, never Python's process-randomized hash. Archive generated permutation/schedule hashes.

S fitting gets a fresh permutation of all valid tokens per recording per epoch. S development uses exactly one fixed permutation per recording keyed by study seed/ID, unchanged across checkpoints. No permutation averaging/selection. Shuffle after encoder extraction, never CQT frames, and never the canonical reference logits. Permutation RNG consumption must not alter initialization, example order or optimizer randomness.

## Fitting and masking

All three heads train on full cached sequences with original hard labels. No crops and no pitch-shift augmentation. This intentionally differs from original encoder training but is identical across head arms. Hidden features are not assumed pitch-equivariant and must not be transposed by a pitch-axis rotation.

Settings: hard CE with equal recording weight; AdamW lr 3e-4, weight decay 1e-3; cosine over 60 epochs to minimum 3e-6; global gradient norm clip 1.0; logical batch size 32 recordings; final partial batch normalized by actual size; identical deterministic recording order across arms; maximum 60 epochs, patience 12 completed epochs.

Split logical batches into deterministic microbatches with at most 8,192 padded tokens. Accumulate per-recording-weighted gradients to preserve the logical-batch objective. A single recording longer than that gets its own microbatch, never truncation; inability to fit under the guard is resource-inconclusive.

Mask padded positions before each temporal convolution, including masking LayerNorm output, and after every residual block. Initial projected activations must be zero outside valid positions. Masking must prevent padded bias/activation values from feeding back into valid convolution positions. Mean-pool only valid tokens. Verify padded-batch output against individual true-length execution before fitting. Do not pad the original encoder input to batch peers.

## Fresh exact-primary checkpoint selection

This is a NEW prospective policy. It does not retroactively reselect checkpoints from old failed MIREX-selected studies or change their outcomes.

At each completed epoch, choose by highest 0.5*BP_exact+0.5*FMAK_exact. Break ties by lower equally weighted domain CE, then earlier epoch. MIREX never selects this study's checkpoint. Evaluate once per epoch and archive complete histories and deterministic selection decisions. Use the same policy in M/O/S. This is reused development data, not independent test evidence.

## Gates, controls and failure meanings

O alone earns confirmation if versus canonical seed-13 K10:

- FMAK exact gain >=2.0 percentage points (at least five net classifications on 250 tracks).
- BP exact does not decrease.
- Symmetric MIREX does not decrease in either domain.

Separately, the order-mechanism gate requires O minus M >=1.0 FMAK exact point and O minus S >=1.0 FMAK exact point (at least three net classifications each on 250 tracks). Report all BP contrasts without hiding regressions. Report asymmetric MIREX, paired exact promotions/demotions, confusion movement and paired uncertainty. Statistical intervals on reused development do not remove selection bias.

Interpretations:

- Candidate and mechanism gates pass: ordered local context merits confirmation.
- Candidate gate alone passes: useful head possible, chronology explanation unsupported.
- M approximately matches O and both improve: nonlinear aggregation is plausible.
- S worsens but O does not beat M/canonical: harmful shuffling alone is not evidence of useful temporal modeling.
- All fail: reject this fixed frozen-feature recipe, not all temporal modeling or other representations.

Keep the original Beatport-only mean comparison as a separate UNCHANGED hurdle, using its existing frozen calculation and thresholds, including FMAK +2 symmetric-MIREX points and its declared BP condition. Do not substitute this new exact-primary gate, alter old metrics, or reopen failed experiments. Candidate qualification earns confirmation/external review, never promotion.

## Matched-cohort confirmation

Only if O passes the candidate gate, freeze a separate confirmation stage using existing joint encoders 37 and 73 with matching head seeds and all three arms. Cohorts are (encoder13,head13), (encoder37,head37), (encoder73,head73), each compared with its own canonical encoder/classifier. Do not hold encoder13 fixed and present head-only seed variation as full-recipe stability.

Use unchanged cache/extraction/reconstruction, head, schedule and checkpoint policy. Confirm mean paired FMAK exact gain >=2 points, mean paired BP exact nonnegative, and mean symmetric-MIREX nonnegative in both domains. Report every cohort. Apply the same mean O-M/O-S mechanism thresholds separately. Original BP-only mean hurdle remains separate. Recompute actual storage/resource preflight for each encoder before extraction; no retrospective enlargement or rescue.

## Resources and conditional CPU stage

Sequential pilot stop budgets: extraction/reconstruction 15 minutes; each head fitting plus development scoring 10 minutes (30 total); conditional CPU stage 10 minutes; aggregate wall maximum 60 minutes. Retain 4% GPU allocator cap, >=8 GiB free before GPU phases and existing vLLM protections. OOM/timeout is inconclusive; unfinished runs cannot count as candidate results. No opportunistic extra epochs or seeds.

CPU export, parity and performance measurement occur ONLY AFTER O passes the accuracy candidate gate. If it fails, stop without this stage. This conditional stage cannot change or rescue accuracy gates.

O arithmetic estimate is approximately 107,136 MACs per hidden token plus 1,536 MACs for the track readout, excluding normalization/activations. This is approximately 8.57M MACs for an 80-token BP recording and 18.63M for the observed mean FMAK development length. Head FP32 weights approximately 427 KiB. These are estimates, not measured latency.

Before measurements, freeze public recording IDs, CPU/thread/ORT settings, repetitions, timing boundaries and percentile definition. One CPU thread; proposed deployment screen: median incremental cached-inference latency <=10% and p95 <=20% versus canonical K10. Measure absolute head latency, full cached-inference latency and RSS as well. Exported full prediction must pass the declared numerical/argmax parity gate versus selected PyTorch head. These are conditional deployment qualification checks, not reasons to modify a failed accuracy result. M5 end-to-end deployment is a separate stage; Linux timings cannot establish it.

The head is linear-time and could stream with a ten-token halo, but implementing/proving streaming is outside this mechanism probe.

## No inner tuning or validation rescue

No grouped inner hyperparameter search here: this is a fixed mechanism probe. Such a search may be justified for a later, independently declared architecture-development campaign, not as rescue for this result. No new optimizer settings, head variants, token selections, shuffled permutations, extra seeds, or reused-development fishing after a failed gate. Old failed MIREX-selected studies retain their original decisions.


## Implementation freeze and exact operational contracts

The version is `frozen-k10-temporal-v1`. Modules are `experiments.temporal_model`,
`temporal_cache`, `temporal_study`, and conditional `temporal_cpu`; no production or
shared training loop changes. The exact source/protocol SHA dictionary, Torch/NumPy
versions, Python/platform/CUDA-build identities, original encoder run/checkpoint/source map and feature manifest are bound
before extraction, and checked again before fitting. All original source673 files
must match their existing approved archived map. Hidden NPY files have little-endian
FP32 dtype and C-contiguous channel/frequency/time layout. The original CQT manifest
is copied byte-for-byte and cache row labels/roles/IDs/hashes must match it.

Preflight uses actual frame counts and exact NPY header lengths, serialized final
cache-manifest size, canonical logits NPY, original-manifest/protocol bytes, largest
single atomic-write temporary and 2 MiB for proof/identity records. It stops above
4 GiB hidden payload or insufficient free disk, never abbreviates a recording.
The frozen encoder's unchanged AdaptiveAvgPool2d and original head reconstruct the
reloaded features. All 2,582 rows must pass before manifest/proof become usable;
record bit identity separately. Archived 376 development predictions/metrics also
must reproduce exactly. Each arm's zero readout is checked across all 2,582 rows
before fitting. A separate nonzero-readout padding probe compares two fitting
sequences of differing lengths with bounded padded-token size; unit tests also
exercise nonzero learned biases and padded input gradients.

Stable permutation seed encoding: compact UTF8 JSON array
`[version, seed, recording_id, "fitting" or "development", epoch or null]`,
`ensure_ascii=False`, separators comma/colon; full SHA256 digest interpreted as
big-endian integer seeds NumPy default_rng/PCG64. Development uses null epoch.
Fitting recording order has its own fresh default_rng(seed) per arm. Hash little-
endian int64 permutations and every epoch recording order; archive all hashes.
No shuffling of canonical logits or CQT occurs.

Exact candidate thresholds are evaluated as integer net-correct counts: five FM
classifications versus canonical, three versus M/S. No-decline symmetric MIREX
uses integer tenths-of-credit counts on the fixed domain denominator. These are
the stated thresholds, not tuned numerical tolerances. Reporting keeps fractions.
The old BP-only mean hurdle retains its literal existing floating mean calculation
and thresholds. Matching-cohort confirmation requires 15 aggregate net FM exact
improvements over own canonical encoders and 9 over each M/S control across three
250-track cohorts; no encoder13-fixed head-seed surrogate.

Paired uncertainty is a fixed 2,000-resample within-domain paired track bootstrap,
NumPy linear 2.5/97.5 percentiles, with independent SHA256 seed of JSON
`[version,"paired-uncertainty",seed,domain]`. It is descriptive, ignores possible
artist dependence and selection bias, and never selects a checkpoint or gate.
Report promotions/demotions and O-M/O-S BP/FM contrasts explicitly.

The aggregate clock begins at extraction preflight and includes gaps between
stages. Root should run the phases sequentially within 60 minutes; no reset or
post-observation extension. OOM/timeout/integrity failure writes an inconclusive
failure artifact. No unfinished arm yields a complete candidate result.

Conditional CPU screen is pinned now: all 376 unchanged development recording IDs
in manifest order, one CPU thread, one warmup and five interleaved baseline/full
PyTorch repetitions per recording. Compare per-recording median full-vs-baseline
cached-CQT inference incremental fractions; report cohort median and linear p95.
Head-only latency is measured separately on cached H; no decode/CQT/file I/O in
timers. Export the full selected encoder+O head and require the same 376 numerical/
argmax parity checks in sequential ORT with intra/inter-op 1 and spinning off.
CPU RSS is whole-screen process peak including export/ORT, not isolated model or
head memory. Export failure is CPU-inconclusive, not an accuracy failure/rescue.
Linux screening does not establish M5 deployment performance.

## Root commands after frozen-source review

```sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
# Root wraps the whole sequential pilot in an aggregate 3600-second backstop.
timeout 950 "$PY" -m experiments.temporal_cache --data "$DATA" --output "$STUDY_ROOT/cache13" --reference-run "$JOINT13/run.json" --reference-source "$ORIGINAL673_SOURCE"
timeout 1850 "$PY" -m experiments.temporal_study --cache "$STUDY_ROOT/cache13" --output "$STUDY_ROOT/heads13" --bp-runs "$BP13/run.json" "$BP37/run.json" "$BP73/run.json"
# Only if heads13/result.json is complete and gates.candidate_pass is true:
timeout 650 "$PY" -m experiments.temporal_cpu --cache "$STUDY_ROOT/cache13" --study "$STUDY_ROOT/heads13" --data "$DATA" --reference-run "$JOINT13/run.json" --reference-source "$ORIGINAL673_SOURCE" --output "$STUDY_ROOT/cpu13"
```

`DATA=/home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/data`.
Each best.pt is adjacent to its archived run.json. `--seed 37/73 --pilot` on cache
extraction is conditional on the passing 13 result; head seed is inherited from its
encoder cache. A separate matched-cohort confirmation stage must retain this
source/protocol and re-evaluate storage/resource preflight. Summary command:
`python -m experiments.temporal_study summarize result13 result37 result73 --output summary.json`.
No implementation-agent SSH, real extraction/fitting, push or automatic promotion.

Export uses the repository's existing explicit time-mean adapter on the export
copy only, because dynamic ONNX export cannot specialize adaptive pooling to one
example length. Original AdaptiveAvgPool2d remains mandatory for cache proof and
PyTorch reference/timing. Export must match both CPU full PyTorch and the selected
cached-GPU O validation logits/argmax under the stated tolerance; otherwise the
CPU screen is inconclusive. No selected-head or accuracy gate is changed.
