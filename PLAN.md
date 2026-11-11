# Drone: a measured path to a strong CPU key classifier for Euphoria

Date: 2026-11-11. **Local planning file: keep untracked; do not stage or commit.**

## Outcome and decisions

**The product is one compact, CPU-friendly musical-key detector for Euphoria.**
Training happens on `lysk-admin@lysk-server`; the desktop app performs local inference.
The small model is the release target. Medium/large profiles are existing research
controls, not promised products, release tiers, or scheduled work.

First establish the corrected small model's real-data accuracy, compare with the
69.4% symmetric MIREX libkeyfinder reference under the same held-out protocol, and
check FMAK generalization. If it underperforms, diagnose data, representation,
architecture and training before assuming more parameters are the answer.

A larger candidate enters the program only after a documented small-model failure
on the predeclared FMAK protocol justifies that experiment. It earns further work
only through a measured accuracy/generalization gain at a stated latency, memory
and compute cost. Distillation and a GPU release are conditional experiments too.
There is no three-model release commitment.

The research objective is the **accuracy/latency/memory frontier for the CPU product**.
A state-of-the-art claim must identify its datasets, data access, scoring convention
and hardware budget. Recent methods and width-scaled profiles do not establish it.

| Candidate | Current status | Decision gate |
|---|---|---|
| small: 331,704 parameters | Product target; corrected real-data baseline completed, external results pending | Establish accuracy and FMAK generalization, then meet CPU release budgets |
| medium: 1,318,744 parameters | Available research profile; unscheduled | Investigate only if small underperforms on FMAK and a larger-model hypothesis is justified |
| large: 5,258,904 parameters / alternative pretrained encoder | Available research direction; unscheduled | Must demonstrate a useful gain before teacher/distillation or optional GPU-release work |

Keep the canonical key output order and a versioned API. If a future candidate needs
different preprocessing, declare it in its artifact rather than silently sharing
an incompatible contract. Larger models never become a normal-startup dependency.
Other song attributes are deferred in [Drone issue #1](https://github.com/lvndry/drone/issues/1)
and do not expand this release or its experiment budget.

Proposed acceptance targets for the small tier, to validate on a named modest x86
laptop and an Apple Silicon laptop before treating them as promises:

- Network forward p95 at most 10 ms, batch 1, one CPU inference thread.
- Initial FP32 weight budget at most 2 MiB; report runtime/library size separately.
- Measure complete decode + features + prediction wall time, CPU-seconds and peak
  resident memory on 2-minute previews and representative full-length files. Aim
  initially for p95 below 1 second for a 2-minute preview on the reference machine;
  revise the sampling strategy if extraction dominates. Do not hide that cost by
  publishing only cached feature or tensor timings.
- For an unchanged cached library, reopening Euphoria starts zero audio decodes
  and zero model forwards. No analysis work and no spinning inference workers
  while idle. Measure CPU relative to the same UI without analysis, rather than
  interpreting a one-core spike as whole-machine saturation.
- Default neural analysis concurrency starts at one job and one intra-op thread;
  increase only after measuring UI responsiveness and total CPU on import.

## Execution status and run register

Last checked: **2026-11-11, approximately 15:44 UTC**. This is a timestamped
snapshot. The run's `run.json` and log are authoritative; update this register at
launch, completion, failure and evaluation. Do not leave “not started” prose in a
plan after a run begins, or describe a completed run as still in flight.

- **Completed:** bug-fix review and rework, committed and pushed as `01bbc88`.
  78 tests pass. The fixes cover semitone indexing, time interpolation, key encoding,
  label/audio augmentation, scoring, inference caching, artifact compatibility,
  data fetching and separate evaluation. uv, versioned features/checkpoints,
  reusable inference, ONNX export and CPU benchmarking are available.
- **Completed:** real-data corrected small baseline, 120 epochs. The latest saved
  record has `device: cuda`, smoothing 0.1, seed 13, batch 64 and LR 0.001.
  Training-loop time: 21.71 seconds. Best validation MIREX: 42.9% at epoch 35.
  Checkpoint: `/home/lysk-admin/key-cnn/data/runs/20261111-154014/drone.pt`;
  record: the adjacent `run.json`; log: `/home/lysk-admin/drone/run.log`.
  The earlier user status described a CPU/16-thread run. Preserve that provenance
  discrepancy: the verified completed artifact records CUDA, not CPU. Do not use
  its timing as a CPU measurement. Its Git revision is `unknown`; source hashes
  are recorded and must be matched to reviewed source before a release claim.
- **Completed:** separate CPU timing sample, 5 epochs with CPU threads=16, smoothing
  0.1 and the same small-model settings. It used 1,237 training / 126 validation
  tracks and kept 604 test tracks separate. Training-loop time: 3.286 seconds;
  process wall time: 5.43 seconds. Record:
  `/home/lysk-admin/key-cnn/data/runs/20261111-154344/run.json`.
  The five-epoch cosine schedule differs from the 120-epoch reference: use this
  run for throughput estimation, not for accuracy comparisons.
- **Pending:** external held-out accuracy and CPU end-to-end measurements, plus
  FMAK evaluation. The 42.9% figure is validation performance and cannot be compared
  directly with the 69.4% held-out reference. Diagnose before broadening the search.
- **Available:** baseline training/test data and corrected caches on the server.
  Record their manifests and audit broader-corpus availability before expansion.

No medium/large training, teacher program, GPU product, or multi-attribute release
is scheduled by this document. Additional techniques below form an evidence-led
backlog; they are not prerequisites for shipping a successful small model.

For each new run, record ID/path, source/lock hashes, device and explicit thread
count, data/split hashes, command/config, status, elapsed time, validation result,
checkpoint hash, and next decision. Current run metadata omits thread count; the
CPU sample's 16-thread setting is established by its launch command. Capture it
in future run records. Keep a stable reference to each log; never overwrite the
baseline record when launching an ablation.

## Training host: verified resources and operating policy

Read-only SSH inspection on 2026-11-11 found:

| Resource | Observed |
|---|---|
| Login | `ssh lysk-admin@lysk-server` |
| CPU | AMD Ryzen Threadripper PRO 9955WX, 16 cores / 32 logical CPUs |
| RAM | 188 GiB total; about 138 GiB available at inspection |
| GPU 0 | RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB total VRAM |
| GPU 1 | RTX 5090, 32,607 MiB total VRAM |
| Driver / compute capability | 580.178.04 / 12.0 on both cards |
| Tooling | Python 3.12.3, uv 0.11.1 |
| Root filesystem | 914 GB total; about 134 GB available at inspection |

The user can free both GPUs for training. Plan dedicated training windows. At the
snapshot GPU 0 was fully utilized and both had substantial allocations; that is a
transient observation, not a permanent resource limit. Identify the owning services
and stop/restart them deliberately when beginning the agreed training window.
No unrelated workloads have been stopped.

If larger-model experiments pass the evidence gate, use GPU 0 for teacher/fine-tuning
and GPU 1 for independent student experiments. No such GPU allocation is required
for the current CPU product.
Prefer independent jobs to distributed training across these unequal cards. Small
models may be launch- or input-bound: benchmark CPU against GPU rather than assuming
GPU speed. Schedule no overlapping jobs on the same device without explicit memory
budgets. Preserve host capacity for I/O and management.

1. Preserve the existing server checkout, data and baseline records. Verify source
   and lock hashes against the reviewed revision before starting each new run.
2. Configure explicit PyTorch CPU/CUDA indexes/extras before locking the server
   training environment. Choose a driver-compatible build supporting Blackwell
   (for example a tested CUDA 12.8+ build); run a real CUDA forward/backward smoke
   test and verify native SM 12.0 support. Do not infer compatibility from an
   installed CUDA toolkit alone. Record the resulting lock and package versions.
3. Reserve at least 50 GiB free disk for the initial corpus, caches and retained
   runs, then estimate requirements before adding larger corpora. Root has limited
   headroom; inventory other usable storage before bulk pretraining downloads.
4. Use the measured timings in the budget below; repeat a short throughput pilot
   whenever data, features, architecture, precision, batch size or device changes.
   Use resumable jobs with saved optimizer/scheduler/RNG state, logs, and resource
   metrics. Add resume support before long jobs; current baseline checkpoints are
   inference artifacts, not full resumable training state.
5. Run jobs under an explicit process/session manager. Monitor loss, utilization,
   VRAM, disk, and nonfinite values. Save a recoverable checkpoint before stopping.

See the [uv PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/)
for backend/index configuration.

## Experiment budget: measured baseline, conditional expansion

These estimates use the actual host and corrected cached features, not a generic
“few hours” estimate. They are planning bounds, not guaranteed runtimes.

| Work | Measured evidence / estimate | Initial allocation |
|---|---|---|
| Small CNN, CPU, 16 threads, 5 epochs | Measured: 3.286 s loop, 5.43 s process wall | Completed timing sample |
| Small CNN, CPU, 120 epochs on current data | Extrapolated loop about 79 s; allow startup/reporting and variability | Reserve 2 minutes per run; roughly 1–2 minutes expected |
| Small CNN, CUDA, 120 epochs on current data | Measured loop 21.71 s; whole process not timed | Reserve 1 minute per run when the GPU is available |
| 3-seed small-model confirmation on current data | About 6 CPU-minutes or 3 GPU-minutes reserved | Only for a selected recipe; same fixed split |
| Feature extraction / downloads | Current caches already exist; a clean preparation time has not been measured | Pilot 100 representative files, extrapolate by duration, then allocate separately; no re-extraction per training run |
| Small CNN on a corpus around 5,489 tracks | At the same frames/batch, linear track-count scaling suggests roughly 5–6 CPU-minutes for 120 epochs | Provisional 10-minute reservation; validate throughput after the actual split/data is fixed |
| Medium/large models and pretrained teachers | Unmeasured; current small-model timings do not establish their costs | Unscheduled; time a pilot before assigning a budget |

The last row is intentionally not a promise of cheap larger-model training. A wider
CNN's convolution cost grows roughly quadratically with channel width, while music
foundation encoders have different inputs and computational costs. Estimate from
a measured representative pilot, including preprocessing and validation.

### First tranche: at most 3 hours of wall-clock work

1. **Diagnostics, up to 30 minutes.** Inspect the completed run and data/feature
   sanity checks; overfit a real-data subset, examine key/mode confusion and
   augmentation behavior. A low validation score triggers diagnosis, not automatic
   model scaling. Use current caches and preserve the reference checkpoint.
2. **Focused screening, up to 1 hour.** At most eight small-model configurations,
   one seed each, on the fixed split. About 16 CPU-minutes of training at the
   measured rate; the rest covers inspection, evaluation and failed runs. Choose
   changes from observed failures, not an exhaustive grid of every paper's trick.
3. **Confirmation, up to 1 hour.** At most three recipes including the control,
   with three matched seeds: nine runs, about 18 CPU-minutes reserved. Report means
   and spread; do not promote a lucky seed. Stop early if no recipe improves the
   predeclared development criterion.
4. **Record and decide, up to 30 minutes.** Summarize runtime and development results,
   select one small-model candidate, and decide the next data/evaluation step.
   Keep test evaluation separate from recipe selection.

This tranche caps training reservations at roughly 34 CPU-minutes plus the already
completed reference/sample runs; it does not reserve a GPU window or launch large
teachers. Faster measured iteration is capacity to test better hypotheses, not a
reason to expand the product scope.

### Further tranches require a measured reason

- Time FMAK acquisition/preparation separately after checking availability, storage
  and audio duration. Run a small pilot before accepting an all-corpus estimate.
- After a documented FMAK failure, allow a single representation or capacity pilot
  with a **30-minute job timeout**. Use its throughput/memory result to propose a
  bounded next batch; a timeout is a cap, not an assertion that it will finish.
- No open-ended optimizer, pretraining, teacher or architecture sweep. Define the
  hypothesis, maximum configurations, seeds, wall/GPU-hour cap and promotion rule
  before launching it. Stop after the cap or when the evidence rejects the idea.
- Reforecast after each tranche using actual elapsed times. Keep preparation,
  training, validation, teacher feature extraction and human diagnosis separate.

## Research conclusions and experiments to borrow

### Classifier training in general

| Direction | Evidence and implication | Drone experiment |
|---|---|---|
| Representation/pretraining | DINOv3 illustrates the importance of data preparation, scalable self-supervision and transferable representations in vision. This is cross-domain evidence, not a music benchmark. | Compare frozen pretrained music-encoder probes and fine-tuning against training from scratch before spending on elaborate optimizers. [DINOv3, 2025](https://arxiv.org/abs/2508.10104) |
| Optimizers and schedules | A 2026 paired Muon/AdamW study includes small vision transformers and finds rankings depend on time/data/tuning budgets. Its findings do not establish a winner for this CNN. | Tune AdamW properly first; compare SGD+momentum, Schedule-Free AdamW, and a matrix-aware candidate only with equal search budgets and real wall-clock measurements. [Paired benchmark, Oct 2026](https://arxiv.org/abs/2610.21555) |
| Schedule-free training | Schedule-Free has distinct train/eval parameter states and special BatchNorm handling. | Add the official implementation as an ablation, with validation and checkpointing in its correct eval state and recomputed BN statistics. Do not just delete cosine decay. [Official implementation](https://github.com/facebookresearch/schedule_free) |
| Averaging | A systematic EMA study reports generalization, robustness and calibration benefits on its evaluated tasks. | Compare raw weights to EMA/SWA, with training-only BN recalibration where necessary. Treat decay as a tuned parameter. [EMA study, TMLR 2024](https://openreview.net/forum?id=2M9CUnYnBA) |
| Loss, augmentation and uncertainty | Recent work continues to study label smoothing, mixing and calibration; the best combination depends on the representation and target task. | Test plain CE and small smoothing separately. Fit temperature on a separate calibration partition, and report NLL/Brier/ECE as well as accuracy. Mixing tracks in different keys may violate the one-key target assumption. [Semantic mixing/calibration, ICML 2025](https://proceedings.mlr.press/v267/luo25k.html) |
| Efficient architectures and distillation | LFM2 couples measured edge hardware costs with a hybrid convolution/attention backbone and a specialized distillation pipeline. These are language-model results. | Borrow hardware-in-the-loop selection and teacher/student methodology. Test small gated temporal-convolution/attention alternatives only if they improve the measured tradeoff. With 24 classes, keep the full teacher distribution before testing top-k approximations. [LFM2 technical report, 2025](https://arxiv.org/abs/2511.23404) |

Fundamental controls apply across these methods: audit labels and duplicates; fix
the split before tuning; inspect per-class failure modes; use equal training/search
budgets; and replicate finalists across seeds. Modern recipes cannot repair label
corruption or test-set selection bias.

### Musical-key and efficient audio work

- **STONE / S-KEY:** investigate octave-aware and transposition-equivariant
  representations, structured key outputs, and unlabeled-data pretraining.
  S-KEY explicitly addresses relative major/minor ambiguity. Its large-corpus
  results are evidence for the method, not a training recipe reproducible from
  1,486 Beatport previews alone. Sources: [STONE, 2024](https://arxiv.org/abs/2407.07408),
  [S-KEY, ICASSP 2025](https://arxiv.org/abs/2501.12907).
- **Myna:** evaluate pretrained music representations and temporal/token masking
  that preserves useful pitch information. Distinguish contrastive pitch
  invariance (potentially harmful for key) from supervised transposition with a
  correctly transformed target. [Myna, 2025](https://arxiv.org/abs/2502.12511).
- **DSP-guided reranking:** a November 2026 preprint reports a 21.5k-parameter
  reranker and cross-corpus checks. This is especially relevant because Euphoria
  already extracts DSP key evidence. Reproduce under our own splits and include
  its DSP/runtime cost; its reported FMAK split is not an all-dataset result.
  [Deployment-constrained key estimation, 2026](https://arxiv.org/abs/2611.02408).
- **Current evaluation practice:** MIREX 2026 reports both symmetric and asymmetric
  weighted scores and notes data-exposure caveats for some entries. Compare
  primary baselines on the same files and protocol, not just a published table.
  [Organizer results](https://music-ir.org/mirex/wiki/2026%3AAudio_Key_Detection_Results).

### System One, decision models, and Liquid models

System One is relevant as a product pattern: return one structured key distribution
in a forward pass, calibrate it, and let ordinary code decide whether to accept,
flag, or escalate. A conventional supervised classifier already supplies much of
this shape. Do not add language generation or a hosted service to ordinary key
inference. TypeSafe describes Jev as a natural-language/image decision model and
its RLCD approach at a conceptual level; that does not provide a reproducible
raw-music training algorithm. Sources: [System One docs](https://docs.typesafe.ai/concepts/system-one),
[training primer](https://docs.typesafe.ai/introduction/machine-learning-primer).

Liquid's LFM2/LFM3 work supports hardware-aware architecture selection, efficient
sequence blocks, and separating training complexity from deployment cost.
LFM3 mixes short convolutions, local attention and global attention; test these
ideas in a small audio encoder rather than assuming a billion-parameter text model
is a better 24-key classifier. [LFM3 announcement, Oct 2026](https://www.liquid.ai/blog/lfm3).

There is an actual audio decision-model candidate: **d1-omni-600M**. Its official
card describes 587M parameters, speech clips capped at 30 seconds, and training on
English speaker/assistant requests. It establishes no musical-key accuracy. Give
it a small, fixed development-only comparison budget, checking transposition
sensitivity and probability calibration; promote it only if it beats music-specific
controls. Do not use its unverified answers as pseudo-ground-truth.
[Official model card](https://huggingface.co/LiquidAI/d1-omni-600M).

LFM3-Audio-2B can be another exploratory GPU/teacher comparison, but speech/audio
understanding benchmarks do not establish tonal accuracy. Prefer a trained
classification head over an encoder, where accessible, to repeated prompting and
free-text key parsing. [Official LFM3-Audio card](https://huggingface.co/LiquidAI/LFM3-Audio-2B).

## Experiment program

### 1. Evaluate and diagnose the completed baseline

- Run locked lint, tests, packaging and ONNX parity checks on both local and server
  environments. Add a supported-Python/platform CI matrix before release.
- Download the verified Beatport and GiantSteps inputs; record actual accepted and
  rejected labels and reasons. Check hashes, IDs and exact content overlap.
- Preserve the completed corrected small-CNN run and inspect its learning curve,
  validation accuracy/MIREX, confusion matrix, and measured CPU cost. Compare to
  Euphoria's existing libkeyfinder detector on exactly the same held-out files.
- Preserve this baseline as the control. Do not claim that it improves accuracy
  before observing the measurements.

Exit: one fully traceable real-data run, no data/label errors, and measured costs.

### 2. Freeze corpus and split policy

- Acquire FMAKv2 annotations and matching FMA audio, with explicit provenance and
  track availability. Add GTZAN-Key and optionally Winterreise as cross-domain
  checks; audit known GTZAN duplicate/artist issues.
- Partition the development corpus by track, and by artist/album/recording family
  where metadata permits. Detect near-duplicates/remixes across corpora. Reserve
  independent validation and calibration partitions. Save immutable split IDs and
  manifest hashes; training seeds must not change the split.
- Keep GiantSteps as an EDM external test. If FMAKv2 contributes training examples,
  only a disjoint reserved portion may be called its held-out test. Label results
  from all-FMAK training or public pretrained models honestly.
- Exclude final test tracks from unlabeled pretraining, distillation, augmentation
  derivatives, pseudo-labeling, and calibration wherever we control exposure.
  Audit teacher training data; document exposure that cannot be determined.
- Preserve ambiguous labels as alternatives for a soft-label experiment instead
  of assuming that the first supplied label is always uniquely correct. Audit
  mode imbalance, key imbalance, modulation, non-Western modes and silent audio.

Exit: reproducible manifests and a predeclared evaluation protocol.

### 3. Shortlist representations and models before large sweeps

Start with the small control and targeted diagnostics. If evidence justifies a new
representation/model tranche, compare relevant candidates under matched data access:

1. The corrected small CNN as the control. Larger profiles are gated by a measured
   FMAK generalization failure and a specific capacity hypothesis.
2. An octave-aware/transposition-equivariant compact audio classifier. Preserve
   pitch-class position in the output; do not pool away the information needed
   to distinguish transposed keys.
3. A compact DSP-feature reranker reusing Euphoria's chroma/key-profile evidence.
4. A strong pretrained music encoder with a frozen linear/MLP probe, then selective
   fine-tuning. Consider S-KEY/ChromaNet and current music foundation encoders;
   retain versions and training-data provenance.

Ablate representation separately: the current 33-band low-frequency baseline,
proper wider-range CQT/VQT or harmonic pitch features, and multi-resolution input.
Compare uniform whole-track compression against fixed-duration excerpts aggregated
over time. Shared model weights do not make different preprocessing equivalent.
Test tuning offsets, codec/compression changes, and quiet/short/corrupt files.

For efficient hybrids inspired by LFM, benchmark gated temporal convolutions,
local attention and modest global context at the same parameter/runtime budget.
Ablate only after a simpler music-aware baseline works. Generic decision models
remain the small exploratory branch described above.

Exit: select a compact candidate on development evidence and measured CPU cost.
A teacher is optional and must independently earn a budget; no test-driven changes.

### 4. Tune training recipes with controlled budgets

Use staged random search or Bayesian optimization with early pruning, logging
failed/pruned runs as well as winners. Begin with LR, batch size, weight decay,
dropout and training duration. Search LR in roughly 1e-4–3e-3 and weight decay in
roughly 1e-6–1e-2 as starting ranges, then adapt to learning curves. These are search
bounds, not recommended optima. Compare no smoothing against small smoothing
(e.g. 0.02–0.1) rather than enabling every regularizer together.

Then run controlled ablations for cosine/warmup vs Schedule-Free, EMA/SWA,
class/mode-balanced sampling or loss, label-noise/soft targets, and task-valid
augmentations. Report natural-distribution accuracy alongside balanced metrics.
Pitch augmentation changes the target and never changes major into minor.
Evaluate time crops/masking for global-label validity; avoid silently assigning a
whole-track key to a modulated excerpt. Treat Mixup/CutMix as hypotheses requiring
label-semantics checks.

Reserve matrix-aware optimizers, SAM-like sharpness methods, and more elaborate
curricula for candidates where the expected benefit justifies added training cost.
Give each optimizer its own tuning budget; do not copy AdamW's learning rate into
Muon/SOAP and call that a fair comparison. BF16/AMP and compilation are throughput
experiments; check numerical stability and cost on these actual small models.

Screen with one seed. Retrain shortlisted recipes with at least three matched
training seeds on the fixed split. Report mean, spread and total compute. Bootstrap
at track/artist-group level for accuracy differences where appropriate; do not
select the one lucky seed for the release table.

### 5. Conditional distillation or larger-model research

Enter this phase only after the small-model/FMAK gate is met and a bounded teacher
comparison is justified. It is not a prerequisite for the first release.

- Establish a teacher that is actually better on the development protocol.
  Frozen pretrained features plus a modest head may be sufficient; an ensemble
  is a training-only option, not the default desktop dependency.
- Compare CE-only students with CE + temperature-scaled full-distribution KL.
  Tune mixture and temperature (initial values such as T in {1,2,4}); multiply
  the KL term by T². Keep teacher outputs aligned by track ID and input view.
- If pitch-shifting student inputs, apply the same transform to the teacher's
  input, or explicitly rotate a justified equivariant teacher distribution.
  Unrelated cached teacher targets corrupt distillation.
- Distill on training/unlabeled-development inputs only. Keep validation and test
  out of pseudo-label generation and teacher-fitting. Audit teacher errors and
  class coverage; retain CE-only controls and compare student/teacher calibration.
- Calibrate finalists on the reserved calibration set. Add abstention and a
  risk/coverage curve for uncertain or out-of-scope audio. Evaluate a small→large
  cascade only as opt-in behavior with measured escalation frequency and cost.
- Ship the small artifact if it meets the product gates. An additional model is
  a separate decision requiring a demonstrated useful gain; medium/large releases
  are not default outputs of the experiment program.

### 6. Deployment, quantization and Euphoria integration

Export each finalist and verify preprocessing and logits on golden PCM/features
across Python, ONNX, and the Rust integration. Include model hash, architecture,
feature parameters, key schema, calibration, supported providers and dataset/run
provenance in the artifact manifest.

Compare FP32 and supported lower-precision paths on the actual CPU backend.
Convolution quantization may require static calibration and backend-specific
operator support; do not assume dynamic INT8 accelerates this CNN. Use training/
calibration data only for PTQ; try QAT if useful compression loses accuracy. Check
per-class, calibrated and cross-domain results after conversion, not just size.

Euphoria already has relevant infrastructure:

- `crates/euphoria-core/src/cache/batch.rs`: location/stamp lookup then content-hash
  lookup before analysis.
- `crates/euphoria-core/src/analysis.rs`: `ANALYZER_VERSION` gates cached results.
- `crates/euphoria-desktop/src/analysis.rs`: background Rayon workers, capped at four.

Extend those paths instead of adding a parallel cache. Model hash/tier, feature
version and calibration must participate in detector identity so switching artifacts
cannot return stale results. Preserve existing DSP/tempo/energy behavior and expose
an intentional fallback for missing/unsupported models.

Use a shared inference budget across Rayon workers. Start with sequential ONNX
execution, intra-op threads=1, inter-op threads=1, and worker spinning disabled.
Reuse model sessions and existing decoded audio/chroma where possible. Profile
feature extraction and decoding too; a 1 ms network does not make a 2 s extractor
cheap. Preserve cancellation and pause/resume; keep UI/playback responsive during
large imports. CPU users never silently escalate to a GPU/remote model.

See [ONNX Runtime threading controls](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).

### 7. Final evaluation and release gates

Freeze selected candidates and evaluate final test sets once for release selection.
Report exact accuracy, both MIREX conventions, per-key/mode and per-genre breakdowns,
confusion matrices, NLL/Brier/ECE, and risk/coverage. State known label ambiguity and
teacher/corpus exposure. Compare against rerun libkeyfinder, relevant open music
models, and reproducible published protocols; purchased tools only where available.

For the small release, and any separately justified future artifact, publish
cold/warm end-to-end latency, forward p50/p95, CPU-seconds,
peak RAM/VRAM, model size and total runtime footprint on named hardware. Test the
small tier on CPU-only machines, cached app restart, new-file import, large import,
GPU absence, cancellation, playback and missing/corrupt artifacts. GPU models need
batch-1 and batch-throughput measurements; GPU availability must not be assumed.

Release requires traceable training records, dataset/model distribution terms,
checksummed versioned artifacts, a model card, reproduction commands, matching native
preprocessing, and a documented fallback. Publish demonstrated results and the
limits of the comparison. Use “state of the art” only for a substantiated scoped
result, not as a synonym for using recent techniques.

## Immediate next actions

1. **Done:** bug-fix review/rework committed and pushed as `01bbc88`; 78 tests green.
   Keep all further source work on a feature branch, never `main`. Keep this plan
   untracked and out of commits/package artifacts.
2. **Done:** corrected small baseline completed; server data and caches exist.
   Preserve its artifact, source hashes and log; verify source provenance and
   capture its outcome in the run register above.
3. **Now:** diagnose the baseline and execute the capped small-model tranche.
   Keep validation tuning separate from held-out evaluation, report matching
   MIREX conventions, and measure CPU cost before considering integration.
4. **Next:** establish the FMAK protocol and measure small-model generalization.
   Only a demonstrated shortfall opens a budgeted larger/teacher experiment.
5. Additional song capabilities stay deferred in
   [Drone issue #1](https://github.com/lvndry/drone/issues/1).
