# Prospective Myna frozen-embedding key pilot

**Source preparation only. No extraction, fitting, scoring or release result is
reported here.** Physical pitch-view qualification is still unresolved. The old
phase-vocoder/polyphase candidate failed amplitude qualification; a RubberBand
candidate is under separate review. Neither is silently admitted by this protocol.
The eventual qualification, cost measurements, outcomes and inline figures belong
in this study's consolidated report, not separate visual-result reports.

## Question and scope

Does one frozen Myna encoder plus a paper-shaped head earn a stronger research
teacher than the existing expanded-data K10 control? This compares systems:
representation, augmentation, objective weighting, head and optimizer differ. It
cannot isolate the representation's causal contribution. The roughly40M-parameter
encoder-plus-head system is a research candidate, not an ordinary-PC release
commitment. No Euphoria integration changes belong to this experiment.

Only approved Beatport1,229/FMA4,033 fitting and Beatport126/FMA250 selection
records are admitted. Reuse the exact plan, numeric-label export, group mapping,
canonical exclusions and existing waveform identities from the
[K10 data pilot](../k10-v2-pilot/PROTOCOL.md) and
[MusicFM pilot](../musicfm-key-pilot/PROTOCOL.md). These selection records are reused
development evidence. No reserved/calibration FMA, GiantSteps, GTZAN, Winterreise,
JSIR, Isophonics or private-crate audio/references enter this pilot. No resplitting,
replacement, label repair or new duplicate selection is allowed.

`experiments/myna_cache.py::PINS` freezes the source artifacts. FMA has797 fitting
components and100 selection components; BP uses recording proxies, not proven
artist-independent groups. The expanded K10 control's selected predictions are
reused without new inference. It has BP89/126 and FMA176/250 exact, with BP symmetric
and asymmetric weighted numerators972/947 over1,260. Restricted K10 and failed
MusicFM are contextual comparisons, not weaker substitute gates.

## Encoder and complete-window cache

The base is reviewed Myna-Vertical,21,375,872 parameters,85,516,720 checkpoint bytes.
Its immutable checkpoint digest is in `myna_cache.ENCODER_SHA`. Source/runtime,
frontend and the eventual qualified physical augmentation are independently bound.
The trainer never loads or forwards the encoder; extraction uses frozen eval mode.
Myna's MIT declaration is recorded in source qualification; do not vendor KeyMyna's
training code or redistribute its separately unqualified pretrained heads.

Local preprocessing is verified existing mono24kHz ffmpeg PCM followed by pinned
TorchAudio16kHz resampling, not asserted equivalence to publisher direct-original
TorchAudio decoding. Use unlogged power mel, centered FFT2048/hop512,128 bins and
196-frame complete nonoverlapping windows, including true neighboring STFT context
when processing blocks. Each window produces384 float32 features. No feature
normalization,19-window cap or independent waveform-crop padding is introduced.
Incomplete final mel windows are discarded and counted. All complete windows is
not literal all-audio coverage; the tail limitation remains visible.

Fit caches require every integer physical shift−6…+6, with identical final sample
counts. Selection requires only shift0. Both−6/+6 remain distinct physical views.
Shift0 must have no trimming or padding. A proposed shifter's raw output, trimming
and padding must reconcile exactly; its method/library/options and numerical
qualification are frozen before extraction. Qualification failures cannot be
resolved by loosening this protocol after scoring.

The complete-cache schema is agreed with the extraction owner:

- Top-level `schema_version="drone-myna-complete-window-cache-v1"`, `complete=true`,
  public `label_schema`, `identities` and `tracks`. A separately sealed config binds
  the exact manifest SHA. Incomplete/extraneous/duplicate records are rejected.
- `identities` exactly binds `encoder_sha256`, `encoder_source_sha256`,
  `environment_sha256`, `frontend_sha256`, `augmentation_sha256`,
  `qualification_sha256`. The augmentation identity binds its method, library,
  parameters and length policy; opaque source identity alone is insufficient.
- Each track records `id`, `domain`, `role` (`fit`/`selection`), frozen `group`,
  original `target`, `source_sha256`, `pcm_sha256`, `samples_24k`, `samples_16k`,
  `mel_frames`, `complete_windows`, `discarded_frames` and `views`.
- Expected sample geometry is `samples_16k=ceil(samples_24k*2/3)`,
  `mel_frames=floor(samples_16k/512)+1`, `complete_windows=mel_frames//196`,
  `discarded_frames=mel_frames%196`. Zero complete windows is a failure, not a
  silently padded input. Actual extraction counts must match the sealed budget.
- Each view records integer `shift`, `raw_samples`, `trimmed_samples`,
  `padded_samples`, `windows`, `dtype="float32"`, `shape=[windows,384]`,
  `window_order="all-complete-196-frame-windows-in-order"`, relative `.npy` `path`,
  `bytes` and SHA256. No pickle/object arrays or nonfinite features are admitted.
  All13 fitting views preserve the track's window count/order. Labels are stored
  once per recording; shifted targets come from `drone.keys.transpose_key`.

Drone already uses tonic-times-two-plus-minor, matching inspected KeyMyna's
interleaved class order modulo enharmonic names. No nonidentity label permutation
is needed. Preserve mode under pitch transposition.

Projected counts are161,841 original fitting windows ×13 =2,103,933 fitting
vectors, plus10,892 original selection vectors. Their float32×384 payload is
3,248,371,200 bytes. The trainer verifies every blob, then loads one bounded
contiguous RAM matrix to avoid repeated random opens of tens of thousands of
files. Its separate label/weight/permutation buffers, head/Adam states and
checkpoint writes also count toward the supervisor's RSS/storage budget. The RAM
limit is not a claim of ordinary-PC inference cost.

## Fixed head, objective and exposure

The independently implemented head is:

`384 → Linear4096 → ReLU → Dropout(.99) → Linear4096 → ReLU → Linear24`.

There is exactly one dropout layer and18,456,600 trainable parameters. Use seed42,
FP32 parameters/activations/gradients/moments, TF32 off, deterministic operations,
Adam lr3e−4, weight_decay1e−5, betas(.9,.999), eps1e−8, foreach/fused false, batch64.
No AMP, standardization, MixUp, smoothing, scheduler, clipping, early stopping,
EMA, compile or sweep. These are declared CLI-default-informed pilot choices, not
an assertion of the paper's winning hyperparameters.

Shuffle every fitting vector once per complete cache pass with an independent
seeded NumPy generator. Keep the final short batch. Long recordings must not gain
more objective weight merely because they contain more windows. Preserve domain
mass BP1230/2206, FMA976/2206, then equal weight per recording within each domain,
then equal weight per shift/window. For N fitting vectors, n_d recordings in
domain d, W_i windows per view and domain mass p_d:

`weight_i = N * p_d / (n_d * 13 * W_i)`.

The batch objective is `mean(weight_i * cross_entropy_i)`. Do not divide by each
random batch's sum of weights. The reported full-pass objective uses the complete
weighted numerator divided by N. This is an explicit local weighting change from
an unweighted window loader. Long tracks consume more compute, not more total
loss weight. Component groups are used for uncertainty, not extra sampling weights.

Root must seal **either10 or100 complete passes before any accuracy is observed**,
using sustained head-update/evaluation timing plus extraction, verification and
artifact costs. Ten passes are21,039,330 presentations/328,740 updates;100 are
210,393,300/3,287,400 at projected counts (61 examples in each last batch). One
selection event follows each complete pass. These options are different exposure
budgets; no10-pass result may automatically continue to100. A100-round one-window
per-record sampler is only526,200 presentations and does not cover long-record
pitch/window grids; it is not implemented as an equivalent shortcut.

## Selection and continuation

At each event, evaluate only original selection embeddings with dropout off. Use
all complete windows and average24-dimensional window logits in FP64, in fixed
window order. Do not average probabilities or embeddings before the nonlinear
head. Select by the existing exact-rational equal-domain exact accuracy, then
symmetric MIREX, retaining the earliest full tie. Both domains are therefore
exposed to adaptive checkpoint selection.

Finish the entire sealed horizon. Replay the actual selected saved checkpoint,
then compare B1 with B64 head evaluation over all376 selected recordings using
`abs(delta) <= 1e-5 + 1e-4*abs(B1)` and zero argmax disagreement. This is a
prebudgeted numerical audit, not another checkpoint-selection opportunity. A
failure preserves the fit but earns no continuation or tolerance adjustment.

The development-only continuation bar is FMA≥181/250 exact and BP≥89/126 exact,
BP symmetric/asymmetric numerators≥972/947. This five-case FMA gain is a practical
research bar, not significance or release evidence. Report exact/MIREX and paired
promotions/demotions against the strongest saved control. A pass earns separate
review; confirmation, distillation and release remain unauthorized by this runner.
Group descriptive uncertainty by existing components/proxies and report native
genre/duration/BPM/missingness with support counts. Do not convert small weak cells
into post hoc experiment-selection rules.

## Execution boundary and retained evidence

`experiments.myna_pilot.run(config_path, config_sha256, watch)` deliberately has no
CLI, implicit default watchdog or automatic retry/resume. It requires a separately
reviewed external guardian, a hash-sealed config and a live watchdog callback.
That guardian must verify runtime/environment/device identity, protect both vLLM
services, restrict one explicitly selected GPU (the process sees `cuda:0`) or CPU,
enforce wall/CPU/process-tree RSS/GPU/disk caps continuously, and bound termination
of only its own process group. This library's cooperative checks do not replace
that supervisor. No guardian or launch authorization is supplied here.

The config schema is `drone-myna-head-pilot-v1`: `passes`10/100, `device`, new owned
`output`, `expected_embedding_bytes`, `max_embedding_bytes`, exact numerical
`identities`, actual `source_sha256` closure from `myna_pilot.SOURCES`, and `inputs`.
Inputs contain `{path,sha256}` bindings for `plan`, `labels`, `groups`, `waveforms`,
`control`, `cache`, `qualification`. The first five have immutable pins in source.
The cache manifest resolves relative blobs against its own directory. No source
roles or fitting horizon can be changed by adding config fields.

A separately reviewed qualification summary must have schema
`drone-myna-cache-admission-v1`, `complete=true`, identical numerical identities
(excluding its own digest), and true `all_pitch_views_passed`,
`long_track_composed_parity_passed`, `sustained_head_cost_measured`. Its digest is
bound in the config/cache. Root must derive that summary from the actual retained
qualification receipts, not manually substitute pass flags for failed checks.
No such passing summary is created by this source preparation.

Each event retains loss, full original-only predictions/logits and checkpoint
selection metadata. Selected/final head files, their hashes, initial-state hash,
full exposure counts, every event and selected-checkpoint batch proof are retained.
Failure writes an inconclusive receipt and preserves existing outputs. A completed
negative rejects this one frozen recipe; partial training is not a negative result.
The external guardian adds resource and retirement receipts. No PLAN.md is tracked.
