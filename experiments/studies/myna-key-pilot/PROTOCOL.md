# Prospective Myna frozen-embedding key pilot

**Source preparation only. No extraction, fitting, scoring or release result is
reported here.** Physical pitch-view qualification is still unresolved. Before
any classifier outcomes, the initial length-preserving augmentation was stopped:
the phase-vocoder/polyphase candidate failed amplitude qualification, RubberBand
R2 failed bass-pitch checks, and offline R3 failed target-band energy retention.
Those failures and the original source proposal remain preserved. They do not
qualify a cache or reject the Myna representation.

**Prospective amendment, November 13:** use a distinct playback-rate/resampling
augmentation, intentionally changing tempo and duration along with pitch. This
is a new local recipe, not a paper reproduction or an unreported substitution.
The eventual qualification, cost measurements, outcomes and inline figures belong
in this study's consolidated report, not separate visual-result reports.

## Question and scope

Does one frozen Myna encoder plus a paper-shaped head earn a stronger research
teacher than the existing expanded-data K10 control? This compares systems:
representation, augmentation, objective weighting, head and optimizer differ. It
cannot isolate the representation's causal contribution. The roughly 40M-parameter
encoder-plus-head system is a research candidate, not an ordinary-PC release
commitment. No Euphoria integration changes belong to this experiment.

Only approved Beatport 1,229 / FMA 4,033 fitting and Beatport 126 / FMA 250 selection
records are admitted. Reuse the exact plan, numeric-label export, group mapping,
canonical exclusions and existing waveform identities from the
[K10 data pilot](../k10-v2-pilot/PROTOCOL.md) and
[MusicFM pilot](../musicfm-key-pilot/PROTOCOL.md). These selection records are reused
development evidence. No reserved/calibration FMA, GiantSteps, GTZAN, Winterreise,
JSIR, Isophonics or private-crate audio/references enter this pilot. No resplitting,
replacement, label repair or new duplicate selection is allowed.

`experiments/myna_cache.py::PINS` freezes the source artifacts. FMA has 797 fitting
components and 100 selection components; BP uses recording proxies, not proven
artist-independent groups. The expanded K10 control's selected predictions are
reused without new inference. It has BP 89/126 and FMA 176/250 exact, with BP symmetric
and asymmetric weighted numerators 972/947 over 1,260. Restricted K10 and failed
MusicFM are contextual comparisons, not weaker substitute gates.

## Encoder and complete-window cache

The base is reviewed Myna-Vertical, 21,375,872 parameters, 85,516,720 checkpoint bytes.
Its immutable checkpoint digest is in `myna_cache.ENCODER_SHA`. Source/runtime,
frontend and the eventual qualified physical augmentation are independently bound.
The trainer never loads or forwards the encoder; extraction uses frozen eval mode.
Myna's MIT declaration is recorded in source qualification; do not vendor KeyMyna's
training code or redistribute its separately unqualified pretrained heads.

Local preprocessing is verified existing mono 24 kHz ffmpeg PCM followed by pinned
TorchAudio 16 kHz resampling, not asserted equivalence to publisher direct-original
TorchAudio decoding. Use unlogged power mel, centered FFT 2048 / hop 512, 128 bins and
196-frame complete nonoverlapping windows, including true neighboring STFT context
when processing blocks. Each window produces 384 float32 features. No feature
normalization, 19-window cap or independent waveform-crop padding is introduced.
Incomplete final mel windows are discarded and counted. All complete windows is
not literal all-audio coverage; the tail limitation remains visible.

Fit caches require every integer physical shift −6…+6; selection requires only
shift 0. Both −6/+6 remain distinct physical views. The fixed method identifier is
`pitch-by-playback-rate-polyphase-v1`. For shift s, use SciPy `resample_poly` with
`up=16000`, `down=round(16000*2**(s/12))`, Kaiser beta 5 and constant boundary
padding. Shift 0 is the original identity path. Output samples are exactly
`ceil(original_samples_16k*up/down)`. There is no output padding, trimming,
length restoration or gain normalization. Record actual playback rate and its
pitch-rounding error in cents. Duration and tempo change deliberately; the
original-only selection audio does not change.

Each view keeps all its own complete windows, so window counts now differ across
shifts. Method/library/options and numerical qualification are frozen before
extraction. No candidate is admitted merely because this metadata plan validates;
qualification failures remain separate and cannot be waived after scoring.

The complete-cache schema is agreed with the extraction owner:

- Top-level `schema_version="drone-myna-rate-window-cache-v2"`, `complete=true`,
  public `label_schema`, `augmentation_method`, `identities` and `tracks`. A separately sealed config binds
  the exact manifest SHA. Incomplete/extraneous/duplicate records are rejected.
- `identities` exactly binds `encoder_sha256`, `encoder_source_sha256`,
  `environment_sha256`, `frontend_sha256`, `augmentation_sha256`,
  `qualification_sha256`. The augmentation identity binds its method, library,
  parameters and length policy; opaque source identity alone is insufficient.
- Each track records `id`, `domain`, `role` (`fit`/`selection`), frozen `group`,
  original `target`, `source_sha256`, `pcm_sha256`, `samples_24k`, `samples_16k`,
  `mel_frames`, `complete_windows`, `discarded_frames` and `views`.
- Track-level geometry describes only the original unshifted recording:
  `samples_16k=ceil(samples_24k*2/3)`,
  `mel_frames=floor(samples_16k/512)+1`, `complete_windows=mel_frames//196`,
  `discarded_frames=mel_frames%196`. Zero complete windows is a failure, not a
  silently padded input. Actual extraction counts must match the sealed budget.
- Each view records integer `shift`, its own `samples_16k`, `mel_frames`,
  `windows`, `discarded_frames`, `rate_up`, `rate_down`, `actual_rate`,
  `pitch_cents_error`, `raw_samples`, `trimmed_samples`, `padded_samples`,
  `dtype="float32"`, `shape=[windows,384]`,
  `window_order="all-complete-196-frame-windows-in-order"`, relative `.npy` `path`,
  `bytes` and SHA256. No pickle/object arrays or nonfinite features are admitted.
  Each view's geometry derives from its actual playback-rate output; all complete
  windows are retained in order. `raw_samples` equals that view's sample count,
  and trimmed/padded samples must both be zero. Labels are stored once per
  recording; shifted targets come from `drone.keys.transpose_key`.

Drone already uses tonic-times-two-plus-minor, matching inspected KeyMyna's
interleaved class order modulo enharmonic names. No nonidentity label permutation
is needed. Preserve mode under pitch transposition.

Under the new playback-rate recipe, projected counts are 2,152,026 fitting
vectors across 13 views, plus 10,892 original selection vectors. Their float32 ×
384 payload is 3,322,242,048 bytes (3.094 GiB). The former length-preserving
projection was 2,103,933 fitting vectors and is historical, not the new budget.
The trainer verifies every blob, then loads one bounded
contiguous RAM matrix to avoid repeated random opens of tens of thousands of
files. Its separate label/weight/permutation buffers, head/Adam states and
checkpoint writes also count toward the supervisor's RSS/storage budget. The RAM
limit is not a claim of ordinary-PC inference cost.

## Fixed head, objective and exposure

The independently implemented head is:

`384 → Linear(4096) → ReLU → Dropout(.99) → Linear(4096) → ReLU → Linear(24)`.

There is exactly one dropout layer and 18,456,600 trainable parameters. Use seed 42,
FP32 parameters/activations/gradients/moments, TF32 off, deterministic operations,
Adam learning rate 3e−4, weight decay 1e−5, betas (.9, .999), epsilon 1e−8,
foreach/fused false and batch 64.
No AMP, standardization, MixUp, smoothing, scheduler, clipping, early stopping,
EMA, compile or sweep. These are declared CLI-default-informed pilot choices, not
an assertion of the paper's winning hyperparameters. The
[KeyMyna paper](https://arxiv.org/html/2604.10021v1) supports the GS head and physical
pitch augmentation; the pinned [argument defaults](https://github.com/echo-cipher/keymyna/blob/7d590820dd93b3adb45be12ebe991b414ae53683/parse_args.py)
and [training source](https://github.com/echo-cipher/keymyna/blob/7d590820dd93b3adb45be12ebe991b414ae53683/train.py)
provide context for the declared optimizer and budget choices. No upstream code
is copied into Drone.

Shuffle every fitting vector once per complete cache pass with an independent
seeded NumPy generator. Keep the final short batch. Long recordings must not gain
more objective weight merely because they contain more windows. Preserve domain
mass BP 1230/2206, FMA 976/2206, then equal weight per recording within each domain,
then equal weight per shift/window. For N fitting vectors, n_d recordings in
domain d, W_i,s windows in track i's specific view s and domain mass p_d:

`weight_i,s = N * p_d / (n_d * 13 * W_i,s)`.

The denominator is the current view's window count, not the original track's
window count. Each of 13 views therefore has equal total objective mass despite
its changed duration.

The batch objective is `mean(weight_i * cross_entropy_i)`. Do not divide by each
random batch's sum of weights. The reported full-pass objective uses the complete
weighted numerator divided by N. This is an explicit local weighting change from
an unweighted window loader. Long tracks consume more compute, not more total
loss weight. Component groups are used for uncertainty, not extra sampling weights.

Root must seal **either 10 or 100 complete passes before any accuracy is observed**,
using sustained head-update/evaluation timing plus extraction, verification and
artifact costs. Ten passes are 21,520,260 presentations / 336,260 updates; 100 are
215,202,600 / 3,362,600 at projected counts (26 examples in each last batch). One
selection event follows each complete pass. These options are different exposure
budgets; no 10-pass result may automatically continue to 100. A 100-round one-window
per-record sampler is only 526,200 presentations and does not cover long-record
pitch/window grids; it is not implemented as an equivalent shortcut.

## Selection and continuation

At each event, evaluate only original selection embeddings with dropout off. Use
all complete windows and average 24-dimensional window logits in FP64, in fixed
window order. Do not average probabilities or embeddings before the nonlinear
head. Select by the existing exact-rational equal-domain exact accuracy, then
symmetric MIREX, retaining the earliest full tie. Both domains are therefore
exposed to adaptive checkpoint selection.

Finish the entire sealed horizon. Replay the actual selected saved checkpoint,
then compare B1 with B64 head evaluation over all 376 selected recordings using
`abs(delta) <= 1e-5 + 1e-4*abs(B1)` and zero argmax disagreement. This is a
prebudgeted numerical audit, not another checkpoint-selection opportunity. A
failure preserves the fit but earns no continuation or tolerance adjustment.

The development-only continuation bar is FMA ≥181/250 exact and BP ≥89/126 exact,
BP symmetric/asymmetric numerators ≥972/947. This five-case FMA gain is a practical
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
that supervisor. No guardian or launch authorization is supplied here. The dependency-light
`experiments/myna_head.py` implements the identical head, optimizer, weighted
training batch and window-logit evaluation kernels used by the trainer and the
separate sustained cost probe. Timings must include transfers, finite checks,
synchronous scalar reads and watchdog overhead, not just matrix multiplication.

The config schema is `drone-myna-head-pilot-v1`: `passes` 10/100, `device`, new owned
`output`, `expected_embedding_bytes`, `max_embedding_bytes`, exact numerical
`identities`, actual `source_sha256` closure from `myna_pilot.SOURCES`, and `inputs`.
Inputs contain `{path,sha256}` bindings for `plan`, `labels`, `groups`, `waveforms`,
`prepared_waveforms`, `control`, `cache`, `qualification`. The first six have
immutable pins in source. The original waveform manifest and its physical disk
replacement must agree on every ID, role, target, sample/byte count and source/PCM
hash. Physical paths may differ. Both use `fit` / `validation`; the embedding
manifest maps these explicitly to `fit` / `selection`.
The cache manifest resolves relative blobs against its own directory. No source
roles or fitting horizon can be changed by adding config fields.

A separately reviewed qualification summary must have schema
`drone-myna-cache-admission-v1`, `complete=true`, identical numerical identities
(excluding its own digest), the same `augmentation_method`, and true
`all_pitch_views_passed`,
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

## Metadata-only extraction plan

`tools.prepare_myna_cache_plan` joins the original and prepared waveform manifests
with the fixed roles, labels and groups without opening PCM or loading a model.
It emits `drone-myna-rate-window-plan-v2`, `complete=false`: expected shapes and all
68,782 track/views are enumerated, while unknown augmentation results, blob hashes
and sizes remain null. The same membership/geometry checks validate this plan,
but the loader explicitly rejects its unmeasured contract. A dry-validation pass
is not extraction or numerical qualification and cannot enable training. Physical
cache paths stay in the private operational plan, not the public repository.
