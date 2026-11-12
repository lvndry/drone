# Prospective paired K10 v2 pilot — frozen review contract

Status: PROPOSED, no preprocessing/training/role activation performed by this study.
The root-reviewed v2 assignment and conflict overlay at ca763c83 remain unchanged.
This is an INTERNAL data experiment, not independent final or release evidence.

## Hypothesis and cohorts

Compare fresh K10 on nested restricted976 versus expanded4035 container-unique FMA
fitting records. Both use the same current exact-hash-guarded1230 BP fitting IDs,
the SAME NEW250 FMA inner-selection IDs and SAME126 BP selection/guard IDs.
Do not substitute the historical v1 FMA250. All identities/expected containers and
numeric targets bind to `data/proposed-plan.json.gz` and its source hashes.
Calibration159 and confirmation998raw/997container-unique remain excluded from
pilot feature preparation, training and checkpoint selection.

BP126 is an adaptive selection AND nonregression guard, used previously by the
program. It is not artist-independent or a fresh test. BP612 unresolved entity
identities are a disclosure/policy limit, not a blocker to this paired internal
FMA comparison. No GS/private audio is used. Previously exposed FMA roles do not
become pristine through reassignment. Only label-independent CQT may be reused;
no canonical learned encoder/checkpoint/teacher is loaded.

## New matched recipe, not old-trajectory reproduction

One seed13, KeyCNN width1 /109128parameters, fresh identical initialization.
Preserve physical CQT (22050Hz,208padded bins,24bins/octave,hop4096,log1p,
float16cache),192core bins,320frames, cached CQT-bin-reslice shift uniform[-4,4],
label transposition by minus shift, uniform temporal crop/right zero pad.
Hard-target CE, AdamW lr0.001/wd0.0001 with inherited defaults; no smoothing,
warmup, EMA/SWA, optimizer/architecture search or physical waveform augmentation.

Both arms complete EXACTLY8280optimizer updates:120virtual epochs ×69updates,
batch32. No independent patience/early stop. The v1 schedule had ceil2206/32=69
updates but only2206draws/full pass; v2 has2208draws/virtual epoch. This is a NEW
matched recipe, not exact v1 reproduction or exact BP draw equality to v1.
CosineAnnealingLR T_max120 steps once after every69updates, exactly the same
progression/evaluation cadence in both arms. CPU4thread training work must be
placed on root's available cores; GPU0 4%allocator/8GiBfree guard remains intact.

Reference BP domain mass is EXACT1230/2206 asymptotically: cumulative BP draws at
updateu=floor(32*u*1230/2206), so batches contain17or18BP, all remaining slots FMA.
Pools shuffle without replacement within each cycle; their order RNGs are
independently keyed by domain. Every domain/draw's augmentation RNG is separately
keyed, independent of other population size/length/RNG consumption. Common BP
sample AND augmentation traces must match between arms. Total draws:147733BP,
117227FMA per arm. FMA record passes are about120.1restricted versus29.1expanded;
undertraining at matched updates is a declared limitation. Matched-pass follow-up
would be a separately frozen experiment, not an automatic rescue.

## Selection and fixed pilot gate

At each of120events evaluate full true-length,shift0 features, bounded batch4,
using existing `evaluate_features` (no peer-dependent padding). Select lexicographic:
1.equal-domain mean EXACT over BP126/FMA250;
2.equal-domain mean symmetric MIREX;
3.earliest event for a full tie (strict > only).
Integer/rational totals settle ties, not floating rounding. BP is therefore exposed
to checkpoint selection; it cannot be described as untouched confirmation.
Record hard-target training CE, both MIREX conventions, ordered per-ID predictions,
selected update/checkpoint hash, source/config/runtime, initial-state hash and
actual sampling coverage/feature-frame exposure. Never pick epochs from confirmation.

Pilot earns SEPARATE root review for confirmation only if both8280-update runs
complete with paired source/config/runtime/device/featuremanifest/initialstate/BP
trace and identical ordered126/250selection IDs, AND expanded FMA exact improves
>=2percentage points (>=5/250 correct), with NO BP exact/symmetric/asymmetric-MIREX
decline. FMA weighted-score changes are reported, NOT an additional gate.
Incomplete/budget/OOM/source/identity failure is inconclusive, not a negative fit or
permission to compare unequal horizons.20min internal wall per arm plus a root
outer timeout; no automatic restart/resume/seed extension. Pilot status/gain is not
release promotion or evidence of statistical significance. Paired group-clustered
uncertainty and error/relation movements follow a completed pair.

`gate` performs no model/feature/audio access and NEVER scores confirmation.
A pass freezes both selected checkpoint hashes and earns a separately authorized
997-unit confirmation evaluation. Before that stage, verify its canonical identities
against pilot fitting/selection. Do not score calibration/confirmation now or reuse
confirmation to choose checkpoints/thresholds. Wider folds/seeds require another
prospective decision after cost/precision are known.

## Numeric labels and normal physical preparation

Post-assignment numeric export contains ONLY5641pilot IDs (5265fit-union +376
selection). No current calibration/confirmation target is exported. Exact per-ID
ref equality is required in feature manifests, alongside the export SHA.
FMA refs bind to prepared metadata.npy SHA94f65bb54329b239028dfffc1e8bf37f7d037f6e1e7c25f6721e73efd2c037bc
and directly equal selected official CSV `key_and_mode` encoded with `encode_key`.
CSV MD5 is3b2d16784ffbda850c8ddf0519478bfd; strong SHA is retained in the export.
BP labels bind to source manifest SHA39fe1842a544bd818fe48c38d9b0916d7f6c04f5531894b0396d6e6d24d1f5f5.
The [publisher](https://zenodo.org/records/12759100) describes expert-labeled FMAK,
with around200v2 label modifications not verified by the original FMAK authors.
Matching this pinned source proves target consistency, not consensus or a noise ceiling.

The preparer follows `drone.features.decode_mp3/read_wav_mono/resample_linear`
EXACTLY: actual mono22050Hz16-bit-WAV decode converted to normalized float32 samples,
then canonical SHA over contiguous little-endian float32 model-input bytes. Hash
current source containers and validate known cache content/shape/dtype/finiteness.
Reuse verified v1 CQT via hardlink/copy; compute missing CQT from that same decoded
waveform without another decode. Record canonical sample count/hash and phase costs.
Full pilot manifests must contain EXACTLY5641allowed IDs, proper physical contracts,
expected refs and cache hashes, no canonical fit/selection identity crossing and no
canonical identical-input label conflicts. New violations stop activation for root
review; no silent union/relabel/drop/replacement. Within-role consistent duplicates
are disclosed, not automatically used to rewrite the container-unit contract.

First16 is a separate non-trainable preparation stage:12missing fitting CQTs,
3per metadata-duration quartile, longest2157s included in upper quartile;4reuse
examples (2BPfit,2restrictedFMAfit) measure canonical decode cost even on cache hits.
Selection uses fixed SHA ranks, never labels/model scores. `first-preprocessing-plan`
contains exact IDs/caps. Two workers,CPU10–11,OMP/MKL/OpenBLAS/NumExpr/Numba1,
CUDA invisible,900s wall/1800s CPU,4GiB aggregate RSS,128MiB CQT and256MiB WAVscratch.
`tools.run_k10_v2_preprocessing` is the actual Linux outer process-group guard:
wall/CPU/RSS/storage checked every0.2s, terminal child CPU/final storage included,
TERM/KILL only its own new group on breach. RSS is sampled summed RSS (shared pages
count perprocess); bounded polling overshoot is reported, not disguised as kernel
memory enforcement. Worker completion checks alone are not the resource guard.

Both preparation and training require separate root-written hash-bound permits.
The first16 helper has a hard16-ID membership guard and always writes complete=false;
it cannot activate training roles or pretend bulk preparation completed. Preserve
partial/failure evidence. Measure first16 before deciding full preparation budget.
