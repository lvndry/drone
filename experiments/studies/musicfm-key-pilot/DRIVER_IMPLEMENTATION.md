# Prospective executable teacher driver

**Source-only review; no model/cost/fit run authorized.** This implementation keeps
the optimizer, seed, fixed horizon, selector and two-control continuation bar in
PROTOCOL.md. Its input strategy supersedes that document's historical RAM proposal.
The complete canonical preparation7d56908d succeeded; its held fitting RAM cache
was later cleaned by the unchanged40GiB guard. Cause of global pressure is unproven.
All376 validation files remain. Read-only audit6fc26efc found all5,638 compressed
source hashes and validation hashes unchanged; only4,283 FMA physical identities
changed. Old manifests are immutable; prospective configuration binds that overlay.

`experiments.musicfm_pilot` provides metadata-only `plan`/`seal` and a private
`worker`. `tools.run_musicfm_teacher` is its source/config/mode-bound outer guardian.
A root approval must explicitly allow_probe=true/allow_training=false for the
cost stage, or allow_training=true for a real fit. There is no resume, automatic
handoff/lease extension, source repair, batch-size rescue or automatic continuation.

## Input providers

Prefer an independently approved complete ordinary-disk manifest from
`tools.prepare_musicfm_disk_cache`. CachedProvider reads exact readonly mmap crops,
closes mappings immediately, and checks cache/source identities before and after.
All full waveform hashes are checked once before model work. No large anonymous
or tmpfs fitting cache is allocated. A cache error aborts; it never switches provider.

The explicit `--provider stream` alternative uses root's full-stream helper2aa1f08:
unchanged6011bf7 full FFmpeg profile, no seeking, full stdout SHA/count before
returning a finite exact byte-index crop. Four decoder threads/FFmpeg processes
use CPUs10–13; main code uses8–9. At most current32 and next32 requests are prefetched
and consumed in fixed draw order. Sources get SHA and stable-identity checks;
labels remain in the main process. Every stderr byte, including empty outcomes,
is represented losslessly in four append-only JSONL logs. Successful temporary
stderr files are removed only after record flush; failed raw stderr stays.
No hundreds of thousands of empty inode files or full-waveform scratch writes.
Thread-safe global watches are throttled while the stream helper samples cheaply.

## Disposable cost versus real fit

Cost uses the original fitting16 (fixed batch hash5fbbc3e2), exactly3×32 synthetic
target examples, original strict-loaded MSD weights and real Adam state. It times
compute/H2D/optimizer AFTER inputs resolve, records provider waits separately, and
then validates all376 complete songs without GT scores or saved/reused weights.
The16 are duration-stratified and include a35-minute source: do not extrapolate
combined stress-case pipeline wall time by8,280. Weighted stream projection55,962
FFmpeg CPU /58,209 serial-service seconds applies only to streaming, not cached
fitting; /4 is an ideal bound. Cached whole-fit budgets require the measured cost.

Real fit always starts from the original pinned MSD checkpoint; its initial state
must match the disposable probe's original initialization. Seed_training(13) runs
AFTER construction/strict load/head/device setup. All8,280 updates and120 full-song
events are mandatory; no independent early stop or accuracy-dependent truncation.
Selection is equal-domain exact first, symmetric MIREX second, earliest tie. Best
and final weights are separate, source/data-bound and hashed. Both MIREX conventions
and ordered predictions are recorded. Continuation needs five FMA exact gains above
EACH frozen K10 control plus BP exact/symmetric/asymmetric nondecline; no promotion.

A canonical BP ID+augmentation-seed trace is replayed from the immutable sampler
and checked against actual teacher draws. Separately, frozen CQT frame metadata
reconstructs the old ID:seed:shift:first trace and must match both executed K10
controls. The two hash formats are never compared as if identical. Context,
augmentation, microbatch BatchNorm and architecture still differ: system comparison.

## Conditional FP32 evaluation batching

Default is B1. B16 requires exact resource proof11c9137e from source554aa50. Each
waveform goes through ORIGINAL preprocessing/normalization individually at B1;
only identical mel shapes are concatenated into encoder batches. No padding,
new frontend, changed context or dropped tail. Out-of-order shape buckets are
reassembled in each song's original window order before FP64 duration aggregation.

With B16 the cost stage performs full376 B1 versus B16 composed comparison after
its synthetic updates. Require absolute logit error≤1e-5 and zero argmax differences,
without GT scores. Fit configuration refuses a failed proof. After the fixed fit,
the selected saved checkpoint receives the same label-blind composed audit; these
two extra passes are not checkpoint selection opportunities. Failure blocks
continuation; no retroactive fallback/selection. Budget these passes explicitly.

## Resource boundary

Preserve the provisioned isolated interpreter/modules; no install/download/runtime
replacement. FP32 master/frontend/gradients/Adam and FP32 evaluation; BF16 training
encoder/head, TF32 off, eager attention. Nonfinite failure aborts the entire fit;
there is no claim that BN/RNG/partial gradients roll back atomically.

The independent guardian enforces fresh root UTC/wall/CPU/RSS/output caps, all four
vLLM start identities, actual global MemAvailable≥40GiB, tmpfs≥1GiB, disk≥32GiB and
GPU free≥8GiB. Its GPU query is sampled at0.5s, disk at1s; cheap direct worker
capacity/PID checks occur at0.2s boundaries and before model forwards. Heartbeat
allowance is10s, compatible with a5s query timeout plus scheduling; observed maximum
age is recorded. Only own workers/decoder descendants are stopped. No old RAM
ownership is assumed or cleaned. Full-fit caps are mandatory and remain pending
measured cost; no source default silently supplies an eight-hour promise.

Fixed duration/genre/group/House support reporting remains required. Five generic
5–10 minute House selection songs across three groups cannot establish the primary
house/deep-house/melodic-house acceptance target. No calibration, confirmation or
private-library selection is possible through these pool-bound interfaces.
