# Prospective canonical disk-cache preparation

Source/config review point only: no allocation or decode authorized by this file.
New ordinary-disk capacity makes a full fitting cache feasible; the earlier
~56 GB constraint no longer describes the observed host. No actor or cause is
inferred from that capacity change. Preserve completed canonical preparation,
held-RAM pressure cleanup, failed seek qualification and successful full-stream
qualification as distinct evidence. No frozen RAM code is patched.

Use canonical manifest7d56908d and separately audited identity overlay6fc26efc.
Their full hashes are constants in `tools.prepare_musicfm_disk_cache`. The overlay
changes physical identities only; IDs, paths, roles, targets, original source SHA,
full float32 SHA, samples and bytes remain exact. The pinned reference configuration
binds the full6011bf7 FFmpeg profile and217 runtime/library files.

Four single-thread decoders on logical CPUs10–13 decode only the5,262 fitting
files, exactly once, into a NEW owned ordinary-disk directory. Source container
SHA and current stable identities are checked before and after; every output must
match its canonical full-waveform SHA and exact byte count and be finite. No
seeking, cropping, resampler change, lossy representation, role repair or file drop.
Make successful cache files read-only. Rehash/recheck all376 already-preserved
validation waveforms and reference their existing paths; do not decode or duplicate
them. No calibration, confirmation or private input is eligible.

Forecast fitting bytes plus64 MiB metadata before allocating; require total≤100
billion bytes and at least32 GiB ordinary-disk reserve after the forecast. Enforce
actual owned bytes≤100 billion, global MemAvailable≥40 GiB, tmpfs reserve≥1 GiB,
all four protected vLLM start identities, aggregate CPU≤6,000 seconds, wall≤7,200
seconds and sampled process-tree RSS≤2 GiB. Maximum single output256 MiB and exact
canonical bytes; numeric threads1/CUDA masked. Guards run during decode and bounded
4 MiB hash scans; global disk walks are cached for1 second and forced at completion.
Atomic progress-rename races are handled by the new disk accounting, not a patch
to the old helper. Shared guard state is locked across decoder threads. Limits
are sampled; a fresh retained outer timeout/controller must bound final execution.

Concurrent per-file child-CPU deltas from the reused helper are explicitly labelled
as overlapping all-child deltas, never exclusive decoder cost. Aggregate process
CPU is the budget evidence. Retain every stderr and per-file record. On failure,
stop/join only these workers; preserve partial owned output and the failed receipt,
with no complete manifest, automatic retry/resume or silent repair. Never alter
source audio or old manifests. Successful prep emits a complete source-bound
manifest with allow_training=false, all5,638 unchanged roles/targets, new fitting
cache identities and reused validation identities. Separate model approval is
required even after the data exist.

The driver will use an explicit CachedProvider if that manifest is approved,
reading exact readonly sample-index crops. Full-stream decoding remains a separately
configured bounded fallback/reproduction tool; a cache error never silently switches
the trained experiment's provider. Full8,280 updates and120 complete-song events
remain unchanged. Full-fit resources await the actual cost stage.
