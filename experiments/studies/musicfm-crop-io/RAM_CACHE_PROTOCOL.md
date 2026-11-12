# Prospective full-decode-once RAM fitting cache

**PROPOSED; no allocation, decoding or training authorized.** This is a new path
after the exact input-seek gate failed. Preserve both the16-file/144-case success
and the5638-scope attempt's2288 passes / first failure / incomplete cache. No file,
label, offset, tolerance or membership repair; FMA046091 remains included.

Use all5638 approved IDs:5262 fitting (BP1229/FMA4033),376 inner (126/250), exact
amended plan c3e59ffb and numeric export b8942b59. `make_config` from frozen9facd95
verifies pinned plan/labels/feature-manifest/runtime bytes before constructing the
embedded ID/role/source/target rows. The new driver checks source hashes, fixed
counts, runtime closure, LinuxCPU8–9/envthreads1/CUDAmasked, root permit and window.
No calibration/confirmation/private inputs or model import.

Decode each source exactly ONCE from the beginning using the literal6011bf7 full
FFmpeg command/profile, mono24kHz F32LE. No input seek, output seek, alternate
resampler, preprocessing quantization or waveform alignment. Write fitting output
directly into one new task-owned `/dev/shm/drone-musicfm-fit-<uuid>` directory;
write validation output to ordinary disk. Hash/check finite values in bounded
blocks, record actual sample counts and source-stability identity, retain every
stderr file, reject any cross-role/conflicting-target full24k hash family. Mark
files read-only and retain device/inode/size/mtime/ctime identities. Mmap crops use
actual sample indices, readonly mapping, an exact-byte copy and immediate unmap.
This eliminates seek-state disagreement by construction; full decoder warnings
and source completeness still require honest interpretation. It does not promise
corpus-quality perfection or new teacher accuracy.

## Capacity before and during preparation

Page-round EACH fitting file separately; add1MiB ownership metadata allowance.
The full fitting forecast must be ≤99,000,000,000B (actual owned page-rounded files,
including ownership file, use the same hard cap). Before any tmpfs allocation,
require `MemAvailable >= forecast + 40GiB`, tmpfs available>=forecast+1GiB,
and ordinary disk available>=projected validation+512MiB receipts+32GiB.
Logs, records and progress live only on ordinary disk. No fitting reference copy
is written to ordinary disk. One sequential decode is capped256MiB per file.

Throughout, poll every50ms during decoding and between bounded hash blocks:
actualMemAvailable≥40GiB (32GiB serving reserve plus8GiB host-model headroom),
tmpfs available≥1GiB, ordinarydisk available≥32GiB, same four protected vLLM start
identities. Guard process-tree CPU6000s/wall7200s/sampled RSS2GiB separately from
owned tmpfs pages. RSS is NOT a proxy for tmpfs memory after mappings close, and
future worker summedRSS can double-count shared cache pages. Ordinary owned
outputs retain8GiB cap; refresh disk accounting at1s and force at terminal. Caps
are sampled and may overshoot between polls. No swap assumption: observed swap
was nearly full. Fresh capacity/start identities and exact sealed source/config
must be rechecked at activation, not copied as permanent availability evidence.

On source/identity/finite/cap/pressure/error or caught termination, kill/join own
decoder, close mappings and unlink ONLY explicitly witnessed owned tmpfs entries.
The cache has a random nonce, saved directory device/inode/UID, matching OWNER
copies on disk/tmpfs and an enumerated filename allowlist. Cleanup rejects changed
ownership, symlinks or unknown files. Preserve disk receipts/failure evidence.
An outer retained controller MUST enforce the final cutoff/abnormal-worker exit
and call `cleanup_from_config` only AFTER its own workers exit; SIGKILL cannot run
in-process cleanup. Never remove old roots, other actors' files, private data or
persistent historical evidence. No automatic fallback/retry/resume or dataset drop.

## Future fitting boundary

A completed manifest has `allow_training=false`. Future fit needs separate root
approval and a fresh capacity/runtime/start-identity readback. Verify every
waveform hash once before activation; use actual sample counts and immutable file
identities for crop access. Maintain the40GiB/tmpfs1GiB/disk32GiB/protected-start
watch continuously for the whole fit, including validation and model initialization,
not just prep. Stop own training/workers, unmap, then witnessed cleanup on pressure.
No sampler/draw/order/domain-mass/window/selection change is implied by the cache.
Read-only crop API alone is not a complete fit guardian. No fitting code is activated
by this proposal.
