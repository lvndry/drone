# MusicFM crop I/O feasibility

Status: prospective waveform qualification, **not authorized to process audio yet**. Original80d4574 and its LRU evidence remain immutable; this prospective amendment changes only stable file-identity checking and the shared decoder recovery profile. The metadata-only LRU simulation has completed. No model, labels, training accuracy or GPU is involved.

## Problem and fixed membership

A 4 GiB full-waveform fitting LRU is not a viable assumed solution to random whole-song crops. The exact amended K10 draw schedule requests 264,960 samples (8,280 updates × 32), retaining nominal BP mass1230/2206. MusicFM would draw uniformly from every legal 12-second start across the full track. Validation still covers complete songs; this experiment changes neither fitting distribution nor validation horizon/cadence.

The approved amendment is b67ad3e868a1aef99fc0f12f723d4c2bd3264f6c: plan c3e59ffb764ca5f51444447b0a18a69543228e60a9e3723d925ddd82b0fc8373; numeric labels b8942b5913b8ebacdf2fc9e0ebca72a934aaa7c56d4e1a97d0f60fac84354323. BP fitting1229 and expanded FMA fitting4033 are used only to generate request IDs; labels are never consulted. All original first16 remain eligible, as independently checked by the data-contract owner. There is no substitution after exclusions.

## Metadata result

[data/lru-projection.json](data/lru-projection.json) uses the actual source sampler SHA bd5f279543c55fc857fd777080e2fbbfd76ed80c324f462f5abe5213285ab65e and immutable full canonical-duration manifest9f2db5efe4cbe2904508d8004f842acd29e90c6bc7f68612679c68993c439de0. A serial, initially empty, exact recency LRU was simulated over every draw. It admits complete waveforms and projects their bytes as `4*ceil(canonical_samples*24000/22050)`; actual24k lengths are not measured yet.

| Quantity | Projected result |
| --- | ---: |
| Fitting working set | 98,757,966,572 bytes |
| 4 GiB cache hits | 1,118 / 264,960 (0.422%) |
| Full-song misses | 263,842 |
| Cache writes | 4,145,741,670,292 bytes |
| Full audio decoded | 43,184,807 seconds |
| Requested12s audio | 3,179,520 seconds |
| Historical1905 CPU seconds scaled by decoded duration | 74,908 seconds |

Adding16KiB/file changes hits to1,117. Worker-prefetch duplication is omitted, so this is not a pessimistic multiworker simulation. The historical1905 CPU seconds include CQT/preparation and are **not a measurement of24k decoding CPU**. Scaling them is a warning against the previous60k-CPU whole-fit budget, not an estimate with validated error bounds. Model/augmentation/validation costs remain additional. The previous pilot I/O budget must be replaced after direct measurements.

## One fixed candidate, no adaptation

Full reference: installed FFmpeg6.1.1-3ubuntu5, binary SHA ed16af623947494a72e284b6eb8ff225f2da22b38b5d5069c2fd4b4ba3384e41, decode the first audio stream to mono24000Hz `pcm_f32le`. No source normalization, S16,22k intermediate or lossy waveform cache. Reference and candidate use identical channel/rate/output flags, one decode thread, one filter thread, and the SAME default recovery profile with no `-xerror`. Recoverable FFmpeg warnings occurred in the fixed corpus during prior full canonical decoding. Preserve per-full/per-crop stderr even on exit0. Such warnings alone prove neither failure nor completeness; finite/length/bit equality remains the qualification gate. Exact library closure is resolved with `ldd`, canonicalized, hashed and sealed before execution; missing libraries abort.

For desired global sample start `s`:

1. Anchor `a=max(0,floor(s/24000)-2)` **integer seconds** before `-i`, with accurate seeking enabled. When `a=0`, omit seek entirely.
2. Decode until `(s-a*24000)+crop_samples+24000` output samples: ≥2 and <3 seconds preroll away from the beginning, plus1 second postroll. Bound with output `-t`, applied after rate conversion. Host trim starts at exact `s-a*24000` and retains exact crop length.
3. Whole-second anchors align rational sample-rate phases for ordinary integer-Hz source rates. This is a hypothesis about the resampling/decoder pipeline, **not proof**: codec reservoir/overlap, encoder delay, timestamp rounding and filter state may still break equality. There is no automatic change to preroll/phase/command after observing mismatches.

FFmpeg's [primary documentation](https://ffmpeg.org/ffmpeg.html#Main-options) explains that input seeking reaches a prior seek point and accurate transcoding decodes/discards the intervening segment. That does not guarantee equality to full decoding after audio resampling. The fixed waveform test is therefore necessary.

Each of the original16 files is source-SHA checked once before decode. Full reference is decoded once, memory mapped, then removed after that file. Subsequent crop requests check unchanged `(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)` identity. Access time is excluded because our own reads can change it. Test starts are:0,1sample,1second+1sample, midpoint, last legal start−1sample, last legal start, plus3 seeded-uniform starts; deduplicate deterministically. Tracks shorter than12seconds use their full length, never fabricated samples. Training's future12s padding rule is outside this test.

Primary gate: **all compared float32 waveform bits equal at the predeclared global offsets**, same lengths and all finite. Report per case max absolute error, RMS error, mismatched samples and fraction; retain first failure reference/candidate crops. Signed-zero differences also fail. No lag correction, amplitude fitting, epsilon, alternate decoder, score gate or waveform quantization can repair this run. Any mismatch means this exact seek path is unqualified; a later source-grounded candidate needs a new protocol/source/window. No model inference is performed, even if the waveform gate fails.

## Resource and launch contract

Prospective sealing window:2026-11-12T11:00:00Z–11:15:00Z (900seconds), metadata sealing authorized; audio execution still requires final root approval of the sealed config SHA. Fixed Linux affinity8–9 and OMP/MKL/OPENBLAS/NUMEXPR threads1; no CUDA/Torch/model import. Aggregate process-tree CPU≤600seconds, summed sampled RSS≤2GiB. Owned output/scratch≤512MiB within a2GiB task allowance. First16 longest reference projects≈207MB; only one full reference exists at once. Two retained failure crops add≤2.3MB. Maximum single reference256MiB; candidate2MiB. Poll every50ms, kill the child process group if a cap or window expires; retain stderr/partial artifacts and exclusive output names. Sampled limits can overshoot between polls, and summed tree RSS counts shared pages conservatively. There are at most16 full decodes and144 cropped decodes. This is qualification, not a throughput claim; subprocess and instrumentation overhead will be recorded.

The original [draft](data/qualification-draft.json) records80d4574. The amended metadata-generated [draft](data/qualification-recovery-draft.json) binds source, first16 IDs/source SHAs, amended plan, supervisor source and requested CPU placement. It cannot run: runtime closure/window are intentionally unsealed. Once root reviews this source/protocol, `seal` adds the exact installed FFmpeg library hashes and the prospective UTC window, without decoding audio. Root then approves the resulting config SHA in a separate receipt `{ "authorized": true, "config_sha256": "..." }`. The run checks config/source/runtime identities before any audio decode. Never overwrite a prior attempt.

```sh
python -m tools.musicfm_crop_io seal --config qualification-draft.json \
  --begin ROOT_REVIEWED_ISO_UTC --deadline ROOT_REVIEWED_ISO_UTC --output sealed-config.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  taskset -c 8,9 python -m tools.musicfm_crop_io qualify \
  --config sealed-config.json --approval root-approval.json --output NEW_OWNED_STAGE
```

No16-file audio experiment has run under this protocol. After qualification, measured decode service time and cache-free crop cost can inform a revised finite pilot budget. A pass on16 files is not proof of corpus-wide equivalence, teacher accuracy, long-house acceptance or production readiness. Full-song validation and5–10min house-domain coverage remain mandatory and separate; the existing labelled corpus is insufficient to establish the latter.
