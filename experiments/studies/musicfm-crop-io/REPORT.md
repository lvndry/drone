# MusicFM waveform crops: a small qualification passes, the corpus expansion fails

**The input-seek path is not qualified for the approved corpus.** The first study
reproduced all 144 crops from 16 fitting files bit for bit. A later preparation
attempt covering 5,638 approved IDs stopped at file 2,289, `fmak:046091`, after
2,288 preceding files passed. It produced no complete manifest or validation cache.
No retry, dataset drop, alignment repair, tolerance change or model scoring followed.

## Problem

Random full-song crops need a feasible waveform path. The proposed 4 GiB LRU cache
would face a projected **98.76 GB fitting working set**. Replaying the exact amended
8,280-update × 32-draw sampler gives only 1,118 hits in 264,960 requests (0.422%),
263,842 full-song misses and 4.15 TB of waveform-cache writes. Adding a 16 KiB
per-file header leaves the conclusion unchanged. This is a serial metadata
simulation, not measured fitting throughput; it omits worker-prefetch duplication.

The [projection](data/lru-projection.json) uses canonical 22,050 Hz sample counts,
the pinned sampler, BP 1,229 / expanded FMA 4,033 fitting IDs and unchanged nominal
BP mass 1230/2206. Scaling historical preparation cost by decoded duration gives
approximately 74,908 CPU seconds, beyond the earlier 60,000-second whole-pilot
allowance. That warning proxy includes historical CQT work and is **not** a measured
24 kHz decoding estimate. Original projection source `80d4574` remains distinct
from the later decoder and report sources.

## Method fixed before waveform outcomes

### Initial 16-file qualification

The [historical protocol](PROTOCOL.md), decoder source
`6011bf7681f0831c4d354f9803797cc874103a9e`, sealed configuration
`30144ca9abf7a8258983856c13cad1be544acad4ad556269769039ec2033721b` and separate
root approval were fixed before execution. The original 16 fitting files remained
eligible after the canonical-identity amendment; none was substituted. No labels,
models or scores selected files or crop offsets.

FFmpeg 6.1.1 decoded each complete source to mono 24 kHz float32 little-endian
samples. The candidate seeks to an integer-second anchor two seconds before the
requested start, retains 2–<3 seconds of preroll and one second of postroll, then
trims at the exact host-side sample offset. Beginning crops omit seeking. Full and
crop commands use the same default recovery profile, without `-xerror`. Source
SHA and stable file identities were checked; 217 runtime/library bindings were
hashed and read back. Every stderr file was retained, including successful calls.

Nine deterministic positions per file cover the beginning, one-sample and
phase-sensitive offsets, midpoint, last legal starts and three seeded-uniform
starts: 144 distinct cases. The gate requires equal length, finite values and
identical float32 bits, including signed zero, at the predeclared offset. There
is no lag correction, amplitude fitting or tolerance-based rescue.

One worker used CPUs 8–9, numeric-library threads set to one and CUDA masked.
Caps were 600 aggregate worker-tree CPU seconds, 900 wall seconds, 2 GiB sampled
summed RSS and 512 MiB owned scratch, with an 11:15 UTC deadline. The run completed
11:02:14–11:02:32 UTC. Both protected vLLM parents and EngineCore start identities
remained live. These are sampled resource limits, not guarantees against brief
overshoot between polls.

### Approved-pool expansion

Frozen preparation sources `598e746` and `9facd95` reused the byte-identical
`6011bf7` full/crop decoder. The [preparation protocol](PREPARATION_PROTOCOL.md)
and separately approved configuration
`b575ed66aab29acad8a6f421055198267554e4d36bbcc7bfe1c27b2a77d1c885` bind all
5,638 IDs: 5,262 fitting and 376 selection. Numeric targets support input-identity
checks only. Each file receives one full decode and one deterministic, label-blind
whole-track crop comparison. Successful fitting references are discarded; selection
waveforms would be retained only as part of a complete cache.

One sequential worker used CPUs 8–9, single numeric threads and CUDA masked.
Caps were 6,000 CPU seconds, 7,200 wall seconds, 2 GiB summed RSS, 8 GiB owned
storage and 256 MiB per reference, within 11:30–13:30 UTC. Recursive disk samples
were cached for one second; CPU, RSS and decoder-size checks remained frequent.
The strict waveform gate stopped execution before any resource cutoff. No selection
waveform was reached, and no calibration, confirmation or private input participated.
Protected vLLM identities remained unchanged.

## Current result: the corpus expansion failed

![The requested crop fails; an array-only diagnostic shift also fails the exact-bit gate](corpus-failure.png)

| Preparation outcome | Observed result |
|---|---:|
| Planned / processed files | 5,638 / 2,289 |
| Passed prefix / first failed file | 2,288 / `fmak:046091` |
| Unprocessed files / validation waveforms | 3,349 / 0 |
| Failed crop maximum absolute / RMS error | 2.02278948 / 0.41690944 |
| Terminal exit / wall / aggregate child CPU | 1 / 714.983 s / 671.239 s |
| Sampled outer-tree peak RSS | 136,564,736 bytes (130.24 MiB) |
| Retained owned bytes | 49,084,818 |

Both commands exited zero and returned finite, equal-length 12-second crops, yet
**all 288,000 samples differed** at the requested offset. Full decoding logged
`Header missing` and a packet-submission invalid-data error; seeking logged
`Header missing`. Seven earlier passing files also had nonempty stderr. Warnings
alone therefore neither identify a failed crop nor establish source completeness.
The exact waveform comparison, not the presence of a warning, rejected the path.

A separately authorized **array-only diagnostic** read the preserved buffers,
without a new decoder or model invocation. Its fixed correlation search and integer
refinement found the best offset 627 samples earlier (−26.125 ms), correlation
0.99992573. Even there, **all 288,000 samples differed**, with maximum absolute error
0.03513932 and RMS error 0.00368060. No shifted crop was admitted. Damaged-frame,
timestamp or resampling-phase behavior is plausible; the exact cause is unproven.
The diagnostic used 0.853 wall / 0.686 CPU seconds and sampled peak RSS 186,613,760
bytes. Its source and wrapper identities are separate from the preparation run.

The passed prefix covered 392,549.506 decoded audio seconds. Full-decode phases
summed 520.179 wall / 504.229 child CPU seconds; crop phases summed 142.236 wall /
110.537 child CPU seconds. These instrumented preparation measurements do not
establish population or random fitting throughput. **The failed full reference
and crop pair remain on the server**; successful fitting references were discarded.
No PCM is committed. The [lossless metadata packet](data/corpus-failure-raw.json.gz),
[original-byte index](data/corpus-failure-index.json.gz) and
[verified summary](data/corpus-failure-result.json) preserve the incomplete attempt.

## Earlier positive scope: 16 files, 144 exact crops

![All 144 fixed crops match; full and bounded-crop decode times by actual song duration](crop-qualification.png)

| Initial qualification | Observed result |
|---|---:|
| Tracks / cases / exact passes | 16 / 144 / 144 |
| Differing samples / maximum absolute error | 0 / 0 |
| Exit / cutoff | 0 / none |
| Supervisor wall / sampled worker-tree CPU | 18.220 s / 15.27 s |
| Sampled peak summed worker-tree RSS | 102,916,096 bytes (98.15 MiB) |
| Full commands, summed wall / child CPU | 6.342 s / 6.285 s |
| Crop commands, summed wall / child CPU | 10.465 s / 7.626 s |
| Retained stderr files / nonempty files | 160 / 0 |
| Owned bytes after reference cleanup | 647,091 |

A full decode and a 12-second crop process different amounts of audio. Timing
includes subprocess launch and 50 ms polling; the graph is a qualification result,
not a representative speed benchmark. The longest temporary reference was about
207 MB, with only one full reference present at a time. Sampled RSS includes
resident mapped pages rather than the full on-disk extent. Outer sampled CPU and
per-command child CPU have different accounting boundaries.

The original report SHA is
`9afddd0a07f3cb4ec0b97041059845ff4e01bc295a1e1da127b424c4e0e33a66`.
The [deterministic packet](data/runtime-raw.json.gz), [index](data/runtime-index.json)
and [verified result](data/verified-result.json) preserve raw JSON, stderr, approval,
wrapper and service-identity receipts. The later failure packet additionally contains
public-corpus numeric target fields used for source-integrity checks; it contains
no model scores. Neither packet contains PCM, compressed audio or model weights.

## Decision and next boundary

The 16-file pass does not extend to the whole pool. The seek candidate cannot
activate a training manifest. Preserve the failure; do not drop `fmak:046091`,
change offsets, loosen the bit gate or retry under the same protocol.

The concrete next design is one full decode per fitting file into an explicitly
owned, read-only RAM-backed float32 cache, with selection waveforms on ordinary
disk and exact sample-index crops. This avoids restarting the decoder/resampler
at a seek point. Capacity forecasts and continuous global-memory, tmpfs, disk and
protected-service guards need separate source/config review and authorization.
This report contains no new cache allocation or future fitting implementation.
Source completeness and decoder warnings remain separate input-quality questions.

Full-song validation, uniform whole-track crop sampling, numeric precision, draw
count and selection cadence remain fixed. The intended acceptance domain is
**5–10 minute house, deep-house and melodic-house tracks**. Current labels do not
provide a sufficient independent panel for that claim. Waveform correctness is
infrastructure evidence, not evidence of improved key accuracy or a qualified teacher.

Regenerate both figures and verify the archived evidence without decoding audio:

```sh
python -m tools.musicfm_crop_io_report --study experiments/studies/musicfm-crop-io
python -m tools.musicfm_crop_failure_report --study experiments/studies/musicfm-crop-io
```

Matplotlib 3.11.2 generated the figures. Original source refs, the separate wrapper
and diagnostic identities, and checksums are recorded in
[publication provenance](data/publication-provenance.json) and [SHA256SUMS](SHA256SUMS).
Publication edits do not alter measured numerical sources or raw evidence.
