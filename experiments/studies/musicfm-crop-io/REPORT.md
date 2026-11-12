# Bounded MusicFM waveform crop feasibility

The fixed16-file qualification reproduced all144 crops bit for bit. **The subsequent approved-pool preparation failed at its2,289th file, fmak:046091, after2,288 prior files passed.** The current seek path is not qualified for the whole pool. The run stopped without retry, repair, dropped files or a complete validation cache. These are waveform-infrastructure results, with no model or accuracy scores.

## Problem

The proposed fitting cache held full24kHz float32 songs in4GiB. The exact amended8,280-update ×32-draw schedule exposes a98.76GB working set. Metadata simulation finds1,118 hits in264,960 requests (0.422%),263,842 full-song misses and4.15TB projected waveform-cache writes. The historical preparation cost scaled by decoded duration is approximately74,908 CPU seconds, already beyond the earlier60,000-CPU-second whole-pilot allowance. This is a warning proxy: historical preparation included CQT, and24kHz decode cost was not directly measured by that projection.

The [projection](data/lru-projection.json) preserves the original80d4574 source identity. It uses actual canonical22050Hz durations and the exact versioned sampler with the checked eligibility amendment: BP1229, expanded FMA4033, fixed nominal BP mass1230/2206. A16KiB per-file header sensitivity leaves the conclusion unchanged. There is no reason to assume useful full-song LRU reuse in the proposed draw stream.

## Method

### Fixed16 qualification

The prospective [protocol](PROTOCOL.md), amended source **6011bf7681f0831c4d354f9803797cc874103a9e**, [sealed config](data/runtime-raw.json.gz) SHA `30144ca9abf7a8258983856c13cad1be544acad4ad556269769039ec2033721b`, and separate root approval were fixed before any waveform outcome. The original first16 public files remained eligible after the canonical-identity amendment; no file was substituted. No labels, models or scores were used.

For each file, FFmpeg6.1.1 decoded one complete mono24000Hz `pcm_f32le` reference. For each crop, the candidate seeks to integer-second anchor `max(0,floor(start/24000)-2)`, decodes with2–<3 seconds of preroll and1 second of postroll, then host-trims at the exact intended sample offset. Start-of-file crops omit seeking. Source files were SHA-checked once before decoding and their device/inode/size/modification/change identities checked before subsequent crops. Access time was excluded. Full and crop decoding shared the default recovery profile without `-xerror`; all160 stderr files were retained, including successful commands. Recoverable warnings would not establish failure or completeness by themselves.

Each song had nine deterministic positions: beginning, one-sample and phase-sensitive offsets, midpoint, final legal starts, and three seeded-uniform starts. The observed membership had no duplicates:16×9=144 cases. The primary gate required equal length, finite values and **identical float32 bits** at the predetermined offset. No tolerance, alignment correction or source change was made after launch. All217 installed FFmpeg/runtime-library bindings were hashed and read back before execution.

One worker ran on Linux logical CPUs8–9, with OMP/MKL/OpenBLAS/NumExpr threads1 and CUDA masked. Polling enforced the earlier11:15UTC deadline,600 aggregate worker-tree CPU seconds,900 wall seconds,2GiB summed sampled RSS and512MiB owned scratch. The run started11:02:14UTC and finished11:02:32UTC. Both protected vLLM engines and their parents remained live with unchanged start identities before and after.

### Approved-pool expansion

The later frozen sources598e746 and9facd95 reused the byte-identical6011bf7 decoder/profile. The separately approved config SHA `b575ed66aab29acad8a6f421055198267554e4d36bbcc7bfe1c27b2a77d1c885` bound all5,638 approved IDs (5,262 fitting+376selection), numeric target export for identity checking only, and217 runtime-file hashes. Every file received one full24kF32 pass and one deterministic label-blind whole-track crop. Fitting references were discarded after their checks; only selection waveforms would be retained for a complete fixed cache. No validation waveform was reached before failure. No model, score, confirmation/calibration ID or private audio entered this run.

One sequential worker ran on CPUs8–9 with numeric threads1 and CUDA masked. The declared caps were6,000CPU seconds,7,200wall seconds,2GiB summed RSS,8GiB owned storage and256MiB per reference, inside11:30–13:30UTC. Recursive disk samples were cached for one second instead of walking growing directories every50ms; CPU/RSS/decoder-size checks remained frequent. The run stopped on its first strict waveform failure before any resource cutoff. Both protected vLLM engines and their parents retained their live start identities.

## Results

### Failed approved-pool expansion — current decision

![The requested crop fails; an array-only diagnostic shift also fails the exact-bit gate](corpus-failure.png)

Both FFmpeg commands for fmak:046091 exited0 and returned finite, equal-length12-second crops. Nevertheless all288,000 samples differed at the predetermined start. The gate remained unchanged, and the source/pipeline was not modified or resumed. No complete manifest or validation-cache receipt exists.

| Corpus preparation outcome | Observed result |
| --- | ---: |
| Planned files /processed files |5,638 /2,289|
| Passed prefix /first failed file |2,288 /fmak:046091|
| Unprocessed files /validation waveforms |3,349 /0|
| Failed crop maximum absolute /RMS error |2.02278948 /0.41690944|
| Terminal exit /wall /aggregate childCPU |1 /714.983s /671.239s|
| Sampled outer-tree peak RSS |136,564,736B (130.24MiB)|
| Retained owned bytes |49,084,818B|

The full decode reported `Header missing` and a packet-submission invalid-data error; the seek decode reported `Header missing`. Seven earlier passing files also had nonempty stderr. Warnings alone therefore neither identify an invalid crop nor establish completeness. Exact waveform comparison is what rejected this file.

A separately authorized **array-only** diagnostic read the preserved full/crop buffers; it did not invoke a decoder or a model. A fixed coarse normalized-correlation search followed by integer refinement found the best offset627samples earlier (−26.125ms), correlation0.99992573. Even at that diagnostic offset **all288,000 samples still differed**, maximum error0.03513932 and RMS0.00368060. No shifted crop was admitted. A damaged-frame/timestamp/resampling-phase mechanism is plausible; the exact causal decoder path has not been proven. The diagnostic took0.853wall/0.686CPU seconds and sampled peak RSS186,613,760B.

The passed prefix covered392,549.506 decoded audio seconds. Its full-decode phases summed520.179wall/504.229childCPU seconds; crop phases142.236wall/110.537childCPU seconds. These are instrumented sequential preparation measurements, not a representative random fitting-throughput benchmark. Full references and the failed crop pair remain on the server; no PCM is committed. [Lossless metadata evidence](data/corpus-failure-raw.json.gz), [original-file hashes](data/corpus-failure-index.json), [verified summary](data/corpus-failure-result.json), [failure figureSVG](corpus-failure.svg).

### Earlier fixed16 qualification — retained positive scope


![All144 crops match exactly; instrumented full and bounded-crop decode times by actual song duration](crop-qualification.png)

The figure describes this qualification run. A full decode and one12-second crop process different amounts of audio. Each command includes process launch and50ms supervisor polling; these measurements do not estimate population throughput or a whole training run. [SVG](crop-qualification.svg).

| Fixed outcome | Observed result |
| --- | ---: |
| Tracks / crop cases | 16 /144 |
| Exact finite/length/bit gate | 144 passed |
| Differing samples / maximum absolute error | 0 /0 |
| Exit / outer cutoff | 0 /none |
| Total supervisor wall | 18.220 seconds |
| Sampled worker-tree CPU | 15.27 seconds |
| Sampled peak summed worker-tree RSS | 102,916,096 bytes (98.15MiB) |
| Full decode commands, summed wall /child CPU | 6.342 /6.285 seconds |
| Crop decode commands, summed wall /child CPU | 10.465 /7.626 seconds |
| Retained stderr files /nonempty files | 160 /0 |
| Owned bytes after reference cleanup | 647,091 bytes |

A complete reference existed only for the current file and was removed after its nine comparisons. The longest reference was approximately207MB; no full-corpus waveform cache was built. Sampled RSS includes resident mapped pages, not the entire reference's on-disk extent. Caps are sampled; they can overshoot between polls. Per-command child CPU is measured around each decode; the outer worker-tree CPU is sampled and includes additional work. These quantities have different boundaries and should not be presented as identical.

The raw report SHA is `9afddd0a07f3cb4ec0b97041059845ff4e01bc295a1e1da127b424c4e0e33a66`. All original JSON,160 stderr files, approval, source-bound outer wrapper and process-identity receipts are preserved losslessly in [the deterministic gzip packet](data/runtime-raw.json.gz), with original decoded-byte hashes in [the index](data/runtime-index.json). [Verified results](data/verified-result.json) include per-track timing points. No PCM, compressed audio, labels or model weights are stored in this repository.

## Decision and limits

The seek candidate passes the earlier16 cases but fails the broader fixed preparation. It cannot become the default whole-pool fitting path or activate a training manifest. A separately reviewed robust I/O method is required. Full-song validation, uniform whole-track crop sampling, numeric precision, draw count and validation cadence remain unchanged. No real MusicFM fit is authorized by either stage.

Seeking remains codec/container/resampler dependent. These16 cases do not cover every eligible file or possible crop offset. Future fitting must establish each track's actual24kHz length and source identity once, without assuming a large waveform LRU will help. The measured crop commands still have process-start overhead; a persistent decoder would be a different implementation requiring its own evidence. Validation decoding, model computation and augmentations still need explicit costing.

The target acceptance domain is5–10-minute house/deep-house/melodic-house songs. The current labelled data do not provide a sufficient independent panel for that claim. Exact waveform reproduction is necessary infrastructure evidence, not evidence that a teacher improves key accuracy on that domain.

Regenerate and verify the figure without audio decoding:

```sh
python -m tools.musicfm_crop_io_report --study experiments/studies/musicfm-crop-io
python -m tools.musicfm_crop_failure_report --study experiments/studies/musicfm-crop-io
```

Use the repository's plot dependencies; this figure was generated with Matplotlib3.11.2. `6011bf7` remains the first-stage measured numerical source;598e746/9facd95 are the later failed-expansion sources; the report/verification tool is a later publication-only addition. The earlier80d4574 metadata evidence and prospective method remain separately identifiable.
