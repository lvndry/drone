# Bounded MusicFM waveform crop feasibility

The tested random-access path reproduced **all144 crops from16 public fitting tracks bit for bit** against full-track mono24kHz float32 decoding. This removes a concrete I/O blocker for the next pilot design. It does not establish corpus-wide waveform equivalence, a training budget, MusicFM accuracy or acceptance on full-length house music.

## Problem

The proposed fitting cache held full24kHz float32 songs in4GiB. The exact amended8,280-update ×32-draw schedule exposes a98.76GB working set. Metadata simulation finds1,118 hits in264,960 requests (0.422%),263,842 full-song misses and4.15TB projected waveform-cache writes. The historical preparation cost scaled by decoded duration is approximately74,908 CPU seconds, already beyond the earlier60,000-CPU-second whole-pilot allowance. This is a warning proxy: historical preparation included CQT, and24kHz decode cost was not directly measured by that projection.

The [projection](data/lru-projection.json) preserves the original80d4574 source identity. It uses actual canonical22050Hz durations and the exact versioned sampler with the checked eligibility amendment: BP1229, expanded FMA4033, fixed nominal BP mass1230/2206. A16KiB per-file header sensitivity leaves the conclusion unchanged. There is no reason to assume useful full-song LRU reuse in the proposed draw stream.

## Method

The prospective [protocol](PROTOCOL.md), amended source **6011bf7681f0831c4d354f9803797cc874103a9e**, [sealed config](data/runtime-raw.json.gz) SHA `30144ca9abf7a8258983856c13cad1be544acad4ad556269769039ec2033721b`, and separate root approval were fixed before any waveform outcome. The original first16 public files remained eligible after the canonical-identity amendment; no file was substituted. No labels, models or scores were used.

For each file, FFmpeg6.1.1 decoded one complete mono24000Hz `pcm_f32le` reference. For each crop, the candidate seeks to integer-second anchor `max(0,floor(start/24000)-2)`, decodes with2–<3 seconds of preroll and1 second of postroll, then host-trims at the exact intended sample offset. Start-of-file crops omit seeking. Source files were SHA-checked once before decoding and their device/inode/size/modification/change identities checked before subsequent crops. Access time was excluded. Full and crop decoding shared the default recovery profile without `-xerror`; all160 stderr files were retained, including successful commands. Recoverable warnings would not establish failure or completeness by themselves.

Each song had nine deterministic positions: beginning, one-sample and phase-sensitive offsets, midpoint, final legal starts, and three seeded-uniform starts. The observed membership had no duplicates:16×9=144 cases. The primary gate required equal length, finite values and **identical float32 bits** at the predetermined offset. No tolerance, alignment correction or source change was made after launch. All217 installed FFmpeg/runtime-library bindings were hashed and read back before execution.

One worker ran on Linux logical CPUs8–9, with OMP/MKL/OpenBLAS/NumExpr threads1 and CUDA masked. Polling enforced the earlier11:15UTC deadline,600 aggregate worker-tree CPU seconds,900 wall seconds,2GiB summed sampled RSS and512MiB owned scratch. The run started11:02:14UTC and finished11:02:32UTC. Both protected vLLM engines and their parents remained live with unchanged start identities before and after.

## Results

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

The fixed seek candidate passes these16 files and can be considered for a separately reviewed corpus qualification and revised pilot I/O budget. Full-song validation, uniform whole-track crop sampling, numeric precision, draw count and validation cadence remain unchanged. No real MusicFM fit is authorized by this result.

Seeking remains codec/container/resampler dependent. These16 cases do not cover every eligible file or possible crop offset. Future fitting must establish each track's actual24kHz length and source identity once, without assuming a large waveform LRU will help. The measured crop commands still have process-start overhead; a persistent decoder would be a different implementation requiring its own evidence. Validation decoding, model computation and augmentations still need explicit costing.

The target acceptance domain is5–10-minute house/deep-house/melodic-house songs. The current labelled data do not provide a sufficient independent panel for that claim. Exact waveform reproduction is necessary infrastructure evidence, not evidence that a teacher improves key accuracy on that domain.

Regenerate and verify the figure without audio decoding:

```sh
python -m tools.musicfm_crop_io_report --study experiments/studies/musicfm-crop-io
```

Use the repository's plot dependencies; this figure was generated with Matplotlib3.11.2. `6011bf7` remains the measured numerical source; the report/verification tool is a later publication-only addition. The earlier80d4574 metadata evidence and prospective method remain separately identifiable.
