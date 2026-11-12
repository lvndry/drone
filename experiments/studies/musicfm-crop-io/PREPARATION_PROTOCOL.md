# Prospective approved-pool24k preparation

**Historical prospective protocol.** The subsequently authorized preparation failed its strict waveform gate; [REPORT.md](REPORT.md) records the incomplete execution. At this protocol freeze no full preparation or GPU execution was authorized. The completed6011bf7 sixteen-file/144-case evidence, raw bytes and figures remain immutable. This follow-up does not relax the failed K10 two-point accuracy gate or consult confirmation/calibration/private audio.

## Fixed method

Use amended plan `c3e59ffb764ca5f51444447b0a18a69543228e60a9e3723d925ddd82b0fc8373`, numeric label export `b8942b5913b8ebacdf2fc9e0ebca72a934aaa7c56d4e1a97d0f60fac84354323` and5638-row prepared derivative `fcbc61081166f1845efe8af1ec2d7f807bb2d0a906dd93728b735e287c855291`. Membership is5262 fitting (1229BP+4033FMA),376 selection (126BP+250FMA); no other input is allowed. Numeric targets are consulted only for canonical-identity invariants, never scoring, model selection or repair. Preserve all same-role/same-target aliases as supplied; any new cross-role or conflicting-target24k identity stops this stage with evidence.

Reuse6011bf7 `tools/musicfm_crop_io.py` SHA `3efaf109c547acaf9479a40d23055680454234560eb17ffad762151a756114ac`: exact FFmpeg command/profile, stable input identity, FP32LE comparison and trimming geometry. Reuse existing `run_cqt_expansion.py` resource/hash helpers. The old bounded16-case function hardcodes512MiB scratch and900seconds; this larger preparation uses a small sequential driver with the SAME decode-command function and parameterized declared caps. It is not a fitting cache or generic data framework.

Every file gets exactly one full24kHz monoF32LE pass, source-SHA/stable-file-identity checks, exact raw-byte waveform SHA, actual sample count and chunked finite checks. Compare ONE deterministic uniform12-second start against that same reference, seeded only by ID and fixed string `musicfm24k-one-crop-v1`. Whole-track legal starts are equally eligible; the choice never sees targets/duration strata/model outputs. Short inputs use their complete available waveform for this infrastructure check; a future training padding rule must be frozen separately. Use the previously qualified integer-second anchor,2–<3s preroll,1s postroll and host sample trim. Gate remains finite, equal length and identicalF32 bits, including signed zero. No tolerance/offset/preroll/decoder change, retries or dropped files after outcomes.

Process fitting IDs first in sorted order, followed by sorted selection IDs. Fitting full references are discarded after their identity/crop checks. Validation full references are renamed into a deterministic immutable disk cache. Only after all5638 identity/crop gates pass and all376 validation files exist can a complete cache receipt and final manifest be written. Preserve full/crop stderr even on successful exit; warnings alone establish neither failure nor completeness. Preserve every per-file record; on first failure retain partial references/crops and stop. No silent resume.

## Cache size and resources

Measured22050 sample counts project24kF32 validation storage as `sum(4*ceil(samples*24000/22050))`:

| Input | Tracks | Projected bytes |
| --- | ---: | ---: |
| BP selection |126|1,451,436,776|
| FMA selection |250|5,189,600,180|
| Total |376|6,641,036,956 (6.185GiB)|

Actual24k decoded sample counts/bytes are authoritative; projection is not relabelled as measurement. Under8GiB total owned storage, projected headroom is1,948,897,636B. Longest existing canonical input projects206,892,204B; one reference is capped256MiB and one crop2MiB. Complete cache data plus transient reference, logs and manifests should fit, but the actual byte guard decides. No full-corpus fitting PCM cache is built. Later consumers must read/mmap a single song/window at a time; on-disk6.185GiB does not imply loading it all into host tensors.

Proposed caps: ONE sequential numeric worker, Linux logical CPUs8–9 only (root must reserve), all four numeric thread variables1, CUDA masked, aggregate process-tree CPU≤6000seconds, wall≤7200seconds, summed sampled RSS≤2GiB, total owned output≤8GiB. Poll CPU/RSS/decoder-target size every50ms. Cache recursive global disk usage and refresh it at an explicit1second requested cadence; report cached bytes and last-sample age. Reuse that sample for progress instead of walking all files again. Force actual disk measurements before publishing completion receipts and at the final boundary. Retain per-phase wall/childCPU and sampled worker resources. Source hashing/finite checking are also bracketed by resource guards. Caps are sampled, including conservative shared-page summation, so brief overshoot between checks is possible. An outer retained supervisor must enforce the final UTC cutoff across the worker and its descendant process groups. Proposed maximum window length is2hours; **no actual window is assigned yet**.

The tool generates a metadata-only draft with all5638 IDs/source SHAs/roles/targets, bound source and217 existing runtime-file hashes from config30144ca9. Before any launch root reviews source/config, grants a fresh time window, and approves the exact newly sealed config SHA in a separate receipt. Fresh runtime-closure rehash and source hashes must match before decoding. GPU and models remain unused. Label-bearing config/manifest stay task-local, not published as report data.

```sh
python -m tools.prepare_musicfm_waveforms plan \
  --plan APPROVED_PLAN --labels APPROVED_NUMERIC_LABELS \
  --manifest CHECKED_ELIGIBILITY_MANIFEST --runtime FROZEN_30144CA9_CONFIG \
  --output NEW_DRAFT
python -m tools.prepare_musicfm_waveforms seal --config NEW_DRAFT \
  --begin ROOT_REVIEWED_ISO_UTC --end ROOT_REVIEWED_ISO_UTC --output NEW_SEALED_CONFIG
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 8,9 \
  python -m tools.prepare_musicfm_waveforms run --config NEW_SEALED_CONFIG \
  --approval SEPARATE_ROOT_APPROVAL --output NEW_OWNED_STAGE
```

Passing these fixed per-file starts will extend source/length and sampled alignment evidence across the approved pool; it is not a proof about every possible offset. It supplies a bounded fitting direct-crop path and complete immutable validation disk cache for a later reviewed pilot. Whole-song coverage,120 validation events, draw count/nominal domain mass, FP32 waveform precision and primary-domain limitations remain unchanged.
