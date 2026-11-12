# Fixed Mac frontend boundary trace: prospective source proposal

This is source preparation, not a completed diagnostic or a new numerical
candidate. The existing native/Mac strict gate remains failed at 375/376; the
Mac/Linux frontend comparison remains 348/376. All recorded predicted keys agree.
No tolerance, formula, backend, model, or release decision changes here.

The one case is index 0308, `fmak:091711`, 399 CQT frames. Its nonwinning G-major
component differed by 1.5899539e-5, exceeding its original 1.2216097e-5 allowance.
Nine of 76,608 native/Mac FP16-roundtrip features differed (maximum 0.0009765625).
These observations do not identify the responsible operation.

## Question and retained evidence

Locate the first *observed boundary* at which the fixed Python and Rust paths
differ: stage input → complex64 STFT before CSR → complex64 CSR response →
recursive resample → stacked response → normalization → magnitude → log1p →
FP16 roundtrip. An earlier differing boundary is evidence about where divergence
has already occurred, not a proof of which internal instruction caused it.

All nine CSR banks already have identical hashes across Mac/Linux. Only three
float64 normalization lengths differ by one ULP between hosts. In the failed
native/Mac comparison, **both implementations used the captured Mac lengths**;
those three cross-host input differences cannot alone explain this pair. Native
component division uses float64 before casting to float32; a hypothetical shared
float32 denominator is not the executed contract.

The surviving `0308.plan/mac-manifest.json` and all 27 CSR artifacts are sufficient
to admit the original plan. Target PCM and full/intermediate feature arrays were
not retained. Recovering the PCM is a separate, bounded step requiring review;
this proposal neither decodes nor transfers it. Original cases/receipts and
binaries stay immutable. The read-only inventory was saved in
`/tmp/drone-mac-portability-revisit.md` (SHA256
`15a784bd440357dd98d8e89854cc38243316919ca7d08ad931010008ffdb8e1a`).

## Fixed inputs and admission

| Artifact | SHA256 |
| --- | --- |
| Original single-case Mac manifest | `462205de24ff451e040afe45a3409cfc3fa9b74afc4f37d1725024cee59dc1fc` |
| PCM: 1,633,354 little-endian float32 samples, 6,533,416 bytes | `0b1234cb5e3c1ec2573d379660d7078ee7bffb1db029b87ffa58605836e2dde0` |
| Archived Python core, 192 × 399 float32 after FP16 roundtrip | `4024b6238ef874f7f6e44a3355f384b4d2bdc881ada425d544107a5f7516023e` |
| Archived native core | `6b6efa2647bcbb63e1f0d0a3d96816ed5c43662d087795393437b02cbf46c390` |
| Original Python runtime receipt | `cda8d97ed71069866768ef0de37296f5bdc3492f49e605f1c62aca3a211d4ea6` |
| Exact surviving old native executable | `9a5b7fc9887e7cf160a95614d26b86f2e65df98d837260177c16ca95841893ad` |
| Exact Soxr split wrapper | `49c7903d21711aff76afaaef122f7e71120f901427acb314f721f5b663b53880` |
| Exact task-local Soxr shared library | `b98d5dcb7214509496968e72f42b1fa18a3365dce251f6eb56331c0c7f618445` |

The source is the existing public file `091711.mp3`, recorded SHA256
`fca8f31887e9306268457b42e54976e1d61bb1e5333456fe10e5c1a4951c7ee9`.
A later PCM recovery must reproduce the fixed PCM hash exactly, using the
archived producer. A hash mismatch stops admission; it does not permit alignment,
resampling, clipping, source replacement, or a different case.

Both trace executables require the hash-bound runtime proof below. Python also
checks its *actual imported* source/extension paths, original versions and all
applicable historical file pins. Each final core must reproduce its own archived
hash. A changed final hash preserves the trace as a failed instrumentation/runtime
reproduction; it cannot be used to explain the historical counterexample.

## Observation paths and complete capture

[`tools/cqt_boundary_trace.py`](../../../tools/cqt_boundary_trace.py) reuses
`tools.export_cqt_goldens.Trace`, which observes actual Librosa private calls.
It records full arrays in addition to all 27 response and 21 resample probes.
Module-local proxies capture the actual normalization square root and final
NumPy `abs`/`log1p` outputs; a wrapper captures `__trim_stack` before in-place
normalization. Every hook returns the original result object unchanged. Original
module attributes are restored even on failure. It verifies newly observed CSR
hashes and stage geometry against the original manifest.

[`src/bin/boundary_trace.rs`](../src/bin/boundary_trace.rs) is a separate
**frontend-only binary**, not the inference `pipeline` entry point. It uses the
existing `PreparedPlan`/Soxr path, with additional observation buffers enabled only
by `PreparedPlan::trace`. It copies the existing complex64 FFT spectrum before the
unchanged ordered CSR sum, and the existing normalization/magnitude/log1p values.
It fixes the historical Mac FMA-ratio magnitude plus platform `ln_1p` path; no SVML
or alternative math option is exposed. Ordinary `compute` and existing diagnostic
calls request no new capture buffers. Frozen executables are never overwritten.

Both paths retain:

- Nine stage inputs, nine complete pre-CSR complex64 STFTs, nine complete CSR
  responses, and seven complete recursively resampled float32 arrays.
- All first/middle/final response probes (16 frames each) and resample probes
  (256 samples each): 27 and 21 respectively.
- Core-bin stacked complex64 values before normalization, actual float64 square
  root scales, normalized complex64 values, float32 magnitude and log1p, final
  FP16 bits, and their widened float32 core. Python also records original lengths;
  native lengths remain hash-bound in the input plan.

Blobs are immutable, little-endian, C-order, content-addressed files
`objects/<sha256>`; metadata includes shape, dtype, byte count and hash. Exact
repeat observations share a blob. Both implementations write sequentially into
one owned output root, retaining complete arrays rather than substituting probes.
The calculated arrays plus PCM are approximately 38 MB; the prospective **64 MiB
whole-output cap**, including receipts/JIT cache, is an admission constraint, not
a measured peak. Writers reserve 256 KiB for final JSON and check existing owned
file sizes before new blobs. An external supervisor must enforce the whole-tree
cap during execution as well. No raw audio/features are added to Git.

No ORT session, checkpoint, key head, logits or label is needed or loaded. The
new bin has no inference arguments/API calls; Python rejects a process with Torch
or ORT imported. The original JSON manifest contains historical model metadata,
but the trace does not use it for computation.

## Runtime proof and known gaps

The retired Python launcher was under the deleted `jazz-worktrees` directory.
The surviving primary Drone venv matches the recorded Python executable SHA256
`cf450e6bc0b00adecd12b7b13024de7000c7350801addc802bd3b45782104e79`, all ten original
source-file pins and all five extension pins. The ORT extension is **not imported**
for this trace. Versions are Librosa 1.0.0, NumPy 2.5.4, SciPy 1.18.1 and Soxr
1.1.0; current Numba/llvmlite are 0.68.1/0.50.0. Version equality alone is
insufficient.

In particular, actual Librosa STFT dispatches through SciPy FFT. The surviving
SciPy uses **DUCC**. The old receipt pinned NumPy PocketFFT and SciPy sparse, but
not the actual SciPy FFT backend binary. Original JIT-compiled code, system-math
closure and per-worker RustFFT dispatch were also not pinned. These are explicit
historical gaps even if the original core hashes reproduce.

A future immutable `cqt-boundary-runtime-proof-v1` JSON must contain:

- `trace_source_sha256` for the actual Python trace source and `trace_support`
  (`path`/`sha256`) for the actual imported `tools/export_cqt_goldens.py`.
- `historical_runtime`: path to the original receipt with the fixed hash above;
  `source_paths`: current paths keyed by all ten historical source names.
- `current_modules`: actual `path`/`sha256` records, including
  `scipy.fft._basic_backend`, `scipy.fft._duccfft.basic`,
  `scipy.fft._duccfft.pyduccfft`, `numba`, `numba._helperlib`, `llvmlite`.
  This module set is exact; unrelated modules cannot be injected through it.
  Add relevant CPU-dispatch/build configuration receipts and versions to this
  reviewed packet; no declaration that they were historically pinned.
- Nonempty `jit_libraries` (`path`/`sha256`), including the actual LLVM library;
  `dispatch_environment` binding numeric threads/NumPy/Numba controls. OMP, MKL,
  OpenBLAS, NumExpr and Accelerate thread limits must explicitly be one. Any newly
  generated JIT files must be task-local, counted under the output cap and
  inventoried after execution, without changing JIT/FFT policy to force parity.
- `native_files`: role-keyed `path`/`sha256` records for `executable`,
  `trace_source`, `plan_source`, `cargo_lock`, `rustc`, `cargo`, `soxr_wrapper`,
  `soxr_library`, `soxr_wrapper_source`, `soxr_source_receipt`. The new executable
  checks its own binary and compiled source hashes. Pin build flags/compiler
  output/target and the wrapper's actual resolved dylib dependencies separately;
  merely hashing an unused library path does not prove what dyld loaded.
- `historical_gaps` exactly `['scipy_fft_backend', 'jit_runtime', 'system_math',
  'rustfft_dispatch']`. This field records unresolved history, not waived proof.

The exact old native binary, archived source/Cargo.lock, wrapper and Soxr library
survive. Apple Clang 21 shared/OpenMP-off/PFFFT-on Soxr build receipts and Cargo
fingerprints survive; original compiler executable hashes/per-worker dispatch do
not. A new trace binary necessarily needs a **separate bounded build review and
new identity**; matching old math source is not an assertion of bitwise build
identity. The original binary has no safe frontend-only trace entry point.

## Fixed prospective execution and interpretation

No execution window or runtime proof has been sealed here. Proposed trace envelope:
one local worker at a time, all numeric threads one, at most 120 aggregate CPU
seconds, 180 wall seconds, 512 MiB process-tree RSS and 64 MiB owned output,
including task-local caches. Build/metadata/PCM-recovery preparation needs separate
finite resource review; no Cargo build or frontend transform was run to prepare this
proposal. Old case supervision took 3.42 CPU/3.40 wall seconds with a 252.2 MiB
worker peak, including inference and extra feature work; that is only a sizing
anchor, not a new timing forecast or ordinary-PC deployment qualification.

After reviewed build/runtime proof/PCM recovery, proposed invocations are:

```sh
python -m tools.cqt_boundary_trace --manifest "$CASE/mac-manifest.json" \
  --output "$OUT" --runtime-proof "$PROOF" --runtime-proof-sha256 "$PROOF_SHA"
boundary_trace --manifest "$CASE/mac-manifest.json" --output "$OUT" \
  --soxr-wrapper "$WRAPPER" --soxr-library "$SOXR" \
  --runtime-proof "$PROOF" --runtime-proof-sha256 "$PROOF_SHA"
```

These are command templates, not a launcher or an authorization file. Future
supervision must retain failures/partial arrays and enforce the fixed caps and
cutoff. No automatic retry, alternative case, formula adjustment, tolerance
change, inference rerun or 376-case expansion follows.

Read retained arrays only after both archived-core guards pass. Compare bytes and
per-stage errors, identifying the earliest differing stage input or pre-CSR STFT.
Equal STFT with differing response narrows attention to sparse arithmetic; equal
responses with differing resample output narrows it to the recursive resampler;
equal stacked values with differing terminal arrays isolates the corresponding
observed operation. Multiple differences can coexist. No change should be chosen
from the winning key or the failed logit alone, and none is implemented here.
