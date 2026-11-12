# Proposed Apple M5 nine-case identical-PCM portability diagnostic

Status: authorized execution window ends 2026-11-12 08:00 UTC, granted before outcomes. This new source identity preserves all historical windows. Exact-version preflight passed before processing.

## Purpose and fixed membership

Test native FMA magnitude + platform log1p on Apple M5/arm64 against BOTH the
canonical Linux reference and actual same-version Mac Python reference. Linux
FMA-only failed one of376 strict gates; it is not treated as qualified on Mac.
The x86 SVML library is never loaded, linked or forced onto arm64.

Use exactly the existing nine cases: six fixed synthetics plus public
FMAK110544/107354/087373, all on their existing canonical FP32 reference PCM.
Copy no private audio and perform no new decoding. Keep original model, class
order, FP16 roundtrip, parameters and finite/shape/logit/key gates unchanged.

## Minimal copied packet and budget

The already recorded manifest permits an exact transfer budget:

| Payload | Bytes |
| --- | ---: |
| Nine canonical PCM arrays |174,658,204 |
| Nine canonical FP32 core arrays |8,193,024 |
| Shared CSR arrays for one invariant plan |38,764 |
| All780 referenced nine-case artifacts, including intermediates |202,779,864 |
| Canonical manifest |457,526 |
| Authoritative full ONNX |463,507 |
| Verified python-soxr1.1.0 source tar |173,067 |

Prefer the complete referenced-artifact packet for direct diagnosis:
203,873,964bytes (~194.43MiB), plus small protocol/source/license records. These
counts include PCM/cores within the780 artifacts (do not add them twice).
No MP3 copying or corpus expansion. A minimal compute-only packet would be
~175MiB but saves little and loses useful stage probes.

Use a NEW task-local root, e.g. `/tmp/euphoria-native-cqt-mac-v1`, with a2GiB
storage stop including copied data, local builds, Mac references and logs. Keep
all copied data outside Git. Reuse the existing installed Rust toolchain and
existing ORT dylib rather than downloading ML runtimes/models again.

## Source and local setup before comparison

Require and record actual Mac Python versions: librosa1.0.0, NumPy2.5.4,
SciPy1.18.1, soxr1.1.0 and ORT1.31.1. If they differ, stop/report before processing;
any environment change needs root's reviewed plan. Record installed source/file
and extension hashes, NumPy git/build/runtime CPU dispatch and platform libm
identity. Do not assume Mac uses Linux's SVML path.

Use existing ORT1.31.1 dylib:
`/Users/lvndry/github/jazz-worktrees/drone-research-tools/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.31.1.dylib`.
Record its SHA, CPU provider/preference/arena and intra/inter1/sequential/no spin;
CoreML must not be selected. Full model SHA stays
`35fc1bfe5d67f7aed2c46221c15519d0f3d5822351932b5bd9899d47d48cb613`.

Build the exact verified soxr sdist locally, source SHA
`9f228ae21c78fa9359ca98d8a5e8e91f30639e438e574133dace62c5b5309e44`.
CMake shared Release, OpenMP off, LSR/examples/tests off, jobs2, no system install
or profile changes. Build a task-local Mach-O split-mono FLOAT32 HQ wrapper with
the same96,000-sample block/final-flush/ceil/scale semantics. Record source tree,
compiler/CMake/options, library/header/license hashes and `otool -L` resolution;
reject accidental Homebrew/system-soxr substitution. Python-soxr/libsoxr are
LGPL-2.1-or-later; preserve notices. Same source revision does not assert binary
identity across architectures or with the wheel.

Use one active reference/native process at a time, ORT and numeric threads1,
plus `VECLIB_MAXIMUM_THREADS=1` for Apple libraries. Build jobs≤2. macOS does not
offer the Linux affinity guarantee here: record that fact and do not make CPU
performance claims. The diagnostic is correctness-only.

## Fixed native plan policy

Observe actual Mac private calls on identical canonical PCM before native
comparison. Capture Mac kernels, normalization/scaling and resampling metadata;
verify observation leaves the original Mac feature output unchanged. Native
ARM uses this **Mac-actual captured plan**, fixed before its outputs are tested,
with FMA-ratio magnitude + platform log1p and the original FP16 boundary.

Record Mac-vs-Linux kernel/CSR/normalization hashes and structural parameters.
No silence-derived track statistics are reused. No choosing between Mac/Linux
plans based on predicted keys. Structural contract changes stop the stage;
last-bit constant differences are diagnosed and remain visible in canonical
comparison outcomes. No additional plan arm is introduced after seeing results.

## Pipelines and separately reported gates

For each of the nine cases produce these named outputs:

- **L:** archived canonical Linux Python CQT + Linux fullORT logits/core.
- **Mref:** actual Mac Python CQT on identical PCM + Mac CPU fullORT.
- **Mnat:** native Mac CQT on identical PCM + Mac CPU fullORT.
- **Mcore:** canonical Linux core fed directly to Mac CPU fullORT, isolating
  model/runtime drift from frontend drift (one additional cheap forward/case).

Report all comparisons: Mnat vs Mref (local frontend preservation), Mref vs L
(platform reference drift), Mnat vs L (canonical portability), Mcore vs L
(ORT-only diagnostic). Each uses finite outputs, identical shapes, every logit
`abs(delta) <= 1e-5 + 1e-5*abs(reference)` and zero argmax disagreements. Feature
max/RMS/FP16 mismatch counts are diagnostics; no tolerance tuning, bias or removal
of FP16. Store original scalar/ufunc arithmetic intermediates if a mismatch
requires attribution, bounded to the first failure and existing probes.

Run one observation/forward per pipeline/case; no repeated timing loop. Continue
collecting all nine bounded outcomes after numerical failures, preserving first
failure evidence, but never promote a failed gate. Both canonical and local
outcomes must be reported; matching Mac Python alone is not a Linux-canonical
parity claim. Passing all nine permits proposing376 Mac validation, not production
or universal portability. Any mismatch leads to a source-grounded mechanism
report and fixed next step, not an adjusted formula fitted to final keys.

## Requested resources and evidence

Cap the complete diagnostic at600 aggregate child CPU /900 wall seconds and2GiB
owned-tree RSS/storage stops, with deadline earlier if root's new window closes.
Compile/copy/setup is separately recorded; no throughput or memory-win claim.
Estimated copy is~194.43MiB; source/build/output overhead should remain well
below the2GiB cap. Supervisor accounting uses Darwin byte units for ru_maxrss,
not Linux kilobytes. Limit one active numerical worker and at most two build jobs.

Bind copied packet/file hashes, PCM/core/model/schema, actual Mac Python/source
versions and binaries, soxr build/wrapper, native source/archive/executable,
compiler/OS/AppleM5 identity, thread settings, canonical Linux manifest/source
refs and the new protocol/window. Keep Linux measured refs and every earlier
artifact/window immutable. Root reviews this protocol before any execution.

The Mac supervisor needs Darwin accounting rather than the Linux `/proc`
implementation: sample owned-tree RSS through `ps` (its KiB converted to bytes),
record Darwin `ru_maxrss` in bytes, aggregate child CPU and stop at the frozen
wall/deadline caps. Sampling is not an instantaneous kernel memory guarantee.
Use one numerical worker, `CARGO_BUILD_JOBS=1`, release codegen-units1/LTOoff for
ordinary compilation, and CMake jobs2. Do not equate build-job limits with a
hard physical-core affinity guarantee on macOS.
