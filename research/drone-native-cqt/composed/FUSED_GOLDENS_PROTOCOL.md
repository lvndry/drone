# Prospective validation of the fixed nine retained golden cases

The single-case fused-product candidate matched all archived Python core values.
This follow-up asks whether the **same arithmetic** transfers to the complete,
previously retained nine-case Mac golden set. It is source/packet preparation,
not an assertion that another build or numerical execution has occurred.

## Frozen membership and comparisons

Use the original Mac reference manifest, in its existing order: silence,
impulses, tone, major chord, minor chord, noise, `fmak-110544`, `fmak-107354`,
`fmak-087373`. The three recordings contain 40.12, 240.40 and 1,682.26 seconds
of PCM; the six synthetic signals span 1.30–5.76 seconds. The manifest identity,
all PCM/core/CSR identities and each original native core identity are bound
before launch. The entrypoint hard-codes membership, order, manifest identity
and original native control identities. No new audio, resampling, reference
regeneration, classifier, label, or genre inference belongs to this stage.

For each case, sequentially:

1. Read its retained PCM and Mac Python reference core and load its captured plan.
2. Compute the ordinary separate-product control and retain its complete core
   and comparison. It must reproduce that case's original native core exactly.
3. If the control matches, compute the same explicitly named fused-product
   candidate used in the successful single-case experiment. Retain its complete
   core and bitwise comparison with the archived Mac reference. A failed control
   withholds only that case's candidate and remains a failed case.
4. Retain every case result, including mismatches. Continue the frozen list
   without replacing a case or searching arithmetic paths; resource or artifact
   admission failures stop the run and preserve partial output.

The verdict is **all nine original controls reproduced and all nine candidate
cores byte-identical to their respective archived Mac Python references**.
Different-value counts, first coordinates and maximum absolute errors explain
failures; none substitutes for exact identity. No Linux reference is substituted
for a Mac mismatch. The original native default, resampler, FFT, accumulation
order, magnitude, logarithm, normalization and FP16 boundary remain unchanged.

## Separate build and runtime admissions

The new `fused_golden_cases` entrypoint requires a new offline build and new
`cqt-fused-goldens-runtime-proof-v1`. Retain immutable prior executables and
source snapshots. Freeze both crate source closures and lockfiles, the exact
existing Cargo cache entries, toolchain/SDK inputs and the new executable.
Use a new task-owned target/cache and the already qualified guardian; no package
acquisition, compiler flags, inherited loader overrides, automatic retry or cap
extension. The build command is `cargo build --offline --locked --release
--jobs 1 --manifest-path "$SOURCE/research/drone-native-cqt/Cargo.toml"
--bin fused_golden_cases`. Cache copy, build and loader inspection share the
existing build envelope: 360 seconds wall, 300 aggregate CPU seconds, 1 GiB
sampled process-tree RSS and 512 MiB whole owned root. No frontend runs in that
build stage. The unchanged product regression already passed in the prior build.

After separate review, the runtime proof binds the new executable, compiled
entrypoint/plan/lockfile, toolchain, exact wrapper/library/source receipt and
actual loaded Soxr identities under the clean environment. Preserve the four
historical runtime gaps. The executable requires its compiled source identities
and the original fixed input/control identities; the guardian verifies all
external inputs before and after launch.

Runtime bounds: **120 seconds wall, 90 aggregate CPU seconds, 1 GiB sampled
process-tree RSS, 64 MiB whole owned output/log root**. Existing inputs remain
read-only, outside the output root; loaded inputs count toward process RSS.
Use one worker, all numeric thread limits one, nice priority 19, one case's
buffers at a time. Retain baseline and candidate cores plus per-case and final
reports. Guardian accounting includes descendants and reaped-child CPU, sampled
RSS, disk reserve, post-exit limits and identity-safe process retirement. No
attempt to reduce evidence or drop the long case after seeing its cost.

Passing establishes only these nine frontend fixtures on this runtime. It is
not a classifier score, genre-balanced validation, ordinary-PC performance
qualification, full 376-recording pass or cross-platform promotion. Any wider
corpus/model/timing step requires its own fixed packet after this result.
