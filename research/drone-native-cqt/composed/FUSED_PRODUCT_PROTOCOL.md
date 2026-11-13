# Prospective fused-product candidate: build, then one fixed core comparison

This is source preparation. The ordinary `PreparedPlan::compute`, existing
diagnostics and boundary trace retain separate component products. Only the
explicit `research_fused_product_core` method selects the new product rule:

```text
cross_re = round32((-ai) * bi)
cross_im = round32(ai * br)
product_re = fused_round32(ar * br + cross_re)
product_im = fused_round32(ar * bi + cross_im)
sum_re = round32(sum_re + product_re)
sum_im = round32(sum_im + product_im)
```

This does not use `Complex::mul_add`, which can fuse the accumulator into a
product. It changes no CSR order, FFT, resampler, normalization, magnitude,
logarithm, FP16 boundary, model, tolerance or math backend. The candidate fixes the
existing FMA magnitude/platform `ln_1p` path and offers no backend selection.

## Stage A: bounded offline build and synthetic regression

Freeze the committed worktree, both local crate manifests, lockfiles, build
scripts and Rust files. Use a new task-owned build root and private Cargo cache;
old binaries, caches and source snapshots remain read-only. Copy only the exact
lockfile's already cached registry packages, archives and sparse-index records,
checking archive hashes and all copied bytes. No package acquisition or shared
target directory is permitted. Retain exact compiler/linker/stdlib identities,
clean environment, argv, source snapshot and immutable logs in the private packet.

The reviewed commands, with paths resolved by the private build packet, are:

```sh
cargo build --offline --locked --release --jobs 1 \
  --manifest-path "$CANDIDATE_SOURCE/research/drone-native-cqt/Cargo.toml" \
  --bin fused_product_case
cargo test --offline --locked --release --jobs 1 \
  --manifest-path "$CANDIDATE_SOURCE/research/drone-native-cqt/Cargo.toml" --lib \
  plan::tests::synthetic_complex_product_rounding_preserves_separate_accumulation \
  -- --exact
```

The single synthetic regression uses analytical cancellation identities, including
the exact residual of `(1+2^-23)(1-2^-23)-1`, to distinguish separate products,
fused products and accidental fused accumulation. It does not read audio, load a
model or run a CQT. Other numerical tests are outside this bounded launch.

Use one Cargo job, numeric thread limits one, nice priority 19, no incremental
build, and explicit offline mode. No inherited compiler wrappers, custom Rust
flags, owner-snapshot option, Python overrides or loader injection. Preserve the
existing HOME value; task caches, target, temporary files and logs live under the
new owned root. The existing ORT dependency is load-dynamic; this entrypoint
constructs no ORT session and requires no model or ORT shared library.

Bounds for cache preparation, build and synthetic test: **360 seconds wall,
300 aggregate CPU seconds, 1 GiB sampled process-tree RSS, 512 MiB whole owned
root**, with copied cache below 128 MiB at admission. Reuse the previously
qualified process-tree supervisor, including reaped-child CPU, PID start identity,
post-exit limits and complete process retirement. No automatic retry or cap
extension. Build success establishes source/binary identity and the synthetic
regression only. Inspect final dependencies; preserve or separately verify the
actual loaded Soxr wrapper/library identities without calling a transform.

## Stage B: separate admission for one fixed frontend comparison

Only after the build is reviewed, prepare `cqt-fused-product-runtime-proof-v1`.
Its `native_files` map binds `executable`, `candidate_source`, `plan_source`,
`cargo_lock`, `rustc`, `cargo`, `soxr_wrapper`, `soxr_library`,
`soxr_wrapper_source` and `soxr_source_receipt` with actual paths and hashes.
The executable verifies its own bytes and compiled source/lockfile identities.
The external packet also retains build flags and actual loader evidence. Preserve
the four previously declared historical runtime gaps; a new build does not erase
them.

The only admitted case is the original `fmak:091711` manifest, 1,633,354 float32
PCM samples and 399 frames. Manifest, PCM, all CSR artifacts, wrapper, Soxr and
archived Python core must match the fixed identities enforced by the entrypoint
and original trace protocol. No decode, source replacement or PCM regeneration
is performed by the candidate. Use the already recovered read-only PCM witness.

Run `fused_product_case` with explicit `--manifest`, `--python-core`,
`--soxr-wrapper`, `--soxr-library`, `--runtime-proof`,
`--runtime-proof-sha256` and a new `--output` directory. There are two sequential
frontend passes within that one worker:

1. Ordinary separate-product computation must reproduce the archived native
   core exactly. Preserve its raw core and comparison report, including the
   original negative comparison with Python. Failure stops candidate admission.
2. The explicitly selected fused-product computation retains a separately named
   complete core and report. Compare all 76,608 float32 values after FP16
   roundtrip against the fixed Python core. Byte identity is the verdict; maximum
   error and differing coordinates are descriptive, not alternative pass rules.

A completed candidate mismatch is preserved as a result. No retry, different
case, additional arithmetic mode, tolerance relaxation or classifier evaluation
follows automatically. No historical executable or result is overwritten.

Prospective execution bounds: one local worker, numeric threads one, **60 seconds
wall, 45 aggregate CPU seconds, 512 MiB sampled process-tree RSS and 16 MiB whole
owned input/output/log root**. External supervision must enforce the same process
and resource contract; the Rust entrypoint does not replace that guardian.

Even an exact fixed-case core would not establish full-corpus or cross-platform
parity, ordinary-PC performance, or improved key accuracy. Those require separate
predeclared validation after this candidate is understood. No build or frontend
execution is asserted by this prospective document.
