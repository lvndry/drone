# Prospective local boundary observer build qualification, v2

The original v1 attempt failed on a transient output-file disappearance and is
preserved in `boundary-build-v1-evidence.tar.gz`. This is a separate prospective
build with unchanged resource limits. Only an `ENOENT` encountered while sampling
an already-enumerated output file is ignored; permissions, I/O failures, symlinks
and every resource cap remain failures. A disappearing object contributes zero
to that sample; the next sample and terminal scan recount surviving files.

One offline build of `boundary_trace` on the local Mac, after the source changes
and this guardian/protocol are committed. This is not permission to execute the
trace, decode audio, transfer PCM, load a model, initialize ORT or adjust numeric
tolerances. The old pipeline binary stays untouched. No Euphoria modifications.

The task-owned output root `/tmp/drone-cqt-boundary-build-v2` must be absent before
preparation. Copy only the exact lockfile's cached registry packages, archives and
sparse-index records into its private `cargo-home`. Verify package archive SHA256
against Cargo.lock and source/copy byte hashes. There are two local packages plus
66 registry packages in the inspected lockfile. The original Cargo cache remains
read-only to this job; do not copy credentials or download anything. Count the
private cache toward512MiB total owned outputs. Preparation rejects a copy at or
above128MiB, leaving at least384MiB for build products; that is an admission bound,
not authority to raise the hard limit. Preserve complete input files, not a trimmed
crate tree. A larger-than-expected copy or unavailable input aborts admission.

Freeze the committed source identity, both local crate manifests/lockfiles/all
Rust files/build.rs, Python observer/support closure/tests, protocol and guardian.
Record exact compiler/linker executable paths, their hashes, Rust stdlib inputs,
LLVM, source SDK resource declarations, old pipeline and Soxr library/wrapper.
The plan binds exact argv, clean environment, private cache source/destination
hashes/byte counts, compiler versions and selected Xcode. No source/toolchain/shared
cache mutation is allowed. No compilation starts unless the ownership/counter
probe confirms exact macOS PID start identity, live RSS and reaped-child CPU.

Build argv (manifest resolves to the isolated committed worktree):

```
/opt/homebrew/bin/cargo build --offline --locked --release --jobs 1 \
  --manifest-path /tmp/drone-cqt-build-output-race/research/drone-native-cqt/Cargo.toml \
  --bin boundary_trace
```

The plan supplies `CARGO_NET_OFFLINE=true`, `CARGO_INCREMENTAL=0`,
`CARGO_BUILD_JOBS=1`, `CARGO_TARGET_DIR=/tmp/drone-cqt-boundary-build-v2`,
`CARGO_HOME=/tmp/drone-cqt-boundary-build-v2/cargo-home`, and
`RUSTC=/opt/homebrew/bin/rustc`. Preserve HOME's existing value. PATH is restricted
to `/opt/homebrew/bin:/usr/bin:/bin`; TMPDIR is owned. No CQT_OWNER_SNAPSHOT,
RUSTFLAGS, CARGO_ENCODED_RUSTFLAGS, compiler wrapper, Python path overrides or DYLD
injection survives the explicit environment. The ORT crate uses load-dynamic,
without download/link features. No shared Cargo build target is used.

The owned process-group guardian is local macOS only, nice19, one Cargo job.
Caps:360 seconds wall,300 aggregate CPU seconds,1GiB sampled process-tree RSS,
512MiB owned target/cache/log/receipt bytes. CPU monitoring uses the installed
SDK's proc_pid_rusage v4 own and reaped-child counters for every live owned
process, plus guardian resource usage; overlapping counters are conservatively
summed. RSS is sampled at approximately0.1 seconds plus scan cost, so transients
between observations may be missed. PID start-abstime and process group identity
must match before group-only termination. Escaped groups or identity uncertainty
are failures. Inherited CPU/file-size limits supplement the guardian. No retry,
cap extension or rescue. Preserve failed logs and report the actual failure.

After a successful compile, inspect Mach-O dependencies with otool. A separate
owned Python process may dlopen the existing Soxr wrapper solely to enumerate the
actual dyld image paths and hash the loaded wrapper/Soxr bytes. It must not call
the transform or any model API. The resulting trace binary is never executed.
All post-build inspection and verification stays inside the same bounded window.
A successful receipt qualifies the build/loader identity, not frontend parity or
ordinary-CPU accuracy. The historical gaps and actual Python trace admission
remain governed by BOUNDARY_TRACE_PROTOCOL.md.

The shared editable test environment currently contributes a second tools
namespace. The strengthened Python admission rejects it; tests isolate their
own import search without modifying that environment or its pth files. A future
trace must bind a genuinely unambiguous actual module closure and exact interpreter
path. Native build qualification may proceed independently of that unresolved
Python trace runtime.
