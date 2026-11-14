# Native K10 cached-feature research

This standalone Rust workspace (Rust 1.92 or later) measures native inference from cached CQT features. Results, with figures, are in the [native report](REPORT.md); the measured source is identified in the [reproducibility record](../report-provenance.json), and later report or CI edits do not replace those measurements.

The workspace leaves the application's production workspace, minimum Rust version, dependencies, detector and cache unchanged. It does not decode, resample, compute CQT, call Python or Torch, score labels or authorize promotion, so it is not an audio-to-key benchmark.

## Runtime and parity contract

`ort = 2.0.0-rc.15` (API 31) dynamically loads a caller-supplied ONNX Runtime shared library, whose C API version string must equal 1.31.1 exactly. No automatic binary or model downloads and no GPU features are enabled. Cargo.lock is committed. Sessions use explicit CPU preference, the CPU arena enabled (matching Python), sequential execution, one intra-op and one inter-op thread and spinning disabled. A full benchmark creates only a full-model session; a chunked benchmark creates only a prefix and a head session. Validation uses both modes in its own process.

Registering the CPU provider alone does not override the Rust wrapper's default efficiency device policy, which can select CoreML on macOS. An initial 8-frame synthetic test showed FP16-rounded full logits and a 4.78e-4 chunked/full discrepancy. Explicit CPU preference fixes the provider choice; the threshold was not relaxed. See the [upstream issue and maintainer explanation](https://github.com/pykeio/ort/issues/662) and the [builder API](https://docs.rs/ort/2.0.0-rc.15/ort/session/builder/struct.SessionBuilder.html).

The prefix keeps 24 frequency bins, temporal stride 8 and receptive field `[8q-14, 8q+21]`. For disjoint cores `[q0, q1)` of 64 outputs, read `[max(0, 8q0-16), min(T, 8q1+16))` and keep local indices `[q0-a/8, q1-a/8)`. Keep the actual T mod 8 tail and only the network's real end padding. Maximum prefix input is 544 frames. Accumulate hidden features in FP64, divide by floor(T/8), cast to FP32 once and apply the unchanged ONNX head once. No quantization, sampling, early exit or input truncation.

Class ID is tonic*2+minor, with C=0. Argmax picks the first index on an exact tie, matching NumPy. All inputs and outputs must be finite; T >= 8 and batch size 1 only. Pass criterion: atol = rtol = 1e-5 and zero argmax disagreements on all 376 inputs, for native full versus reference, native chunked versus reference and native chunked versus native full. Equivalence in exact arithmetic does not promise bitwise floating-point parity.

## Fixture format (exported by Drone)

`manifest.json` contains:

```json
{
  "complete": true,
  "tracks": 376,
  "selected_tracks": 20,
  "format": "k10-core-f32le-v1",
  "feature_spec": "cqt24-8oct-c1-pad4-hop4096-v2",
  "label_schema": "tonic-times-two-plus-minor-v1",
  "layout": "bands,time",
  "dtype": "float32-le",
  "bands": 192,
  "full_model_sha256": "64 lowercase hex characters",
  "checkpoint_sha256": "64 lowercase hex characters",
  "reference_onnxruntime": "1.31.1",
  "validation_report_sha256": "64 lowercase hex characters",
  "export_tool_sha256": "64 lowercase hex characters",
  "rows": [{
    "id": "existing-prefixed-public-id",
    "domain": "fmak",
    "frames": 320,
    "path": "features/public-id.f32",
    "sha256": "hash of this raw f32 file",
    "audio_sha256": "hash of the public source audio",
    "cache_sha256": "hash of original padded FP16 NPY cache",
    "source_sha256": "same original cache hash when that is the source",
    "reference_logits": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "selection20": true
  }]
}
```

Require complete=true, tracks=376, selected_tracks=20, exactly 376 unique domain/id
rows and exactly 20 selected unique audio identities. Relative feature paths
contain normal components only and canonical resolution must stay within the
manifest directory, including through symlinks. Raw files
are headerless, contiguous band-major `[192,T]` little-endian f32: take the
original padded cache `[8:200]`, widen its FP16 values to FP32, **without recomputing
or renormalizing features**. The raw hash is separate from source/cache hashes.
Reference logits come from the unchanged authoritative full ORT artifact; the
exporter reads no ground-truth key references. No labels appear in Rust structs.

Bundle `contract.json`, prefix/head ONNX hashes, dimensions, class order,
checkpoint identity, feature schema and aggregation are checked. The caller's
full-model hash must equal the fixture manifest's full-model hash.

## Build and run

```sh
cargo test --locked --manifest-path research/drone-native-cached/Cargo.toml
cargo clippy --locked --manifest-path research/drone-native-cached/Cargo.toml --all-targets -- -D warnings
cargo build --release --locked --manifest-path research/drone-native-cached/Cargo.toml
```

Measure with the built executable directly, not through Cargo. Use the ONNX Runtime 1.31.1 CPU library from the Python environment that produced the references. The measurements used Rust 1.99.1 installed in a task-local directory (no shell profile or system changes), with the installer and channel manifest checked against their published SHA-256 files; the downloads were about 148 MB compressed and 664 MB installed. Build jobs were capped at 2.

Example arguments (set these paths and the released full-model digest first):

```sh
NATIVE=research/drone-native-cached/target/release/drone-native-cached
ORT_LIBRARY=/path/to/python-env/onnxruntime/capi/libonnxruntime.so.1.31.1
FIXTURES=/path/to/export/manifest.json
BUNDLE=/path/to/unchanged/chunk-bundle
FULL_MODEL=/path/to/authoritative/drone-v1.onnx
FULL_SHA256=the_frozen_64_character_full_model_digest
```

Each command runs in a fresh process. Set CPU affinity externally; the process enforces a 600 CPU-second limit and the outer timeout bounds wall time. Existing output files are refused. Run validation first, then the full and chunked measurements (one warmup and 20 timed repetitions per selected track).

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 600 taskset -c 0 "$NATIVE" --mode validate --manifest "$FIXTURES" --bundle "$BUNDLE" --full-model "$FULL_MODEL" --full-model-sha256 "$FULL_SHA256" --ort-library "$ORT_LIBRARY" --output /tmp/native-validation-v1.json
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 600 taskset -c 0 "$NATIVE" --mode full --selection-only --manifest "$FIXTURES" --bundle "$BUNDLE" --full-model "$FULL_MODEL" --full-model-sha256 "$FULL_SHA256" --ort-library "$ORT_LIBRARY" --validation-report /tmp/native-validation-v1.json --output /tmp/native-full-v1.json
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 600 taskset -c 0 "$NATIVE" --mode chunked --selection-only --manifest "$FIXTURES" --bundle "$BUNDLE" --full-model "$FULL_MODEL" --full-model-sha256 "$FULL_SHA256" --ort-library "$ORT_LIBRARY" --validation-report /tmp/native-validation-v1.json --full-report /tmp/native-full-v1.json --output /tmp/native-chunked-v1.json
```

Reports record the executable, compiled source, lockfile, ONNX Runtime library, model, bundle and input hashes, CPU affinity and thread environment, plus CPU model and capabilities, kernel version, machine architecture and a hashed machine identity, so reports from different machines cannot be paired by accident. Changing code or runtime requires validating again. Timings include input validation and tensor construction; file reading, hashing and parsing are reported separately. RSS covers the whole native process, including feature loading. Linux reports `ru_maxrss` in KiB; macOS bytes are converted to KiB. CPU time includes all threads in the process. No child processes run. Thread counts are reported on Linux.

Only paired fresh reports can set `performance_target_met`: at least 2x lower RSS than both the fresh full run and the earlier 535 MiB Python measurement, a median paired latency ratio of at most 1.1, and parity still passing. This is a model-runtime target, not an audio pipeline or production readiness claim. `promotion` is always false.

## Explicit runtime integration test

Set `DRONE_ORT_LIBRARY`, `DRONE_BUNDLE`, `DRONE_FULL_MODEL` to trusted local artifacts:

```sh
cargo test --locked --manifest-path research/drone-native-cached/Cargo.toml pinned_runtime -- --ignored --nocapture
```

This test runs synthetic inputs only: all remainders, the 8-frame minimum, seams, tails, constants, negatives, impulses and amplitudes from 1e-3 to 1e3. It calls no Python. Ordinary tests separately cover geometry, row-major slicing, class mapping and ties, rejection of non-finite or short inputs, integrity and provenance checks.
