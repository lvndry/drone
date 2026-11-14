# Drone native CPU research

These studies test whether Drone's K10 key model can run natively (in Rust) on an ordinary CPU with the same predictions as the Python reference, and what that saves in latency and memory. The crates, reports, protocols, source notices and evidence were moved here from the application repository without rerunning any experiment or changing the detector, model weights, Python dependencies or production defaults.

- [Cached-feature inference](drone-native-cached/REPORT.md): native full and chunked inference from precomputed CQT features; prediction parity on 376 tracks and an 8.76x peak-memory reduction from chunking.
- [Native CQT frontend](drone-native-cqt/REPORT.md): native CQT on identical decoded PCM reproduces every feature value and logit on 376 tracks after two arithmetic fixes.
- [Composed CQT and inference, Linux and Mac](drone-native-cqt/composed/REPORT.md): the complete native pipeline on Linux (parity on 376 tracks, latency and memory on 20) and the Apple M5 portability investigation.
- [Build, evidence and unit-test commands](drone-native-cqt/README.md).

## Provenance

[MIGRATION.json](MIGRATION.json) records, for every imported file, its original identifiers, SHA-256 value and path history. Formatting-only Python changes (with identical syntax trees), a test-only lint fix and README and CI relocation edits keep both original and current hashes. Raw evidence and license notices are byte-identical to the originals. [NATIVE_EVOLUTION.json](NATIVE_EVOLUTION.json) records every later change to these folders, such as documentation rewrites, and [verify_migration.py](verify_migration.py) checks both files. Rebuilt executables have their own source and binary identities and never replace the measured artifacts. Original source identities, execution paths, artifact hashes and copyright attributions are historical records and are not relabeled as Drone runs.

Each crate keeps its own Cargo workspace. `drone-native-cqt` depends only on its sibling `drone-native-cached`, with no path back into the application repository. The optional adapter for another contributor's historical experimental frontend is opt-in; normal builds, tests and evidence checks do not read any other checkout.

## Checks

From the repository root, these match the research CI:

```sh
cargo fmt --manifest-path research/drone-native-cached/Cargo.toml --check
cargo test --locked --manifest-path research/drone-native-cached/Cargo.toml
cargo fmt --manifest-path research/drone-native-cqt/Cargo.toml --check
cargo test --locked --manifest-path research/drone-native-cqt/Cargo.toml
python3 -m unittest discover -s research/drone-native-cqt -p 'test_*.py'
python3 -m unittest discover -s research/drone-native-cqt/composed -p 'test_*.py'
python3 research/verify_migration.py
```

Rust CI pins the 1.99.1 research toolchain on Linux and macOS. These checks need no model, audio, ONNX Runtime, SoXR or SVML download; they read saved results without rerunning measurements. The explicitly ignored runtime tests and the reproduction commands in each report need the original pinned runtime, model and audio.

`/research` is excluded from Python build artifacts. All research scripts are covered by the normal root Ruff and ty checks, with two explicit module search paths for the standalone report scripts; no file is excluded.

## Licensing

The imported source stays scoped research: the GPL-3.0-or-later crates and their LGPL (SoXR) and BSD (SVML) notices are not relicensed by Drone's project license.
