# Drone native CPU research

The standalone crates, reports, protocols, source notices and lossless evidence
belong to Drone. They were moved from Euphoria without rerunning experiments or
changing the detector, model weights, Python dependencies or production defaults.

- [Cached-feature inference](drone-native-cached/REPORT.md): model-only contracts
  and memory/latency evidence.
- [Native CQT feasibility and arithmetic](drone-native-cqt/REPORT.md).
- [Composed CQT → inference and Mac376 diagnostics](drone-native-cqt/composed/REPORT.md):
  Linux qualification/timing and separately preserved strict Mac gate failures.
- [Build, evidence and unit-test commands](drone-native-cqt/README.md).

[MIGRATION.json](MIGRATION.json) records original Euphoria commit/tree/blob IDs,
SHA256 values and path history for every imported file. The initial import is a separate commit; formatting-only Python changes (with
identical ASTs), a test-only Clippy repair, and README/CI relocation adjustments
retain original and current hashes separately. All raw evidence and notices are
byte-identical. Rebuilt executables have their own source/binary identities and
cannot replace the recorded measured artifacts. Original source commits,
execution paths, tags, artifact hashes and licensing notices remain historical
identities. They must not be relabelled as Drone executions. Copyright notices
retain their original attribution. The crates retain their standalone Cargo
workspaces; `drone-native-cqt` depends only on its imported sibling, with no path
back into Euphoria. The optional historical owner-snapshot adapter remains opt-in;
normal builds, tests and evidence verification do not read another checkout/WIP.

From this repository root, the research-only CI equivalents are:

```sh
cargo fmt --manifest-path research/drone-native-cached/Cargo.toml --check
cargo test --locked --manifest-path research/drone-native-cached/Cargo.toml
cargo fmt --manifest-path research/drone-native-cqt/Cargo.toml --check
cargo test --locked --manifest-path research/drone-native-cqt/Cargo.toml
python3 -m unittest discover -s research/drone-native-cqt -p 'test_*.py'
python3 -m unittest discover -s research/drone-native-cqt/composed -p 'test_*.py'
python3 research/verify_migration.py
```

The imported source/licensing remains scoped research: the GPL-3.0-or-later
crates and their LGPL/SVML notices are not relicensed by Drone’s project license.
`/research` is excluded from Python build artifacts. All active research scripts remain under normal root Ruff/ty checks. Two explicit
module search paths resolve standalone report-script imports; no file is excluded.
The dedicated research workflow additionally runs the Rust/evidence checks.

Rust CI pins the existing 1.99.1 research toolchain on Linux and macOS. These tests
need no model, audio, ONNX runtime or SoXR/SVML download. Explicit ignored runtime
checks and reproduction commands require the original pinned runtime, model and
audio artifacts. The normal CI and evidence commands above read saved results
without downloading those artifacts or rerunning the measurements.
