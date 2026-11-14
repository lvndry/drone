# Standalone native CQT research harness

This crate is the native (Rust) CQT frontend used in the [native CQT study](REPORT.md) and the [composed frontend and inference study](composed/REPORT.md). It is research code: the production workspace, minimum Rust version, default dependencies and key detector are unchanged.

Read [REPORT.md](REPORT.md) for the method and results. The fixed protocols are [PROTOCOL.md](PROTOCOL.md) (nine cases), [EXPANSION.md](EXPANSION.md) (376 tracks) and [CANDIDATE2.md](CANDIDATE2.md) (the two arithmetic arms). On the tested Linux machine only the FMA magnitude plus SVML logarithm arm passes both its nine-case and 376-track checks.

## Export reference files

From a Drone checkout, the Python pipeline exports the reference files (PCM, kernels, features and logits). Set the paths for your machine first; `FULL_MODEL_SHA256` is the released full model's digest, listed in the [reproducibility record](../report-provenance.json).

```sh
python -m tools.export_cqt_goldens \
  --data-manifest "$DATA_ROOT/data/manifest.json" \
  --selection "$CPU_STUDY_ROOT/selection.json" \
  --feature-source drone/features.py \
  --full-model "$FULL_MODEL" \
  --full-model-sha256 "$FULL_MODEL_SHA256" \
  --protocol experiments/studies/native-cqt/PROTOCOL.md \
  --output "$EXPERIMENT_ROOT/goldens"
```

Run it with numeric thread variables set to 1 and the process restricted to two logical CPUs (for example `taskset -c 2,3`). The output must be a new directory. Exported files contain audio-derived data and are never committed. `tools.record_cqt_environment` records the actual build and binary identities.

## Build

```sh
bash research/drone-native-cqt/build_soxr.sh "$EXPERIMENT_ROOT"
cargo build --release --locked --jobs 2 --manifest-path research/drone-native-cqt/Cargo.toml
```

The soxr build checks both the archive checksum and the extracted source against the 167-file inventory in [evidence/build-evidence-v2.json](evidence/build-evidence-v2.json) before CMake runs. Missing, extra or changed files and symlinks leaving the source tree are rejected; the one original internal symlink is allowed.

The first nine-case measurements used Rust 1.99.1; the executable identity is in [evidence/index.json](evidence/index.json).

An optional external snapshot of another contributor's experimental frontend (`cqt.rs` and `halving.rs`) can be enabled with `CQT_OWNER_SNAPSHOT=/absolute/frozen/snapshot`. It is never committed or published. Without it, the default unit tests and the exported-kernel frontend work normally. Its source hashes are recorded in the first comparison reports.

## Run the nine-case comparison

```sh
ulimit -t 600
taskset -c 2,3 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  timeout 610 \
  research/drone-native-cqt/target/release/drone-native-cqt \
  --candidate exported \
  --soxr-wrapper "$EXPERIMENT_ROOT/libcqt_soxr_split.so" \
  --manifest "$EXPERIMENT_ROOT/goldens/manifest.json" \
  --full-model "$FULL_MODEL" \
  --ort-library "$ORT_LIBRARY" \
  --output "$EXPERIMENT_ROOT/exported-new.json"
```

`ORT_LIBRARY` is the ONNX Runtime 1.31.1 CPU shared library from the Python environment that produced the reference. Use a new output name; existing evidence is never overwritten. Use `--candidate owner` without `--soxr-wrapper` for the external experimental frontend. A failing candidate exits nonzero after saving all per-case diagnostics; a zero exit for the exported candidate covers only the nine fixed cases.

The 376-track two-arm run is recorded in [evidence/candidate2-launch.sh](evidence/candidate2-launch.sh), with all arguments, thread and CPU settings. It passes `--numpy-magnitude` to both arms and the checksum-verified `--math-library` only to the SVML arm. Reproducing it needs a new output directory; keep the old evidence.

## Tests and figures

```sh
cargo fmt --manifest-path research/drone-native-cqt/Cargo.toml --check
cargo test --manifest-path research/drone-native-cqt/Cargo.toml
cargo clippy --manifest-path research/drone-native-cqt/Cargo.toml --all-targets -- -D warnings
python3 -m unittest discover -s research/drone-native-cqt -p 'test_*.py'
uv run --script research/drone-native-cqt/plot_results.py
```

The two 376-track summaries are stored as deterministic `.json.gz`. `evidence_io.py` verifies both the compressed and the original decoded-byte checksums against [evidence/index.json](evidence/index.json), and `plot_expansion.py` reads through it; compression preserves the original JSON bytes exactly. The unit tests check roundtrips, repeatability and rejection of altered evidence. Both plot scripts verify their inputs against the index, and the feasibility plot also requires unique case IDs in the reference order, so valid file hashes cannot hide mispaired rows.

Later runner checks validate captured plan dimensions and accept a manifest file name relative to the working directory. They do not change the recorded numerical evidence; any new numerical result is tied to its own source and binary.

## Licensing

`libsoxr` and the Python-SoXR-derived wrapper are LGPL-2.1-or-later; keep [the upstream Python wrapper notice](LICENSE-python-soxr). The local source archive also contains libsoxr and its component notices. No conclusion about distributing a dynamic product bundle is drawn here.
