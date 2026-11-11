# Research tools

Commands stay runnable as `python -m tools.<name>`. Training recipes and inference
math belong to `experiments/` and `drone/`; these tools prepare audited inputs,
compare recorded runs, export artifacts or measure resources.

| Purpose | Commands |
|---|---|
| Validate the unchanged-weight chunk runtime | `export_chunked`, `evaluate_chunked` |
| Measure CPU resources | `benchmark_k10_pipeline`, `benchmark_k10_ort_memory`, `benchmark_k10_memory_curve` |
| Prepare native runtime inputs | `export_native_fixtures` |
| Render existing evidence | `plot_research` |
| Audit/prepare datasets | `inventory_audio`, `fmak_protocol`, `audit_fmak_groups`, `prepare_fmak` |
| Compare or summarize trained runs | `compare_*`, `summarize_*` |

`tools/research/` provides shared, Torch-free building blocks:

- `contracts.py`: immutable validation-track identities, safe feature-cache
  loading and the fixed public performance selection. Reference labels do not
  enter these contracts.
- `evidence.py`: model/data/bundle identities and validation/paired-report gates.
  Every support module is hashed, so moving an implementation behind a stable
  entry point cannot reuse stale parity evidence.
- `measurements.py`: explicit RSS units, execution environment and repeated
  prediction timing with exactly one warmup.

The existing `evaluate_chunked` input helper imports remain compatibility
wrappers for older fixture callers. New tools should import the typed contracts
directly rather than importing another command-line program.

After an implementation/helper change, rerun validation before resource
measurements. Historical reports remain immutable evidence of their recorded
sources; they are not permission to benchmark new code. Paired measurements
require identical artifacts, sources, runtime/environment, selected inputs and
order. Invalid or failed pair evidence is rejected before loading a model.

## Native fixtures

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
taskset -c 0 uv run --extra export python -m tools.export_native_fixtures \
  --manifest <prepared/manifest.json> --selection <existing-public-20.json> \
  --bundle <chunk-bundle> --full-model <unchanged-full.onnx> \
  --validation-report <new-passing-validation.json> --output <new-directory>
```

The exporter requires ONNX Runtime 1.31.1 and the same passing 376-track
validation identity. It writes contiguous `[192,T]` little-endian FP32 files by
widening the original FP16 cache core, plus full-model reference logits and
source/hash metadata. `selection20` marks the existing performance subset.
No audio or ground-truth labels are exported. Incomplete exports retain an
explicitly incomplete manifest; an existing output directory is never replaced.
The CLI enforces 600-second CPU and wall watchdogs.
