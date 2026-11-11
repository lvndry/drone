# Full-track K10 CPU baseline

[Graphs and interpretation](VISUAL_RESULTS.md#frontend-time-and-repeated-run-memory).

Measured on 11 November 2026, after the K10 layout and variable-length export
fixes. This is a reference for CPU optimization, not an application performance
guarantee or another key-accuracy evaluation.

## Method

Twenty public FMAK validation recordings were selected at evenly spaced duration
ranks among the existing 250 validation rows. Their CQT lengths correspond to
approximately 40 seconds–28 minutes. No reference keys were used, and the private
DJ library, calibration set, historical finals and SWD were not evaluated.

The process and its decoder children were pinned to one logical CPU of an AMD
Ryzen Threadripper PRO 9955WX. Torch/ORT each used one intra-op thread; ORT was
sequential with worker spinning disabled. Numerical-library environment thread
budgets were one. The loaded models used the unchanged shipped K10 parameters.

Each track began with an empty temporary feature cache. Ten repeated timings
were then taken for each full-track backend and the warm feature-cache + ONNX
path. The table summarizes those twenty per-track medians, except feature
preparation, which was measured once per track. Its p95 is a quantile over these
duration-rank examples, not an estimated library-wide latency guarantee.

## Timing results

| Phase | Median | p95 across cases | Longest observed |
|---|---:|---:|---:|
| Decode + CQT + feature-cache creation | 353 ms | 1,002 ms | 2,376 ms |
| Full-track Torch forward | 59.7 ms | 155 ms | 646 ms |
| Full-track ONNX forward | 35.6 ms | 79.4 ms | 278 ms |
| Warm feature cache + ONNX forward | 40.7 ms | 91.2 ms | 316 ms |

Torch checkpoint loading took 7.95 ms and ONNX session creation 50.06 ms in this
process. Python imports are outside those load timings. The first frontend call
also includes library initialization and may reuse existing compiled/JIT caches.
"Cold" here does not mean cold OS page caches. Separate phase quantiles must not
be added to manufacture an end-to-end p95.

All twenty Torch/ONNX key decisions agreed. Maximum absolute logit difference
was **3.82e-6**, within the existing export parity tolerance. This is backend
agreement on these inputs, not agreement with ground-truth keys or a universal
floating-point equivalence guarantee. Total benchmark wall time was 50.70 seconds.

## Memory results

The combined Python/Torch/ORT/CQT benchmark process reached **1,614 MiB** peak
RSS. An isolated second process imported no Torch and ran only NumPy/ORT on the
same twenty cached inputs:

| Isolated process phase | Peak RSS |
|---|---:|
| Before session creation | 53.2 MiB |
| After session creation | 61.8 MiB |
| After full-track inference suite | **534.9 MiB** |

These are process high-water marks, not incremental per-track measurements or
idle retained memory. The isolated probe excludes decoding/CQT and Euphoria.
The longest input had 9,057 frames; model weights alone are a poor predictor of
the memory needed for its full-track activations/workspaces. Neither number is
a Rust application memory measurement.

## Decisions and next tests

The feature frontend dominates the measured cold path. Reusing one initialized
session and caching final analysis results are important for startup behavior;
a feature cache alone still reruns the network. A cache identity must include
audio identity, frontend/decoder contract, model hash and inference version.

Next test aligned, overlapping inference with every temporal output retained,
then aggregate the original hidden features before applying the unchanged head.
Do not replace this with equal-weight crop voting, input truncation, quantization
or an early exit. The first target is lower activation memory without changing
key decisions; decoder/CQT streaming requires a separate frontend parity check.

No deployment speedup or memory reduction has been measured by this baseline.
Real Euphoria behavior and target-laptop resource use remain to be measured.

## Reproduction artifacts

- [Raw phase timings](cpu-k10-2026-11-11/pipeline.json),
  [isolated memory probe](cpu-k10-2026-11-11/ort-memory.json), and
  [source/environment provenance](cpu-k10-2026-11-11/provenance.json).
- [Pipeline benchmark](../tools/benchmark_k10_pipeline.py) and
  [Torch-free ORT probe](../tools/benchmark_k10_ort_memory.py).
- Server root: `~/drone-experiments/k10-cpu-20261111-v1/`; `selection.json` records
  input IDs/content hashes and the source development manifest hash. It contains
  server paths and stays with the server artifacts; no audio is committed.
- Pipeline source commit: `5c551a3e4875f5649e737720f79be6db40d8b1de`, based on
  main `c522557`. The invocation supplied label `cpu-audit-v1` in the raw
  `source_commit` field; `provenance.json` binds that label to the actual revision,
  source archive hash and raw result hash.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0 \
  uv run --extra export python -m tools.benchmark_k10_pipeline \
  --selection <selection.json> --checkpoint weights/drone-v1.pt \
  --onnx weights/drone-v1.onnx --source-commit <revision> --output <pipeline.json>

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0 \
  uv run --extra export python -m tools.benchmark_k10_ort_memory \
  --selection <selection.json> --features <prepared-k10-data> \
  --onnx weights/drone-v1.onnx --output <ort-memory.json>
```

The commands use Linux CPU affinity; on another platform use its equivalent or
report that only library thread counts were constrained. Versions are in the
provenance file. Both vLLM instances remained running; no GPU was used or freed.
