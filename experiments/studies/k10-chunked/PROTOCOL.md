# K10 unchanged-weight chunking protocol

Declared before evaluation on cached public tracks, 2026-11-11. Research only;
the existing full `weights/drone-v1.onnx` remains authoritative and is never overwritten.

## Function and boundaries

The eval-mode prefix removes only the final adaptive pool. BatchNorm is frozen
and Dropout2d is off. Its output is `[1,C,24,Q]`, where `Q=floor(T/8)`.
The six Conv3 and three Pool2 layers give receptive field 36 and stride 8:
output q depends on input indices `[8q-14,8q+21]`.
For output core `[q0,q1)`, use input
`[max(0,8q0-16), min(T,8q1+16))` and retain local output
`[q0-a/8,q1-a/8)`. Every global q must appear exactly once. Preserve the real
input tail T%8. Let the layers pad only at the real ends; do not append raw zeros.

Sum retained hidden features in FP64, divide by total Q, cast once to FP32,
then run the unchanged Flatten+Linear head once in CPU ORT. In real arithmetic,
linearity proves equivalence to the original mean+linear model. Reassociation
and shape-dependent floating-point kernels can differ; this is not a universal
bitwise or accuracy theorem. Default core Q=64 bounds prefix input to 544 frames.
This does not bound decoder, CQT, or full-feature-cache memory.

## Fixed promotion gate

Use all 376 public validation cached features from the prepared multidomain
manifest, selecting role=validation only. Read no ground-truth references for
scoring or selection. Slice the existing padded 208-bin cache as [8:200], FP32.
Require finite outputs, logits allclose with atol=1e-5 and rtol=1e-5, and ZERO
argmax disagreements against the unchanged full ORT artifact. Any failure blocks
promotion; do not alter labels, tolerance, or membership to make the gate pass.
Record individual discrepancies, absolute error and reference top-two margin.

Separate comparisons: original Torch/full ORT; full Torch prefix/full ORT prefix;
full-prefix pooled head/full ORT; stitched/full-prefix hidden tensors; final
chunked/full ORT. Torch diagnostics run separately from isolated resource probes.
Boundary tests cover every T%8, minimum T=8, cores 1/2/7/64, seams/endpoints,
random/constants/negative/high-dynamic-range inputs and impulses.

Synthetic diagnostic observation (macOS CPU, random untrained checkpoint with
nonidentity frozen BN): input amplitudes spanning 1e-3 to 1e3 produced individual
Torch/full-ORT hidden differences up to approximately 5.1e-5 and one
full-prefix/chunk-prefix hidden difference of 1.14e-5 near cancellation. Therefore
hidden-element allclose is a reported diagnostic, not the logit promotion gate.
All stress cases remain in tests; same-model FP64 chunk/full checks independently
test halo geometry, and final logits retain the declared 1e-5 numerical check.
No public validation had run when this diagnostic was recorded; no gate tolerance changed.

## Resource target and budget

Only after validation passes, benchmark the same existing 20 public selected
tracks, cached features, unchanged full ORT versus chunked ORT. Use separate fresh
processes, sequential ORT, intra/inter threads=1, spinning off, one CPU affinity
set by the caller, and environment thread counts=1. Collect 20 timed repetitions
per track, compare paired per-track medians, and report RSS units and platform.
Tentative target: >=2x peak-process-RSS reduction relative to the measured
535 MiB ORT-only baseline AND <=10% median latency regression. Also compare fresh
paired full-process RSS; do not claim a performance win if either target fails.
Each validation or performance stage has a 600 CPU-second hard watchdog. No
quantization, early exit, input truncation, private audio, or GPU inference.

Cache result identities include feature file hash, feature contract, label order,
source checkpoint hash, prefix/head hashes, original full ONNX hash, pooling
precision and chunk contract. Runtime metadata is integrity checked before use.
Benchmark gates bind the current runtime/evaluator source hashes to the validation
report. Paired resource reports also require matching source hashes, thread
environment, platform, ORT/NumPy versions and CPU affinity.

## Commands (repository root, caller selects new output paths)

Export uses Torch on CPU; the validate and benchmark commands never import Torch.
Run diagnose in its own process; it must not precede an isolated benchmark in the
same process. Export refuses an existing directory and reports refuse existing
paths. Use the unchanged deployed checkpoint/full-ONNX pair:

```sh
uv sync --all-extras
uv run python -m tools.export_chunked --checkpoint weights/drone-v1.pt --output /tmp/k10-chunk-bundle
```

The following Linux commands assume this source is deployed at the repository
root. All output names are new paths; change them for a new run. The 600-second
CPU timer is built into each command; outer timeout also bounds wall time.

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 600 taskset -c 0 uv run python -m tools.evaluate_chunked --mode validate --manifest /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/data/manifest.json --bundle /tmp/k10-chunk-bundle --full-model weights/drone-v1.onnx --output /tmp/k10-chunk-validation.json
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 600 taskset -c 0 uv run python -m tools.evaluate_chunked --mode diagnose --manifest /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/data/manifest.json --bundle /tmp/k10-chunk-bundle --full-model weights/drone-v1.onnx --checkpoint weights/drone-v1.pt --output /tmp/k10-chunk-diagnostics.json
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 600 taskset -c 0 uv run python -m tools.evaluate_chunked --mode benchmark --bench-mode full --manifest /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/data/manifest.json --bundle /tmp/k10-chunk-bundle --full-model weights/drone-v1.onnx --validation-report /tmp/k10-chunk-validation.json --selection /home/lysk-admin/drone-experiments/k10-cpu-20261111-v1/selection.json --output /tmp/k10-chunk-full-benchmark.json
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 600 taskset -c 0 uv run python -m tools.evaluate_chunked --mode benchmark --bench-mode chunked --manifest /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/data/manifest.json --bundle /tmp/k10-chunk-bundle --full-model weights/drone-v1.onnx --validation-report /tmp/k10-chunk-validation.json --selection /home/lysk-admin/drone-experiments/k10-cpu-20261111-v1/selection.json --full-benchmark-report /tmp/k10-chunk-full-benchmark.json --output /tmp/k10-chunk-benchmark.json
```

`performance_target_met` is emitted only when both fresh benchmark reports are
paired. It requires >=2x RSS reduction against both the historical 535 MiB value
and the fresh full-process peak, and median paired latency ratio <=1.1. It does
not authorize deployment; `promotion` remains false in all research reports.
