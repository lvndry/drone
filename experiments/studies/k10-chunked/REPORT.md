# K10 chunked CPU inference: development gates passed

## Problem

The [CPU baseline](../k10-cpu/REPORT.md) showed that small weights can still need
substantial memory for long-track activations. Can we bound those activations
while preserving all evidence and key decisions, with no more than a 10% median
paired latency regression?

## Method

The [frozen protocol](PROTOCOL.md) uses aligned overlapping chunks: 64 output
positions per core, 16-frame halos and at most 544 input frames per prefix call.
Every global output is counted once. Hidden values accumulate in FP64, are
averaged/cast to FP32, then pass through the unchanged ONNX head once. Weights,
track evidence and the full ONNX reference stay unchanged.

The numerical gate requires finite logits, absolute/relative tolerance 1e-5 and
zero key disagreements on all 376 public validation inputs. Reference keys are
not used for scoring/selection. Only a passing gate permits resource measurement.
Boundary, remainder, impulse and extreme synthetic tests supplement this check;
no universal bitwise floating-point equivalence is claimed.

Resource probes use the same 20 selected FMAK validation recordings, one CPU
core on the AMD 9955WX, all library/model threads one, sequential ORT and spinning
disabled. Fresh full/chunked processes each run one warmup and 20 repetitions per
track. The target is at least a two-times RSS reduction against both paired full
and historical 535 MiB single-pass probes. No training job ran during these
probes; both vLLM instances stayed running.

## Results

All 376 key decisions matched. Maximum logit difference was **1.90735e-6**.

| Cached-feature inference | Full ONNX | Chunked ONNX |
|---|---:|---:|
| Median of 20 per-track median latencies | 35.52 ms | 36.29 ms |
| Peak process RSS | 865.34 MiB | 129.32 MiB |
| Loaded-runtime peak RSS before tracks | 62.59 MiB | 61.34 MiB |

![Paired latency and repeated-run peak memory](figures/k10-chunked-summary.png)

*Figure 1. Separate fresh processes each handle all 20 tracks. Median paired
latency rises 1.91%; peak RSS falls 6.69 times. Both gates pass. The bar medians
and median of paired ratios are distinct summaries.*

RSS is also 4.14 times below the historical single-pass probe; different allocator
history prevents a direct cross-protocol speed claim. Both processes stayed
Torch-free and used 20.02/19.55 CPU seconds. Separate Torch diagnostics found zero
key differences, maximum logit error 5.72205e-6, and identical stitched/full ORT
hidden tensors here. Extreme synthetic rounding discrepancies remain documented.

### Memory versus track length

A follow-up starts a fresh process for **each track and runtime**, avoiding
retained allocator history across tracks: 40 processes, the same 20 inputs and
repetition count. It took 44.53 seconds and passed paired numerical/key parity.

![Per-track memory and latency versus track duration](figures/k10-memory-length.png)

*Figure 2. Full-track RSS grows strongly with duration; chunking bounds CNN
activations. RSS still includes Python, ORT and the full input feature array.
Connecting lines guide the eye rather than fit a scaling law.*

| Approximate duration | Full peak RSS | Chunked peak RSS |
|---|---:|---:|
| 40 seconds | 75.8 MiB | 76.0 MiB |
| 4 minutes | 158.3 MiB | 110.7 MiB |
| 28 minutes | 739.6 MiB | 120.7 MiB |

Ordinary song lengths save less than the longest case. Durations are CQT frame
proxies; no cases were measured between roughly 7 and 28 minutes. These per-track
peaks and Figure 1's all-tracks high-water use different protocols.

## Discussion and decision

Chunking passes the development gates and earns native integration testing.
Observed parity is not a universal accuracy guarantee. This bounds CNN activation
memory, not decoding, CQT or input storage. Neither probe measures Rust/Euphoria,
target laptops or idle application memory. Next: native parity and end-to-end
profiling, persisted result caching and bounded concurrency. No release artifact
was promoted.

## Reproduction and evidence

The first resource attempt failed before measurement: its reader expected `sha`
instead of canonical `sha256`. A regression fixture fixed that bug. Validation
was rerun before successful paired probes; original diagnostics and the failed
report remain preserved.

[Paired reports/provenance](data/provenance.json) bind source archives, contracts
and environment. Source v1: `0fd31faaa6bc199e1705588eea756ade7e2fb22b`; v2:
`8663145914cfa370a9d1170e05f8a05b2455badc`. The unchanged bundle remains on the
server. [Per-track data](data/memory-curve.json) and
[provenance](data/memory-curve-provenance.json) identify follow-up source
`ef69ecbedb6453d0e3b00e585920a426b5ffb218` and its archived validation gate.
Replay with `python -m tools.benchmark_k10_memory_curve --help`, Linux affinity
[0], thread budgets one and a fresh output. The tool requires exact passing
artifacts/source evidence and enforces a 600-second wall budget.

Regenerate figures without audio/server access:

```bash
uv run --extra plots python -m tools.plot_research
```

Checked-in JSON supplies all figures. [Figure inputs](../figure-inputs.json)
record data/generator/lockfile hashes and the plotting-library version.
