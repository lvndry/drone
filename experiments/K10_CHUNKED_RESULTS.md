# K10 chunked CPU inference: development gates passed

The unchanged-weight prototype passed all 376 public validation inputs without
reading their reference labels for scoring. Every predicted key matched the
shipped full ONNX artifact. Maximum logit difference was 1.90735e-6, within the
fixed absolute/relative tolerance of 1e-5. This is observed decision parity,
not a universal floating-point equivalence guarantee.

| Cached-feature inference | Full ONNX | Chunked ONNX |
|---|---:|---:|
| Median of 20 per-track median latencies | 35.52 ms | 36.29 ms |
| Peak process RSS | 865.34 MiB | 129.32 MiB |
| Loaded-runtime peak RSS before tracks | 62.59 MiB | 61.34 MiB |

Median **paired** latency ratio was 1.0191, a 1.91% regression, within the fixed
10% allowance. Peak process RSS fell 6.69 times in these paired probes and was
4.14 times below the earlier 535 MiB single-pass audit. Both memory checks passed
the two-times target. The current full baseline uses 20 repetitions per track;
its allocator high-water measurement differs from the historical single-pass
probe. Do not replace paired evidence with a cross-protocol speed or RSS claim.

Fresh, separate Python/NumPy/ONNX Runtime processes used CPU affinity [0] on the
AMD 9955WX server, all library/model thread counts one, sequential execution and
spinning disabled. Each of the same 20 preselected FMAK validation recordings had
one warmup and 20 timed repetitions. Both processes stayed Torch-free. The full
and chunked probes took 20.02 and 19.55 CPU seconds respectively. No training job
was active during these resource probes; vLLM services remained running.

The [fixed protocol](CHUNKED_PROTOCOL.md) uses 64 output positions per core,
16-frame aligned halos and at most 544 input frames per prefix call. It counts
every output exactly once, accumulates hidden values in FP64 and applies the
unchanged FP32 ONNX linear head once. Track evidence and deployed weights are
unchanged. Separate Torch diagnostics over all 376 inputs found zero key
differences versus full ORT, maximum logit error 5.72205e-6, and exactly matching
stitched/full ORT hidden tensors on these inputs. Extreme synthetic stress tests
still document shape/runtime rounding differences; they are not hidden or used
to relax the final gate.

The first resource attempt stopped before measuring because the selection reader
expected `sha` instead of the existing canonical `sha256` field. That bug was
fixed with a regression fixture. Validation was rerun against the corrected
evaluator source before fresh paired measurements. The bundle/runtime were
unchanged; the original diagnostic and failed resource report are retained.

[Raw reports and provenance](chunked-k10-2026-11-11/provenance.json) bind the source
commits, archives, model/bundle/cache contracts, reports and runtime environment.
Source v1 is `0fd31faaa6bc199e1705588eea756ade7e2fb22b`; v2 is
`8663145914cfa370a9d1170e05f8a05b2455badc`. Original full ONNX and checkpoint
artifacts were never overwritten. Exported research bundles remain on the server.

This bounds CNN activation memory, not audio decoding, CQT extraction or full
feature-cache storage. It does not measure native Rust/Euphoria or laptop CPU
behavior. The next performance step is native integration and end-to-end
measurement, preserving result caching and bounded background concurrency.
No release artifact has been promoted by this experiment.
