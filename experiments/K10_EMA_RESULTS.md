# K10 EMA pilot: rejected

[Accuracy graph and interpretation](VISUAL_RESULTS.md#ema-a-negative-result-worth-keeping).

The fixed seed-13 probe completed on lysk-server on 11 November 2026. Its ordinary
training history and every selected checkpoint tensor exactly reproduced the
archived joint control. The EMA candidate failed the predeclared method gate;
seeds 37/73 were therefore not run. Release weights remain unchanged.

| Selected model | Beatport MIREX | Beatport exact | FMAK MIREX | FMAK exact |
|---|---:|---:|---:|---:|
| Ordinary RAW | 76.11% | 68.25% | 75.48% | 68.40% |
| Ordinary + BN recalibration | 72.06% | 63.49% | 73.84% | 66.00% |
| EMA + identical BN recalibration | 71.27% | 63.49% | 72.00% | 64.00% |

EMA minus the matched recalibrated control was -0.79 Beatport / -1.84 FMAK MIREX
points, with FMAK exact accuracy down 2.00 points. The separate safeguard against
ordinary RAW also failed. Recalibration itself reduced accuracy in this recipe;
the matched control is necessary to distinguish that effect from averaging.
These results reject this fixed recipe, not every possible averaging method.

The [protocol](EMA_PROTOCOL.md) was frozen before launch: parameter-only EMA
decay 0.99 initialized at theta-zero, selected at the ordinary RAW best epoch,
then one fitting-only BN pass per copy. The unchanged 2,206 fitting and 376
validation identities were used; calibration, final, SWD and private DJ audio
were not scored. Epoch 37 was selected; patience stopped at epoch 62, after
4,278 optimizer updates. Total runtime was 228.61 seconds, including 3.43 seconds
for both BN passes. A CPU audit was running concurrently, so this is not a clean
training-throughput comparison against the earlier run.

Run source: `407b9024b5b88277d18bb66ce4cbad925de7fead` from base
`c52255761412ef48f2f9af814df67c3d3214ef4f`. Archived reference source:
`673e280ca244d4187d8c20f9fdaecaa1c75281fd`. Full reviewed source maps, protocol,
schedule and artifact hashes are in [the raw result](ema-k10-2026-11-11/ema-result.json)
and [identity](ema-k10-2026-11-11/ema-identity.json). Checkpoint file hashes differ
because their metadata differs; exact state-tensor equality passed, including
dtype. All generated logits, schedules, model states and reference files remain
under `/home/lysk-admin/drone-experiments/k10-ema-20261111-v1` on the server.

GPU 0 was used under the existing 4% allocator cap and free-memory guard. Both
vLLM instances remained running. No promotion, optimizer sweep or decay search
followed this negative result.
