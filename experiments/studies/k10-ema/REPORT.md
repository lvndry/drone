# K10 EMA pilot: rejected

## Problem

Can training-time weight averaging improve the existing small classifier without
adding deployment cost? EMA is plausible, but its transfer to this AdamW recipe
is unproven. BatchNorm statistics can confound the comparison, so averaging and
recalibration must be distinguished.

## Method

The [protocol](PROTOCOL.md) was frozen before launch: parameter-only EMA decay
0.99 initialized at theta-zero, selected at the ordinary RAW best epoch, then
one fitting-only BN recalibration pass per copy. Compare RAW, RAW with
recalibration, and EMA with identical recalibration. Ordinary RAW alone controls
checkpoint selection and stopping; there is no EMA checkpoint search.

The unchanged 2,206 fitting and 376 validation identities were used. Calibration,
final, SWD and private DJ audio were not scored. Before interpreting EMA, fresh
RAW history and selected checkpoint tensors had to reproduce the archived control
exactly. The pilot needed at least one FMAK MIREX point over the matched BN
control, no Beatport MIREX decline and no exact decline in either domain to earn
confirmation. A separate safeguard compares against RAW.

## Results

The seed-13 probe completed on lysk-server on 11 November 2026. Exact RAW
reproduction passed, including checkpoint dtype/value equality.

| Selected model | Beatport MIREX | Beatport exact | FMAK MIREX | FMAK exact |
|---|---:|---:|---:|---:|
| Ordinary RAW | 76.11% | 68.25% | 75.48% | 68.40% |
| Ordinary + BN recalibration | 72.06% | 63.49% | 73.84% | 66.00% |
| EMA + identical BN recalibration | 71.27% | 63.49% | 72.00% | 64.00% |

![RAW, recalibrated RAW and recalibrated EMA accuracy](figures/k10-ema.png)

*Figure 1. Recalibration and averaging both reduce this pilot's point estimates.
The matched RAW + BN comparison isolates averaging; comparing EMA only with RAW
would combine two interventions.*

EMA minus matched BN control was -0.79 Beatport / -1.84 FMAK MIREX points;
FMAK exact fell 2.00 points. The RAW safeguard also failed. Epoch 37 was selected;
patience stopped at epoch 62, after 4,278 updates. Total runtime was 228.61 seconds,
including 3.43 seconds for both BN passes.

## Discussion and decision

The fixed recipe failed, so seeds 37/73 were not run. Recalibration itself hurt;
averaging hurt further against its matched control. This rejects this recipe,
not every possible averaging method. A CPU audit ran concurrently, preventing a
clean training-throughput comparison. One seed on reused development data cannot
establish generalization.

Release weights remain unchanged. No optimizer/decay sweep or automatic promotion
followed. GPU 0 used the existing allocator/free-memory guards; vLLM stayed running.

## Reproduction and evidence

The [frozen fitting source](https://github.com/lvndry/drone/tree/407b9024b5b88277d18bb66ce4cbad925de7fead)
and [archived control](https://github.com/lvndry/drone/tree/673e280ca244d4187d8c20f9fdaecaa1c75281fd)
remain available. Full source maps, protocol/schedule and
artifact hashes are in [raw results](data/ema-result.json) and
[identity](data/ema-identity.json). Checkpoint file hashes differ because metadata
differs; exact tensor equality passed. Generated logits, schedules, states and
references remain under `/home/lysk-admin/drone-experiments/k10-ema-20261111-v1`.
