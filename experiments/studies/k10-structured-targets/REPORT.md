# Structured key targets: fixed pilot rejected

## Problem

Can a musically structured training objective improve a small classifier without
changing its architecture or inference cost? Nearby harmonic mistakes receive
partial MIREX credit, but encouraging those classes may sacrifice exact keys.
This study tests that tradeoff explicitly instead of assuming weighted-score gains.

## Method

The [frozen protocol](PROTOCOL.md) changes only the loss: probability mass 0.8 on
the true key and 0.2 on both fifth directions, relative and parallel keys in
proportion to existing relation weights. This is structured soft-target CE,
not direct optimization of expected MIREX. No smoothing or neighbour-weight sweep.

Both arms use the same 2,206 fitting / 376 validation identities, actual CQT
caches, K10 width 1, crops/augmentation, AdamW, cosine horizon and checkpoint rule.
The seed-13 hard-CE control had to exactly reproduce archived history, predictions
and checkpoint tensors before candidate fitting. A pilot needed at least one
FMAK symmetric-MIREX point, no Beatport MIREX decline and no exact decline in
either domain to earn seeds 37/73. No final/calibration/SWD/private data was scored.

## Results

The control reproduced exactly. The candidate failed every improvement condition.

| Arm | BP symmetric / asymmetric MIREX | BP exact | FMAK symmetric / asymmetric MIREX | FMAK exact |
|---|---:|---:|---:|---:|
| Verified hard-CE control | 76.11 / 73.33% | 68.25% | 75.48 / 74.48% | 68.40% |
| Structured targets | 74.44 / 70.87% | 65.87% | 74.40 / 73.00% | 67.20% |

![Paired control and structured-target validation scores](figures/accuracy.png)

*Figure 1. The fixed pilot reduces both domain scores and exact accuracy.
Symmetric-fifth MIREX is the predeclared gate; the asymmetric convention remains
separately reported rather than substituted after inspection.*

Symmetric MIREX changes were **−1.67 Beatport / −1.08 FMAK points**; exact changes
were −2.38 / −1.20 points. On FMAK the candidate promoted 13 non-exact predictions
to exact but demoted 16 previously exact predictions. Beatport had nine promotions
and 12 demotions. Both scoring conventions' complete paired category matrices
are retained in the raw candidate result.

The control selected epoch 37 and stopped at 62 in 259.48 seconds. The candidate
selected epoch 76 and stopped at 101 in 359.63 seconds. Epoch counts, initialization
and system load make total times unsuitable as an inference or throughput claim.

## Discussion and decision

Do not run confirmation seeds or promote weights. A relation-aware target did not
help this recipe; it does not follow that every use of harmonic structure fails.
The experiment changes targets, not the model's equivariance or access to pitch
evidence. Exact-demotion counts prevent partial-credit gains from hiding damage.
Soft-target and hard-target training losses also have different entropy floors,
so their numeric values should not be compared as an optimization diagnosis.

This is one pilot on repeatedly used development data, not an independent test.
No new smoothing variant, lucky seed, fresh final score or automatic release
followed. Existing release weights remain unchanged.

## Reproduction and evidence

Actual fitting source: `36d9e4164dcffc1192c2aebb60aa6d80dc2a24d1`, based on
`40afede7ca6af96d740086eab2616f9ec98d3321`; archived control source `673e280`.
Use that frozen source/archive for fitting replay: the published combined runner
has subsequently accumulated other study hooks and source guards intentionally
reject unapproved context. The scientific protocol/maps and recorded artifacts
are preserved, rather than relabelled as a new fit of current main.

[Control proof](data/control-result.json) and [candidate result](data/candidate-result.json)
bind source, feature, protocol, target-table, checkpoint and validation-logit
hashes. Full run/logit/checkpoint artifacts remain under
`/home/lysk-admin/drone-experiments/k10-structured-targets-20261111-v1`.
GPU 0 used the existing allocator guard; both vLLM instances stayed running.
Regenerate Figure 1 with `uv run --extra plots python -m tools.plot_research`.
