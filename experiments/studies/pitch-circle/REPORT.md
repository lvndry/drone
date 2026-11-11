# Pitch-circle sharing: a smaller model missed the accuracy gate

## Problem

Can sharing musical patterns across transpositions improve accuracy with fewer parameters? Increasing K10 capacity and softening its targets both failed earlier pilots. This study tests a distinct architectural hypothesis: retain octave channels but share convolutions around a pitch-class circle.

## Method

The fixed architecture has **73,506 parameters** versus K10's 109,128. Fold the 192 CQT bands into eight octave channels and 24 half-semitone positions. Six convolutions use circular frequency padding and dilations 1/1/2/2/4/4, with the same BN/ELU and final dropout policy as the baseline. Pool only in time; a two-mode readout produces the canonical 24 tonic/mode logits.

This is equivariance to rotations of the folded representation. A real transposition of finite-band CQT also moves energy between octave channels and band edges; exact physical transposition equivariance is **not** established.

Use the same 2,206 fitting / 376 validation identities, padded CQT caches, 320-frame crops, ±4-semitone augmentation, hard CE, batch size 32, AdamW, 120-epoch cosine horizon and patience 25. The shared loop changes only the model factory/checkpoint hooks. Different parameter shapes consume different initialization/dropout draws; identical hidden RNG draws are not claimed.

The seed 13 K10 control was reused only after exact history, prediction and checkpoint-tensor comparison with the archived joint run and verification of the original completed control proof. The pilot needed at least one FMAK symmetric-MIREX point, no Beatport MIREX decline and no exact decline in either domain. GPU 0 retained the 4% allocator cap and 20-minute fit/scoring budget; both vLLM instances remained running.

## Results

| Arm | BP symmetric / asymmetric MIREX | BP exact | FMAK symmetric / asymmetric MIREX | FMAK exact |
|---|---:|---:|---:|---:|
| Verified ordinary joint K10 | 76.11 / 73.33% | 68.25% | 75.48 / 74.48% | 68.40% |
| Pitch-circle | 73.81 / 71.83% | 65.87% | 74.72 / 73.52% | 68.00% |

![Paired ordinary K10 and pitch-circle scores](figures/accuracy.png)

*Figure 1. The fixed pilot reduced both domain scores. Exact accuracy is reported separately from harmonic partial credit. These are repeatedly used development tracks, not independent generalization evidence.*

Symmetric MIREX changed by **−2.30 Beatport / −0.76 FMAK points**; exact accuracy changed by −2.38 / −0.40 points. The selected epoch was 20 and fitting stopped at 45. Total fitting/scoring wall time was 163.21 seconds. Peak Torch allocation was 409,924,608 bytes and reservation 515,899,392 bytes; these are training allocator measurements, excluding some runtime memory, not application RSS.

## Discussion and decision

The candidate failed its fixed gate. Do not run confirmation seeds 37/73, change release weights, or infer that its smaller parameter count makes it faster on CPU. Earlier analytic MAC/tensor estimates are not measured inference performance.

The result rejects this architecture under this recipe, not every use of musical symmetry. It does not isolate whether octave-channel boundary handling, model capacity, optimizer compatibility or another factor caused the loss. The shared-loop tests establish implementation contracts, not a causal explanation of this negative result.

No final/calibration/SWD/private data was scored. No smoothing, dilation, width or learning-rate search followed. The 90% exact objective remains a stretch target without supporting results so far.

## Reproduction and evidence

Actual fitting source is preserved at [`research/pitch-circle-fit-v1`](https://github.com/lvndry/drone/tree/research/pitch-circle-fit-v1), commit `a804ee7018f4b1883da9a3b30b7cf0ec32816859`. Its [fixed protocol](https://github.com/lvndry/drone/blob/research/pitch-circle-fit-v1/experiments/studies/pitch-circle/PROTOCOL.md), architecture specification and strict research-checkpoint loader remain there; replay from that frozen source rather than current main. This report does not add an unused production model or duplicate training loop.

[Raw candidate result](data/candidate-result.json) binds source/protocol/feature/control/checkpoint/logit hashes, actual memory policy, both score conventions and complete paired error movements. Full runs/checkpoints/logits remain at `/home/lysk-admin/drone-experiments/pitch-circle-20261112-v1`; archived source tar SHA256 is `d08563c680af54534ada71c319a8cbb17e5c00c40f6f6f40aea3f8735cb0d3f0`.

Regenerate Figure 1 with `uv run --extra plots python -m tools.plot_research`.
