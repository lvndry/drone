# Pitch-circle v1: a shared tonic and mode scorer

**Status: architecture and CPU verification only. No fitting protocol is pinned.**
This study does not train a model, alter checkpoint formats or production model
dispatch, export weights, or change the independent capacity study `679b7fd`.

## Question

Can a shared tonic readout use limited labels better than K10's frequency-specific
linear head while retaining useful register information? The design tests an
architectural bias. It establishes neither novel methodology nor a route to the
user's stretch objective of 90% exact accuracy.

## Fixed architecture

Fold the existing C-based, 192-bin K10 CQT into eight octave channels and 24
half-semitone pitch positions. Do not immediately average the octaves.

| Stage | Output channels | Frequency dilation | Downsampling |
|---|---:|---|---|
| First convolution pair | 16, 16 | 1, 1 | Time only, factor 2 |
| Second convolution pair | 32, 32 | 2, 2 | Time only, factor 2 |
| Third convolution pair | 64, 64 | 4, 4 | Time only, factor 2 |
| Tonic/mode readout | 2 modes | — | Frequency only, factor 2 |

Each convolution is 3×3 with circular frequency padding, ordinary zero temporal
padding, time dilation 1, BatchNorm2d and ELU. A single Dropout2d(0.2) follows the
third temporal pool, matching actual K10's activation and dropout choices. The
readout is Conv2d(64, 2, kernel=(2,1), stride=(2,1)). Average over **time only**,
then flatten tonic and mode in C major, C minor, C♯ major, C♯ minor order.

There is no absolute tonic embedding, tonic-specific bias, arbitrary 24-class
linear head, or frequency-wide pooling. The model has **73,506 parameters**.
Its convolutional frequency receptive field is 29 half bins before readout and
30 including readout; temporal receptive field is 36 frames with stride 8.
These are inference-path receptive fields; training BatchNorm uses spatial
statistics. Inputs must be nonempty `(batch, 192, time)` finite floating tensors
with at least eight frames. FP64 tests verify geometry; they do not propose a
production precision change.

## What the symmetry guarantees

Rotate the folded 24-bin pitch axis by `2s` while leaving octave channels fixed.
The 12-tonic output rotates by `s` and preserves its mode axis. Circular
convolutions, shared normalization and activation, temporal pooling and the
aligned two-bin readout support this C12 feature-space symmetry. Geometry tests
use evaluation mode with dropout disabled. Channel dropout shares masks spatially;
independently sampled training forwards need not have identical outputs.

This is **not exact physical CQT or audio transposition equivariance**. Real
reslicing carries energy across octave-channel and finite-band boundaries. A test
constructs a concrete counterexample at an octave boundary: physical reslicing
and within-octave circular rotation give different retained channel tensors.
A later fitting protocol must retain the existing physical reslice augmentation
and report its empirical consistency separately. No temporal circularity is used.

Keeping two bins per semitone preserves more tuning-phase information than an
early 12-bin collapse, but the model adds no detuning correction. Its output is
aligned to the fixed CQT grid. Frequency shifts of one half bin have no guaranteed
mapping onto the 12 discrete tonic classes.

## Relationship to the earlier octave model

[OctaveNet](../../octave_model.py) already used temporal-only downsampling and a
shared two-mode frequency classifier followed by octave/time averaging. The
published [S-KEY architecture](https://arxiv.org/html/2501.12907v2) also provides
precedent. A future controlled run would revisit this bias under today's corrected
frontend and data protocol; it would not establish the idea from scratch.

The old inputs used 12 bins per octave from nnAudio VQT, two 32-second segments,
84-bin crops, relative-dB compression and A-based pitch order. This model consumes
the current 24-bin-per-octave librosa features with C-based order and variable
full-track support. Historical score differences cannot isolate architecture
across those frontend and training changes.

Early folding changes locality: neighboring pitch classes interact regardless
of octave, while register becomes an input channel. It can retain register cues,
but also suppress useful cross-octave adjacency or learn octave-specific weights
that weaken physical-shift robustness. The full-circle receptive field avoids
losing harmonic context simply by removing frequency pooling. Whether the bias
helps exact recognition remains untested.

## Verification and estimated cost

The [diagnostic data](data/geometry-and-cost.json) records source hashes, Torch
version, seed, tested lengths and numerical tolerances. On the recorded CPU,
192 rotation comparisons per precision covered all 12 rotations and every temporal
remainder at lengths 8–15 and 31–38. Maximum absolute discrepancy was **0** for
both FP64 and FP32. Declared tolerances remain FP64 `rtol=1e-10, atol=1e-12` and
FP32 `rtol=1e-5, atol=1e-6`; this is not a bitwise guarantee on every backend.

| Estimate at batch 1, 320 frames | K10 | Pitch-circle |
|---|---:|---:|
| Parameters | 109,128 | 73,506 |
| Convolution and linear MACs | 575,115,264 | 185,917,440 |
| Largest observed FP32 convolution input/output tensor | 3,932,160 bytes | 655,360 bytes |

These are arithmetic and tensor-size estimates, **not latency or peak RSS**.
They include explicit circular-pad tensors but exclude convolution workspace,
autograd storage, normalization/ELU/pooling work, finite-input checks and CQT
extraction. Dilated convolution and circular-padding kernels can behave poorly
on a CPU. Measure the actual runtime and whole pipeline before claiming a speed
or memory improvement.

Tests also check canonical label mapping, zero temporal padding, invalid shapes,
nonfinite rejection before BatchNorm mutation, minimum length, noncontiguous
inputs, finite gradients and an optimizer update. Analytic MACs are checked against
executed convolution shapes at 8, 15, 16, 319, 320 and 321 frames.

```sh
uv run pytest tests/test_pitch_circle_model.py -q
```

## Later integration proposal — not implemented or approved

Reuse the shared fitting loop through explicit research model construction and
checkpoint callbacks. Keep existing data, caches, crops, labels, hard CE,
optimizer, horizon and gates fixed. Use a separate research checkpoint writer and
loader with an explicit architecture identifier, feature/label contract, model
specification, source identity and state dictionary. Never serialize this model as
`KeyCNN(width=...)` or change production dispatch as part of the experiment.

Before any fitting, pin the protocol, source, paired control proofs, resource
budget and stopping rules in a separate reviewed step. No folding, kernel,
activation or dropout variants are authorized by this architecture-only study.
