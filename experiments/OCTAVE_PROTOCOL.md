# One supervised octave-aware accuracy pilot

Declared before CQT extraction or fitting. The prior fixed-data candidate improves
FMAK validation by 6.64 points over three seeds but trails native Euphoria on
Beatport validation. Test broader pitch evidence and an octave-aware inductive
bias before scaling model width. This is one complete candidate; gains cannot be
attributed separately to the frontend, architecture, segment sampling or batch
size without subsequent ablations.

- Same 2,206 fitting identities and 376 validation identities as the data pilot:
  Beatport 1,230/126 plus artist-purged FMAK 976/250. No calibration/final audio
  extraction, no GiantSteps scores, no private DJ-library usage.
- For each selected track, decode two 32-second float32 mono/22,050 Hz segments
  centered at 25% and 75% of duration, clamped within the track; pad short audio.
- nnAudio 0.3.4 VQT with gamma=0 (constant-Q), fmin 27.5 Hz, 99 bins, 12 bins per
  octave, hop 512. Relative 80 dB amplitude compression to [0,1], silence zero.
  Dependency versions and source/data identities are recorded. Optional research
  packages live in a task-local directory, outside all serving environments.
- S-KEY/ChromaNet-inspired independent backbone: temporal-only downsampling,
  depthwise ConvNeXt blocks, channels 2/3/40/40/30/10/3, kernels 7/7/7/7/7/5/5,
  layer scale 0.1, drop-path 0.1, octave pooling and 12-tonic × 2-mode logits.
  Train from scratch with labels, not the paper's self-supervised objectives.
  No pretrained weights. Source: https://github.com/deezer/skey and
  https://arxiv.org/abs/2501.12907.
- Each training example chooses one segment and a uniform crop start 0..15,
  retains 84 bins, and transposes the target by minus the crop offset. Validation
  fixes crop start 12 and averages the two segment probability distributions.
- One configuration, seed 13, FP32, AdamW LR 0.0003 / weight decay 0.0001,
  smoothing 0.1, batch 16, cosine schedule, at most 120 epochs, early stopping after 25 epochs without a new best. Combined validation
  per-track symmetric MIREX selects checkpoints; ties retain the earlier epoch.
- Compare each domain with the frozen adapted seed-13 artifact on identical rows.
  Qualify only with FMAK gain >=2 points and Beatport loss <=2 points. Confirmation
  is conditional; no recipe sweep, larger-model tier or product promotion.
- Preparation cap 20 minutes, fitting cap 15 minutes; shell cap 40 minutes for the
  sequential stage. Save preparation progress and epoch-boundary optimizer,
  scheduler and RNG state. Budget exhaustion is not a completed accuracy result.
- Use GPU 0 only with at least 8 GiB currently free; PyTorch allocator fraction
  0.04 of its 96 GiB (~3.8 GiB, excluding CUDA context/external allocations).
  Preserve all vLLM instances. Check current headroom before each stage.

The experimental checkpoint format differs from the deployed v2 feature contract;
ordinary Drone loading must reject it. Accuracy comes first; Rust runtime work
and final real-library performance testing follow a validated quality candidate.

## Confirmation after the qualifying pilot

The pilot stopped at epoch 66 (best 41) in 432.39 seconds. Beatport validation
MIREX 72.94%, FMAK validation 68.76%: gains 11.51 and 4.80 points versus the
frozen adapted seed-13 reference, respectively. This passes the fixed gate.

Confirm the same recipe at seeds 13, 37, 73, repeating 13 with the seed-selectable
CLI source used for the other seeds. Reuse identical immutable CQT arrays and
identity lists. Report per-domain means/sample SD and verify the repeated pilot
history/predictions. Per-run cap remains 15 minutes; sequential stage cap 50
minutes. No new architecture, optimizer, frontend, crop or stopping-rule search.
Canonical artifact remains seed 13.

Before seeing seeds 37/73, predeclare one optional inference candidate: equal-
probability averaging of all three models (each already averages its two segments).
No seed subset/weight search. Choose it only if combined validation MIREX gains
at least 1 point over canonical seed 13 and neither domain loses more than 0.5
points. Otherwise retain canonical 13. Three models total 141,852 parameters /
567,408 FP32 parameter bytes, excluding buffers and metadata.

A model is eligible for later final evaluation only if the confirmation mean
meets the native validation references in both domains (65.16% Beatport, 66.76%
FMAK) and the chosen canonical/ensemble candidate also meets both. Passing this
is permission to freeze/evaluate, not a state-of-the-art or product claim.
Calibration and final 1,000 rows remain unused during confirmation/selection.
