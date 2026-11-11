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
