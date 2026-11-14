# Method directions from the literature

This document reviews published methods that could improve Drone's key accuracy
or CPU cost, and records whether each one is likely to transfer to a
109,128-parameter key classifier. It was last updated on 12 November 2026. It
does not restate experiment results: those are in the
[research index](../experiments/README.md), and the release recipe is in
[training.md](training.md).

The goal is higher musical-key accuracy with efficient CPU execution inside
Euphoria; novelty is not a goal. Training-time complexity is acceptable if the
deployed classifier stays small, but every transferred method needs its own
controlled result on our data.

## Directions and transfer limits

| Direction | Primary evidence | Assessment for Drone | Tested here |
|---|---|---|---|
| Weight averaging (EMA) | [EMA study](https://arxiv.org/abs/2411.18704), TMLR 2024 | Plausible, but the paper's SGD evidence does not establish transfer to our AdamW recipe. Drone has six BatchNorm layers, and [PyTorch's averaging documentation](https://docs.pytorch.org/docs/main/generated/torch.optim.swa_utils.AveragedModel.html) notes their statistics need recalibration, so averaging and recalibration must be separated. | [Rejected](../experiments/studies/k10-ema/REPORT.md) |
| Sharpness-aware optimization | [Spectral SAM + Muon](https://arxiv.org/abs/2607.26001), July 2026 | ImageNet ViT/ResNet results. Changing perturbation geometry and optimizer together would hide which change helped; a single fixed SAM recipe is the cleaner test. | [Fixed SAM rejected](../experiments/studies/k10-sam/REPORT.md) |
| Train large, deploy small (distillation) | [EfficientAT](https://arxiv.org/abs/2211.04772), ICASSP 2023; [S-SONDO](https://arxiv.org/abs/2604.24933), ICASSP 2026 | Promising once a teacher that is actually better at key detection exists. Audio tagging and embedding results do not show transfer to absolute pitch. S-SONDO retains up to 96% of teacher performance, which is not lossless, and distilling a weaker teacher preserves its errors. | No qualified teacher yet ([ensemble](../experiments/studies/k10-ensemble/REPORT.md), [MusicFM](../experiments/studies/musicfm-key-pilot/REPORT.md)) |
| Hardware-aware model design | [LFM2 report](https://arxiv.org/abs/2511.23404), November 2025 | Transfer the practice of measuring on target hardware. Its language and speech models (350M+ parameters) are a different scale and task. | [CPU studies](../experiments/README.md#cpu-and-native-inference) |
| Explicit edge constraints | [DCASE 2025 low-complexity results](https://dcase.community/challenge2025/task-low-complexity-acoustic-scene-classification-with-device-information-results) | Evaluate accuracy, memory and compute together. Scene classification results are not key benchmarks. | [CPU studies](../experiments/README.md#cpu-and-native-inference) |
| Harmonic-structure loss | MIREX weighting of related keys | Partial credit for neighbouring keys may trade exact accuracy for weighted score; this needs a direct test rather than an assumption. | [Rejected](../experiments/studies/k10-structured-targets/REPORT.md) |
| Chunked inference | Standard overlap-and-trim for convolutional networks | K10's six temporal 3-wide convolutions and three stride-2 pools give output stride 8 and receptive field 36, so aligned chunks with trimmed halos can reproduce full-track inference while bounding memory. Real-arithmetic equivalence does not imply bitwise FP32 equality. | [Passed](../experiments/studies/k10-chunked/REPORT.md) |

## Further sources reviewed

[Key lime](https://arxiv.org/abs/2611.02408) (2 November 2026) is directly
relevant: a 21,528-parameter transformer combines a chroma sequence with DSP key
scores. Its reported 0.32 ms is model inference only, excluding about 486 ms of
DSP analysis. Its FMAK split is track-disjoint; our stricter audio, artist and
album separation and different fitting pool prevent direct score comparison. A
reproduction must use our split. Our failed DSP-prior fusion pilot (see
[RESEARCH_TRAINING.md](RESEARCH_TRAINING.md#what-came-next)) used a different
architecture, so neither result settles the other.

[USAD 2.0](https://arxiv.org/abs/2606.06444) (Interspeech 2026) weights teacher
losses by input domain and includes a music expert. Its smallest encoder has 25M
parameters, so it is a possible training-time representation source, not a CPU
replacement. Pitch preservation and training-data overlap would need checking.

[Teacher selection for fine-grained recognition](https://arxiv.org/abs/2605.15689)
(CVPR FGVC13 2026) studies compatibility across teachers, students and datasets.
It supports testing teacher suitability rather than choosing by size; its image
selection metric is only a hypothesis for music.

[Layer-wise music representation analysis](https://arxiv.org/abs/2608.14819)
(ISMIR 2026) finds that standard intrinsic quality metrics fail on tonal tasks
and proposes pitch-transposition equivariance as a diagnostic. Teacher and layer
compatibility should be checked this way before any distillation.

[MT²](https://arxiv.org/html/2507.12996v1) reports 73.3 key MIREX for its
single-task equivariant ViT-CQT versus 71.3 for averaged multitask tokens (Table 1).
Broad multitask gains do not imply the best key classifier.

## Evaluation constraints for any new direction

Repeated use of the validation panels is adaptive, so a passing pilot only earns
a frozen evaluation; it does not show state of the art or replace release
weights. Seeds measure training variability, not coverage of unseen recordings or
artists. The earlier GiantSteps and FMAK final sets are exposed. The Schubert
Winterreise Dataset is a 24-work classical stress test, not broad deployment
evidence. The private DJ library is reserved for final performance testing and
manual review and has no trusted labels.
