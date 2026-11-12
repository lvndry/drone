# Classifier research decisions, updated 12 November 2026

The product objective is higher musical-key accuracy with efficient CPU execution
inside Euphoria. Novelty is not a success criterion. Training-only complexity can
be useful if the deployed classifier remains small; every proposed transfer needs
its own controlled result. Three specialist passes covered classifier training,
small CPU/decision models and evaluation, followed by cross-critique.

## Current evidence

The completed [K10 v2 matched-data pilot](../experiments/studies/k10-v2-pilot/REPORT.md)
uses nested976/4033 FMA fitting pools, shared1229 BP and new common inner panels.
Its +1.60 FMA exact points missed the fixed +2.00 gate; no confirmation or promotion.
The primary5–10minute house use remains undercovered: only5 inner House182 tracks /
3 groups, and no verified deep/melodic-house panel. Canonical preparation required
an explicit3-input eligibility overlay; the failed original receipt is preserved.

The shipped K10 model remains the release candidate. The completed
[multi-domain comparison](../experiments/studies/k10-multidomain/REPORT.md) gained 2.38
Beatport / 1.60 FMAK MIREX points over its corrected control, but missed its fixed
two-point FMAK gate. Neither a promising pilot nor a newer paper overrides that
decision. The [CPU baseline](../experiments/studies/k10-cpu/REPORT.md) also shows that
frontend work and long-track activation memory need attention, not just weights.

## Methods and transfer limits

| Direction | Primary evidence | Decision for Drone |
|---|---|---|
| Weight averaging | [EMA study](https://arxiv.org/abs/2411.18704), TMLR 2024 / arXiv November 2024 | Fixed EMA pilot completed and rejected below. The study's SGD evidence did not establish transfer to our AdamW recipe. |
| New optimizer geometry | [Spectral SAM + Muon](https://arxiv.org/abs/2607.26001), July 2026 | Its ImageNet ViT/ResNet results are interesting; changing perturbation geometry and optimizer together would obscure attribution here. Defer. |
| Train large, deploy small | [EfficientAT](https://arxiv.org/abs/2211.04772), ICASSP 2023; [S-SONDO](https://arxiv.org/abs/2604.24933), ICASSP 2026 | Strong future direction once an independently stronger key teacher exists. Audio tagging/embedding results do not prove absolute-pitch/key transfer. |
| Hardware-aware model design | [LFM2 report](https://arxiv.org/abs/2511.23404), November 2025 | Transfer the practice of measuring target hardware and using training-time distillation. Its language/speech workloads and 350M+ parameter family are a different scale and task from this 109k classifier. |
| Typed, calibrated decisions | [System One documentation](https://docs.typesafe.ai/concepts/system-one) | Retain explicit key classes and validated uncertainty. A structured output interface or calibrated group probabilities does not establish local acoustic accuracy or guarantee an individual result. |
| Explicit edge constraints | [DCASE 2025 low-complexity results](https://dcase.community/challenge2025/task-low-complexity-acoustic-scene-classification-with-device-information-results) | Evaluate accuracy, memory and compute together. Scene/device classification results are not musical-key benchmarks. |

S-SONDO's reported compression retains **up to 96%** of teacher performance; this
is not evidence of lossless compression. Distilling a teacher that is not better
at key detection can simply preserve its errors. General acoustic embeddings may
emphasize timbre or invariance that loses the pitch information this task needs.

The [EMA paper](https://arxiv.org/abs/2411.18704) supports a practical averaging
hypothesis, while [PyTorch's averaging documentation](https://docs.pytorch.org/docs/main/generated/torch.optim.swa_utils.AveragedModel.html)
explicitly calls out BatchNorm statistics. Drone has six BatchNorm layers. A
comparison of EMA plus recalibration against ordinary weights without equivalent
recalibration would confound two interventions.

## Completed accuracy experiment

Use one fixed EMA decay and unchanged training data, augmentation, optimizer,
schedule and architecture. Ordinary weights alone select the training checkpoint
and stopping epoch. Save the contemporaneous EMA, then recalibrate each saved
copy once on an identical frozen fitting-only crop/batch schedule, with dropout
disabled and only BatchNorm updating. Compare ordinary raw, ordinary recalibrated
and EMA recalibrated results to separate the effects.

The review rejected per-epoch recalibration and independent EMA checkpoint search:
they add compute, can change the wall-clock horizon and introduce more validation
selection. It also rejected distillation as the immediate next step because no
independently stronger key teacher is currently established in our artifacts.

The EMA increment has its own predeclared hurdle, while the original two-point
FMAK improvement versus the corrected Beatport control remains required before
further qualification. The small historical data-only gain is not reclassified
as a pass. Canonical seed 13 is fixed in advance; no decay/seed/ensemble sweep.

The [completed EMA pilot](https://github.com/lvndry/drone/pull/8) reproduced RAW
exactly but failed: EMA plus recalibration lost 1.84 FMAK MIREX points against
ordinary weights with identical recalibration. Recalibration also hurt ordinary
weights. No other seeds were run and no release weights changed.

The next proposed accuracy experiment changes only the existing K10 loss:
structured smoothing with 0.8 on the true class and 0.2 divided among harmonic
neighbours using the existing relation weights. This is an unproven hypothesis,
not direct expected-MIREX optimization. Keep the same data, seed, augmentation,
optimizer and inference. A seed-13 improvement of at least one FMAK MIREX point
with no MIREX or exact decline in either domain earns confirmation on 37/73;
three-seed means must retain the separate two-point original control hurdle.
No smoothing sweep, architecture change, curriculum or teacher is bundled in.

## Completed performance experiment

K10's six temporal 3-wide convolutions and three stride-two pools give output
stride eight and receptive field 36. Align chunks to the pooling grid, preserve
the true final input tail, trim halo outputs and count every final temporal
position exactly once. Sum hidden features, perform the original global average
and apply the unchanged linear head once. This retains all track evidence.

The review rejected averaging equal-weight crop predictions and applying the
linear head independently at every temporal position. The latter can be
mathematically valid but multiplies head computation unnecessarily. The prototype
uses a separate small ONNX head to avoid introducing another BLAS implementation.

Real-arithmetic equivalence does not imply bitwise FP32 equivalence. The unchanged
full ONNX artifact remains the reference. Boundary/remainder/seam tests and
label-blind parity on the existing public validation inputs must pass, including
zero key disagreements. Measured logit tolerances are engineering checks, not
universal error bounds for future near-ties. No quantization or input truncation.

The [completed chunk test](https://github.com/lvndry/drone/pull/9) passed all 376
validation inputs with zero changed keys. Paired cached-feature probes reduced
peak process RSS from 865 MiB to 129 MiB with a 1.91% median paired latency
increase. This passes the fixed development gate; native frontend/application
measurements remain necessary before making an Euphoria performance claim.

## Additional source review

[Key lime](https://arxiv.org/abs/2611.02408), submitted 2 November 2026,
is a directly relevant candidate: a 21,528-parameter transformer combines a chroma
sequence with DSP scores. Its reported 0.32 ms is added model inference, excluding
roughly 486 ms DSP analysis. Its FMAK split uses track disjointness; our stricter
audio/artist/album separation and different fitting pool prevent direct score
comparison. A future reproduction must preserve our split. Our failed DSP-prior
pilot does not test this sequence architecture; neither result settles the other.

[USAD 2.0](https://arxiv.org/abs/2606.06444), Interspeech 2026, weights teacher
losses by input domain and includes a music expert. Its smallest reported encoder
is 25M parameters, so it is a possible training-time representation source rather
than a justified replacement for the CPU default. Key-specific transfer, pitch
preservation and training-data overlap would need auditing before use.

[Teacher selection for fine-grained recognition](https://arxiv.org/abs/2605.15689),
CVPR FGVC13 2026, studies compatibility across teachers, students and datasets.
This supports testing teacher suitability instead of choosing solely by size.
Its image-domain selection metric is a hypothesis for music, not an established
key-classification recipe.

[Layer-wise music representation analysis](https://arxiv.org/abs/2608.14819),
ISMIR 2026, finds standard intrinsic quality metrics fail on tonal tasks and
introduces pitch-transposition equivariance as a tonal diagnostic. This makes
teacher/layer compatibility worth checking explicitly before any distillation.

[MT²](https://arxiv.org/html/2507.12996v1) is relevant to future multitask work,
but Table 1 reports 73.3 key MIREX for its monotask equivariant ViT-CQT versus
71.3 for averaged multitask tokens. Broad transfer gains do not imply the best
key classifier. Later capabilities remain separate from the current release.

## Evaluation constraints

Repeated validation use remains adaptive. Seeds measure training variability,
not coverage over unseen recordings or artists. A passing development experiment
only earns further frozen evaluation; it does not demonstrate state of the art
or automatically replace release weights. The prior GiantSteps/FMAK finals are
exposed. Prepared SWD is a 24-composition classical stress test, not broad
deployment evidence. The private DJ library remains for final performance/manual
review and supplies no trusted ground-truth labels.
