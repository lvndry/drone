# Pitch-circle joint fit v1: fixed architecture comparison

Pinned before fitting. This protocol authorizes one architecture, the exact
73,506-parameter `pitch-circle-v1` model from `1545d2f`. The earlier SPEC/geometry
verification did not authorize fitting; this separately reviewed protocol does.
No kernel/folding/activation/dropout variants, teacher, smoothing, width sweep,
private DJ data, calibration, final-set or SWD scoring, or automatic promotion.
90% exact recognition remains a stretch objective, not a promised outcome.

## Hypothesis and controlled recipe

Test the shared-tonic/retained-octave architecture against ordinary joint K10.
Only model architecture and its research checkpoint representation change. Preserve
all clean identities and audio hashes: BP 1,230 plus FMAK 976 fitting; BP 126 plus
FMAK 250 validation, including previous artist/album/audio quarantine. Reuse the
same validated 208-padded/192-core librosa CQT cache, 24 bins per octave, hop 4096.

Use FP32, batch 32, random 320-frame crops, uniform shifts −4…4 semitones with label
minus shift, hard cross entropy, AdamW LR 0.001 / weight decay 0.0001, cosine horizon
120 epochs and patience 25. Select the earliest best combined per-track symmetric
MIREX checkpoint. Full-track true-length validation and all RNG seed rules stay
unchanged. Canonical pilot is seed 13. Different architecture initialization consumes
different Torch draws; do not claim identical hidden tensors/dropout draws across
architectures. NumPy data-order/crop/shift policy remains identical.

## Source and control evidence

Freeze integration from architecture commit `1545d2f`; generic paired helpers are
copied from executed capacity source `679b7fd`, with one optional research-loader
extension. Pin source maps and normalized fitting-math components in
`data/fit-reviewed-sources.json`. Component audit covers data selection, crop/shift
sampling, optimizer/scheduler, hard CE, validation score, horizon, selection and
stop condition. It is not arbitrary program-equivalence proof. The actual source
review, tests and mandatory tensor/history replay comparison provide other evidence.
Generic replay/scoring helper audit permits only path relocation and explicit
budget injection while retaining the default 1,200-second limit.

Reverify reused control 13 from `36d9e4` against archived joint `673e280`: complete
status, base config, feature SHA, fit/validation IDs, every epoch loss/score, selected
and stop epochs, early-stop status, predictions/metrics, checkpoint file hashes,
state keys, tensor dtypes/shapes and values must match exactly. Check the original
structured control result/run/checkpoint/validation hashes as well. Do not trust a
pass flag alone or weaken tolerances. Archived raw artifacts stay immutable.

A passing pilot earns seeds 37 and 73. Before each candidate, fit an unchanged
ordinary joint control with the same frozen driver if no approved exact-reproduced
control already exists. The shared default path remains literal KeyCNN/hard CE and
production checkpoint metadata. Reverify each control against its archived joint
reference before candidate fitting. At most three candidates and three controls;
verified control 13 can be reused, so normally five additional fits at most.

## Research checkpoint and shared loop

One shared fitting loop accepts typed optional factory/writer/loader hooks. Defaults
remain KeyCNN and the existing production saver. Only `--pitch-circle-probe` supplies
the fixed research hooks. No second training loop or production model dispatch is
introduced. Research checkpoints use kind `drone-key-research-checkpoint`, format
version 1, architecture `pitch-circle-v1`, exact feature/label/model specs, FP32 state,
selected epoch, seed/data/source identities. Strict loading checks schema, source
identity, keys, shapes, dtypes and finite values. They cannot be loaded as
`KeyCNN(width=...)`; production checkpoints are rejected by the research loader.

Save protocol, source/feature/control hashes, model spec and audit before fitting.
Save actual selected logits, IDs, audio hashes, references, metrics and checkpoint
hash at completion. Preserve Torch CPU/CUDA RNG during diagnostic loading/scoring.

## Budget and safeguards

GPU 0 only, existing `.04` allocator fraction, at least 8 GiB free before setup,
four CPU threads, deterministic settings and TF32 off. Preserve vLLM. No raised cap,
precision/batch change or reduced horizon. Hard 20 minutes (1,200 seconds) per fit,
including final scoring/artifact work; outer wrapper timeout 1,250 seconds. No
extension after observing results. Budget/resource failures are inconclusive and
cannot qualify. Log caps, available memory, CUDA peaks, epochs and timings.

Pilot requires at least **1.0 FMAK symmetric-MIREX point** above ordinary joint
RAW control, no Beatport MIREX decline, and no exact-accuracy decline in either
domain. Only that passing pilot earns the fixed 37/73 confirmation. Final three-seed
means must pass the same method gate and separately gain at least **2.0 FMAK MIREX
points over archived BP-only controls**, with no Beatport decline and no domain
MIREX/exact regression against ordinary joint. Canonical seed remains 13, never a
lucky seed. Passing permits a later audited external freeze, not automatic promotion.

Report both existing symmetric-fifth MIREX (gate) and asymmetric MIREX
(`score_mirex(..., symmetric=False)`: ascending same-mode fifth only), exact accuracy,
and paired exact/fifth/relative/parallel/other movements under both conventions,
including exact promotions/demotions. Reused development observations and seed means
are not independent generalization evidence.

## Symmetry and performance interpretation

The model is equivariant to abstract rotations inside the folded pitch channels.
Real finite-band CQT reslicing also moves energy between octave channels and band
edges; it is not that exact group action. Preserve this distinction in all reporting.
The earlier MAC/tensor estimates do not establish CPU latency or peak RSS. Benchmark
CPU deployment only if the accuracy gate earns further work; no claimed speedup or
product tier follows this fit automatically.

## Pilot command

Run from this frozen integration source, using the established server environment:

```sh
timeout 1250 python -m experiments.k10_multidomain train \
  --arm joint --seed 13 --device cuda:0 --pitch-circle-probe \
  --data /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/data \
  --reference-run /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/joint-seed13/run.json \
  --reference-bp /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/control-seed13/run.json \
  --paired-control /home/lysk-admin/drone-experiments/k10-structured-targets-20261111-v1/control-seed13/run.json \
  --output /home/lysk-admin/drone-experiments/pitch-circle-20261112-v1/candidate-seed13
```

Only a complete passing `pitch-circle-result.json` earns confirmation. For seeds 37
and 73, first run unchanged ordinary joint fitting with this driver (omit all probe
and paired-control flags), then pass that control's `run.json` to the candidate.
Replace seed-dependent paths and add `--pilot-run` pointing to candidate 13's
`pitch-circle-result.json`. Retain the same data and all settings.

Summarize with `python -m experiments.pitch_circle_fit --results <candidate13-result>
<candidate37-result> <candidate73-result> --output <summary.json>`. No follow-up
architecture variant is authorized by a negative or inconclusive result.
