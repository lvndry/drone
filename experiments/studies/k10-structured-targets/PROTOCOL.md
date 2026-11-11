# K10 structured-targets v1 — pinned before fitting

One loss-only experiment, never an accuracy-for-speed trade. No automatic weight
promotion, architecture/frontend/teacher/curriculum/data/augmentation changes.
This is inspired by key lime (arXiv:2611.02408, 2026-11-02), whose study does not
isolate structured targets versus hard CE. It tests a hypothesis, not a reproduced
paper result. The negative47k DSP-prior trial used uniform0.1 smoothing instead.

Freeze from main40afede7ca6af96d740086eab2616f9ec98d3321. Archived joint and BP-only
references come from673e280ca244d4187d8c20f9fdaecaa1c75281fd. Pin both source maps
in data/reviewed-sources.json and record every archived/current hash change.
Only shared-loop hooks and the new helper may differ from reviewed base.

Control uses original hard cross entropy unchanged. Candidate uses q(true)=0.8;
remaining0.2 distributed to BOTH same-mode fifth directions, relative and parallel
classes in proportion to existing key_relation(...,symmetric=True)/WEIGHTS values.
Thus each fifth receives0.2*(0.5/1.5), relative0.2*(0.3/1.5), parallel0.2*(0.2/1.5),
all other classes0. Existing tonic*2+minor contract defines all neighbours. Assert
normalization and transposition equivariance over every class/shift in tests.
This is structured soft-target CE, not direct expected-MIREX optimization.
Record loss_spec separately: base config is exactly unchanged; loss is the only
controlled pair difference. No lambda sweep or EMA/BN intervention.

Preserve fixed Beatport1230+FMAK976 fitting and Beatport126+FMAK250 validation
identities, content hashes, true librosa24-bin/octave CQT208/192/hop4096, full-track
validation, random320frame training crops, uniform shifts-4..4/label-minus-shift,
K10 architecture109128params, initialization/order/RNG, FP32 AdamW1e-3 wd1e-4,
batch32, cosine120, patience25, earliest best combined symmetric-MIREX selection.
Do not shorten horizon. Candidate selection uses its own validation trajectory
under that same fixed rule. No new holdout scoring, calibration, SWD or private DJ.

Run exactly paired control13 first. Require exact archived RAW config/identities/
featureSHA/history(loss and scores)/best&stop epochs/early-stop flag/per-track
predictions and metrics, plus tensor dtype/value equality of archived best.pt.
No relaxed tolerances or alternate controls after failure. Existing EMA replay is
supporting evidence; this fresh shared-driver control must independently reproduce.
Only successful control result + matching run hash permits candidate13 fitting.
Save protocol, source hashes, target matrix and archived input hashes before fit.
Report both existing symmetric-fifth MIREX (the gate) and existing score_mirex
symmetric=False (ascending same-mode fifth only, directional relative relation).
Do not switch conventions or add a threshold. For candidates report paired
exact/fifth/relative/parallel/other movement matrices under BOTH conventions,
including exact promotions/demotions, to expose weighted-score gaming.
Save actual selected validation logits/IDs/audio hashes/refs/metrics and checkpoint
hashes at end. Preserve Torch CPU/CUDA RNG around final diagnostics.

Pilot candidate13 needs >=1.0 FMAK MIREX point over fresh ordinary joint control,
no Beatport MIREX decline, and no exact decline in either domain to earn seeds37/73.
For each confirmation seed first reproduce its fresh hardCE control exactly, then
fit candidate with unchanged protocol/source/config. Require pilot artifact as
explicit input. At most6fits total (three controls and three candidates), no sweep.
Final three-seed means must satisfy that same method-specific gate, separately
>=2.0 FMAK MIREX points over archived BP-only/no Beatport decline, and no MIREX or
exact decline in either domain versus ordinary RAW joint. Never choose best seed;
canonical13. Passing means eligibility for later external freeze, not promotion.

GPU0 only; preserve vLLM and device_setup>=8GiBfree/allocator0.04/TF32off/fourCPU
threads. Hard20min per fit INCLUDING final scoring/artifacts, external timeout1250s.
Record training/total timings, epochs, checkpoint/artifact versions and all input
hashes. Failure/budget exhaustion cannot qualify. Reused development results and
three seed means do not establish independent generalization or frontier accuracy.

## Pilot invocation

Run from this frozen source copy with the existing server Python environment.

```sh
timeout 1250 python -m experiments.k10_multidomain train \
  --arm joint --seed 13 --device cuda:0 --structured-control \
  --data /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/data \
  --reference-run /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/joint-seed13/run.json \
  --reference-bp /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/control-seed13/run.json \
  --output /home/lysk-admin/drone-experiments/k10-structured-targets-20261111-v1/control-seed13
```

Only after `structured-result.json` reports complete and control_reproduced=true:

```sh
timeout 1250 python -m experiments.k10_multidomain train \
  --arm joint --seed 13 --device cuda:0 --structured-targets \
  --data /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/data \
  --reference-run /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/joint-seed13/run.json \
  --reference-bp /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/control-seed13/run.json \
  --paired-control /home/lysk-admin/drone-experiments/k10-structured-targets-20261111-v1/control-seed13/run.json \
  --output /home/lysk-admin/drone-experiments/k10-structured-targets-20261111-v1/candidate-seed13
```

Only a passing candidate13 earns the same control/candidate pair for37 and73.
Replace every seed-dependent reference/output path, retain `--data`, and add
`--pilot-run /home/lysk-admin/drone-experiments/k10-structured-targets-20261111-v1/candidate-seed13/structured-result.json`
to EACH confirmation control/candidate invocation. No other configuration changes.
Summarize via `python -m experiments.structured --results <candidate13-result>
<candidate37-result> <candidate73-result> --output <summary.json>`.
