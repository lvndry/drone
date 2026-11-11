# Fixed K10 width-2 capacity test v1

Declared BEFORE fitting. One capacity hypothesis, hard CE, clean joint data.
Width1 has109128 parameters; width2 has361080. Test whether additional supervised
capacity yields a stronger own-key teacher before any distillation. No product
size tiers, width/hyperparameter sweeps, soft targets, EMA, curriculum, teacher
supervision, new fitting data or automatic weight promotion.90% EXACT is a user
stretch objective, not a forecast or a promised result from this experiment.

## Controlled design

Freeze from maina1abac9ddad927830b0409f80d6908502eeab2cd. The archived references
use673e280ca244d4187d8c20f9fdaecaa1c75281fd. Reusable seed13 control was fitted from
36d9e4164dcffc1192c2aebb60aa6d80dc2a24d1 and exactly reproduced archive history,
selected/stop epochs, per-track predictions/metrics and checkpoint tensors.
Re-verify those exact comparisons and all run/checkpoint/result/validation hashes
before fitting the capacity candidate; never merely trust a stored pass flag.

Pin all three source maps and normalized component-level training-math audits in
data/reviewed-sources.json. Current and archived source hashes/differences and
math audit are recorded before fit. Generic EMA evidence helpers have a separate
AST audit permitting path-only loader relocation and explicit scoring-budget
injection while preserving replay/scoring/RNG math and default1200s budget.
The normalization scope is named; no archived source/protocol bytes are edited.
Audit covers data selection, sampling/crops,
seed, optimizer/scheduler, batch math, score, horizon, selection and stop condition;
it is not a proof of arbitrary program equivalence. Reviewed source and CPU tests
complement it. New capacity hooks permit only width/model-config, memory guard,
budget and diagnostic metadata differences. Archived raw artifacts stay immutable.

Candidate model is KeyCNN(width=2.0), widths32/64/128 instead16/32/64, otherwise
same layers/head structure. Save width2 in versioned checkpoint model_config and
verify load_checkpoint reconstructs its361080-parameter shape. Control uses the
literal existing KeyCNN() and hard-CE call. Both use existing torch seed13/37/73;
changed parameter count necessarily changes random initialization draws and later
Dropout RNG state, so do not claim shared channel tensors are initialized equally.
NumPy data-order/crop/transpose schedule and recipe stay the same.

Preserve frozen Beatport1230+FMAK976 fitting, BP126+FM250 validation, content hashes,
artist/album/audio quarantine, true librosaCQT208/192bins/24bins-octave/hop4096,
320frame random crops, uniform shifts-4..4 and label-minus-shift. Hard CE FP32,
AdamW1e-3/wd1e-4, batch32, cosine120epochs, patience25, earliest best combined
per-track symmetric-MIREX checkpoint, full-track true-length validation unchanged.
No BN recalibration, precision switch, shorter horizon, batch adjustment, sweep,
private DJ data, calibration/final/SWD scoring or fishing for another test set.

## Budgets and memory fixed upfront

Candidate hard45min (2700s) per fit INCLUDING final scoring/artifact work; control
hard20min (1200s). This change is approved upfront because extra capacity may reach
120epochs and exceed20min; do not extend budgets after observing results.
At most3candidates and3controls, using verified reusable control13 when available.
A timeout is budget-INCONCLUSIVE, not an accuracy-negative experiment. Do not
resume beyond cap, shorten horizon or pick another width after timeout/OOM.

GPU0 only, vLLM preserved. Existing device_setup sets deterministic/TF32off settings,
four CPU threads, free>=8GiB and allocator0.04. ONLY width2 may raise fraction to
0.08 after a separate free>=12GiB guard. Log prefit free/total bytes, fraction and
actual cap bytes plus CUDA peak allocated/reserved and final free bytes. Report
resource failure separately; never stop serving workloads to make the fit work.

Forecast, not measured CUDA peaks: local CPU autograd saved-storage inspection at
batch1/192x320 gave47,139,008B width1 and94,603,648B width2. Scaling tobatch32 gives
1439 versus2887MiB upper estimates of stored tensors, excluding CUDA workspace,
allocator and full-track validation effects. Model/gradient/Adam states add only
~1.67 versus5.51MiB. Thus7.68GiB cap on96GiB GPU is plausible, not guaranteed.
Same-epoch runtime estimated2–4x;120epochs could exceed20min, motivating45min cap.

## Gate and reporting

Canonical candidate13 qualifies only at >=1.0 FMAK MIREX point over ordinary RAW
joint control, with no Beatport MIREX decline and no EXACT decline in either domain.
Only a passing pilot earns37/73, each preceded by an exact-reproduced hardCE joint
control if not already verified. No confirmation after negative/inconclusive pilot.
Final three-seed means require the same method gate, separately >=2.0 FM MIREX
points over archived BP-only controls/no BP decline, and no domain MIREX or exact
decline versus ordinary RAW joint. Never choose lucky seed: canonical13. Passing
means eligibility for later independently audited external freeze, not promotion.

Report both existing symmetric-fifth MIREX (gate) and existing asymmetric score
(score_mirex symmetric=False: ascending fifth only), exact, and paired category
movement matrices exact/fifth/relative/parallel/other under both conventions,
including exact promotions/demotions. Save actual logits/IDs/audioSHA/refs/metrics,
source/protocol/data/input/checkpoint/artifact hashes, epochs, timing and memory.
Three seed means on reused development identities are not independent evidence of
broad generalization,90% exact performance or frontier novelty.

## Invocation

Use frozen source with existing server environment. If reusing structured control13:

```sh
timeout 2750 python -m experiments.k10_multidomain train \
  --arm joint --seed 13 --device cuda:0 --capacity-probe \
  --data /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/data \
  --reference-run /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/joint-seed13/run.json \
  --reference-bp /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/control-seed13/run.json \
  --paired-control /home/lysk-admin/drone-experiments/k10-structured-targets-20261111-v1/control-seed13/run.json \
  --output /home/lysk-admin/drone-experiments/k10-capacity-20261111-v1/candidate-seed13
```

Only on pilot pass run seeds37/73, adding --pilot-run pointing to candidate13's
capacity-result.json. First fit each fresh control using --capacity-control,
timeout1250 and its archived seed-specific references; omit --paired-control.
Then fit width2 with timeout2750, same references and --paired-control pointing at
that fresh control's run.json. Maximum6fits; no extension/sweep after any result.
Summarize with python -m experiments.capacity --results <candidate13-result>
<candidate37-result> <candidate73-result> --output <summary.json>.
