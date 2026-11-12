# Fixed vanilla Euclidean SAM with existing AdamW

Frozen before any control/candidate fit. One optimization hypothesis only: K10 may
learn more useful key evidence if every update minimizes CE near a worst-case
parameter perturbation. This is an established method transferred experimentally,
not a new algorithm or proof of audio-domain benefit. A negative canonical pilot
ends optimizer variants and moves priority to auditable data/self-supervision.

## Evidence and limitations

Original SAM (October 2020, revised April 2021) provides a fixed rho=.05 sensitivity
study: https://arxiv.org/html/2010.01412v3 . The official implementation couples the
PRNG key across passes and retains only unperturbed-pass BN state:
https://github.com/google-research/sam/blob/main/sam_jax/training_utils/flax_training.py .
Balanced feature learning (May 2024) motivates possible suppression of overly easy
features, but its DomainBed results use target-domain trained probes and parameter
sweeps: https://arxiv.org/html/2405.20439v1 . This is not direct evidence of music-key
or zero-shot generalization. July 2026 spectral SAM/Muon uses different geometry,
optimizer routing and tuning; no such changes enter this study:
https://arxiv.org/html/2607.26001v1 . August 2026 EMASAM introduces EMA-guided
perturbations; it is also excluded: https://arxiv.org/html/2608.15105v1 .
Prior negative EMA, structured targets, width 2, PitchCircle and fixed tonal-ranker
studies do not establish the reason for K10 errors or imply all alternatives fail.
A SAM failure rejects this fixed recipe, not every SAM radius or optimization.

## Frozen data and ordinary recipe

Existing cache only: 2,206 fitting (1,230 Beatport + 976 FMAK), 376 validation
(126 + 250). Same frozen manifest, artist/album/audio quarantine, cache hashes and
feature identities. No raw decoding, additional audio, teachers, calibration159,
exposed GS/FMAK finals, SWD or private tracks. Full-track validation remains
unchanged; reuse of development data cannot establish independent generalization.

K10 109,128 parameters; existing CQT24 192 core / 208 padded bins, hop4096.
Train crops320 frames, exact ±4-semitone reslice and label minus shift, batch32
including final partial batch. Hard CE. Existing AdamW lr1e-3, weight_decay1e-4,
existing betas/epsilon/defaults unchanged. Cosine120 epochs, earliest strict-best
combined symmetric-fifth MIREX checkpoint, patience25. No shortened horizon,
clipping, smoothing, mixup, EMA, final BN recalibration, warmup or extra variants.
The ordinary forward/backward/AdamW block remains literal in the disabled branch;
its AST and executable multi-step parity are checked against the pre-change base.

## Exact candidate operation

One optional typed training-step hook in the same shared fitter; no copied loop.
Fixed rho=.05, Euclidean global norm over gradients of **all trainable parameters**,
including biases and BN affine parameters. For the same already constructed batch:

1. Save CPU/active-CUDA RNG; ordinary first forward, hard CE, backward. Save
   post-first RNG, exact original parameter tensors and post-first model buffers.
2. Perturb theta by `.05*g/(global_l2_norm(g)+1e-12)`, with no differentiation
   through the perturbation. First gradient has CE only; AdamW decay stays outer.
3. Clear gradients, restore pre-first RNG, second forward/CE/backward on the same
   inputs/targets and same Dropout mask. BN stays in training mode and uses batch
   statistics on both passes, with existing momentum unchanged.
4. In `finally`, copy exact original tensors back (not add/subtract roundtrip),
   restore post-first buffers, including all6 BN running means/variances/counters,
   and restore post-first RNG. Apply one original AdamW step using second gradients.
   No step is made if the second pass raises or gradients/loss/norm are nonfinite.

Only first-pass BN buffers persist; the next batch sees one-forward RNG advancement.
NumPy augmentation is outside the hook and is never repeated. Training history
records ordinary first-pass CE, not perturbed CE. Radius0 is a CPU formula/parity
test facility only; no runtime study CLI exposes it. Candidate inference/model
parameters/graph remain K10, so no inference performance benefit is presumed.

## Fresh control proof BEFORE SAM fitting

Canonical seed13 first runs this SAME new frozen driver with hook disabled. It must
exactly reproduce archived source673 joint history, selected/stopped epochs,
validation predictions/metrics, config, feature-manifest SHA and fit/val IDs.
Actual selected checkpoint state-dict keys, dtypes and every tensor must be exactly
equal; verify checkpoint hashes against run artifacts. Existing approved archived
source map binds673; source/annotation differences are recorded, not pretended
identical. No tolerances or historical-map weakening. Current model/features/key/
checkpoint/evaluation/evidence helpers match pinned base6f443b8 direct hashes.
Shared-loop source diff and literal ordinary-batch AST are recorded for review;
empirical full replay remains mandatory before interpreting SAM.

Candidate accepts only that fresh control's matching new source dictionary,
protocol SHA and completed disabled-hook proof, plus actual run/checkpoint and
validation artifact hashes. Prior structured36d9 replay is NOT sufficient here.
Failing control reproduction stops qualification; no SAM fit should be launched.
If pilot passes, fixed37/73 each require the same fresh-control-first sequence.
At most three candidates plus three controls. No retry with changed settings.

## Resources and artifacts

Fixed BEFORE fit: controls1,200 seconds; candidate2,400 seconds (40 minutes), each
including final selected-checkpoint scoring/artifact construction. No extension
after results. Preserve existing device_setup: GPU0, allocator fraction.04,
currently free>=8 GiB guard, TF32 off, deterministic/cuDNN flags, torch CPU threads4.
Log cap/free/total, peak allocated/reserved and final free bytes; keep vLLM running.
Two sequential backward passes approximately double training compute, not inference
or peak activation memory; record measured time/memory. Budget/OOM is inconclusive,
never negative accuracy. Preflight cache/source/control reading is outside the
shared trainer's unchanged fit clock, and is recorded through source identities.

Before fitting: protocol, base audit, source hashes and research identity copied to
output. Record seed, exact SAM spec, source diff vs archived673, references and
versions. Source snapshots remain immutable; base/protocol hashes are not rewritten
after a fit. Run.json ordinary config is unchanged; SAM identity is separate.
Save raw376×24 selected checkpoint logits with IDs/audiohashes/refs and hashes;
checkpoint/state/run/proof references form the provenance chain.
Report both repository symmetric-fifth MIREX and existing asymmetric ascending-fifth
MIREX, exact, mode accuracy and paired exact/fifth/relative/parallel/other movements.

## Predeclared decisions

Seed13 pilot requires FM symmetric MIREX >=1 pp over ordinary joint RAW, no BP
MIREX decline, no exact decline either domain. Only a passing complete pilot earns
37/73; canonical13 fixed. Negative: no radius/BN/optimizer rescue variants.
Final three-seed means retain the >=1 FM-pp method gate independently, plus original
>=2 FM-pp vs archived BP-only/no BP MIREX loss, and no domain MIREX/exact regression
vs ordinary joint. Both MIREX conventions reported; primary gate stays repository
symmetric convention. No automatic promotion; stretch90% exact is not promised.

## Root launch

Use immutable archived joint/BP runs and a fresh source directory from this commit.
Root reviews code/protocol first, launches control, verifies complete/reproduced
sam-result.json, THEN launches candidate. Existing shared server interpreter is fine.

```sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
# Control: outer backstop1250 / inner1200 seconds.
timeout 1250 "$PY" -m experiments.samstudy --data "$DATA" --output "$ROOT/control-seed13" --reference-run "$JOINT13/run.json" --reference-bp "$BP13/run.json"
# Candidate only AFTER exact fresh-control proof: outer2450 / inner2400 seconds.
timeout 2450 "$PY" -m experiments.samstudy --candidate --data "$DATA" --output "$ROOT/candidate-seed13" --reference-run "$JOINT13/run.json" --reference-bp "$BP13/run.json" --paired-control "$ROOT/control-seed13/run.json"
```

`DATA=/home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/data`.
`--reference-checkpoint` optionally supplies separately stored archived joint best.pt;
BP best.pt must be adjacent to its run. No fitting/SSH/push by implementation agent.
