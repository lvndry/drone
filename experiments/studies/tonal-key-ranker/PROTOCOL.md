# Whole-track candidate-conditioned tonal ranker: frozen falsification protocol

Status: fixed before feature extraction or fitting. Canonical seed 13; at most three
candidate fits, with 37/73 earned only by the pilot gate. No sweeps, feature revisions,
validation early stopping, teacher, or promotion. This is a representation and
classifier **product comparison**, not a causal classifier-only ablation.

## Question and evidence

Can a small joint nonlinear candidate scorer learn mode/register/segment evidence
from 2,206 clean labeled tracks better than the crop-trained 109,128-parameter K10?
This tests a broad classification lesson: useful fixed summaries plus a shallow
nonlinear decision model may outperform a larger learned representation on scarce
labels. This is established methodology, not a novelty claim. It differs from our
negative 48-parameter DSP-score fusion: features here are direct candidate-relative
chroma/register/segment measurements, and all 128 trees train jointly with a hard
24-way grouped objective. Unlike Keylime, it has no transformer or short-sequence
DSP front end. Earlier EMA, structured loss, width 2 and PitchCircle pilots were
negative; none licenses further variants within this study. On failure, prioritize
auditable data/self-supervision rather than nudging these features.

Primary package and mathematical contract: official [CatBoost 1.2.10 release](https://github.com/catboost/catboost/releases/tag/v1.2.10),
[QuerySoftMax](https://catboost.ai/docs/en/concepts/loss-functions-ranking#QuerySoftMax),
[training parameters](https://catboost.ai/docs/en/references/training-parameters/).
Version 1.2.10 is pinned in an isolated optional `ranker` extra and lockfile, verified
from official releases/PyPI on 2026-11-12 (Europe/Paris; UTC 2026-11-11); Apache 2.0 package, no external weights.
A synthetic-only three-group API fit/export/reload verifies compatibility before
real extraction; its accuracy is neither measured nor evidence for this study.

## Data, grouping and provenance

Only immutable existing CQT cache: 1,230 Beatport+976 FMAK fitting; 126+250 validation.
Prepared cache manifest must match archived source 673 joint reference's exact SHA,
feature spec, label schema and frozen development manifest
`5f995f516430332a37f591ec299a36f9a46c16b2cd01cd887700837fd600bac9`.
Existing artist/album/audio quarantine remains authoritative; no re-splitting.
Every cached file hash, float16 shape 208 × T, core 192 bins and finite nonnegative
values are checked. Raw audio paths are never opened. No final GS/FMAK, SWD,
calibration 159, private audio, downloads, decoding or teachers. Immutable audiohashes
bind these caches to the previously quarantined manifest; no claim of re-auditing
artists from audio is made.

One track is one group: 24 candidate rows in canonical interleaved C-major,C-minor,
C#-major,… order; exactly one positive. 2,206 groups/52,944 rows fit; 376/9,024 score.
Candidates from a track can never cross roles. Effective sample size is 2,206, not
52,944. No fitted feature transforms; CatBoost numerical borders fit on fitting
rows only. No categorical CTR/tonic identity feature or fitted tonic prior.

Before extraction, save protocol bytes/source/package/control identities. Prepared
artifact contains feature/row/cache identities, manifest SHA, feature-array SHA,
resources. Before fitting reverify current identities and actual control artifacts.
Reuse structured 36d9 control 13 only by rechecking its history/selected and stopped
epochs/predictions and actual checkpoint tensors against archived 673, approved
historical source map, proof/result/logit/checkpoint hashes. Do not weaken historical
math/source guards or pretend the current main can replay frozen fits. Standalone
feature/scoring files get direct new SHA bindings; no expanded training allowlist.
Confirmation requires the same actual replay proof for 37/73 from frozen 36d9 source
if needed. Historical tensor proof is stronger than matching comments/annotations.

## Exact 24 audio-only features

Input: shift 0 full-track 192-bin cached log1p CQT, T>=8, in 2,048-frame chunks.
Recover amplitude `A=expm1(float32(core))`, reshape `[8,24,T]`. Full halfbin profile
`Q=sum(A,octave)`; bass uses lowest 3 octaves. For semitone j,
`P_j=.5 Q_(2j)+.25(Q_(2j-1 mod24)+Q_(2j+1 mod24))`, then normalize each
frame by sum(P)+1e-8 (silent frame zero). Average normalized profiles over frames;
no energy threshold/label-informed segmentation. Quarter bounds follow
`np.array_split` (first remainder quarters one frame longer); ending is last
ceil(T/8) frames. Bass normalization is independent of full normalization.

For each candidate tonic t=ID//2, mode m=ID%2, features in this fixed order:

1. 0–11: whole-track mean full chroma, rotated by−t.
2. 12–14: whole-track mean bass at t, t+third, t+7; third = 3 minor / 4 major.
3. 15–18: each quarter's mean candidate-triad chroma minus mean off-scale chroma.
   Major scale 0,2,4,5,7,9,11; natural minor 0,2,3,5,7,8,10. Off-scale has 5 tones.
4. 19: ending mean bass tonic.
5. 20: minor bit 0/1.
6. 21: whole-track mean `(sum_even Q−sum_odd Q)/(sum Q+1e-8)`.
   This is a halfbin phase proxy, not a cents estimate or retuning.
7. 22–23: population mean/std across 4 quarters of candidate contrast minus relative
   key contrast. Relative mapping uses repository key relation, e.g. C major ↔ A minor.

Features never take target/ID/domain as arguments. Mode is a candidate hypothesis,
not the target. No CNN logits, native fitted key profiles or absolute tonic.
Cyclic feature rotation respects folded pitch-class geometry only; actual finite
CQT reslicing changes octave boundaries/register. Summary pooling loses temporal
order/harmony and may leave relative-major ambiguity; bass/tuning proxies may be
unreliable under overtones or unusual arrangements. These are falsification risks.

## Fixed decision model and resource bounds

CatBoostRanker 1.2.10, all numeric 24 features. `QuerySoftMax:beta=1`, 128 trees,
depth 4, learning_rate .05, l2_leaf_reg 3, grow_policy SymmetricTree,
boosting_type Plain, bootstrap_type No, random_strength 1, border_count 64,
task_type CPU, thread_count 4, random_seed 13/37/73, use_best_model False,
allow_writing_files False. No eval_set, best-tree selection or early stopping.
Unit weights and one positive/group make QuerySoftMax hard 24-way listwise CE.
Split-score randomness is seeded; no resampling of candidates. Resolved parameters
and package versions are saved; same CBM reload must reproduce scores exactly.

Linux pin CPU 6–9, OMP/MKL/OPENBLAS/NUMEXPR thread environment all 4 (env4 is not
an environment name). CUDA disabled in launch. Caps fixed before fit: 600 seconds
feature preparation; 300 seconds complete fit/export/reload/validation scoring;
2 GiB peak process RSS, 1 GiB new study output. Stream/mmap caches, no audio copies.
RSS/time/storage sampled per chunk and tree; external timeout adds hard backstop.
Budget exceptions are resource-inconclusive, never negative accuracy. Max 128 × 16
leaf values; theoretical 12,288 depth comparisons/track, no runtime claim. Native
export/end-to-end performance work only if accuracy passes.

## Decisions and artifacts

Save actual 376 × 24 raw scores (logits for grouped softmax), IDs/audiohashes/refs,
repo symmetric-fifth MIREX, existing asymmetric ascending-fifth MIREX, exact,
mode accuracy and paired exact/fifth/relative/parallel/other movements for both
conventions. No validation evaluation is used in fit. Evaluate final 128 trees once.
Pilot requires FM repoMIREX gain >=1 pp vs ordinary joint RAW, BP repoMIREX no loss,
exact no loss in either domain. Negative means stop, no 37/73/no feature nudges.
Passing earns only fixed 37/73; final three-seed means separately require original
FM >=2 pp vs archived BP-only, BP MIREX no loss, and no domain MIREX/exact loss vs
ordinary joint. The final three-seed mean method gate also independently requires
FM repo MIREX gain >=1 pp vs ordinary joint, with no BP MIREX or either-domain
exact loss; passing the original hurdle cannot substitute for this method gate.
Canonical seed 13; no automatic promotion. Reused development evidence
cannot establish unseen-domain generalization. 90% exact remains a stretch goal.

Own tiny artifact: `model.cbm`+`artifact.json`, architecture/study ID,
feature/label/version/config/source identity and model SHA; strict research loader,
never production dispatch. Prepared/features/control/model/validation SHA recorded.

## Root launch (replace reference paths with immutable archived paths)

```sh
uv sync --extra ranker --frozen
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
taskset -c 6-9 timeout 650 uv run --frozen --extra ranker python -m experiments.key_ranker prepare --data /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/data --output "$STUDY_ROOT/prepared" --reference-run "$JOINT13/run.json" --reference-bp "$BP13/run.json" --paired-control "$STRUCTURED_CONTROL13/run.json"
taskset -c 6-9 timeout 350 uv run --frozen --extra ranker python -m experiments.key_ranker fit --prepared "$STUDY_ROOT/prepared" --output "$STUDY_ROOT/candidate-seed13" --reference-run "$JOINT13/run.json" --reference-bp "$BP13/run.json" --paired-control "$STRUCTURED_CONTROL13/run.json"
```

No study data extraction or fitting is authorized for this implementation agent;
root reviews frozen commit/protocol and launches. Reference-run defaults to adjacent
best.pt; `--reference-checkpoint` can supply separate archived checkpoint. All paths
must bind actual frozen artifacts. Source snapshot must include optional lockfile,
protocol, both new modules and existing capacity reviewed-source map byte-for-byte.

Implementation compatibility evidence is in `data/compatibility.json`: synthetic
three-group fit with the exact fixed configuration and exact CBM reload scores.
The new optional packages are CatBoost 1.2.10, graphviz 0.21, pandas 3.1.1,
plotly 7.1.0 and six 1.17.0; no existing locked package/version was removed or
upgraded. Resolved transitive/platform markers differ as required by the extra.
The server may use its existing compatible Torch/NumPy interpreter plus isolated
CatBoost dependencies: actual runtime versions are recorded before extraction and
must remain identical during that prepared study, rather than matching the local
machine. Only CatBoost 1.2.10 is a hard package-version compatibility requirement.
