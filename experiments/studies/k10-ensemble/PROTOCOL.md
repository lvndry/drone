# Fixed three-seed joint K10 probability ensemble

Frozen before extraction. Inference only, one candidate, no new training, model
selection, calibration, temperature/weight search or automatic promotion.
Cohort and order are exactly joint seeds 13/37/73 from clean source 673 runs, each
selected by its original training protocol. Temperature 1, weights 1/3 each.

## Question and motivation

Can independent optimization outcomes combine into a stronger own-data teacher?
The rejected width 2 model and within-trajectory EMA did not test an ensemble of
independently initialized checkpoints. Probability averaging is established:
[Deep Ensembles, 2016/2017](https://arxiv.org/html/1612.01474v3).
Independent seeds can provide diversity beyond one optimization trajectory:
[Fort et al., 2019/2020](https://arxiv.org/abs/1912.02757).
Neither finding guarantees useful diversity for our musical-key errors.
Distillation of ensembles is established, including acoustic models:
[Hinton et al., 2015](https://arxiv.org/html/1503.02531v1).
A teacher may remain difficult to imitate, and fidelity need not improve student
accuracy: [Stanton et al., 2021](https://arxiv.org/abs/2106.05945).
This study only qualifies a teacher for a NEW separately controlled distillation
proposal. No student, native export, product tier or production dispatch is added.
Existing weights add no external-weight license/pretraining-overlap dependency;
inherited dataset and annotation rights remain unchanged.

## Cohort, data and immutable provenance

Use original archived source 673 source directory and six run/checkpoint pairs:
joint 13/37/73 and BP-only 13/37/73. Their seeds/arms/status/config, source dictionary,
feature manifest SHA, selected epochs, fit/validation identities, checkpoint SHA,
format 2, feature spec, label schema, crop 320, width 1 and checkpoint metadata must
agree. Joint models each have 109,128 parameters. Archived source file bytes must
match EVERY file in the existing approved source 673 map; no historical map changes
or expanded allowlist. Import the verified archived `drone/model.py` constructor
for inference, with current N_CORE explicitly checked 192; load state dict strictly.
No current optimized model graph is silently substituted.

Only existing immutable prepared CQT cache, same quarantined 2,206 fitting/376
validation manifest. Verify feature spec/label schema/frozen manifest hash
`5f995f516430332a37f591ec299a36f9a46c16b2cd01cd887700837fd600bac9`, exact
fit/validation IDs and domain counts 1,230/976 fitting, 126/250 validation. Check all
376 validation cache content hashes and paths beneath the data root. Fitting IDs
are checked against runs but fitting features are NOT extracted. No additional,
private, calibration 159, GS/FMAK final or SWD data; no raw-audio decode/download.
Existing artist/album/audio quarantine stays fixed; no claim of a new group audit.

Before inference, save protocol bytes and identity containing source/protocol/map/
run/checkpoint/cache-manifest hashes, schema/spec, fixed members/combiner and actual
Torch/NumPy versions. Save each member's actual 376 × 24 float32 logits, unchanged
IDs/audiohashes/refs and metrics. Hash each artifact and final probabilities/report.
No target is used in extraction, softmax, weighting or tie resolution.

## Exact extraction and combination

After the authoritative terminal SAM state only: require its candidate terminal
result plus root wrapper's candidate.exit receipt. Missing/unfinished/contradictory
receipts reject before device setup. Record both receipt hashes. A successful
SAM result requires exit 0; terminal inconclusive/failure does not authorize a retry
or change to the ensemble cohort.

Sequential GPU 0 inference, one model resident at a time. Original true-length
full-track inference: order-preserving groups by exact frame length within each
chunk of 4, float16 cached 208-bin CQT, shift 0 core rows 8:200 converted float32.
No chunked neural inference, test-time augmentation, padding to other tracks'
lengths, training mode, crop selection or BN updates. Batch statistics are frozen.
Before ensemble scoring, EACH single's predictions and complete both-domain metrics
must EXACTLY reproduce its recorded run.json domain_metrics. Failure stops the
study; no tolerance tuning, CPU/GPU substitution or member replacement.

For each member use stable float64 softmax at T=1; arithmetic mean of the three
probability vectors in fixed 13/37/73 order. This is not softmax of mean logits or
majority voting. Lowest canonical class index wins exact ties (NumPy argmax).
Interleaved tonic*2+minor class order remains unchanged; no transpositions.

## Resources and safeguards

One 20-minute total budget from preflight through final scoring/artifacts, no
post-result extension. Outer 1,250-second root timeout. Preserve existing device_setup:
GPU 0 currently free>=8 GiB, allocator .04, TF32 off, deterministic/cuDNN flags,
Torch CPU threads 4; keep vLLM running. Record free/total/actual cap and allocated/
reserved peaks. Stream at most 4 true-length feature arrays rather than all tracks.
Three sequential backbones approximately triple network work, while CQT may be
shared; no end-to-end CPU latency claim. Budget/OOM is inconclusive, never a
negative accuracy result. No implementation-agent SSH/extraction/push.

## Fixed decisions and fair reporting

Report seed 13/37/73 singles, their descriptive mean/range and ONE ensemble for
BP/FM exact, repository symmetric-fifth MIREX and existing asymmetric ascending-
fifth MIREX, plus mode accuracy. Show ensemble deltas and paired exact/fifth/
relative/parallel/other movements against EVERY member under both conventions.
Report uniform nonregression vs every single without selecting a new cohort or
replacing the canonical comparison. Teacher gate: >=1 FM symmetric-MIREX pp over
canonical joint 13; no BP symmetric-MIREX loss, no exact loss in either domain.

Separately labelled BP-reference hurdle: ensemble >=2 FM symmetric-MIREX pp above
the frozen three-BP-control mean, no BP MIREX loss against that mean. This is ONE
ensemble vs a mean of BP runs, not the earlier three-independent-candidate mean
experiment. Both results remain separate; teacher qualification does not imply
original-hurdle passage or permission to promote/distill.

Exactly one fixed cohort/result: no ensemble seed standard error, no claim of
three ensemble replications. Member mean/range is descriptive only. Validation 376
has already selected every member checkpoint and supported prior development;
no independent generalization/SOTA claim. A canonical 13 gain can coexist with
regression vs another member, which must be visible. Correlated errors, overconfidence
and limited genre coverage may defeat averaging. A negative result ends this
combination; no weight/temperature/seed rescue.

## Root launch after review and terminal SAM receipt

```sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
timeout 1250 "$PY" -m experiments.ensemble --data /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/data --output "$STUDY_ROOT/result" --reference-source /home/lysk-admin/drone-experiments/k10-multidomain-20261111-v1/source --joint-runs "$JOINT=13/run.json" "$JOINT37/run.json" "$JOINT73/run.json" --bp-runs "$BP13/run.json" "$BP37/run.json" "$BP73/run.json" --sam-result /home/lysk-admin/drone-experiments/k10-sam-20261112-v1/candidate-seed13/sam-result.json --sam-exit /home/lysk-admin/drone-experiments/k10-sam-20261112-v1/candidate.exit
```

All checkpoints best.pt must be adjacent to immutable run.json. Source archive must
include this protocol, extractor and unchanged approved source 673 map. No training
loop, production dependencies, weights or dispatch changed by this study.
