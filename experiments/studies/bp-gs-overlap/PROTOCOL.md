# BP-to-GiantSteps cached overlap audit — prospective v1

Status: source/metadata build only. Root must review the frozen source, plan SHA,
count ledger and resources before any real fingerprint query. No role assignment,
training, key scoring, audio decode/download/hash, feature extraction or index
mutation is authorized. All previous failed methods and v1 roles remain intact.

## Question and count lineage

Audit recording overlap between the full original BP inventory and GiantSteps604,
and map any positive evidence back to actual model fitting/selection exposure.
Different counts, titles or filenames alone prove neither overlap nor independence.

The official spreadsheet ID column and cached reference inventory each contain
1,486 unique BP IDs. The archived canonical parser accepts 1,363 source-label
files; 123 fail parsing. Source-bound metadata exports retain each readable
filename, ID, text hash and branch reason, without exporting target keys or
repairing eligibility. The released checkpoint declares 1,237 fitting IDs and
126 validation IDs (1,363 union). Its fitting IDs have 1,232 distinct container
hashes and two container identities shared with its validation IDs. Five
within-fit duplicate aliases plus two validation aliases explain the seven-ID
removal to the corrected 1,356-ID cohort: 1,230 fitting and 126 validation.
The seven original IDs remain queried and individually attributable here.

Do not invent a 1,077/1,347 denominator or call all 130 missing current rows
exact duplicates. The source-label audit explains 123 exclusions; seven are
alias removals. Checkpoint-declared generic `training_audio_sha256` covers the
prepared corpus and does not independently identify every gradient exposure.
Separate direct fitting IDs, selection IDs, container aliases and cached-PCM
aliases for release and six actual BP-control/joint checkpoints (seeds13/37/73).
Run IDs are accepted only with actual checkpoint SHA and matching source/arm/seed
witnesses. Missing provenance is unknown, never clean.

## Immutable reuse and scope

Reuse reference result SHA-256
`5b3d7296f8277d8b43470736e670d35e58c2c94230b0f95dc53172d33787ed14`
and index SHA-256
`b2609bd2b7184291e7604c74887feae6dd399c1bd4ca696c6f616ec27d155148`
under `/home/lysk-admin/drone-experiments/fma-audio-admission-20261112-v1`.
The existing 7,579-reference index covers 5,489 FMA, 1,486 BP and 604 GS
recordings. All reference fingerprint hashes, full descriptor, extraction source,
reference/posting counters and the complete `(ref,id,hashes)` SQL dictionary must
verify before matching. The index opens read-only/query-only; verify its SHA again
at completion. Never rebuild it or create a filtered index.

The extraction engine and all-offset matcher are copied byte-for-byte from
`c3177bf`. The original verified factory and original `match` voting code remain
unchanged. A typed SQLite connection adapter adds only a verified reference-ID
interval to the exact frozen postings query. Unexpected postings SQL fails.
Actual GS scope is all IDs6975…7578, contiguous and 604 records. Filtering happens
before votes, not after choosing a best reference. No top-k, best-offset pruning,
posting cap, wraparound or duration/title filtering is introduced. Every original
3-anchor/2-second qualifying offset/bin/phase survives within the target scope,
including below-review weak alignments. Four-frame bins, phases0/2, distinct
query anchors, original two-million vote-cell cap and deadlines are unchanged.

BP audit queries reuse their **reference** NPZ landmarks (one reference shift),
not newly extracted four-shift query fingerprints. The extraction-contract
witness is unchanged, but this role difference limits sensitivity. Do not claim
the transformed four-shift query calibration proves recall for this one-shift
reuse, pitch/tempo alterations or all excerpt matches. A completed negative means
only no detected overlap under this scoped cached recipe.

Frozen decision rule: identity evidence requires20 anchors/8 seconds plus
Chromaprint distance<=0.15 at retrieved landmark offsets. Review requires10
anchors/5 seconds; missing/disagreeing CP remains review. CP is not searched
independently. Existing exact-container or cached-PCM equality is metadata
identity evidence; no audio is rehashed. Zero current BP/GS numeric-ID, container
or PCM intersections is only a metadata finding, not a near-duplicate clearance.

## Target-scope controls, audit and incomplete outcomes

First run180 existing transformed positive controls, each scoped to its expected
reference in the same full index. Then run all604 cached GS references as self
queries within the actual GS604 target scope, followed by64 existing synthetic
negatives within GS604. Require all848 controls complete, all784 expected
positive identities retrieved and zero negative identity flags. Report review
flags separately. This operational scope check does not undo the historical
failed zero-review admission gate or authorize data admission.

Only after that gate, query all1,486 original BP recording IDs against GS604.
Do not collapse aliases or drop unparseable-label rows; their actual fitting
and selection exposure remains separately recorded. Retain every qualifying
offset with CP corroboration, decisions and query/target dictionaries. No label
scores are computed. A deadline, uninformative query, vote cap, corrupt/missing
artifact or global budget failure is explicit insufficient/incomplete evidence;
unattempted IDs are never negative. Preserve all completed raw evidence and
failure logs. Positive evidence can establish recording overlap; it does not
numerically establish an affected model's score inflation without a separately
authorized re-evaluation design.

## Prospective resources and plan

One CPU process, affinity restricted to4–5; OMP/MKL/OPENBLAS/NUMEXPR threads1.
No model/GPU calls in the matcher. Sixty-minute wall cap, two CPU-hours,
two-GiB process RSS and512-MiB new output cap including partial/atomic evidence.
The index/cache is existing storage. Each landmark query has at most120 seconds
within the aggregate deadline; CP/serialization also remain under the aggregate
watchdog. Monitor every0.25 seconds; sub-interval resource spikes can be missed.
Before each evidence write enforce the output cap with eight-MiB reporting
reserve. Stop incomplete rather than pruning alignments or tuning thresholds.

Total planned queries:180 transformed +604 self +64 negative +1,486 BP =2,334.
Historical full-index244-control wall time was303.048 seconds; count scaling
projects approximately2,899 seconds (48.3 minutes), within the one-hour cap but
not a guarantee. Scoped posting density, hash counts, one/four-shift extraction
and cache conditions differ. No real audit timings are claimed by this plan.

`prepare` consumes metadata exports only and writes literal query/reference IDs,
source/container/PCM/fingerprint hashes, official/current/release lineage and
all seven model witnesses. `run` requires its reviewed SHA, checks source and
policy identities, verifies existing NPZ/index/dictionary/control/model witnesses,
then executes the fixed stages. No resumption, alternative thresholds or hidden
negative filtering. Failure outputs retain unresolved queries and cannot qualify
as complete. Root owns scheduling, execution and any later interpretation.

## Proposed invocation after source/plan review

The source commit and plan are frozen before query access. `prepare` is a
metadata-only step; its local original prepared-manifest witness must have the
exact frozen SHA39fe1842… . The retained input index fingerprints are not copied
or regenerated. Metadata collector v1 and the separately identified reason-
breakdown v2 preserve their exact source bytes, command, exit status and output
hashes. Collector v2 reports104 wrong-token-count,17 unsupported-mode and two
unsupported-tonic exclusions; no invalid UTF-8. These categories follow parser
branches and do not decide whether an exclusion was intentional or reparable.

```sh
python -m tools.audit_bp_gs prepare \
  --references experiments/studies/bp-gs-overlap/data/reference-result-v1.json.gz \
  --prepared /tmp/drone-fitting-feature-manifest.json \
  --collector experiments/studies/bp-gs-overlap/data/collector-v2.json \
  --collector-provenance experiments/studies/bp-gs-overlap/data/collector-provenance-v2.json \
  --release experiments/studies/bp-gs-overlap/data/release-metadata.json \
  --catalog experiments/studies/bp-gs-overlap/data/official-id-catalog.json \
  --witness experiments/studies/bp-gs-overlap/data/control-plan-v1.json.gz \
  --historical-controls experiments/studies/bp-gs-overlap/data/historical-controls.json \
  --output /tmp/bp-gs-plan.json
```

Only after root reviews the actual plan SHA and source:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Root owns the outer timeout and CPU scheduling; this is not a launch request.
timeout -s INT -k 3 3650 taskset -c 4,5 python -m tools.audit_bp_gs run \
  --plan /path/to/reviewed/bp-gs-plan.json --plan-sha256 REVIEWED_PLAN_SHA256 \
  --output /home/lysk-admin/drone-experiments/bp-gs-overlap-v1
```

All original positives remain controls outside the scientific GS-only target
scope. They do not become BP/FMA overlap findings. The scientific audit targets
only GS604. The target-self controls exercise the real target dictionary; the
transformed controls exercise the retrieval implementation. Neither proves
arbitrary transformed-GS recall. No empirical claim or admission follows merely
from passing controls.
