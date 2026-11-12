# PROPOSED v2 data contract: nested fitting with many evaluation groups

**Decision: reviewable proposal only. No roles are current, no training is authorized,
and v1 data, weights and gates remain unchanged.** This study designs a component-first
FMA campaign for a paired restricted-versus-expanded data experiment. It does not
establish accuracy, a pristine final test or fully artist-independent Beatport.

## Problem and fixed method

V1's 1,000 historical-final FMA rows touch components containing another 3,104
development rows. Reusing that coverage requires explicitly retiring historical
roles. An exact-size 4,080/250/159/1,000 certificate could put 1,000 evaluation rows
in only eight groups. This proposal instead preserves all old fitting components
and requires many complete groups in each evaluation panel.

The [predeclared protocol](PROTOCOL.md) fixes one seed **20261112**, one greedy
metadata rule, panel order and group constraints before assignment. It does not
search seeds or inspect model scores. All 1,157 saved components over 5,489 FMA
records are preserved, including bridges through omitted records. The old 976
fitting IDs force their 490 complete components into expanded fitting. Remaining
groups are chosen toward approximate row, publisher-duration and coarse-genre
targets, with 60/100/200 calibration/selection/confirmation components and per-group
size ceilings. No target labels enter the assignment schema or objective.

The first rule produced **4,082 / 250 / 159 / 998** raw rows in expanded fitting,
inner selection, calibration and development confirmation. These counts were
preserved; no second seed/rule was tried to improve class support. The assignment
ledger was pinned **before labels** at SHA256
`64fa2acbe2901ffb7ff0cf9fd723a7fcedba2da1fc716422377f6af69943d3ed`.

## Source-integrity overlay and usable counts

A separate, later diagnostic reads only the prepared NPZ's metadata member and
inherited integer targets. It found one identical-container-SHA group with
conflicting key labels: **fmak:098877 and fmak:099017**, component `artist:17999`.
Both were old quarantined development and proposed expanded fitting. This is an
observed target conflict within fitting, **not** cross-role leakage.

Root approved a general [eligibility overlay](ELIGIBILITY_OVERLAY_PROTOCOL.md):
quarantine every record in any conflicting exact-SHA group across every inspected
FMA role and both arms. Both records are flagged; there is no first-label choice,
model adjudication, replacement, seed change or role reassignment. Independent
source adjudication is required before either can be recovered. Every old fitting
record remains retained and source-check eligible.

| Proposed cohort/panel | Assigned raw / unique SHA | Source-check eligible raw / unique SHA | Components after overlay | Largest component | Concentration equivalent* |
|---|---:|---:|---:|---:|---:|
| Restricted FMA fitting | 976 / 976 | 976 / 976 | 490 | 12 | 281.5 |
| Expanded FMA fitting | 4,082 / 4,036 | **4,080 / 4,035** | 797 | 272 | 71.3 |
| Inner selection | 250 / 250 | 250 / 250 | 100 | 5 | 83.3 |
| Calibration | 159 / 159 | 159 / 159 | 60 | 4 | 56.8 |
| Development confirmation | 998 / 997 | 998 / 997 | 200 | 12 | 154.4 |

*`N² / sum(component_size²)`, descriptive row-mass concentration; not validated
statistical effective N. Evaluation panels use far more groups than the size-only
certificate. Expanded fitting remains strongly concentrated: its largest artist/
album/audio component contains 272 rows. More recordings do not imply proportionally
more independent musical works. Unique evaluation counts matter for metric weighting;
matching-label exact duplicates require a single deterministic evaluation unit.

With the same 1,230 BP fitting records, prospective **unique** combined fitting is
**5,265 versus 2,206**, a 2.387× increase. The eligible raw combined count is 5,310.
The incidental eligible raw FMA count of 4,080 results from source-integrity exclusion,
not forcing the desired cardinality. No checkpoint or data loader uses this proposal.

## Balance, class support and precision limits

Publisher duration is populated for every FMA row. Eligible metadata durations are
223,585 s restricted fitting, 893,216 s expanded fitting, 53,972 s inner selection,
34,320 s calibration and 215,523 s confirmation. These are raw-row sums, including
same-SHA duplicates; they are not decoded duration, unique listening hours or measured
training work. Equal optimizer steps do not guarantee equal passes or audio coverage.

Coarse genre is missing for **3,065 / 5,489** source rows and explicitly treated as
MISSING. Missing counts are 439 restricted, 2,257 expanded, 143 inner, 92 calibration,
573 confirmation. Complete histograms are in the proposal/eligibility summaries.
Known classes include a single Easy Listening record and only eight Blues records
in the full corpus; a claim that every genre is balanced across panels is impossible.
The rule balances the available coarse metadata against the non-forced reservoir,
not a known deployment distribution or every underlying work/genre label.

Class support was **diagnostic after fixed assignment**, not stratification or model
selection. All 24 inherited key classes occur in every panel after the conflict
rule. Minimum unique per-class support is 9 restricted fitting, 47 expanded fitting,
**1 inner selection, 2 calibration, 14 confirmation**. No row was moved to improve
these counts. Those small cells limit classwise conclusions/calibration. Targets
are inherited prepared FMAK-v2 annotations, not evidence of human consensus or a
known label-noise ceiling. The historical before-overlay diagnostic remains recorded;
its arbitrary first-representative class counts for conflicting audio are not used
as eligible targets.

For planning only, a 20% paired-disagreement rate on 997 independent recordings
would already yield an approximate ±2.8-point 95% interval for a near-zero difference;
actual clustered uncertainty can be wider. A +2-point pilot is therefore a practical
screen, not assured significance or proof of no effect when unresolved. Future
analyses must pair predictions and resample actual groups; folds/seeds/crops do not
multiply independent recordings.

## Historical exposure is retained, not reset

The complete 5,489-row compressed exposure ledger retains each ID, full component,
old role, proposed role, restricted membership and historical exposure category.
The separate full-row eligibility overlay changes only source-integrity status.

| Old FMA role | → expanded fit | → inner selection | → calibration | → confirmation |
|---|---:|---:|---:|---:|
| Fit | 976 | 0 | 0 | 0 |
| Validation | 94 | 49 | 29 | 78 |
| Calibration | 64 | 29 | 17 | 49 |
| Historical final | **631** | 72 | 46 | 251 |
| Quarantined development | 2,317 | 100 | 67 | 620 |

These are immutable pre-overlay assignment counts; two of the 2,317 proposed
quarantined-development→fit records are subsequently ineligible. Prior program-level
final/development exposure is not erased by partitioning. Legacy development's
individual earlier fitting exposure is not cleared by this audit. Confirmation is
**internal exposed development confirmation**, not independent SOTA/release proof.
Both future data arms must be freshly trained and evaluated on the same panels;
canonical learned encoders/features fitted on an inner holdout cannot serve as fresh
inner-validation inputs. Only label-independent verified preprocessing may be reused.

## Observed guarantees versus unresolved identities

The checker finds **zero observed cross-role FMA primary-artist IDs, album IDs or
previously recorded exact-container SHAs**. Every saved component is assigned whole;
all 976 old fitting IDs remain in the expanded source-check-eligible cohort. This
guarantee covers the saved primary-credit graph, not every alias, remix, composition
or acoustic near-duplicate. No audio was read/hashed/decoded/fingerprinted here.

BP is a shared fitting-pool proposal, with **no new BP inner roles assigned**. Official
structured artist/release identity exists for only 618/1,230 fitting tracks; **612
remain unresolved**. Spreadsheet credits cover all but differ from filename credits
on 242 records. This is an explicit policy/disclosure decision for a paired internal
FMA data experiment, not proof of an observed BP cross-role violation or an automatic
indefinite veto. Sharing identical BP exposure across arms controls that exposure for
the data comparison; it does not justify artist-independent BP or independent-final
claims. The separate BP→GS overlap investigation must report its actual cached scope.

## Provenance, reproduction and remaining decision

Compressed sanitized input retains all FMA IDs, components, old roles, existing SHA
strings, artist/album IDs, coarse genres and publisher durations. It contains **no
key labels, features, audio or HTTP response cookies**. Source file paths/hashes and
public BP acquisition checksums are retained. Original prepared metadata-member SHA
is `94f65bb54329b239028dfffc1e8bf37f7d037f6e1e7c25f6721e73efd2c037bc`.
Assignment uses the captured full graph, not rebuilt subset groups. Old calibration
is the 159 eligible indices left after the observed 976 fitting/250 validation IDs,
consistent with the saved v1 split policy; no old role is guessed from model scores.

Publication sanitization removed HTTP response headers from source provenance only.
The original before-label input/witness remains preserved privately in `/tmp`; a
separate SHA derivation proves **identical canonical track payload before/after**.
The historical witness was not silently rewritten. Class diagnostics bind the original
before-label input and ledger; the checker verifies the sanitized derivative and
unchanged assignment. Executed diagnostic source/stdio and metadata-member hashes
are recorded; only metadata was read on CPU4–5, with no persistent server job.

Reproduce locally without network, labels or models:

```sh
python -m tools.check_v2_contract
python -m pytest tests/test_v2_contract.py -q
```

Six regressions cover whole components/nesting/order invariance, bounded-group
failure, same-audio cross-component rejection, old-ID-preserving deduplication,
strict rejection of target fields in assignment inputs, and generic conflict
quarantine across restricted/expanded membership with no feature member read.
The checker reproduces proposal/ledgers, verifies the before-label/sanitization chain,
source hashes, all-role overlay coverage, observed ID/SHA separation and counts.

**Root decision remains required** on role retirement, narrow BP guarantees,
duplicate metric units, rare-class/genre coverage, clustered precision and prospective
paired training budgets. No automatic release promotion, training-role activation,
new seed/fold search or external panel scoring follows this metadata proposal.
