# Internal v2 data contract and coverage provenance

The assignment sections below retain their original proposal-time scope and counts.
The [current coverage/provenance chapter](#current-genre-duration-and-reference-provenance)
uses the later frozen eligibility amendment and metadata only. It changes no split,
reference key, model outcome or running experiment.

**Original proposal decision (historical): no new role assignment or training
activation; v1 data, weights and gates remained unchanged.** The original study
designed a component-first FMA campaign for a paired restricted-versus-expanded data experiment. It does not
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

**At that proposal review point, decisions remained** on role retirement, narrow BP guarantees,
duplicate metric units, rare-class/genre coverage, clustered precision and prospective
paired training budgets. No automatic release promotion, training-role activation,
new seed/fold search or external panel scoring follows this metadata proposal.


## Current genre, duration and reference provenance

### Problem, method and result

A broad DJ-key classifier needs evidence across hip-hop, contemporary Afrobeats,
house/techno and other club styles, and across short and extended recordings.
Counts alone establish neither accuracy nor representative independent testing.
This metadata-only audit uses the current amended plan `c3e59ffb…`, frozen exposure
ledger `64fa2acb…`, captured FMA master/taxonomy, and existing BP metadata. It reads
no audio or private files, makes no model predictions, selects no reference-key
fields, and changes no roles. Full source hashes are in
[coverage-inputs.json](data/coverage-inputs.json).

**BP genre coverage is stronger than an FMA-only inventory suggests:** all 1,229
current BP fitting and 126 selection IDs have a workbook genre, including explicit
Deep House/Tech House/Progressive House/Techno. Those inputs are roughly two-minute
previews, not evidence on their full commercial versions. FMA supplies longer
recordings, but specific current evaluation cells are sparse or unidentified.

![Observed native-genre duration coverage in current selection](coverage-genre-duration.png)

Figure: FMA cells are **tracks/frozen components** from published durations; BP
cells are tracks from actual prepared-preview sample counts, with artist/album
independence unverified. All 9 FMA House-tagged selection tracks are also Chill-out.
**Afrobeat is not contemporary Afrobeats**; the latter cannot be identified by
these tags. Zero matching tags do not prove the music is absent. BP's populated
2–<5 min cells hug the 120 s boundary, not a spread of longer songs. Native taxonomies
are kept separate; overlapping FMA tags are not additive. [SVG](coverage-genre-duration.svg).

Duration bands are `<30 s`, `30 s–<2 min`, `2–<5 min`, `5–10 min`
(including 600 s), and `>10 min`. FMA's publisher
`track_duration` is not newly decoded support; metadata rounding and a clipped
rendition can change the applicable band. BP uses the existing canonical 24 kHz
sample-count metadata `7d56908d…`; no waveform was reopened. Neither source's genre tag is a
verified listening judgment.

### FMA coverage and missingness

The master table has 106,574 records; 5,489 are in the FMAK exposure ledger. Current
Expanded fitting has 4,033 records from 797 components; selection has 250 from
100; unique reserved confirmation has 997 from 200; excluded calibration has 159
from 60. Restricted fitting (976 records, 490 components) is nested inside expanded
fitting. The raw partition also includes 49 inactive/quarantined
former-fit IDs and 998 raw reserved records; those are not added back into fitting.

All 163 taxonomy tags are inventoried in the aggregate. Every master tag count
reproduced the taxonomy's published count. Below, each cell is tracks/components;
artist/album counts for every role/tag/duration are retained in the aggregate.

|Native FMA membership or explicit slice|Expanded fit|Inner selection|Reserved unique|Calibration excluded|
|---|---:|---:|---:|---:|
|Hip-Hop|269/50|16/8|41/13|16/6|
|Rap|57/5|0/0|0/0|2/1|
|Afrobeat|4/2|0/0|0/0|0/0|
|House|71/14|9/4|9/3|5/2|
|Chill-out|67/10|9/4|5/2|5/2|
|House minus Chill-out|4/4|0/0|4/1|0/0|
|Techno|17/11|6/3|11/3|2/1|
|Named dance-tag union|184/57|40/15|60/18|9/4|

`genres_all` includes ancestors, not only direct leaf tags. 86/94 House-taxonomy
FMAK tracks also carry Chill-out; House minus Chill-out is a declared set difference,
not a new genre. Its four reserved records occupy one component. Techno includes
Bigbeat; no claim of complete house/techno subgenre coverage follows. No Deep House,
Melodic House or contemporary Afrobeats tag exists in this FMA taxonomy. African
or Afrobeat cannot silently fill the Afrobeats gap. Garage (85) belongs under Rock,
so it is not counted as UK garage.

The named dance-tag union is exactly House, Techno, Disco, Dance, Drum & Bass,
Jungle, Dubstep, Breakbeat and Breakcore - Hard—not all music DJs play. It has 293
FMAK records/94 components overall. Its selection support is 40/15: four <2 min,
27 two–five-minute and 9 five–ten-minute tracks, and zero <30 s or >10 min. Current FMA
Hip-Hop selection has 16 tracks/8 components, but none ≥5 min; Rap-specific selection
and reserved support are both zero. Other named styles remain individually
available in the aggregate rather than inferred from generic Electronic.

FMAK has 3,065/5,489 missing top-level genre strings, but only 84 empty all-genre
lists. Selection has 143/250 missing top genre and 16 empty all-genre lists. A
coarse-genre-only analysis loses detail; empty tags must not become proof of
nonmembership. Known artist/album counts are 942/1,200 in expanded fitting,
118/118 in selection and 239/295 in reserved unique. Credits and aliases remain
imperfect proxies for musical independence.

### BP native genre and preview support

Workbook extraction selects only A:id/F:genres, resolving shared strings only for
those cells and rejecting formulas. G:M key/confidence/comment fields are not
selected. Store JSON members are filtered by the current 1,355 IDs; only
id/genres/sub_genres/duration are selected. Historical bare lowercase nan is safely
read as null outside quoted strings. No store `key` is used as a reference label.

|Native workbook genre, preserved verbatim|Current fit|Current selection|
|---|---:|---:|
|Big Room|2|0|
|Breaks|77|10|
|Deep House|63|12|
|Drum & Bass|73|7|
|Dubstep|80|5|
|Electro House|77|11|
|Electronica / Downtempo|159|11|
|Funky / Groove / Jackin' House|9|0|
|Hard Dance|81|8|
|Hip-Hop / R&B|79|6|
|House|66|9|
|Minimal|69|3|
|Progressive House|82|8|
|Psy-Trance|83|9|
|Tech House|74|6|
|Techno|70|14|
|Trance|84|7|
|Trap / Future Bass|1|0|

API JSON exists for 671/1,229 fit and 61/126 selection; 558/65 are missing. Missing
JSON does not mean unknown workbook genre. API subgenres appear for only 15 fit and
2 selection records. Workbook/API genre strings disagree on 41 fit + 2 selection records;
both native descriptions are retained, with no forced FMA conversion or silent
choice. `Hip-Hop / R&B` stays combined; `Minimal` is not automatically Techno.

Actual preview support is 43.3–120.007583 s fitting and 119.2–120.000750 s selection.
Strict bands put 35 fit + 5 selection below 120 s, 1,194 fit + 121 selection at/above 120 s; these
values closely straddle the boundary. **None reaches 5 min.** Store durations carry
explicit milliseconds and minutes text. Of 676 known values, 500 describe 5–10 min,
but 673/676 differ by >1 s from the supplied preview. Those durations cannot turn
preview evidence into full-track coverage.

The current uncertainty file assigns all 1,355 BP entries record-unknown-artist
singleton proxies. These are not 1,355 verified independent artists/albums. This
narrow extraction does not inspect artist/release fields; earlier identity metadata
exists, so the proxy does not imply the original source lacks artist information.
BP improves known genre coverage, not the independence guarantee.

### Exposure, larger-universe bridges and reference quality

The active frozen FMA fit/selection/reserved/calibration sets share zero ledger
components. A separate graph over all master artist/album credits yields two
components connecting fit↔selection and two fit↔reserved via outside-cohort
records. This wider-universe metadata diagnostic is **not proven acoustic
leakage**, an audio-identity test or grounds to repair the frozen split after
outcomes. It matters when choosing a future independent panel. Of 101,085 master
rows outside FMAK, 53,624 remain unlinked to FMAK under this conservative graph;
they are metadata candidates, not acquired audio or independently labelled data.
The aggregate contains counts only, not source rows or credit-path exports.

Reserved in the current split does not mean never exposed historically. The
998 raw reserved IDs include 251 program-exposed historical-final and 78 adaptive-
development entries; remaining exposure statuses are preserved in the ledger.
Component separation cannot erase previous program use. Pretrained-teacher
recording overlap also remains unknown rather than assumed absent.

The [original FMAK release](https://zenodo.org/records/10719860) describes expert
song-level references. [FMAKv2](https://zenodo.org/records/12759100), used by the
prepared research corpus, changes around 200 annotations; its release explicitly
says the original FMAK authors did not verify those modifications. **That does not
establish that the revisions are wrong.** Describe this as references derived from
expert-annotated FMAK, not 5,489 uniformly re-adjudicated expert labels. Internal
“v 2 split” and the FMAKv2 annotation version are separate identifiers. The
[author README](https://github.com/stellaywong/fma_keys) describes Spotify URI matches
for analyzer comparison—not Spotify-generated ground truth. No reference CSV was
re-extracted, compared, switched or corrected for this coverage audit.

The [BP release](https://zenodo.org/records/1101082) credits human annotations,
comments and confidence to Eduard Mas Marín, revised/expanded by Ángel Faraldo.
These references are distinct from store metadata keys. Its single-key encoding
can lose ambiguity; the audit does not certify all annotations equally certain.
The [maintained loader](https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/beatport_key.py)
also documents the source and annotation terms. Metadata/annotation availability
is not blanket rights to redistribute underlying recordings or a proof that a
future public model has completed its provenance review. Pinned targets and all
completed/predeclared comparisons remain intact.

### Next independent-panel decision

Keep the live fixed model comparison on its original terms. A separate small
feasibility check could use four operational examples: contemporary Afrobeats,
hip-hop, a 5–10 min house/deep-house/melodic-house extended version, and a separate-
artist techno/club recording (prefer >10 min if a lawful exact version is available).
These are workflow examples, **not a statistical sample-size or accuracy claim**.
Select exact versions and access provenance before predictions; preserve rejected
candidates/reasons. Do not use private corrections or DJ libraries for fitting.

Two independent style-qualified annotators should be blind to models, vendor keys
and each other's answers, cover the whole recording, and retain disagreements,
multiple interpretations and no-stable-key cases. Measure access/annotation effort
before setting the later panel's size/precision budget. Workflow examples become
development-exposed and must not later masquerade as a sealed panel. A fresh panel
needs its own exact-version, exposure, ambiguity and scoring rules before outcomes.
No paid access, human annotation, contact or acquisition is started by this study.

### Reproduction and limits

Only the small [aggregate archive/index](data/coverage-index.json), input hash/path
receipts and scripts are committed—no source CSV/XLSX/ZIP, raw source rows, audio
or key-label export. Metadata readers use the standard library; the figure uses
the repository's existing Matplotlib plotting environment, with no new dependency.
Input paths may be relocated while retaining hashes; no downloader runs.

```sh
python -m tools.dj_coverage --inputs experiments/studies/v2-data-contract/data/coverage-inputs.json --output /tmp/coverage-fma-replay
python -m tools.dj_coverage_bp --inputs experiments/studies/v2-data-contract/data/coverage-inputs.json --output /tmp/coverage-bp-replay
python -m tools.dj_coverage_report --replay-fma /tmp/coverage-fma-replay/coverage.json --replay-bp /tmp/coverage-bp-replay/coverage-bp.json --plot
python -m pytest -q tests/test_dj_coverage.py
```

The verifier checks compressed/decoded/aggregate hashes, current cohort identity,
role and genre-band totals, native workbook completeness, preview/store separation
and metadata-only scope. Reader regressions exercise column/field allowlists,
quoted-nan preservation and formula rejection. These are metadata checks, not
musical-key accuracy, sample-support equivalence or corpus-wide licence clearance.
