# Internal v2 data contract and dataset coverage

This study fixes how the 5,489 FMA key-annotated records are divided into
fitting, selection, calibration and confirmation roles for a paired
restricted-versus-expanded training-data experiment, and then audits what the
resulting data actually cover. Whole artist, album and audio components were
assigned with one seed before any labels were read, and no cross-role artist,
album or exact-audio link was observed. The coverage audits show strong
electronic-genre metadata for Beatport but almost no full-length DJ tracks,
contemporary Afrobeats or tempo data. A GTZAN short-clip panel was prepared and
audited for audio integrity and recording overlap, but no recording has been
admitted for scoring. Nothing in this study is an accuracy result. Study dates:
November 12–13, 2026.

## Question

1. How can FMA records be split so that a model trained on a larger ("expanded")
   fitting pool can be compared fairly with one trained on the original
   ("restricted") pool, without artist, album or audio leakage between roles and
   without erasing which records were already used in earlier work?
2. Across Beatport, FMA and GiantSteps, which genres, recording lengths and
   metadata fields (tempo, duration, artist) are actually covered, so later
   accuracy breakdowns are not misread?
3. Can the public GTZAN key annotations serve as an external short-clip panel:
   are the references well defined, do the downloaded recordings match the
   published index, and do any overlap with recordings used in model
   development?

The study also records a defect in an existing KeyFinder wrapper and the plan for
a Mixed In Key comparison.

## Method

### Split assignment

The FMA records form 1,157 components, connected by shared primary artist,
album or exact audio file hash, over 5,489 records (bridges through omitted
records included). In v1, the 1,000 historical-final FMA records touch components
containing another 3,104 development records, so reusing that coverage means
explicitly retiring historical roles. An exact-size 4,080/250/159/1,000 split
could have put the 1,000 evaluation records into only eight groups; this design
instead keeps every old fitting component in fitting and requires many complete
groups in each evaluation panel.

The [protocol](PROTOCOL.md) fixes one seed, **20261112**, one greedy rule, the
panel order and the group constraints before assignment. No seeds were searched
and no model scores were inspected. The 976 old fitting IDs force their 490
complete components into expanded fitting. The remaining groups are assigned
toward approximate targets for record count, publisher duration and coarse genre,
with 60, 100 and 200 components for calibration, inner selection and confirmation
respectively, and per-group size ceilings. Key labels are not part of the
assignment input or objective. The assignment ledger was fixed
[before labels were read](data/before-label-witness.json). Old calibration is the
159 eligible records left after the 976 fitting and 250 validation IDs, consistent
with the saved v1 split policy.

The published input retains every FMA ID, component, old role, audio hash,
artist and album ID, coarse genre and publisher duration, but no key labels,
features, audio or HTTP headers. Removing HTTP response headers changed the
compressed file; a separate check proves the track payload is identical before
and after, and the original before-label record is unchanged. Two later
appendices to `PROTOCOL.md` changed its checksum; the
[archived snapshots](data/protocol-snapshots/) and
[amendment chain](data/protocol-amendments.json) show that each appendix only
added text to the original, and the checker verifies this byte for byte.

### Source-integrity overlay

A later diagnostic read only the prepared metadata and inherited integer targets.
It found one group of byte-identical audio files with conflicting key labels:
**fmak:098877 and fmak:099017** (component `artist:17999`), both formerly
quarantined development records proposed for expanded fitting. This is a label
conflict within fitting, not cross-role leakage. Under the
[eligibility overlay protocol](ELIGIBILITY_OVERLAY_PROTOCOL.md), every record in
any byte-identical group with conflicting labels is quarantined, across all FMA
roles and both arms. Both records are flagged; there is no label choice, model
adjudication, replacement, seed change or role reassignment, and recovering
either needs independent source adjudication. Class support was computed after
this fixed assignment as a diagnostic, not for stratification or selection.

### Coverage audits

The genre and duration audit uses only metadata: the current amended plan, the
frozen exposure ledger, the FMA master and taxonomy tables, and existing Beatport
metadata. It reads no audio or private files, makes no predictions, selects no
reference-key fields and changes no roles. Beatport workbook extraction reads
only the ID and genre columns (key, confidence and comment columns are not read),
rejects formulas and treats bare lowercase `nan` as null. Store JSON is filtered
to current IDs and to ID, genre, subgenre and duration fields; store keys are
never used as references.

Duration bands are `<30 s`, `30 s–<2 min`, `2–<5 min`, `5–10 min` (600 s
inclusive) and `>10 min`. FMA durations come from the publisher's
`track_duration`; Beatport durations from the prepared 24 kHz preview sample
counts.

The later collection catalogs organize **1,355 current-plan Beatport records**,
**5,489 FMA records** and all **604 GiantSteps previews** with stable IDs, source
identities, native genre tags, observed artist, album and release links, frozen
memberships and exposure. They contain no key targets or predictions. Every count
uses the full collection as denominator, unknown values stay unknown, and
plotted duration is the recorded 24 kHz cache sample count divided by the sample
rate.

### GTZAN reference metadata, audio integrity and overlap

GTZAN references come from the
[annotation release](https://zenodo.org/records/5813199) at a
[pinned repository revision](https://github.com/audiocontentanalysis/dataset-gtzan-key/tree/9ef1591f81ee2f7821c7ce41e774ea7ac7f94cbe).
The [primary DAFx paper](https://www.dafx.de/paper-archive/2013/papers/03.dafx2013_submission_48.pdf)
describes manual annotation that excluded classical and ambiguous or modulating
excerpts; it gives no inter-annotator agreement. Sturm's
[fault lists](https://github.com/boblsturm/GTZAN/tree/fedc781026cf1a20034830f8071c98d93a422016)
define an eligibility filter, and the mirdata
[index 1.0](https://zenodo.org/records/13993311/files/gtzan_genre_index_1.0.json?download=1)
gives expected per-file MD5 checksums. The [protocol amendment](PROTOCOL.md#gtzan-reference-metadata-amendment-2026-11-12)
fixed the cohorts and rules before any scores.

For audio integrity, the single pinned archive from a
[public mirror](https://huggingface.co/datasets/marsyas/gtzan/tree/d2146561ecc7df707d9e6b8318885fe6a39668a2/data)
was downloaded once and streamed without extraction: its byte length and
SHA-256 were checked against the publisher, each WAV's MD5 was compared with the
mirdata index, and each WAV header was parsed. No classifier, key finder or
feature extraction ran. The audio is not redistributed.

For recording overlap, both GTZAN cohorts were fingerprinted against the existing
**7,579-recording reference index: 1,486 Beatport, 5,489 FMA and 604 GiantSteps**.
The [overlap protocol](GTZAN_OVERLAP_PROTOCOL.md) and
[query manifest](data/gtzan-overlap-manifest.json.gz) fixed membership before
matching. The matcher keeps all qualifying landmark offsets, not only the best.
A "review" match needs at least ten distinct anchors spanning five seconds; an
"identity" match additionally needs at least twenty anchors spanning eight seconds
and Chromaprint distance at most 0.15 at that offset. These thresholds predate
GTZAN and were not tuned on it. Before the main run, all **250 controls** behaved
as expected: 180 cached positive queries found their reference, 64 synthetic
negative queries produced no identity flags and five review flags, and six newly
extracted 30-second positives all found their reference. These are retrieval
checks, not a calibrated false-positive rate on music.

## Results

### Split counts

The rule produced **4,082 / 250 / 159 / 998** records for expanded fitting, inner
selection, calibration and development confirmation. These counts were kept; no
second seed or rule was tried to improve class support.

| Cohort | Assigned records / unique audio | Eligible after overlay, records / unique audio | Components | Largest component | Concentration equivalent* |
|---|---:|---:|---:|---:|---:|
| Restricted FMA fitting | 976 / 976 | 976 / 976 | 490 | 12 | 281.5 |
| Expanded FMA fitting | 4,082 / 4,036 | **4,080 / 4,035** | 797 | 272 | 71.3 |
| Inner selection | 250 / 250 | 250 / 250 | 100 | 5 | 83.3 |
| Calibration | 159 / 159 | 159 / 159 | 60 | 4 | 56.8 |
| Development confirmation | 998 / 997 | 998 / 997 | 200 | 12 | 154.4 |

*`N² / sum(component_size²)`, a descriptive concentration measure, not a
validated effective sample size.

Evaluation panels use far more groups than an exact-size split would. Expanded
fitting is still concentrated: its largest component holds 272 records, so more
recordings do not mean proportionally more independent works. Exact duplicates
with matching labels should count as one evaluation unit. With the same 1,230
Beatport fitting records, unique combined fitting grows from **2,206 to 5,265**, a
2.387× increase; the eligible raw combined count is 5,310. The eligible count of
4,080 FMA records results from the integrity exclusion, not from targeting that
number. The later amended pilot used a 4,033-record eligible fitting set; see its
[report](../k10-v2-pilot/REPORT.md).

### Balance and class support

Publisher duration is present for every FMA record. Eligible summed durations
are 223,585 s (restricted fitting), 893,216 s (expanded fitting), 53,972 s (inner
selection), 34,320 s (calibration) and 215,523 s (confirmation). These are
record sums including duplicates, not decoded or unique listening time, and equal
optimizer steps do not mean equal passes over the audio.

Coarse genre is missing for **3,065 / 5,489** records and treated as its own
MISSING category: 439 restricted, 2,257 expanded, 143 inner selection, 92
calibration and 573 confirmation. The corpus contains one Easy Listening and
eight Blues records, so balance across every genre is impossible. The rule
balances available coarse metadata within the non-forced pool, not a known
deployment distribution.

All 24 key classes occur in every panel after the overlay. Minimum unique
per-class support is 9 (restricted fitting), 47 (expanded fitting), **1 (inner
selection), 2 (calibration) and 14 (confirmation)**. No record was moved to improve
these counts, so per-class conclusions and calibration are limited. Targets are
the inherited FMAK-v2 annotations, not a measure of human consensus.

For planning, a 20% paired-disagreement rate on 997 independent recordings gives
roughly a ±2.8-point 95% interval for a near-zero difference; clustering can
widen it. A +2-point pilot is therefore a screen, not assured significance.
Future analyses must pair predictions and resample groups; folds, seeds and crops
do not multiply independent recordings.

### Historical exposure is kept

The full 5,489-record [exposure ledger](data/exposure-ledger.csv.gz) records each
ID's component, old role, proposed role, restricted membership and historical
exposure. The overlay changes only integrity status.

| Old FMA role | → Expanded fitting | → Inner selection | → Calibration | → Confirmation |
|---|---:|---:|---:|---:|
| Fitting | 976 | 0 | 0 | 0 |
| Validation | 94 | 49 | 29 | 78 |
| Calibration | 64 | 29 | 17 | 49 |
| Historical final | **631** | 72 | 46 | 251 |
| Quarantined development | 2,317 | 100 | 67 | 620 |

These are pre-overlay counts; two of the 2,317 quarantined-development records
moved to fitting are later ineligible. Re-partitioning does not erase earlier
program use, so confirmation here is internal, exposed development confirmation,
not independent proof. Both data arms must be trained fresh and evaluated on the
same panels, and encoders or features fitted on an inner holdout cannot be
reused as fresh inner-validation inputs.

### Identity guarantees and their limits

The checker finds **zero cross-role FMA primary-artist IDs, album IDs or recorded
exact audio hashes**. Every component is assigned whole, and all 976 old fitting
IDs remain eligible in expanded fitting. This covers the saved primary-credit
graph, not aliases, remixes, compositions or acoustic near-duplicates; no audio
was read here.

Beatport is a shared fitting pool with no new Beatport inner roles. Official
artist and release identity exists for only 618 / 1,230 fitting tracks; **612 are
unresolved**. Spreadsheet credits cover all records but differ from filename
credits on 242. Sharing identical Beatport data across both arms controls it for
the FMA comparison but does not support artist-independent Beatport claims.

A wider graph over all FMA master artist and album credits finds two components
linking fitting with selection and two linking fitting with confirmation through
records outside the 5,489. This is a metadata diagnostic, not proven acoustic
leakage or grounds to change the frozen split, but it matters when choosing a
future independent panel. Of 101,085 master records outside FMAK, 53,624 are
unlinked to FMAK under this graph; they are metadata candidates only. The 998
reserved confirmation IDs include 251 historical-final and 78 adaptive-development
records. Overlap with pretrained models' training data is unknown.

### Genre and duration coverage

A broad DJ key classifier needs evidence across hip-hop, contemporary Afrobeats,
house, techno and other club styles, and across short and extended recordings.
All 1,229 current Beatport fitting and 126 selection IDs have a workbook genre,
including explicit Deep House, Tech House, Progressive House and Techno, but they
are roughly two-minute previews. FMA supplies longer recordings, but specific
evaluation cells are sparse or unidentifiable.

![Observed native-genre duration coverage in current selection](coverage-genre-duration.png)

Each cell counts selection tracks per native genre and duration band (FMA
cells show tracks / components from published durations; Beatport cells show
tracks from prepared preview lengths). Look at the 5–10 minute column: FMA has
only a few tracks per genre there and nothing over ten minutes, while every
Beatport preview falls in the 2–<5 minute band (hugging 120 s) or just below it.
All 9
FMA House-tagged selection tracks are also Chill-out. Afrobeat is not
contemporary Afrobeats, and zero matching tags do not prove absence. Overlapping
FMA tags are not additive. [SVG](coverage-genre-duration.svg).

**FMA.** The master table has 106,574 records, of which 5,489 are in the FMAK
exposure ledger. In the current amended plan, expanded fitting has 4,033 records
from 797 components, selection 250 from 100, unique reserved confirmation 997
from 200 and excluded calibration 159 from 60; restricted fitting (976 records,
490 components) is nested in expanded fitting. The raw partition also includes 49
inactive former-fit IDs and 998 raw reserved records, which are not added back.
All 163 taxonomy tags were inventoried, and every tag count reproduced the
taxonomy's published count. Cells are tracks / components:

| Native FMA tag or explicit slice | Expanded fitting | Inner selection | Reserved unique | Calibration (excluded) |
|---|---:|---:|---:|---:|
| Hip-Hop | 269/50 | 16/8 | 41/13 | 16/6 |
| Rap | 57/5 | 0/0 | 0/0 | 2/1 |
| Afrobeat | 4/2 | 0/0 | 0/0 | 0/0 |
| House | 71/14 | 9/4 | 9/3 | 5/2 |
| Chill-out | 67/10 | 9/4 | 5/2 | 5/2 |
| House minus Chill-out | 4/4 | 0/0 | 4/1 | 0/0 |
| Techno | 17/11 | 6/3 | 11/3 | 2/1 |
| Named dance-tag union | 184/57 | 40/15 | 60/18 | 9/4 |

`genres_all` includes ancestor tags. 86/94 House-taxonomy FMAK tracks also carry
Chill-out; "House minus Chill-out" is a set difference, and its four reserved
records form one component. Techno includes Bigbeat. There is no Deep House,
Melodic House or contemporary Afrobeats tag in this taxonomy, and Garage (85)
sits under Rock, not UK garage. The dance-tag union is exactly House, Techno,
Disco, Dance, Drum & Bass, Jungle, Dubstep, Breakbeat and Breakcore - Hard: 293
FMAK records in 94 components. Its selection support of 40/15 splits into four
tracks under 2 min, 27 at two to five minutes and 9 at five to ten minutes, with
none under 30 s or over 10 min. FMA Hip-Hop selection has 16 tracks in 8
components, none five minutes or longer; Rap selection and reserved support are
both zero.

FMAK has 3,065 / 5,489 missing top-level genres but only 84 empty all-genre
lists; selection has 143 / 250 and 16. Known artist / album counts are 942 / 1,200
in expanded fitting, 118 / 118 in selection and 239 / 295 in reserved unique.

**Beatport** native workbook genres, kept verbatim:

| Workbook genre | Current fitting | Current selection |
|---|---:|---:|
| Big Room | 2 | 0 |
| Breaks | 77 | 10 |
| Deep House | 63 | 12 |
| Drum & Bass | 73 | 7 |
| Dubstep | 80 | 5 |
| Electro House | 77 | 11 |
| Electronica / Downtempo | 159 | 11 |
| Funky / Groove / Jackin' House | 9 | 0 |
| Hard Dance | 81 | 8 |
| Hip-Hop / R&B | 79 | 6 |
| House | 66 | 9 |
| Minimal | 69 | 3 |
| Progressive House | 82 | 8 |
| Psy-Trance | 83 | 9 |
| Tech House | 74 | 6 |
| Techno | 70 | 14 |
| Trance | 84 | 7 |
| Trap / Future Bass | 1 | 0 |

API JSON exists for 671 / 1,229 fitting and 61 / 126 selection records (558 / 65
missing); missing JSON does not mean an unknown workbook genre. API subgenres
appear for only 15 fitting and 2 selection records, and workbook and API genres
disagree on 41 fitting plus 2 selection records; both are kept. `Hip-Hop / R&B`
stays combined, and `Minimal` is not assumed to be Techno.

Preview lengths are 43.3–120.007583 s in fitting and 119.2–120.000750 s in
selection: 35 fitting and 5 selection previews fall below 120 s and 1,194 and 121
at or above it. **None reaches 5 minutes.** Of 676 known store durations, 500
describe five to ten minute releases, but 673 / 676 differ by more than 1 s from
the preview, so they cannot turn previews into full-track coverage. All 1,355
Beatport entries currently use single-record placeholder groups, which are not
1,355 verified independent artists.

### Collection metadata availability

![Metadata availability in the organized Beatport and FMA collections](figures/collection-metadata-availability.png)

Colored bars are records with the field present and pale bars the unknown
remainder, with counts beside each row. The tempo rows stand out: most Beatport
and all FMA records have no BPM. Availability says nothing about annotation
correctness.

| Metadata available | Beatport, n=1,355 | FMA, n=5,489 |
|---|---:|---:|
| Any native genre tags | 1,355 | 5,405 |
| Workbook / native top genre | 1,355 | 2,424 |
| Observed artist ID | 676 | 5,489 |
| Observed album or release ID | 676 | 5,459 |
| Publisher BPM | 716 | 0 |
| Recorded 24 kHz cache duration | 1,355 | 4,283 |
| Publisher recording duration | 676 | 5,489 |
| Separately probed source-file duration | 0 | 0 |

**639 Beatport records and all 5,489 FMA records have no BPM.** The 716 Beatport
values are archived publisher reports, not estimated tempo; half/double-time
ambiguity is unresolved.

![Recorded training-cache duration counts, retaining unknown FMA rows](figures/collection-recorded-duration.png)

Counts of saved training-cache duration per band, with unknown FMA records kept
as their own bar. Beatport's single tall bar is the 2–<5 minute band, made of
roughly 120-second previews. Scales differ between the two panels.

| Recorded cache duration | Beatport | FMA |
|---|---:|---:|
| <30 seconds | 0 | 18 |
| 30 seconds–<2 minutes | 40 | 543 |
| 2–<5 minutes | 1,315 | 3,058 |
| 5–10 minutes, inclusive | 0 | 630 |
| >10 minutes | 0 | 34 |
| Unknown | 0 | 1,206 |

FMA has cached audio for the **4,033 expanded fitting and 250 inner-selection
records**; the other **1,206 reserved or excluded IDs** have none in the cache, and
their publisher durations are not substituted. FMA's 5,489 records contain 5,442
distinct source hashes in 1,157 components.

The [K10/MusicFM retrospective diagnosis](../musicfm-key-pilot/REPORT.md#retrospective-diagnosis-by-genre-recording-duration-and-bpm)
joins these catalogs to the 376 development recordings through a separate
[analysis file](data/development-slices.json.gz) that holds model predictions and
selected references; the catalogs themselves stay target-free. It verifies IDs,
source identities and sample support before scoring, keeps empty and unknown
cells, and does not change splits or thresholds. Its first attempt was
inconclusive and was rerun with incremental catalog reading
([execution record](data/development-slice-execution.json)).

### GiantSteps: all 604 previews

The earlier scored panel used only 45 GiantSteps recordings. The
[GS604 collection](data/giantsteps604-collection.json.gz) organizes all **604**,
joining the publisher annotation archive to the overlap plan and the saved
fingerprint decodes, without extracting key targets or predictions. All 604 have
a native genre and recorded decode support. The 23 native labels include **77
deep-house, 47 house, 88 progressive-house, 81 tech-house, 34 techno, 38
drum-and-bass, six glitch-hop and four hip-hop** recordings. No record has BPM,
verified artist or work grouping, or full-song duration.

Saved decodes span **44.70–120.01 seconds** at 11,025 Hz mono float32, the audio
used for fingerprinting rather than the 24 kHz training cache. There are **602
distinct audio hashes across 604 IDs**. All 604 have been used in release
evaluation; direct training membership is not inferred.

### GTZAN references

The pinned release reproduces **837 single-key references, including 81 hip-hop
excerpts**:

| Genre | Single-key references | Sturm filtered union | Both (eligible) |
|---|---:|---:|---:|
| Blues | 98 | 100 | 98 |
| Classical | 0 | 99 | 0 |
| Country | 99 | 98 | 97 |
| Disco | 98 | 93 | 91 |
| Hip-hop | 81 | 92 | 73 |
| Jazz | 79 | 87 | 67 |
| Metal | 93 | 91 | 84 |
| Pop | 94 | 84 | 78 |
| Reggae | 97 | 86 | 83 |
| Rock | 98 | 100 | 98 |
| **Total** | **837** | **930** | **769** |

Sturm's fault metadata has disjoint filtered lists of 443, 197 and 290 IDs; their
union is used only as an eligibility filter. The eligible intersection excludes
68 of the 837 references. Of all 1,000 IDs, 163 have annotation −1 (including all
100 classical excerpts) and 70 fall outside the filtered union; these overlap.
`jazz.00054`, flagged as corrupt by the published loader, already has annotation
−1. The [reference ledger](data/gtzan-reference/reference-ledger.json.gz) keeps all
1,000 IDs with annotation hashes, class mapping, expected archive members, MD5
checksums and status fields, and the [summary](data/gtzan-reference/summary.json)
records the pinned sources. The 837 and 769 denominators are fixed and cannot be
changed after seeing detector outputs.

### GTZAN audio integrity

The 1,226,192,050-byte archive matches the publisher checksum. Every recording in
both reference cohorts matches its expected checksum and parses as a mono
22,050 Hz 16-bit PCM WAV:

| Cohort | Recordings | Matching MD5 | Parseable WAV |
|---|---:|---:|---:|
| Complete inventory | 1,000 | 999 | 999 |
| Single-key references | 837 | 837 | 837 |
| Fault-filtered eligible | 769 | 769 | 769 |

The only failure is `jazz.00054`, already excluded: its checksum differs from the
index and its bytes do not start with a RIFF header. This creates no new
exclusion and does not show the mirror is byte-equivalent to the historical
archive. The strict full-inventory check therefore fails; a separate inventory
mode recorded all 1,000 outcomes. The mirror's archive hash differs from the
historical Torchaudio archive and its audio license is unspecified.

The [1,000-record GTZAN catalog](data/gtzan1000-collection.json.gz) joins the
reference ledger and integrity results by ID, keeping reference inclusion, file
identity, header parsing and scoring admission as separate fields:

| Observation | Records |
|---|---:|
| Complete source inventory | 1,000 |
| In single-key references | 837 |
| In fault-filtered eligible references | 769 |
| Acquired MD5 matches the index | 999 |
| WAV header parseable | 999 |
| Excerpt duration known (from WAV header) | 999 |
| Model-analyzed audio support | 0 |
| Admitted for classifier scoring | 0 |

The identity and header counts coincide but are independent checks. The 769
cohort contains 73 hip-hop excerpts and zero classical.

### GTZAN overlap audit

All 837 reference recordings were queried. The first pass stopped after 377
attempts when `reggae.00086` hit the matcher's fixed memory limit; its 376
completed queries and the one incomplete query are kept, and that query was not
retried. A continuation then processed the 460 remaining recordings in the same
order with the same matcher, index and rules. The 769-recording primary cohort
completed in full; the wider cohort has that one incomplete query. No query
produced an identity-level match, but **167 primary recordings have review-level
matches**.

| Cohort | Attempted | Complete | No overlap detected | Needs review | Incomplete | Identity matches |
|---|---:|---:|---:|---:|---:|---:|
| Primary (fault-filtered) | 769 | 769 | 602 | 167 | 0 | 0 |
| Sensitivity (all single-key references) | 837 | 836 | 656 | 180 | 1 | 0 |

The cohorts overlap, so rows must not be added. "Complete" means the query ran,
not that exposure is cleared. In the machine-readable output, `unresolved` counts
incomplete queries only, not open review flags.

![GTZAN overlap outcomes by genre and actual analyzed excerpt duration](figures/gtzan-overlap.png)

Per-genre bars separate no detected overlap, review flags and incomplete queries
for both cohorts (classical is zero); look at how unevenly the orange review
flags fall across genres, from about half of disco to none of jazz.
The duration histogram uses the 836 retained extraction records. These are
recording-overlap outcomes, not classifier errors. [SVG](figures/gtzan-overlap.svg).

Review flags concentrate in a few genres: **48/91 disco**, **25/84 metal** and
**22/97 country**, against **11/73 hip-hop** and **0/67 jazz** in the primary cohort.
Dropping flagged recordings would change the genre mix. Analyzed audio spans
**29.93–30.65 seconds** (median **30.01 seconds**).

For the released Drone model specifically, **80 primary queries** have
review-level matches to its known fitting IDs or exact aliases and **23** to its
selection IDs or aliases (83 and 24 in the wider cohort). These categories can
overlap and are unresolved flags, not confirmed exposure. The released Drone-v1
was trained on Beatport only; other reference-index matches matter for research
models with different training data, so any future score must say which model's
exposure it addresses. No GTZAN loader or scoring artifact was found in the
inspected Drone code, reports or history, which does not prove GTZAN was never
used elsewhere. The [evidence index](data/gtzan-overlap/index.json) publishes every
query and all 55,474 review- or identity-level alignments; all 1,704,541 original
alignments, including those below screening level, were checked against the fixed
rule, and the full raw set is retained outside the repository.

### KeyFinder wrapper defect

The existing wrapper used for the [93-recording four-way score report](../key-sota-4way/REPORT.md)
compares libkeyfinder's key enum against `C_MINOR` as a range instead of treating
major and minor as alternating, and also shifts several major tonics. **Its
KeyFinder accuracy on those 93 tracks is invalid as a KeyFinder comparison.** The
original report is kept as a historical record; old outputs cannot be remapped
because several raw enums collapse to the same wrong label. The other detectors
in that report are not affected by this finding.

A new standalone reference CLI maps every libkeyfinder 2.2.8 key enum and silence
correctly, verified at compile time and by a runtime self-test against the real
library ([25 self-test lines](data/keyfinder-reference/self-test.jsonl) covering
24 keys and silence). Tests also require compilation to fail if A major and A minor
are swapped, and reject malformed WAV before any classification. The CLI accepts
one mono PCM16 WAV path and prints one JSON object (schema
`drone.keyfinder-reference.v1`) with the raw enum, status `key` or `silence`, and
for keys a C-first interleaved `drone_class` 0–23, sharp tonic and mode. A
nonzero exit is an error and must never be dropped silently from a denominator.
The library's default `keyOfAudio` algorithm is unchanged. No recording was
decoded or classified; this qualifies the interface only. The
[protocol](data/keyfinder-reference/PROTOCOL.md), [frozen inputs](data/keyfinder-reference/frozen.json),
[execution record](data/keyfinder-reference/receipt.json) and
[library inspection](data/keyfinder-reference/linked-libraries.txt) record the
build.

## Decision

- **Split.** No new role assignment or training was activated by the original
  proposal; v1 data, weights and thresholds stayed unchanged. Open questions at
  that point were role retirement, the narrow Beatport identity guarantee,
  duplicate scoring units, rare-class and genre coverage, clustered precision and
  paired training budgets. The later amended pilot is reported in
  [K10 v2 pilot](../k10-v2-pilot/REPORT.md).
- **Coverage.** Keep the live model comparison on its original terms. The main
  data gaps are full-length (five to ten minute) house, deep-house and
  melodic-house, contemporary Afrobeats and hip-hop, reliable tempo, direct genre
  semantics and recording/work relationships. A small feasibility check could use
  four worked examples (contemporary Afrobeats, hip-hop, a five to ten minute
  house/deep-house/melodic-house extended version, and a separate-artist
  techno/club recording, preferably over ten minutes): exact versions and access
  chosen before predictions, two independent style-qualified annotators blind to
  models, vendor keys and each other, covering whole recordings and keeping
  disagreements and no-stable-key cases. These examples would become development
  data, not a sealed panel, and measuring their annotation effort would size a
  later panel. No acquisition, annotation or contact has started.
- **GTZAN.** Keep both cohorts and all outcomes intact. Review flags need a
  separate blind disposition policy; absence of a detected match is not
  admission. Before any paired classifier evaluation, freeze model and key-finder
  versions, scoring conventions, missing-output handling and exposure-stratified
  reporting. No recording is admitted for scoring yet.
- **Mixed In Key.** The next product comparison asks whether Drone is more
  accurate than Mixed In Key on the same independently annotated full recordings.
  The [comparison protocol](MIK_COMPARISON_PROTOCOL.md) fixes separate development
  and final panels, blinded annotation, identical audio editions, a named
  comparator build, paired uncertainty, genre and length coverage, and separate
  accounting of feature-extraction and model cost. No comparison has run. A local
  [access inventory](data/mik-comparison-access.json) found no Mixed In Key
  installation and no license is available (Rekordbox being installed does not
  provide it). Independent full-length audio, annotators and a powered final
  panel are also unresolved. Model research continues meanwhile. FMA roles and
  the private DJ library's restriction to manual review remain unchanged.

## Limitations

- The split guarantees cover saved primary-artist, album and exact-audio links
  only, not aliases, remixes, compositions or near-duplicate audio. Beatport has
  612 fitting tracks with unresolved identity, so no artist-independent Beatport
  claim is possible.
- Confirmation records were exposed in earlier program work; confirmation is
  internal development evidence, not an independent or state-of-the-art test.
- Some panels have very small per-class support (one in inner selection, two in
  calibration).
- FMA references are derived from expert-annotated FMAK; the
  [FMAKv2 release](https://zenodo.org/records/12759100) changed about 200
  annotations that the [original FMAK](https://zenodo.org/records/10719860)
  authors did not verify. That does not make the revisions wrong, but the
  references are not 5,489 uniformly re-adjudicated labels. The internal "v2
  split" and the FMAKv2 annotation version are unrelated names. The
  [author README](https://github.com/stellaywong/fma_keys) describes Spotify URI
  matches for analyzer comparison, not Spotify-generated ground truth.
- The [Beatport key release](https://zenodo.org/records/1101082) credits human
  annotations to Eduard Mas Marín, revised and expanded by Ángel Faraldo, distinct
  from store metadata keys. Single-key encoding can hide ambiguity. The
  [mirdata loader](https://github.com/mir-dataset-loaders/mirdata/blob/master/mirdata/datasets/beatport_key.py)
  documents source and annotation terms. Annotation availability is not a right
  to redistribute recordings.
- Genre tags are native and unverified by listening. FMA's `track_genres_all`
  mixes direct and ancestor tags without saying which; empty tags do not prove
  absence. Neither source is mapped to a modern normalized taxonomy.
- Publisher durations and preview lengths answer different questions; neither is
  a newly measured full-song duration, and metadata rounding can shift a band.
- GTZAN clips are about 30-second historical excerpts. The 73 eligible hip-hop
  clips do not represent contemporary Afrobeats or five to ten minute DJ tracks.
  GTZAN audio rights, the mirror's equivalence to the historical archive and
  overlap with MusicFM/MSD pretraining data are unresolved.

## Reproduce

All commands run locally without network access, audio, labels or models unless
noted. Checksums for all evidence files are in [SHA256SUMS](SHA256SUMS).

Split and overlay:

```sh
python -m tools.check_v2_contract
python -m pytest tests/test_v2_contract.py -q
```

The checker reproduces the proposal, ledger and overlay (comparing decompressed
CSV content, since gzip bytes vary by platform), verifies the before-label and
sanitization chain and the append-only protocol amendments, and checks overlay
coverage, ID and audio-hash separation and counts. Tests cover whole-component
assignment, ordering invariance, bounded groups, same-audio rejection, rejection
of label fields in assignment inputs, generic conflict quarantine, and rejection
of invented ancestors or rewritten protocols. The checker was also run
successfully on Linux with Python 3.12.3.

Genre and duration coverage:

```sh
python -m tools.dj_coverage --inputs experiments/studies/v2-data-contract/data/coverage-inputs.json --output /tmp/coverage-fma-replay
python -m tools.dj_coverage_bp --inputs experiments/studies/v2-data-contract/data/coverage-inputs.json --output /tmp/coverage-bp-replay
python -m tools.dj_coverage_report --replay-fma /tmp/coverage-fma-replay/coverage.json --replay-bp /tmp/coverage-bp-replay/coverage-bp.json --plot
python -m pytest -q tests/test_dj_coverage.py
```

Only the [aggregate index](data/coverage-index.json) and
[input descriptors](data/coverage-inputs.json) are committed, not source rows or
key labels; input files may be relocated as long as their checksums match.

Collection catalogs:

```bash
uv run python -m tools.report_collection_coverage
uv run python -m tools.report_collection_coverage --write --plot
uv run python -m tools.export_giantsteps_collection --external-sources sources.json
uv run python -m tools.export_gtzan_collection
```

The first command verifies the stored [counters](data/collection-coverage-analysis.json);
the second regenerates them and the figures (4.48 s wall, 4.15 s CPU, 238,665,728
bytes peak memory, 487,991 bytes of output). For GiantSteps, `sources.json` maps
`giantsteps_annotations` to the pinned archive (its location may change, its
checksum may not), and omitting `--write` checks the committed files (0.34 s wall,
0.42 s CPU, 89,800,704 bytes peak memory). The GTZAN catalog export took 0.76 s
wall, 0.69 s CPU and 44,515,328 bytes peak memory, writing 127,531 bytes.

GTZAN references, integrity and overlap:

```sh
python -m tools.import_gtzan_reference --check
python -m tools.check_gtzan_audio_report
python -m pytest -q tests/test_gtzan_audio.py tests/test_gtzan_reference.py
uv run python -m tools.prepare_gtzan_overlap --check
uv run python -m tools.report_gtzan_overlap
uv run --extra plots python -m tools.report_gtzan_overlap --plot
# Optional, with the retained raw alignment archive; no audio is opened:
uv run python -m tools.report_gtzan_overlap --raw-audit /path/to/audit-metadata.tar
```

With the downloaded archive, `python -m tools.check_gtzan_audio ARCHIVE --output FILE`
runs the strict check (and fails on `jazz.00054`); `--inventory-only` records all
mismatches without admitting anything for scoring. The
[integrity evidence](data/gtzan-audio-integrity/evidence.tar.gz) contains no audio.
The archive download took 13.04 s; the inventory took 6.96 s wall, 6.93 s CPU and
47,501,312 bytes peak memory. The overlap driver's control run took 306.25 s wall,
309.56 s CPU and 872.51 MiB peak memory; the continuation over 460 recordings took
445.20 s wall, 455.06 s CPU and 839.06 MiB, writing 26.94 MiB; the first pass
recorded 294.12 s CPU and 1.10 GiB peak memory. KeyFinder self-test compilation
took 0.623288 s wall, 0.313313 s CPU and 94,969,856 bytes peak memory.
