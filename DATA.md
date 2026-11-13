# Dataset inventory and provenance

The purpose of this inventory is to establish **where the released model is
strong or weak by genre, analyzed length, BPM and other supported attributes**,
and distinguish an accuracy result from a hole in our data. Bundled-model
provenance remains in [weights/README.md](weights/README.md).

Two related views now organize the existing evidence:

- The [reference-panel catalog](experiments/studies/v2-data-contract/data/reference-panel-catalog.csv)
  describes the exact 93 scored recordings and joins their existing predictions.
- The [Beatport collection](experiments/studies/v2-data-contract/data/beatport1355-collection.json.gz)
  contains all 1,355 current development IDs; the
  [FMA collection](experiments/studies/v2-data-contract/data/fmak5489-collection.json.gz)
  contains all 5,489 source-reference IDs. The
  [GiantSteps collection](experiments/studies/v2-data-contract/data/giantsteps604-collection.json.gz)
  now covers all 604 source previews. These collection views contain metadata,
  frozen role history and exposure witnesses, **without key targets or predictions**.

Both reuse typed identity, measurement, native-genre and evidence primitives.
The [collection contract](experiments/collection_catalog.py) permits explicitly
missing observations; the [reference contract](experiments/corpus_catalog.py)
retains the stricter identity requirements of the scored panel. Source archives
are resolved by explicit local mappings and verified hashes. A local path is not
an identity. See the [coverage study](experiments/studies/v2-data-contract/REPORT.md)
for methods, figures and limitations. The GTZAN collection adapter remains outstanding.

## Current corpus inventory

Counts describe different stages and must not be added as if every row were a
unique, usable, independent evaluation recording.

| Corpus and version | Existing population / usable scope | Current roles and evidence |
|---|---|---|
| Beatport EDM Key, [pinned Zenodo release](https://doi.org/10.5281/zenodo.1101082) | 1,486 source IDs. The current research plan has 1,229 fitting + 126 selection IDs after its eligibility amendment. | Released `drone-v1` instead declares 1,237 fitting + 126 selection IDs. Preserve both histories; aliases can carry both roles. [Release metadata](experiments/studies/bp-gs-overlap/data/release-metadata.json), [research amendment](experiments/studies/k10-v2-pilot/REPORT.md). |
| [FMAKv2](https://zenodo.org/records/12759100), derived from [FMAK](https://zenodo.org/records/10719860) | 5,489 key-reference recordings. Current expanded fitting has 4,033 eligible IDs; the nested restricted fit has 976. | Current roles also include 250 inner selection, 998 raw / 997 container-unique reserved confirmation, 159 excluded calibration and 49 inactive/quarantined fitting candidates. These are research roles, not a new release-model training history. [Exposure ledger](experiments/studies/v2-data-contract/data/exposure-ledger.csv.gz). |
| [GiantSteps Key](https://github.com/GiantSteps/giantsteps-key-dataset) | 604 source previews; the reference panel uses a fixed 45-recording sample. | Explicit evaluation input, but development-exposed for released Drone. It is not globally “held out” for every model or experiment. Downloader annotation revision: `6bcd492c825ac9b8597bc650a5f6fd18b6c43d2b`. |
| [Schubert Winterreise Dataset](https://zenodo.org/records/10839767), run-declared v2.1 | 48 available labeled performances: 24 works × 2 performance versions, HU33 and SC06. | All 48 are in the reference comparison. Recording IDs and actual input identities are preserved; a directory version string is not a substitute for a corpus acquisition receipt. Work-level grouping is essential. |
| GTZAN, pinned key annotations / Sturm filtered lists / recording-index v1.0 | 1,000 source IDs; 837 raw single-key references; 769 named filtered eligible references, including 73 hiphop. | Metadata eligibility, audio integrity, overlap review and score admission are separate. The diagnostic audio inventory matches 999/1,000 source identities; the already flagged `jazz.00054` is outside the 837/769 cohorts. No independence is inferred from the filtered lists. [Metadata ledger](experiments/studies/v2-data-contract/data/gtzan-reference/summary.json), [integrity evidence](experiments/studies/v2-data-contract/data/gtzan-audio-integrity/evidence.tar.gz). |

The SWD publisher identifies v2.1 at the linked release; the analyzed run's
source archive still needs its own acquisition/checksum receipt.

The downloader writes `data/sources.json`; annotation/archive checksums are pinned
in [drone/download.py](drone/download.py). Missing or mismatching downloads are
not verified inputs. Preserve raw source archives, version/commit, member path,
checksum, retrieval receipt and the exact adapter that produced each catalog
field. Annotation rights and recording rights are separate.

## Which metadata actually exists

| Corpus | Genre, artist and work metadata | Duration support | BPM / tempo status |
|---|---|---|---|
| BP | Workbook genre is complete for the current 1,355 research IDs. Existing original store JSON includes native genre/subgenre, artists, remixers, release, mix and label fields, with incomplete source coverage. Do not equate workbook and store taxonomies. Artist/album independence remains unverified. | Canonical preview sample counts exist. Store JSON can describe the full catalog track instead; keep that duration separately. | A `bpm` field exists in the pinned store archive. It is **store-reported metadata**, not a manually validated tempo reference. The metadata-only adapter admits 716/1,355 current research IDs (656/1,229 fitting; 60/126 selection), range 66–195 BPM. The other 639 are unknown: 623 missing source records, 11 missing/null and five zero sentinels. The older genre/duration allowlist did not admit BPM. |
| FMA | The existing master CSV includes artist/album IDs and names, track title/composer, native top/all genre fields, license and subset. Artist/album components and historical exposure are recorded. Missing top-level genre does not imply missing native subgenre. | Metadata track duration and existing actual waveform/cache support are distinct witnesses. Some FMA distributions are 30-second excerpts, others full audio. | No BPM column exists in the master CSV used here. Upstream FMA offers other feature tables, including partial Echonest coverage; those are separate analyzer-derived sources and are not silently joined or treated as tempo truth. |
| GS | The full 604-row collection preserves 23 native publisher genres from the pinned annotation archive. Artist/work IDs are not established. The archive also contains `sources.xlsx`; its artist strings are not verified identities. | Saved fingerprint decode support spans 44.70–120.01 seconds at 11,025 Hz. Its 604 IDs have 602 distinct decoded PCM hashes. Publisher JAMS duration describes a preview; full-song duration stays unknown. | No BPM source is admitted. The separate GiantSteps Tempo dataset is not assumed to have identical membership or recording versions. |
| SWD | Work and performance IDs are explicit; “classical” describes this vocal/piano song cycle, not a representative sample of all classical music. Global/local key and measure annotations are separate source types. | The 48 analyzed recordings range 44.07–325.33 seconds, median 169.12 seconds. Two performances of a work are related observations. | No scalar BPM is admitted in the current panel. Score tempo, measure timing and performed tempo are different quantities; do not invent one whole-song BPM from a classical work label. |
| GTZAN | Native genre follows stable recording IDs. Reliable artist/work linkage is incomplete; known repetitions, versions and faults require separate identity evidence. | The pinned inventory supplies per-WAV frame counts, rate, bytes and hashes. The clips are approximately 30 seconds, not full-song support. | No BPM source is admitted in the current GTZAN catalog preparation. Genre and filename do not supply BPM. |

The quantitative BP/FMA coverage witnesses are
[coverage-inputs.json](experiments/studies/v2-data-contract/data/coverage-inputs.json)
and [coverage-index.json](experiments/studies/v2-data-contract/data/coverage-index.json).
The separate [BP tempo metadata](experiments/studies/v2-data-contract/data/beatport-tempo-metadata.json.gz)
is produced by [its adapter](tools/beatport_tempo_metadata.py) from the pinned store
archive and current 1,355-ID plan, not the released model's 1,363 fitting/selection IDs.
Reuse [tools/dj_coverage.py](tools/dj_coverage.py),
[tools/dj_coverage_bp.py](tools/dj_coverage_bp.py) and
[tools/dj_coverage_report.py](tools/dj_coverage_report.py); do not create competing
counts from a second taxonomy interpretation. Their old absolute acquisition
paths can point at retired worktrees. Resolve repository copies by the recorded
content hashes; never substitute a different file because its basename matches.

The [FMA publisher](https://github.com/mdeff/fma#data) distinguishes `tracks.csv`,
`genres.csv`, computed `features.csv`, partial `echonest.csv`, 30-second subsets
and untrimmed full audio. The [GiantSteps publisher](https://github.com/GiantSteps/giantsteps-key-dataset/blob/master/README)
distinguishes manual key corrections, site-derived genres and preview audio.
Neither provides a reason to call every metadata value expert ground truth.

## Stable organization and joins

The catalog contract owns exact field names and validation. Each corpus adapter
must preserve these distinctions rather than change the meaning to fit a table:

- **Recording identity:** corpus namespace + native ID + source version; original
  container hash, decoded identity under an explicit decode contract, and known
  aliases. A file path is a locator, not the identity. Duplicate recordings,
  artist/album components and work/performance links are separate relations.
- **Source provenance:** source URL/revision, archive/member, byte hashes,
  acquisition receipt and adapter/version. Retain original native genre IDs and
  taxonomy hierarchy and source-native tag lists without duplicating recordings.
  The schema supports source-native tag lists with direct/ancestor relations;
  the FMA adapter preserves hierarchy-inclusive source tags without inventing direct-versus-ancestor assignments. Any display bucket is an explicitly
  versioned derivation.
- **Analysis support:** full catalog duration, actual supplied audio duration,
  analyzed start/end/support, codec, sample rate, channel handling, preprocessing
  and truncation policy. A two-minute preview of a seven-minute catalog track belongs in the two-minute analyzed-duration slice.
- **Reference provenance:** raw key annotation, original annotator/revision,
  normalized key class/mode, label-source version and ambiguity/alternative
  policy. Preserve unknown/modulating/multiple labels and conflicts; do not silently repair them after seeing predictions.
- **Tempo provenance:** reported or estimated value, units, source method,
  temporal support and any supplied uncertainty. Missing, invalid, unmeasured
  and conflicting values remain distinguishable. Never guess a tempo from genre.
- **Role and exposure:** dataset partition and model/checkpoint-specific fitting,
  selection, historical evaluation, confirmation and overlap status. Unknown
  exposure stays unknown. A new split cannot erase earlier use of a recording.
- **Prediction linkage:** checkpoint and evaluator identity, exact recording/input
  identity, prediction/status and complete-denominator relation score. Predictions
  belong to a run; they are not reference labels or intrinsic corpus attributes.

Artist names alone do not establish artist identity, and matching song titles
do not establish recording identity. SWD work links are known; BP record-only
proxies are not verified artist groups. Preserve separate evidence/status for
exact container, decoded PCM, fingerprint, work/version and artist relationships.
Passing an available overlap check does not establish absent unknown exposure.

Versioned source indexes, deterministic adapters and checksum validation follow
existing [mirdata practices](https://mirdata.readthedocs.io/en/stable/source/mirdata.html).
Our catalog remains a project-specific view over source witnesses, not a claim
that corpus versions, taxonomies or independence have been reconciled completely.

## What an accuracy slice must show

Lock cohort membership before examining model scores. Report the checkpoint,
corpus/cohort and exposure status before interpreting a slice. For every genre,
analyzed-duration or sourced-BPM slice, report:

- Total eligible recordings, metadata-known/missing/invalid counts and, when
  defensible, independent artist/work/component counts.
- Exact correct count/fraction and the existing symmetric-fifths MIREX score
  under the named key convention. Keep errors and abstentions in the fixed
  denominator, with valid/error/abstention counts shown separately.
- The reference source and any ambiguity/revision limitations; native genre
  membership and versioned bucket definitions; measured analyzed support.
- Whether the slice was defined before examining outcomes, and whether its
  observations selected this model. Three to five exposed tracks describe those
  tracks; they do not establish a genre-level strength or weakness.

Only 60/126 BP selection recordings have admitted BPM, so BPM-slice accuracy
would describe metadata-known recordings; it cannot be extrapolated to all 126
or establish a causal effect of missing metadata.

Keep an explicit unknown-metadata slice. Otherwise a BPM report can silently
exclude the recordings with missing BPM and look more representative than it is.
Overlapping genre tags produce overlapping slices, so their counts need not sum
to the full cohort. Descriptive intervals should use appropriate saved groups;
recording-level resampling is not a work-level uncertainty estimate for SWD.
Descriptive findings do not authorize optimizing on the same held-out cohort.
Vocals, tuning and key changes may be useful future attributes, but remain unknown
until a documented source or measurement is admitted.

The [93-recording reference report](experiments/studies/key-sota-4way/REPORT.md)
already separates Winterreise from development-exposed GS and preserves the
corrected comparator evidence. This is an initial coverage/accuracy map, not an
all-genre diagnosis. Its legacy GS buckets also merge drum-and-bass into
`dubstep` and glitch-hop into `hip-hop`; the latter is not broad hip-hop coverage.
Preserve native tags alongside these historical display buckets. The reference panel does not justify a full-corpus BPM, artist-independent or
long-track accuracy claim. Collection metadata must be joined to the appropriate
scored cohort before interpreting model strengths or weaknesses.

## Coverage gaps and next adapters

The current evidence cannot answer independent accuracy for **five–ten-minute
DJ tracks**. BP catalog metadata has many long tracks, but its available analyzed
previews top out at about 120 seconds. The FMA inner panel contains only **five
actual five–ten-minute House (182) recordings**; the 93-recording reference panel
contains no long DJ tracks. A larger source-duration count does not fill this gap.

FMA's pinned taxonomy includes **Afrobeat (81)** under African (92) and House (182) under
Electronic (15). Neither an African parent tag nor a legacy Afrobeat tag establishes
coverage of contemporary **Afrobeats**, Afro house or current DJ production.
Keep those names and coverage questions distinct. Missing tags are not proof
that no relevant music exists, and a few available recordings do not establish
broad contemporary coverage.

The BP and FMA adapters now join the existing sources without changing roles:

| Collection | Source rows | Native genre metadata | Artist / album or release IDs | Recorded 24-kHz cache support | Admitted BPM |
|---|---:|---|---|---|---|
| Current Beatport development cohort | 1,355 | All workbook genres; API taxonomy remains separate | 676 with artists and release IDs; 145 with remixers | 1,355 | 716 publisher-reported; 639 unknown |
| FMA source-reference cohort | 5,489 | 5,405 with hierarchy-inclusive tags; top-level field missing for 3,065 | 5,489 with artist IDs; 5,459 with album IDs | 4,283; 1,206 unknown | 0; no admitted tempo source |

FMA has 5,442 distinct saved source-container hashes and 1,157 existing frozen
components. Multiple rows therefore need not be independent recordings. The
1,206 missing cache-support witnesses comprise 998 reserved confirmation rows,
159 calibration rows and 49 excluded fitting candidates. Publisher duration does
not fill those unknowns. Existing 24-kHz cache observations describe that training
preparation; they are not a claim that released Drone used those exact PCM bytes.

Reproduce with [the BP adapter](tools/export_beatport_collection.py) or
[the FMA adapter](tools/export_fma_collection.py). The BP adapter accepts an
`--external-sources` JSON mapping with `workbook`, `store_json_zip` and
`canonical_counts` paths. The FMA adapter accepts explicit `--master-metadata`,
`--taxonomy` and `--cache-metadata` paths. Both verify the pinned source hashes;
no source audio is opened. Recorded source-file and PCM identities remain identified
as historical witnesses, not newly verified files.

The [GS adapter](tools/export_giantsteps_collection.py) accepts an
`--external-sources` JSON mapping with the single key `giantsteps_annotations`
pointing to the pinned publisher metadata archive. Its separate collection
records historical 11,025-Hz fingerprint decode support, not the BP/FMA 24-kHz
cache support in the table above. All 604 GS recordings are marked as already
evaluated in release development; that does not assert direct fitting membership.

Next work:

1. Use the completed [development slice report](experiments/studies/musicfm-key-pilot/REPORT.md#retrospective-diagnosis-by-genre-recording-duration-and-bpm)
   to identify coverage and error patterns for the next prospective experiment.
   The 376-record K10/MusicFM joins are descriptive analysis of exposed data,
   not independent confirmation; all small, empty and unknown cells are retained.
2. Add GTZAN's frozen 837/769 membership, integrity and overlap status without
   changing its cohorts. GiantSteps metadata now covers all 604 previews.
3. Assess gaps in long DJ recordings and contemporary genre coverage before
   choosing additional public data or an annotation effort. Preserve native
   taxonomies and distinguish reported tempo from measured or verified tempo.

These adapters preserve the existing fitting roles and reference labels. The
private DJ library remains reserved for performance testing and manual review.
Future acquisition should fill a measured coverage gap with documented
identity, labels, metadata and use terms; it should not be chosen to rescue a
particular observed score.

## Reference trust and use terms

Beatport human annotations by Eduard Mas Marín, revised/expanded by Ángel Faraldo,
are distinct from store metadata keys. Published annotations are CC BY-SA 4.0;
that does not establish audio redistribution rights. FMAKv2 reports about 200
revisions not verified by the original FMAK annotators; this does not prove those
revisions wrong. Preserve the released targets and revision provenance.
[Spotify URI matches](https://github.com/stellaywong/fma_keys) support comparison
with an analyzer, not Spotify-generated reference keys.

SWD has separate score, annotation and performance rights; consult the
[release description](https://zenodo.org/records/10839767) and performance-specific
notices. GTZAN annotation licensing does not grant rights to its source recordings.
Store corpus/source/annotation/audio terms separately, preserving missing terms
as unknown. Preview or research audio is not bundled in package builds. Source
availability, checksum integrity, scientific independence and redistribution
permission are four different properties.

## Personal corrections

Human annotations are optional explicit inputs, separate from Drone's public base
model data. `train.py prepare-corrections` accepts portable schema 1 JSONL produced
by `euphoria corrections export` or another tool implementing that adapter. Drone
does not collect annotations automatically, read private libraries unless their
folder is explicitly supplied, or upload audio/labels. Only corrected keys are
fitting labels; tempo-only records are counted and skipped.

Each line is a JSON object with `schema_version: 1`, a lowercase 64-hex-character
`content_hash` (full-file BLAKE3), `corrected_key` (`tonic` integer 0–11 and `mode`
`major`/`minor`, or null for a tempo-only record), and integer Unix-seconds
`corrected_at`. Optional `detected_key` uses the same key object or null. Additional
Euphoria display/tempo fields are accepted without becoming key-training targets.
The format and existing helper/CLI names are unchanged. File-byte matching needs
`uv sync --extra corrections` / `drone-key[corrections]`; schema parsing and core
model use remain independent of BLAKE3. Files are matched recursively by
Euphoria's full-file BLAKE3 identity,
including tags. The export has no audio paths; preparation manifests contain local
paths, SHA-256 identities and corrected labels. These personal manifests/audio are
local inputs, not bundled package data. Keep only audio and labels you are
entitled to use; this workflow confers no new redistribution rights.

Matching GiantSteps file bytes is rejected before correction preparation. The
hash split assigns roughly one fifth of matching files to corrections-held-out;
both sides must be nonempty. The split is stable for identical file bytes only.
Retagging/re-encoding may move unchanged musical audio between sides; artist,
album, recording, remix and decoded-audio independence are not established.

Fine-tuning checks known starting-checkpoint train/validation IDs and training
SHA-256 witnesses and actual new fitting data against both evaluation sets before
scoring or gradients. Known overlap fails without silently changing membership.
Missing legacy checkpoint witnesses remain unknown and are listed in
`finetune.json`; passing available checks does not prove unknown historical
exposure absent. Genuine inherited provenance is retained in tuned checkpoints,
while new hashes describe only the filtered base and correction-fit audio actually
sampled. Pre/post scores are descriptive; no measured real-world or public-model
improvement is claimed. Public-model training would require a separate, explicit
annotation provenance/rights and evaluation decision; local inputs are not
silently folded into a published model.
See [the workflow](docs/training.md#11-optional-annotation-fine-tuning).
