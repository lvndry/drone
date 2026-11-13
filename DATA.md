# Dataset inventory and provenance

The purpose of this inventory is to establish **where the released model is
strong or weak by genre, analyzed length, BPM and other supported attributes**,
and distinguish an accuracy result from a hole in our data. Bundled-model
provenance remains in [weights/README.md](weights/README.md).

The corpora are not yet one fully unified dataset. The shared
[catalog contract](experiments/corpus_catalog.py) and
[exporter](tools/export_corpus_catalog.py) provide the organization boundary.
The initial [reference-panel catalog](experiments/studies/v2-data-contract/data/reference-panel-catalog.json.gz)
and [CSV](experiments/studies/v2-data-contract/data/reference-panel-catalog.csv) cover **the existing 93-recording reference panel only**: 48 Winterreise
recordings and 45 GiantSteps previews. They do not represent full BP, FMA,
GiantSteps or GTZAN admission. See the
[coverage/provenance study](experiments/studies/v2-data-contract/REPORT.md) for
the broader existing metadata and its limitations.

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
| GS | The 45 panel JAMS files contain `key_mode` and `tag_open`; their artist/release fields are empty. Native genre tags are available. Publisher `sources.xlsx` is another metadata source requiring its own version/hash and join audit. | Most source previews are about two minutes, with known shorter exceptions. Use measured input duration for length slices. | Bounded inspection found no BPM/ tempo fields in those 45 JAMS files. The publisher workbook has not been admitted as a BPM source here. The separate GiantSteps Tempo dataset is not assumed to have identical membership or recording versions. |
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
  The current 93-row pilot has a scalar genre field; broader FMA admission needs
  multi-tag support. Any display bucket is an explicitly versioned derivation.
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
Preserve native tags alongside these historical display buckets. The current
pilot catalog does not justify a full-corpus BPM, artist-independent or long-track
accuracy claim.

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

Concrete next adapters, using existing sources before any acquisition:

1. **BP:** join the complete current ID/genre workbook, admitted store BPM and
   source artists/remixers/release metadata to actual preview support. Publish
   join coverage, conflicting taxonomies and missing values by role. Keep released
   and current research exposures separate.
2. **FMA:** join all 5,489 pinned key-reference IDs to the existing master metadata,
   native genre hierarchy, component/exposure ledger and actual analyzed support.
   Preserve current eligibility exclusions. Admit tempo features only through an
   explicit independent source adapter if such features are already available.
3. **GS/SWD:** expand the 93-row pilot using the exact source cohorts and references;
   preserve GS development exposure and SWD work/performance grouping. Inspect
   existing publisher metadata for additional fields without guessing their meaning.
4. **GTZAN:** join the frozen 837/769 membership to its integrity and overlap
   evidence; retain eligibility and unknown identity status regardless of outcomes.
   Do not turn the artist/fault lists into a claim of complete independence.

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
