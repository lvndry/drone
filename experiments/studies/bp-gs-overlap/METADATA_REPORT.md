# BP source and checkpoint lineage, before overlap queries

This report contains metadata findings only. No real landmark query, audio
read/decode/hash, new feature extraction or GS key-label scoring has occurred.
The [protocol](PROTOCOL.md) declares the separate prospective overlap audit.

| Stage | Verified count | Evidence |
|---|---:|---|
| Official spreadsheet IDs / indexed original BP recordings | 1,486 / 1,486 | Spreadsheet column A only; frozen reference descriptor |
| Source labels accepted by archived parser | 1,363 | Source-bound text metadata collector |
| Source labels excluded by parser | 123 | 104 wrong token count;17 unsupported mode;2 unsupported tonic |
| Released checkpoint fitting / selection IDs | 1,237 /126 | Actual published PT metadata, CPU weights-only read |
| Corrected fitting / selection IDs | 1,230 /126 | Frozen prepared manifest |
| Corrected union | 1,356 | Seven removed exact-container aliases, retained in audit |
| Indexed BP container / normalized PCM identities | 1,478 /1,477 | Previously recorded hashes only |
| GS targets / normalized PCM identities | 604 /602 | Previously recorded descriptor; not604 independent signals |

The seven removed release fitting IDs are 298532→298431,298535→298429,
299976→297696,299977→297697,4998705→4334231,5057312→5124850 and
5296203→5038642. Five are duplicate aliases within fitting; two share audio with
selection IDs. The published release therefore declares 1,232 distinct fitting
container hashes, with two also present in its selection corpus. This is
historical exposure evidence; no checkpoint or old gate was changed.

The 123 source exclusions explain the first count reduction. Parser branch
categories are not a decision to repair labels or a claim that every exclusion
was intentional. Neither a smaller denominator nor matching titles establishes
BP/GS overlap. Existing numeric IDs, container hashes and cached PCM hashes have
zero BP/GS intersections; excerpt/transcode overlap still requires the prospective
fingerprint audit. All 1,486 raw BP records will be queried, including excluded
labels and aliases, with exposure attributed separately.

Six actual archived best.pt hashes were verified against their run.json files;
source/arm/seed witnesses agree. Each BP-only control fits 1,230 BP IDs; each joint
model fits 2,206 IDs including the same 1,230 BP records. All use 376 selection IDs,
including 126 BP. The metadata plan retains each actual model's fitting and
selection identities plus exact-container/normalized-PCM alias witnesses. The
release's generic `training_audio_sha256` field covers the prepared corpus; it
must not be interpreted as an independent per-gradient exposure list.

Collector v1 and v2 outputs/provenance are preserved separately. Their provenance
JSON embeds exact executable source text, command, exit status and stdout/stderr
hashes; v2 adds parser-branch reasons without changing eligibility. The original
label ZIP's official provenance is not inferred from its filename. The official
spreadsheet ID inventory is separately bound to its downloaded file SHA.

## Additional proposed-FMA-role PCM check

[The reproducible metadata join](data/pcm-role-check.json) covers all 5,489 FMA
records and 5,413 cached PCM identities. It finds **zero cross-role PCM families**
and zero current/release BP-fitting matches into proposed FMA evaluation roles.
No roles, components or eligibility were changed.

| Proposed role | Rows | Distinct normalized PCM | Duplicate families | Excess rows | Families bridging saved components |
|---|---:|---:|---:|---:|---:|
| Expanded fitting | 4,082 | 4,012 |70 |70 |0 |
| Development confirmation |998 |992 |5 |6 |0 |
| Inner selection |250 |250 |0 |0 |0 |
| Calibration |159 |159 |0 |0 |0 |

These are normalized-signal families, distinct from existing container-SHA
training deduplication (4,036 fitting and 997 confirmation container identities).
Every repeated PCM family remains within one saved artist/album/container
component. Preserve these witnesses for clustered uncertainty; do not silently
change weighting or deduplication. The proposal is not activated by this check.

The cached PCM hash is SHA-256 of **mono 11,025-Hz little-endian float32 ffmpeg
stdout**, not the canonical 22,050-Hz model input. It uses the frozen fingerprint
decoder with threads/filter_threads 1, `-ac1 -ar11025 -f f32le`, and rejects output
longer than one hour rather than accepting truncation. Mono/downsample equality
does not prove equality of original stereo or canonical model-input samples.
The report retains exact command/source/input hashes and every duplicate
family's IDs, roles and saved components. No near-duplicate/work clearance is
claimed by zero exact-PCM cross-role matches.

## Proposed next step

The reviewed plan comprises 180 transformed controls, 604 GS self controls,
64 synthetic negatives and 1,486 BP audit queries: 2,334 total. The scientific
audit's targets are GS604 only; the transformed positives use their separately
declared known source scopes. Sixty-minute wall/two CPU-hour/two-GiB RSS/512-MiB
new-output caps are prospective. Source, metadata and plan must freeze before
root authorizes any real query; the local tool's existence is not a launch.
