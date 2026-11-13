# PROPOSED v2 component-first data contract

This is a metadata-only design, NOT current/admitted roles or training authorization.
V1 manifests/weights remain unchanged. No audio bytes, model predictions or seed sweep.

Assignment version `fmak-v2-proposal-20261112-v1`, deterministic seed `20261112`.
Use every saved FMA component over all5489 rows, including bridges through omitted
records. Force the490 components containing old976 fitting rows into expanded
fitting; restricted fitting remains exactly those976 IDs. BP old1230 fitting IDs
remain a shared prospective pool; BP inner roles are NOT assigned here.

Fixed evaluation panel order: calibration, inner-selection, development-confirmation.
Approximate target rows159/250/1000; minimum complete components60/100/200;
maximum single-component size floor(5% of target rows). No exact-size subset-sum.
For each panel select exactly its minimum component count from remaining eligible
components. At step j, choose the candidate minimizing squared normalized distance
of cumulative rows and official duration to j/min_groups of target, plus mean
squared coarse-genre histogram distance to that progressive target. Reference genre
and duration distribution is the whole non-forced component reservoir. Ties resolve
by SHA256(version:seed:component). Never move selected components to repair outcomes;
leftovers enter expanded fitting. If too few bounded components, fail explicitly.

All target labels are excluded from assignment inputs. Class support may be reported
later as a separately hashed diagnostic, never used to choose seed/roles, with no
automatic repair. Missing class metadata is an explicit activation blocker.
Genre missingness is an explicit MISSING category. Duration is publisher metadata,
not decoded duration. Every old role→proposed role is retained with historical
exposure; newly assigned panels are development/confirmation, never pristine final.

Keep exact-audio groups together. Report raw rows and unique previously recorded
SHA values; prospective fitting deduplication prefers old-fitting IDs, then smallest
ID, to preserve nesting. Duplicate/conflicting-label diagnostics and feature content
verification must be reviewed before actual use. No fingerprints are run here.

Report achieved counts, complete components, max share, concentration N²/sum(n²)
(descriptive, not statistical effective N), genre/duration, unknowns and old exposure.
The fixed greedy rule is a reviewable proposal, not guaranteed globally optimal or
representative. A failure is evidence for a separately declared revision, not seed
shopping. Root reviews output before any training/data-role activation.

## GTZAN reference metadata amendment (2026-11-12)

This separate prospective external diagnostic concerns **released BP-only Drone-v1
versus a newly run, pinned libkeyfinder**, not MusicFM teacher accuracy. It changes
no v2 role, fitting population, checkpoint selection or existing evidence. Only
annotation/repository/recording-index metadata may be acquired in this stage.
No waveform, audio archive, fingerprint, detector output or model is opened.

Pin annotation repository v1.0 commit
`9ef1591f81ee2f7821c7ce41e774ea7ac7f94cbe`, tree
`59faf8174287d4a714f0875ee78ffb88e692533b`; pin Sturm fault repository commit
`fedc781026cf1a20034830f8071c98d93a422016`, tree
`35c4f075c4bd3c8a8dfe772fb08acbc6080d80e5`. Preserve the complete small codeload
archives and reconstruct their full Git trees, including unselected files.
Archive hashes and raw Git commit witnesses are retained. This is a codeload
acquisition, not the differently packaged Zenodo annotation release archive;
its deposit MD5 is not claimed to verify our ZIP. Preserve the annotation MIT
notice; that notice does not license the underlying recordings. Record the
mirdata index version 1.0, publisher MD5 and acquired strong SHA-256 separately.
The index supplies filename/member-MD5 witnesses only: no audio was checked.

Import all 1,000 stable genre/index IDs. Read only each `.lerch.txt` integer as
published; accept 0–23, exclude −1, and fail on any other value. Map A-based
major/minor enumeration to Drone's C-based interleaved classes without relabeling.
All classical records remain unannotated. Keep the raw single-key reference
ledger. The **named filtered intersection** is that raw reference intersected
with the union of the author's three filtered lists (not the author's train/test
roles reused as our roles). Prespecify `jazz.00054` as a published loader
corruption flag; exclude it from metadata eligibility, whether or not already
excluded by its annotation. No replacement, model-dependent exclusion, inferred
key, class balancing or optimization of this rule is allowed. Every ID retains
all applicable exclusion reasons. This intersection is not called “MIREX837.”

All metadata rows remain `admitted=false`, `audio_verified=false`, and
`overlap_cleared=false`. Future audio/version/terms and cross-corpus identity
review is separate. Before scoring, freeze exact release and libkeyfinder
versions, shared admitted inputs and decode contract, failure denominators,
exact/symmetric/asymmetric-MIREX outputs and recording/artist uncertainty rules.
Audio checks must consider actual release fitting/selection aliases, not assume
that current research fitting manifests describe the published release. Unknown
identity coverage remains unknown; filtered lists do not prove complete artist
or recording independence. No scoring or audio acquisition is authorized by
these metadata outputs. Existing GTZAN literature exposure remains disclosed.

### GTZAN archive integrity stage — November 13, before any scores

Acquire exactly the pinned Marsyas mirror archive at Hugging Face commit
`d2146561ecc7df707d9e6b8318885fe6a39668a2`: 1,226,192,050 compressed bytes,
SHA256 `b28cc067ff6199bd826f5d1a6931458586d64acae7f44b2e882b7a97af057531`.
Use it only for the already authorized, private research evaluation; public hosting
and annotation licensing do not establish redistribution or commercial clearance.
Do not commit or redistribute audio. Preserve a failed transfer without automatic
retry or switching mirrors.

This stage verifies the archive, all 1,000 per-recording MD5 witnesses from the
pinned mirdata index, and WAV structure. Stream the tar without extracting audio;
reject links, traversal, duplicates, unexpected members, missing recordings and
payloads over fixed limits. Preserve the existing jazz.00054 corruption flag;
it does not authorize excluding other new failures. No classifier or keyfinder
runs, training, feature cache, reference changes or score admission occur.
Matching recordings still need cross-corpus overlap review and a locked scoring
protocol. Thirty-second excerpts do not establish long-track quality.

Server acquisition bounds: 900 wall seconds, 300 aggregate CPU seconds, 512 MiB
whole-tree RSS, two GiB owned outputs; use CPU affinity 10–11, nice 19 and no GPU.
Keep at least 40 GiB host memory and 32 GiB ordinary disk free, in addition to the
remaining archive reservation. Preserve all four original vLLM PID/start identities.
Network reads have a 30-second timeout. Stop only owned processes on a cap or
identity failure. Keep source, configuration, download and audit receipts. A
passing audit proves recording identity and format only, not label correctness,
license clearance, exposure independence or accuracy.

The first acquisition downloaded the exact pinned archive, then its verifier
stopped on bundled AppleDouble/Finder files before checking recordings. Preserve
that failed controller and the verified archive. A packaging correction admits
only exact `._` companions of expected recordings/directories and `.DS_Store`
files under known directories, capped at 16 KiB per file and 16 MiB total. Links,
unknown paths and every recording MD5 requirement remain unchanged. This changes
archive parsing, not reference eligibility or track-failure exclusions.

A separate audit-only pass may reuse those exact retained bytes without another
download: 120 wall seconds, 90 CPU seconds, 256 MiB tree RSS and two MiB new
outputs, CPU 10–11/nice 19/no GPU, with unchanged host/disk/service reserves.
Keep both verifier source versions and their results; no automatic retry.

The corrected strict parser stopped on an MD5 mismatch at jazz.00054, already
flagged corrupt by the source metadata. This remains a failed strict audit.
A separate `--inventory-only` mode may enumerate all recording identities and
format faults without treating mismatches as a pass, changing exclusions, or
admitting tracks for scoring. The same retained archive and audit-only bounds
apply. Report full-universe and frozen eligible-subset counts separately; archive
integrity, recording identity, parseability and scoring admission are distinct.
