# GTZAN overlap-query preparation — archived prospective amendment

This text was frozen before overlap execution on the research branch. It is
published here separately to preserve the existing base protocol and its original
witness. The completed report does not retrospectively change these rules.


`python -m tools.prepare_gtzan_overlap` reproduces the existing metadata eligibility
and terminal diagnostic inventory, then writes
`data/gtzan-overlap-manifest.json.gz` with gzip timestamp zero and no filename.
`--check` verifies exact compressed bytes; decompression preserves the existing JSON
serializer bytes. The CLI reports both compressed and decoded SHA256 witnesses.
This is metadata preparation only; it performs no acquisition, audio extraction,
fingerprinting, matching, classification or scoring, and grants no resource permit.

Freeze the 769 named filtered recordings, including all 73 eligible hiphop IDs,
as the primary cohort. Also freeze the complete 837 raw single-key references as
a separately reported sensitivity cohort. Query rows expose only stable IDs,
archive members, verified WAV MD5/SHA256, bytes, frame counts, sample rates and
durations. Reference keys and annotation integers are absent. Their membership
comes from `tools.import_gtzan_reference.produce`, without changing its rules.
The archive URL, compressed size and SHA256 remain the existing integrity pins;
the terminal inventory remains a diagnostic inventory, not a strict audit pass.

Anchor released-model exposure to the preserved `drone-v1` checkpoint metadata
(SHA256 `dcf8a7a292b7c12076ffff63c8c6550cb09a6c41ce4d000fd7949fc6bb4b452c`),
including its 1,237 declared fitting IDs and 126 selection IDs. Join those IDs
and their exact container/decoded-PCM aliases against the preserved full reference
metadata: 1,486 BP, 5,489 FMA and 604 GiantSteps recordings. Preserve both roles
when a known alias carries both. Current candidate fitting manifests do not
replace released-model exposure. These are known exact aliases under the existing
cached decode contract, not exhaustive recording, work or artist clearance.
The old BP-to-GiantSteps review flags do not clear any GTZAN query.

A later separately authorized pass must bind the frozen query manifest and actual
reference/index artifacts before results, preserve one frozen overlap pass and
bounded blind adjudication, and report model-specific exposure strata (confirmed,
unresolved, no detected overlap). Flags do not change either frozen cohort.
Unresolved exposure limits independence claims; it does not by itself prevent
fixed-benchmark arithmetic after the separate scoring contract is satisfied.
This preparation selects no new overlap thresholds using GTZAN and makes no
fingerprint admission, accuracy or independence claim.
