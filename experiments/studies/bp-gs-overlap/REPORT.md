# Cached BP → GiantSteps audit leaves 539 queries unresolved

## Question and result

Do the original BP recordings overlap GiantSteps 604, and which published or
candidate models were exposed to the flagged sources? The fixed cached audit
completed all 2,334 planned queries. Among 1,486 BP recording IDs, it returned
**zero fixed-policy identity-evidence outcomes, 539 reviews, zero insufficient-
information outcomes and 947 no-detected-overlap outcomes**. The 539 reviews are
36.3% of the scientific population. This does **not** establish that BP or any
model's training corpus is overlap-free.

All 848 controls completed:784/784 expected positives were recovered and 64
synthetic negatives produced no identity or review flags in GS scope. Synthetic
negatives do not estimate a false-positive rate on unrelated music. The old
broader-scope failed admission decision remains unchanged. No audio was decoded,
new fingerprints extracted, key labels scored, thresholds adjusted or data
admitted. Release weights and all data roles remain unchanged.

![Scientific outcomes and review support](figures/audit-outcomes.png)

*Figure 1. Left: outcomes for 1,486 BP recording IDs, including original aliases and
excluded labels. Right: retained review offset/phase alignments, which are
correlated rather than independent observations. Every review alignment's CP
distance exceeds 0.15. CP disagreement is not a new clearance rule.*

## Population, lineage and method

The [prospective protocol](PROTOCOL.md) froze source, population, thresholds,
controls, scopes and budgets before query execution. The
[metadata report](METADATA_REPORT.md) explains the count lineage:
1,486 original BP IDs →1,363 parser-eligible source labels →1,356 corrected
prepared IDs (1,230 fitting and 126 selection). The first 123 exclusions are
104 wrong-token-count, 17 unsupported-mode and two unsupported-tonic annotations.
The remaining seven are exact-container aliases. They all remain in this audit;
no 1,077/1,347 denominator is substituted.

The released checkpoint declares 1,237 fitting IDs and 126 selection IDs. Its
fitting corpus has 1,232 distinct container hashes, with two also in selection.
Six actual BP-control/joint checkpoints and their run/source/feature witnesses
were verified separately. This distinguishes published release exposure from
current corrected candidate exposure. The generic release audio-hash field
covers the prepared corpus; it is not an independently recovered gradient log.

The immutable 7,579-reference index contains 5,489 FMA, 1,486 BP and 604 GS recording
IDs. The GS targets represent 602 distinct normalized PCM hashes. The scientific
queries target only GS 604; transformed controls use their separately declared
known source scopes. All reference fingerprints, index bytes and the complete
SQL reference dictionary verified before matching. The original extraction
engine and corrected all-offset voting code are byte-identical copies from
[the archived matcher implementation](https://github.com/lvndry/drone/tree/c3177bf); a typed SQL adapter filters target postings before voting.

Every qualifying 3-anchor/2-second offset/bin/phase is retained. Identity evidence
requires 20 distinct anchors spanning 8 seconds plus CP distance≤0.15 at that
landmark offset. Review requires 10 anchors/5 seconds; absent or disagreeing CP
remains review. Both four-frame-bin phases survive; there is no best-offset,
top-k or weak-match pruning. Anchor span measures the first-to-last anchor
extent, not continuous shared audio.

BP queries reuse cached **reference** fingerprints with one extraction shift,
rather than newly extracted four-shift query fingerprints. The passing
transformed controls exercise the latter path; self controls exercise actual
GS target indexing. Neither proves recall for arbitrary transformed GS matches
under the cached BP reference-query path. A no-detected result remains conditional
on this particular representation, scope and fixed policy.

## Outcomes and model exposure

| Scientific outcome | BP IDs | Share |
|---|---:|---:|
| Fixed-policy identity evidence | 0 | 0.0% |
| Unresolved review | 539 | 36.3% |
| Insufficient information | 0 | 0.0% |
| No detected overlap under this recipe | 947 | 63.7% |

The run retained 2,657,001 scientific alignment records; including controls, the
independent checker verified 4,315,263. Review support involves 3,173 distinct
BP/GS recording-ID pairs. These counts do not measure independent matches,
precision, score inflation or new model accuracy.

The following counts concern BP source IDs. Joint-model FMA fitting sources were
outside this scientific query population.

| Actual model | Direct fitting IDs with review | Direct selection IDs with review | Additional non-direct fitting source alias |
|---|---:|---:|---:|
| Published release | 459 | 43 | 0 |
| BP control 13 | 458 | 43 | 1 |
| BP control 37 | 458 | 43 | 1 |
| BP control 73 | 458 | 43 | 1 |
| Joint 13 | 458 | 43 | 1 |
| Joint 37 | 458 | 43 | 1 |
| Joint 73 | 458 | 43 | 1 |

Definite and insufficient-information exposures are zero for all seven models.
The additional candidate fitting source is `beatport:299977`, whose exact
container and normalized PCM both match a declared fitting source. Those are
two witnesses for the **same** source, not two extra exposed recordings.
Normalized PCM equality alone would not guarantee canonical model-input equality.
The direct fitting and selection categories describe metadata exposure, not a
causal estimate of training influence. Full source-ID lists and actual checkpoint
hashes appear in [the diagnostic](data/diagnostic.json.gz); root query outcomes are
preserved in [the original result](data/execution/result.json.gz).

## Why the audit abstains so often

All 162,928 review alignments have CP disagreement under the frozen 0.15 cutoff:
139,756 fail at least one identity-landmark requirement, while 23,172 meet both
20 anchors and 8 seconds but still lack CP corroboration. All 539 review queries
have the former category; 195 also have the latter. Query categories overlap.
No review alignment has missing or corroborating CP in this run.

This is consistent with numerous landmark coincidences that the corroborating
fingerprint does not confirm. It does not establish that these pairs are
unrelated or justify converting review flags into negatives. Shared musical
material, sparsely distributed landmarks, extraction alignment and fingerprint
sensitivity remain possible explanations. No audio inspection or truth-labelled
unrelated-music panel was performed.

The examples below are the first three distinct query IDs in lexical order per
support category, using their first retained alignment. They illustrate the
recorded support; they are not a representative sample or a new selection rule.

| Support category | BP source → GS target | Anchors | Span, seconds | CP distance |
|---|---|---:|---:|---:|
| Review landmark support only | 1017875 →3443052 | 10 | 99.219 | 0.448 |
| Review landmark support only | 1019315 →1026478 | 10 | 8.638 | 0.401 |
| Review landmark support only | 1051007 →1092676 | 11 | 22.500 | 0.453 |
| Identity landmark support, CP disagrees | 1019315 →1678455 | 21 | 71.517 | 0.351 |
| Identity landmark support, CP disagrees | 1056047 →4824580 | 21 | 49.203 | 0.464 |
| Identity landmark support, CP disagrees | 1115626 →1353830 | 20 | 39.335 | 0.417 |

The [diagnostic examples](data/diagnostic.json.gz) carry exact query-evidence hashes,
reference IDs, offset bins and phases. Their five original source query files are
retained under [execution/examples](data/execution/examples/); full raw evidence
retains every weaker offset as well. Nothing was reclassified after inspection.

## Validation, resources and provenance

The authoritative wrapper exited 0 on 2026-11-12 at 09:36:31 UTC, after starting at
09:17:33 UTC. Pipeline wall time was 1,137.377 seconds, CPU time 1,156.810 seconds,
and process peak RSS 276,832 KiB (270.3 MiB). These are cached fingerprint-audit
costs, not model inference or canonical audio-frontend benchmarks. All fixed
60-minute wall/two CPU-hour/two-GiB RSS/512-MiB output caps passed. The audit ran once on CPUs 4–5 with native threads limited to one and used no
GPU. CPUs 4–5 were released after terminal status.
The actual index SHA was unchanged after execution.

Root independently validated all 2,334 original evidence files, per-query gzip
and decoded hashes, all 4,315,263 alignment decisions, control/population scopes,
and a fresh actual index SHA. The original checker executed once and exited 0.
Its byte-exact source is preserved as
[the original checker](data/execution/original-root-checker.py.txt).
The [original receipt](data/execution/original-root-check-receipt.json) preserves
the command, executor, exit status and exact stdout/hash. It explicitly identifies
itself as a copy of the observed tool result. This validation is not attributed
to the later typed report helper, and the original checker was not reformatted
or rerun for this report.

Full metadata collector receipts/source bytes and original index/feature
witnesses remain adjacent in this study. The
[execution index](data/execution/artifact-index.json) binds original logs,
wrapper, staging readback, result, checker and example bytes.

All 2,334 raw query files remain in the task-owned server directory recorded in
the [execution index](data/execution/artifact-index.json).
The complete 64,615,187-byte packed query archive is retained externally;
it is not included as a 64.6-MB Git blob. The index identifies its retained local
path, and the original result retains each raw/decoded query hash. Reproducing
the complete diagnostic requires that retained archive; the repository includes
the compact result, diagnostic and illustrative original files. No audio,
checkpoints or landmark arrays are committed with the execution report.

## Decision

The fixed audit is complete and validated. It supplies no fixed-policy identity
evidence, leaves 539 queries unresolved, and does not clear benchmark recording
independence. Preserve all flags and weak offsets. Any audio review, improved
negative calibration, new query extraction or benchmark re-evaluation requires
its own approved protocol; none is silently added here. The data/teacher program
can use the exposure ledger while keeping these limitations explicit and
maintaining earlier failed model gates.

The original result and compact diagnostic are deterministically gzip-compressed;
encoded and decoded hashes are retained in their artifact indexes. The diagnostic's
original generator is preserved byte-for-byte as
[original-report-helper.py.txt](data/execution/original-report-helper.py.txt).
The current typed report helper also reads gzip result inputs. Compression did
not rerun matching or the original independent validation.

Publication provenance: the exact measured FAA source/archive remains private.
Its metadata included transient HTTP `Set-Cookie` fields; the published download
provenance removes only those fields. The
[sanitization witness](data/official-provenance-sanitization.json) records original
and derivative hashes plus equality of all remaining semantic content. The
sanitized publication snapshot is not reattributed as the executed source.
Original result, checker/receipt and matching-source identities remain unchanged.
