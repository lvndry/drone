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
