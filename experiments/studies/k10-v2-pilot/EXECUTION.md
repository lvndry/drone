# Frozen execution amendments and terminal evidence

This record supersedes historical readiness statements in PROTOCOL.md and
BULK_PREPARATION_PROTOCOL.md without changing their frozen source or gate.
The report leads with the completed result; original plans remain historical.

1. First16 source37f17a1, plan7e831158 and batch5fbbc3e completed under its
   preprocessing-only permit. No fit was authorized by that permit.
2. Full preparation source58fe806/config368d4226 used8workers on CPUs4,5,8,9,10–13,
   envthreads1/CUDAmasked:2700wall/5400CPU seconds,16GiB groupRSS,3GiBCQT,
   1GiBWAVscratch,4GiB totalowned. All5641 rows were processed in302.20s wall /
   1905.73CPU seconds; sampledRSS6,001,221,632B, CQT2,461,121,184B.
   It failed readiness on canonical aliases (`complete=false`, exit1), not resource caps.
3. Before any fit, a general source-integrity eligibility overlay preserved all
   selection IDs, removed only fitting-side canonical aliases crossing selection,
   and quarantined every conflicting canonical-target family. No replacement,
   resplit, prediction access, extra extraction or label choice. Sourceb67ad3e
   yields planSHAc3e59ffb764ca5f51444447b0a18a69543228e60a9e3723d925ddd82b0fc8373,
   labelsSHAb8942b5913b8ebacdf2fc9e0ebca72a934aaa7c56d4e1a97d0f60fac84354323,
   decodedruntimefeatureSHA9f94f6a5c6a9e78b4c5d1013cf97b44d6d2fd69665348ebb497e2c817560d6cc.
   Original assignment and failed preparation stay unchanged. Both fitting arms
   share this overlay; nominal BP mass1230/2206 and all numerical training code stay fixed.
4. Actual-duration reporting strata <120s,120–<300s,300–600s,>600s were declared
   before model outcomes. Use canonical sample counts. No gate/selection change.
5. Wholepair permit authorized restricted then expanded under source58fe806 and
   guardfa20244/config4712a6e9. Window2026-11-12T10:47–11:47UTC; eacharm1250s outer/
   1200s internal wall,6000CPU seconds,8GiBRSS,1GiB outputs; GPU0sameUUID,≥8GiBfree,
   4%allocator, protected4vLLMstarts, CPU4,5,10,11/Torch4/env1/TF32off.
   No retry/resume, accuracy-dependent stop or reserved panels.
6. Root /home/lysk-admin/drone-experiments/k10-v2-pair-20261112-v1;
   launcher962288 at10:52:08UTC, restricted962297, expanded1025966;
   terminal11:04:41UTC, exit0/BOTH_FIXED_ARMS_COMPLETE. Both8280updates, gatefalse.
   Confirmation/calibration unscored. Checkpoint hashes are in raw run artifacts;
   checkpoints remain server-side and were not reloaded for report production.

`data/results/archive-index.json` records exact per-file and decodedSHA256, including
compressed raw run/results and historical receipts. New supplementary gzip streams
use mtime0; decoded content is unchanged. `SHA256SUMS` covers the publication files.
`recorded-analysis-source.py.gz` retains the original local one-off analysis source
as evidence (historical absolute paths, not the portable reproduction interface).
The typed `tools/check_k10_v2_report.py` verifies the published prediction joins,
metrics, selector, pair bindings, gate and intervals without external data access.

Frozen source refs (original execution commits, not publication cherry-picks):
`research/k10-v2-numeric-20261112`, `research/k10-v2-eligibility-20261112`,
`research/k10-v2-guard-20261112`. The source commits and per-file hashes remain
authoritative; these names do not describe a new execution.
