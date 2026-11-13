# Prospective Drone versus Mixed In Key comparison

**Status: preparation protocol, November 13, 2026. No panel is admitted and no
comparison has run.** The objective is a public, reusable Drone model that is
more accurate than Mixed In Key on independent complete recordings and practical
on ordinary CPUs. Mixed In Key is the named primary competitor; other detectors
are contextual comparisons, not substitutes for beating it.

This companion protocol adds a future comparison to the existing data study. It
does not edit the frozen FMA assignment, its protocol amendment chain, historical
scores, or earlier promotion decisions. A completed execution packet must supply
the unresolved fields below before final-panel predictions are generated.

## Available access and the immediate preparation step

A read-only application-directory inventory found no Mixed In Key application in
`/Applications` or the user's `~/Applications`. This is a scoped observation, not
proof of no installation elsewhere. The user separately confirmed that they have
no Mixed In Key license. Rekordbox is present, but its installation does not
establish Mixed In Key access or activation. No application
was opened, license store inspected, music scanned, or preference changed. The
[access inventory](data/mik-comparison-access.json) retains the observation scope
and installed comparator metadata.

Mixed In Key's current official documentation describes version 11 and a
supported **Initial Key tag** output. It does not establish a supported CLI,
automation API or CSV export, so none is assumed. Once a licensed executable is
available, qualify analysis and output capture on disposable synthetic fixtures
outside the final panel. Confirm all 24 key-label mappings, notation, missing and
multi-key output behavior, fresh-analysis behavior, supported containers and tag
roundtrip. Synthetic fixtures test transport and mapping, not musical accuracy.

Use owned copies in a dedicated benchmark directory and conventional tonic/mode
labels for scoring. Disable title/artist/comment overwrites, cue export and all
DJ-library integration. If the supported application requires tag writing, allow
only the agreed result field on disposable copies. Keep source recordings
immutable. Do not use Euphoria as an intermediary, modify the user's library,
read a proprietary application database, or infer undocumented output fields.

Official workflow references, checked November 13:

- [Adding Key and BPM tags](https://mixedinkey.com/workflows/add-key-and-bpm-tags/)
  describes Tag Options and the Initial Key field.
- [Missing-tag diagnostics](https://mixedinkey.com/workflows/fix-missing-key-or-bpm-tags/)
  distinguishes completed analysis, file identity, tag writing and library refresh.
- [Rekordbox cue export](https://mixedinkey.com/workflows/export-cue-points-to-rekordbox/)
  is a separate integration workflow; it is not evidence of a general key-results
  CSV or a requirement to alter a DJ collection.

## Establish independent recordings and references

Create two disjoint collections: a development collection for candidate/recipe
selection and an untouched final collection for the named claim. Assign whole
recording/work/artist components before model comparisons; shared versions,
remixes and alternate performances cannot cross roles merely because their file
hashes differ. Preserve uncertain relationships and an exclusion ledger.

The current BP/FMA development data, exposed GS604 evaluations and 93-track
GS/SWD comparison are not a fresh full-length DJ panel. Existing role history is
not erased by renaming a split. The user's private DJ crate remains restricted
to performance testing and manual review; it supplies no fitting labels, final
accuracy references or model-selection cases under this protocol.

For new audio, record the lawful acquisition source and use terms, exact edition
and mix, original container identity, decoded-audio identity and fingerprint
comparisons against every known fitting/selection/previous-test corpus. Include
known teacher pretraining exposure; undocumented commercial/pretraining exposure
remains unknown. Do not call the panel independent of Mixed In Key's undisclosed
training data. Keep audio private unless separate redistribution rights exist.

Community catalogs are acquisition leads, not admitted truth. In particular,
[KeyBPM's consensus pipeline](https://github.com/bascurtiz/keybpm/blob/main/docs/data-pipeline.md)
includes FMAKv2, HookTheory and KeyFinder among its sources and documents
section/arrangement matching issues. Three agreeing sites are not necessarily
three independent annotators. MIK pseudo-labels may be considered only in a
separate development/training experiment with exposure disclosed; they cannot
serve as the final reference for a claim that Drone beats MIK.

Use at least two independent musically qualified annotators, blind to model
predictions, existing key tags, and the identity of the evaluated systems. Give
them the same full audio edition, neutral IDs and written annotation guidance.
Retain their original independent answers and confidence before adjudication.
An adjudicator resolves disagreements while still blind to system outputs.
Report raw agreement, the confusion structure, adjudication rate and unresolved
cases; do not assume a universal human-agreement ceiling.

Record global tonic/mode, tuning observations, plausible alternatives and key
changes with approximate section boundaries. A stable global-key reference needs
agreement on the predominant tonal center over the complete supplied recording.
For a modulating recording, a global reference is admitted only if the blinded
annotators can agree on a predominant key under that same rule; otherwise it is
retained as an ambiguous/local-key case, not forced to the opening key or the
first entry in a multi-label string. No post-prediction reference repair occurs.

Freeze the 24-class global-key eligibility decision before model outputs exist.
Report all acquired records, excluded/unresolved references and reasons beside
the scored denominator. Ambiguous, modal, percussion-only, severely detuned and
no-predominant-key recordings remain visible as separate coverage categories.
A secondary local-key annotation set can support later research; this comparison
does not claim that a global classifier detects modulation.

## Freeze coverage and the claim population

The target claim is about complete DJ recordings, with explicit priority coverage
for hip-hop/rap, contemporary Afrobeats, house (including deep and melodic house),
techno and other club styles. Keep Afrobeat, Afrobeats and Afro house distinct.
Preserve native tags; fix any mutually exclusive primary reporting groups and
overlapping secondary tags before scoring. Include an explicit unknown-genre
category. A separately reported non-DJ collection checks broader generalization;
it cannot silently change the weighting of the primary DJ comparison.

Record actual decoded duration and supplied/analyzed time support, not just store
track length. Show `<2 min`, `2–<5 min`, `5–10 min` (including 600 seconds), and
`>10 min` complete-recording groups, with previews in a separate panel. Disclose
native genre by duration intersections, work/artist-component counts, codec,
sample rate, channel count, vocal/instrumental annotation when available, and
BPM provenance plus missingness. Analyzer-derived BPM is metadata, not tempo
truth. Never infer tempo or contemporary genre from a filename or ancestor tag.

The final execution packet fixes the acquisition frame, inclusion order, group
assignment, genre/length quotas, aggregate weights and sample-size calculation.
Choose them for product coverage and power before candidate outcomes are known;
50–80 tracks per cell is not automatically adequate for a superiority claim.
Estimate disagreement/cluster structure using a disjoint development pilot,
predeclare the smallest practically worthwhile gain, and calculate the required
number of independent components. If that number cannot be acquired, report an
underpowered descriptive comparison rather than relax the rule after scoring.

## Lock systems and run on identical audio

Freeze one deployable Drone artifact, preprocessing and aggregation recipe before
opening final references or outputs. If three training seeds are used for recipe
confirmation, report all three on development data and choose the deployment seed
by a rule fixed there; do not choose a final-panel winner. A seed average is not
the accuracy of a shipped single artifact. A new ensemble or last-epoch averaging
rule is a new recipe requiring development evaluation.

Record the installed MIK product/edition, exact build, executable identity, OS,
all analysis and output settings, analysis date and any observed network/cache
requirements. An advertised version name alone does not bind the evaluator.
Preserve settings and supported raw output evidence, including failures. No
update, settings search, manual result correction or alternate MIK version is
selected using final scores. A future version is a new comparison.

Supply both systems byte-identical, full-length benchmark input editions. Make
neutral copies without existing key labels and provide a separate copy to each
system; retain the reversible source-to-benchmark identity mapping. Lock any
container conversion once, verify decoded PCM equivalence, and report that input
boundary. After tag writing, verify audio payload/decoded identity remains intact.
Record actual analysis support when observable; internal MIK cropping is unknown
unless documented or measured. Full input duration does not prove internal
whole-track use. Log terminal status for every scheduled ID in a fixed order.

## Metrics and decision rule

Primary: **24-class exact tonic-and-mode accuracy**, at one recording per frozen
metric unit, for the named Drone artifact versus the pinned MIK build. Normalize
enharmonic spellings using [Drone's shared key contract](../../../drone/keys.py);
class order is C-first, `2 * tonic + minor`. A missing, invalid or failed system
prediction scores zero and remains in the denominator. Do not drop hard cases
because a system times out or declines. Preserve the original output string and
parser status separately from the normalized class.

Secondary: per-recording weighted MIREX with the existing weights (exact 1,
fifth 0.5, relative 0.3, parallel 0.2). Publish **both** symmetric-fifths and
ascending-fifth-only variants with those names and identical inputs. Exact and
weighted scores answer different questions; MIREX is not exact accuracy. Also
show the complete relation/error counts. Tonic-only 12-class accuracy is a
secondary diagnostic, not a substitute for mode correctness or mixability.

The primary comparison uses paired outcomes on the same recordings. Report the
exact paired win/loss table, absolute accuracy difference and a two-sided 95%
confidence interval from a whole-component bootstrap, preserving each component
and the predeclared population weighting. Use 10,000 replicates with analysis
seed 20261113. Do not resample individual tracks as independent when artist/work
components repeat, or independently resample two slices sharing a component.
Weighted MIREX differences use the same paired resamples. Exact-binomial
McNemar may additionally be reported where independent paired recording units
are defensible; it is not a test of weighted MIREX and does not cure clustering.

A superiority claim requires the primary interval's lower bound above zero,
plus the predeclared practical-gain requirement in the execution packet. Fix
whether that requirement applies to the point estimate or interval before
outcomes, and justify its power calculation. Retain secondary-metric and
priority-slice regressions visibly; a broad claim cannot conceal a failed
prespecified noninferiority safeguard. Any such safeguard and multiplicity rule
must be in the same packet. Do not replace these rules by “gain exceeds twice
seed SD,” use post-hoc one-sided tests, select a comparator from final scores,
or infer equality from a nonsignificant difference.

One completed final comparison is a reportable result. Using its weak slices to
choose training data, distillation or architecture converts that panel into
development evidence for subsequent models; a later claim needs a new final
panel. Additional detectors are descriptive unless separate hypotheses and
multiplicity handling were preregistered.

## CPU and long-track measurement

Measure Drone on named ordinary-PC hardware, with CPU model, RAM, OS, power mode,
thread settings and device/provider recorded. Use complete recordings including
5–10-minute and longer tracks; a preview result cannot stand in for them. Separate
decode/resample, feature extraction, model inference and end-to-end elapsed time.
Report cold-start separately from warm-session latency, CPU time, sampled process
RSS with sampling interval, thread count and failure counts. Full-matrix versus
chunked runs must share the same audio and prediction-parity check.

MIK's supported GUI can support a timed fresh-analysis batch when available.
Record the observed start/end events and include its actual application work;
do not pretend a batch wall time is a per-track kernel time or divide overlapping
work into fabricated stage timings. Record internal stages as unavailable where
not exposed. Accuracy comparison remains possible even if only end-to-end MIK
timing is observable. The CPU release budget applies to Drone independently of
MIK's implementation. No app library integration is required for this work.

## Execution dependencies still open

| Required item | Current state | Completion evidence |
|---|---|---|
| Licensed MIK executable and output capture | User confirms no license; no app found in scoped inventory | Exact installed build, usable activation and disposable-fixture output qualification |
| Independent full-length audio | No final-panel manifest exists | Acquired editions, rights record, full support, overlap/group audit and frozen membership |
| Blinded annotators/adjudicator | Not assigned; budget unknown | Named roles privately recorded, guidance, independent answers and sealed references |
| Sample size and practical gain | Not yet estimable from a valid independent pilot | Frozen power assumptions, component count, quotas, weighting and decision packet |
| Candidate and comparator versions | No candidate frozen for this new claim | Artifact/settings records and output parser qualification |
| Ordinary-PC timing host and bounds | Hardware/run packet not selected here | Measured stage pilot, declared duration/thread/memory limits and run configuration |

The comparison and any claim of beating MIK are blocked by comparator access.
Model research is not blocked. No purchase, trial activation or upload is
authorized by this preparation document.

Preparation can continue while access is resolved: source/coverage inventory,
annotation instructions, output schema and stage-cost design do not require
opening final references. This document is not evidence of a completed freeze,
a new accuracy result, or an executable training allocation.
