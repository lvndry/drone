# Key detection references on Winterreise and GiantSteps

The initial comparison mixed a valid Drone/madmom run with an invalid
libkeyfinder wrapper. Its **37.6% exact / 47.6% weighted libkeyfinder result is
withdrawn as a baseline**. The original artifacts remain preserved in
[data/historical/](data/historical/); they are not silently replaced or repaired
by relabeling predictions. A corrected wrapper rerun is pending publication.

The question is whether the released compact detector agrees with established
references on two specific populations. This is a descriptive reference comparison,
not a state-of-the-art ranking, deployment gate or architecture decision. MusicFM
was not run, so this is not a completed four-model comparison.

## Problem and comparison

Drone targets ordinary-CPU inference. Before drawing conclusions about its
coverage, compare it with madmom's pretrained KEY_CNN and libkeyfinder's native
DSP algorithm using the same decoded recording and the same reference convention.
Different corpora answer different questions; their combined average is not a
population estimate of general music accuracy.

The fixed panel contains **93 recordings**:

| Corpus | Recordings | Coverage | Exposure and interpretation |
|---|---:|---|---|
| Schubert Winterreise Dataset (SWD) v2.1 | 48 | One 24-song cycle, performances HU33 and SC06 | Narrow vocal/piano classical repertoire; paired performances are not 48 independent works |
| GiantSteps | 45 | Deterministic round-robin sample across the harness's 12 genre buckets | Development-exposed for Drone; descriptive, not independent evaluation |

The SWD archive's 48 available labeled recordings have a median duration of
**169.12 seconds**, ranging from **44.07 to 325.33 seconds**. Twenty-six are
under three minutes, twenty are three to five minutes, and two are five to eight
minutes. All 45 GiantSteps items are under three minutes. Across the complete
panel the duration counts are therefore 71 / 20 / 2 / 0 for under three,
three–five, five–eight and eight-plus minutes. This panel does not establish
accuracy on five–ten-minute DJ tracks.

The released `drone-v1` checkpoint declares **1,237 fitting IDs and 126 selection
IDs** from Beatport. These are separate exposures, not 1,363 fitting recordings.
GiantSteps has already been used for development reporting; a high score on this
sample should not be interpreted as an independent generalization result.

## Method and evidence

One canonical mono 44.1-kHz PCM16 WAV is supplied to each active detector, without
loudness normalization or trimming. Drone subsequently uses its native 22.05-kHz
frontend. Detector-native preprocessing remains part of the comparison. Decode
and model execution costs must be reported separately; inference latency does
not include the initial shared decode.

References use Drone's interleaved 24-class encoding. The existing
`drone.keys.score_mirex` and `score_breakdown` define exact agreement and weighted
agreement: exact 1.0, symmetric same-mode fifths 0.5, relative mode 0.3,
parallel mode 0.2, and other 0.0. This symmetric-fifths convention is stated
explicitly; results under different published scoring conventions need not match.
Ambiguous GiantSteps reference strings retain their first listed key under the
existing parser, not an outcome-dependent choice.

The repaired harness and metadata reproducer share `aggregate_rows`. Every
scored corpus, genre and duration slice keeps its complete input denominator;
errors and explicit abstentions contribute zero agreement. Valid prediction,
error and abstention counts are reported separately. A failed detector is not
allowed to improve its score by removing difficult recordings.

The original 93-row CSV, summary and manifest are archived byte-for-byte in
deterministic gzip files with decoded/compressed hashes. The active artifact
index binds the published CSV, summary and manifest; the metadata analysis
replays their joins and scores without audio or detector execution. The figure
receipt binds its analysis, plotting source, library version and rendered files.
These hashes establish evidence identity, not retroactive execution provenance
that the initial run did not record.

## Results and interpretation

The initial error-free Drone and madmom rows remain available for descriptive
comparison: Drone recorded **63/93 exact (67.7%) and 75.7% weighted agreement**;
madmom recorded **72/93 exact (77.4%) and 82.9% weighted agreement**. The combined
numbers mix 48 classical performances and 45 development-exposed electronic
tracks. Interpret the corpus breakdown first.

![Exact and weighted agreement by corpus](figures/corpus-comparison.png)

*Figure 1. Exact and symmetric-fifths weighted agreement on each fixed corpus.
Numbers above bars are percentages of all recordings in that corpus. The initial
libkeyfinder wrapper result is excluded; a corrected result appears only after
its new recording-level evidence is published. No confidence interval or
independent SOTA ranking is implied.*

Madmom's recorded advantage on Winterreise does not establish superiority on
classical music generally. Conversely, the GiantSteps sample cannot establish
Drone's independent EDM performance. Genre buckets contain only three to five
GiantSteps recordings each. Report their counts alongside descriptive values;
they do not justify claims such as “strong on house” or “weak on trance” for
those genres as populations.

![Weighted agreement by genre, with recording counts](figures/genre-comparison.png)

*Figure 2. Descriptive weighted agreement by the existing genre buckets. Every
row shows its recording denominator. The nonclassical rows contain only three
to five development-exposed GiantSteps recordings, so visible gaps are not
population-level genre conclusions.*

Duration and corpus are confounded: every three–five-minute recording is SWD,
and the five–eight-minute bucket contains just two SWD recordings. This is not
evidence that duration caused any accuracy difference. Both SWD performances
also repeat the same underlying works, so recording-level uncertainty must not
be presented as independent-work uncertainty.

### Invalid historical libkeyfinder result

The original wrapper result was 35/93 exact and 47.6% weighted agreement. It is
retained as an implementation-failure record, excluded from valid-detector
plots, and must not support DSP-versus-neural conclusions. Corrected wrapper,
binary/library identity, self-test and fresh per-track results must accompany
any replacement baseline. This defect does not invalidate the separate older
69.35% / 60.60% baseline produced through Euphoria's Rust path; its detector path
and evaluation population differ.

### MusicFM scope

The initial run recorded MusicFM+MLP as blocked because the published submission's
required fine-tuned checkpoint was unavailable. Its reported literature numbers
are not measurements on this panel. The repository's separate MusicFM teacher
experiment is a different trained artifact and task head; it is not substituted
for that submission. Availability can change, so a future run must record the
actual source revision and acquired checkpoint identity rather than reuse a
hard-coded historical availability statement.

## Reproduction and remaining limits

Metadata and figure reproduction, without inference:

```bash
uv run python -m tools.report_key_references
uv run python -m tools.report_key_references --plot
```

The first command verifies pinned active and historical evidence and recomputes
the stored metadata analysis. `--plot` regenerates the inline figure. A separately
recorded execution can publish new CSV/summary/manifest evidence and reseal their
index using `--seal-index corrected --write --plot`; resealing alone does not
prove that a detector was corrected or rerun.

For a new detector run, follow `tools.benchmark_sota_4way --help` with the actual
SWD/GiantSteps paths, explicit checkpoint and corrected CLI paths, and a fresh
work directory. Preserve source/dependency/model/CLI and input identities, valid
counts and failures with the run. A missing optional detector is reported as
unavailable rather than silently replaced by another algorithm.

No weights, architecture, production model, dataset eligibility or release gate
changes as a result of this study. Broader genre coverage, long recordings,
independent exposure accounting and ordinary-CPU resource measurement remain
separate work.
