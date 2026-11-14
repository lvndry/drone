# Key detection references on Winterreise and GiantSteps

On a fixed panel of 93 recordings (48 Schubert Winterreise performances and 45
GiantSteps tracks), madmom's KEY_CNN agrees exactly with the reference key on
**72/93 (77.4%)**, Drone v1 on **63/93 (67.7%)** and a corrected libkeyfinder on
**57/93 (61.3%)**. madmom leads on this panel, mainly on Winterreise. This is a
descriptive comparison on two narrow populations, not a state-of-the-art ranking,
and GiantSteps was used during Drone's development. The first libkeyfinder result
(35/93 exact, 47.6% weighted) came from a broken wrapper and is withdrawn.
Study date: 13 November 2026.

## Question

Does the released compact detector agree with established key detectors on the
same audio and the same reference convention? Drone targets ordinary-CPU
inference; before drawing conclusions about its coverage, we compare it with
madmom's pretrained KEY_CNN and libkeyfinder's DSP algorithm. MusicFM was planned
as a fourth detector but was not run (see Limitations).

## Method

### Panel

| Corpus | Recordings | Coverage | Exposure and interpretation |
|---|---:|---|---|
| Schubert Winterreise Dataset (SWD) v2.1 | 48 | One 24-song cycle, performances HU33 and SC06 | Narrow vocal/piano classical repertoire; paired performances are not 48 independent works |
| GiantSteps | 45 | Deterministic round-robin sample across 12 legacy benchmark groups | Development-exposed for Drone; descriptive, not independent evaluation |

SWD recordings have a median duration of **169.12 seconds** (range **44.07 to
325.33 seconds**): 26 under three minutes, 20 from three to five minutes and two
from five to eight minutes. All 45 GiantSteps items are under three minutes.
Across the panel the counts are 71 / 20 / 2 / 0 for under three, three to five,
five to eight and eight-plus minutes. The panel says nothing about five to ten
minute DJ tracks.

The released `drone-v1` checkpoint declares **1,237 fitting IDs and 126 selection
IDs** from Beatport (separate exposures, not 1,363 fitting recordings).

### Inputs and scoring

Each detector receives the same canonical mono 44.1 kHz PCM16 WAV, without
loudness normalization or trimming; Drone then applies its own 22.05 kHz
frontend. Detector-specific preprocessing is part of the comparison. Decode cost
is reported separately from detector cost.

References use Drone's 24-class encoding. `drone.keys.score_mirex` and
`score_breakdown` define exact agreement and weighted agreement: exact 1.0,
same-mode fifth in either direction 0.5, relative 0.3, parallel 0.2, other 0.0.
Results under other published conventions need not match. Ambiguous GiantSteps
labels keep their first listed key.

Every corpus, genre and duration slice keeps its full input count as the
denominator; errors and abstentions score zero, and valid, error and abstention
counts are reported separately, so a failing detector cannot improve by dropping
hard recordings.

### Correcting libkeyfinder

The original libkeyfinder wrapper was faulty. libkeyfinder was rerun with a
corrected wrapper and default settings, while the original Drone and madmom
predictions were reused. Reuse is valid because all 93 freshly decoded WAVs
matched the original PCM bytes and complete WAV file bytes. The original 93-row
results, summary and manifest are archived unchanged in
[data/historical/](data/historical/).

## Results

All three detectors return 93 valid predictions, with no errors or abstentions.

| Detector | Exact (of 93) | Weighted agreement | Predictions |
|---|---:|---:|---|
| Drone v1 | 63/93 (67.7%) | 75.7% | Original run |
| madmom KEY_CNN | 72/93 (77.4%) | 82.9% | Original run |
| libkeyfinder 2.2.8 | 57/93 (61.3%) | 73.7% | Corrected rerun, default settings |

The totals mix 48 classical performances and 45 development-exposed electronic
tracks, so read the corpus breakdown first.

| Corpus | Drone exact / weighted | madmom weighted | libkeyfinder exact / weighted |
|---|---:|---:|---:|
| Winterreise, 48 | 30/48 (62.5%) / 72.5% | 88.5% | 28/48 (58.3%) / 75.2% |
| GiantSteps, 45 | 33/45 (73.3%) / 79.1% | 76.9% | 29/45 (64.4%) / 72.0% |

![Exact and weighted agreement by corpus](figures/corpus-comparison.png)

*Figure 1. Exact and weighted agreement for each detector on each corpus, as a
percentage of all recordings in that corpus. Look at the Winterreise group, where
madmom's lead is largest. No confidence intervals are shown.*

madmom's advantage on Winterreise does not establish superiority on classical
music generally, and the GiantSteps sample cannot establish Drone's independent
EDM performance. Each legacy benchmark group contains only three to five
GiantSteps recordings, too few for claims like "strong on house".

![Weighted agreement by genre, with recording counts](figures/genre-comparison.png)

*Figure 2. Weighted agreement by legacy benchmark group, with the recording count
on each row. The non-classical rows hold only three to five recordings each, so
visible gaps are not genre-level conclusions.*

### Where Drone succeeds and misses

The slices below were defined **after** the benchmark results were known. They
describe these recordings and are not a pre-registered test. All original genre
labels, all 24 reference keys and the original duration bins are kept, including
empty slices. The [slice table](data/slice-analysis.csv) gives each detector's
input, valid, error, abstention and exact counts and both agreement scores;
[structured results](data/slice-analysis.json) also keep intersections.

By original tag, Drone gets trance 0/4 exact (27.5% weighted) and dj-tools 0/2
(25.0% weighted); hard-dance and pop rock 4/4 each; breaks and drum-and-bass 4/5
each. These are review targets in small samples, not genre-wide estimates.

![Exact and weighted agreement for every original genre label](figures/original-genre-slices.png)

*Figure 3. Every original genre label with its recording count and all three
detectors. Look for rows with very few recordings before reading any gap.
GiantSteps labels are archived site tags; SWD's "classical" is corpus context,
not a per-recording annotation.*

The legacy grouping folds **drum-and-bass into dubstep** and **glitch-hop into
hip-hop**; it is not a modern genre taxonomy and cannot speak to contemporary
hip-hop, Afrobeats or house subgenres. Original labels and mappings are in the
[metadata catalog](../v2-data-contract/data/reference-panel-catalog.csv).

| Metadata / coverage question | Known | Unknown or absent | Interpretation |
|---|---:|---:|---|
| Original genre / corpus context | 93 | 0 | 18 GiantSteps tags plus SWD classical context |
| Analyzed audio duration | 93 | 0 | PCM frames divided by sample rate |
| Verified full-recording duration | 0 | 93 | Preview duration is not full-song duration |
| BPM | 0 | 93 | Not measured; no tempo slice |
| Underlying work identity | 48 | 45 | SWD has 24 works with two performances each; GiantSteps grouping unknown |
| Analyzed audio at least 8 minutes | 0 | Entire slice absent | Accuracy is undefined, not zero |

GiantSteps consists of previews of about two minutes
([dataset documentation](https://github.com/GiantSteps/giantsteps-key-dataset));
the [SWD v2.1 description](https://zenodo.org/records/10839767) identifies the
24-song cycle and its performances.

![Corpus by analyzed duration coverage and exact and weighted agreement](figures/duration-coverage-slices.png)

*Figure 4. Coverage and agreement by duration bin (<3, 3 to 5, 5 to 8 and 8+
minutes, plus an explicit unknown bin). Note that every recording of three
minutes or more is SWD, and the two five-to-eight-minute recordings are two
performances of the same song (Schubert_D911-01).*

Drone's exact / weighted agreement is 70.4% / 77.6% on the 71 recordings under
three minutes, 55.0% / 66.5% on the 20 from three to five minutes, 100% / 100% on
the two from five to eight minutes, and undefined for eight-plus (n=0). These
numbers mix corpus, repertoire and duration and do not isolate a length effect.

| Corpus and reference mode | n | Drone exact | Drone weighted |
|---|---:|---:|---:|
| GiantSteps major | 13 | 9/13 (69.2%) | 73.1% |
| GiantSteps minor | 32 | 24/32 (75.0%) | 81.6% |
| SWD major | 16 | 7/16 (43.8%) | 56.3% |
| SWD minor | 32 | 23/32 (71.9%) | 80.6% |

Across the panel, Drone agrees exactly on 16/29 major references (55.2%) and
47/64 minor references (73.4%); the gap is larger within Winterreise. Mode, key,
instrumentation and work identity are not separated experimentally.

![All 24 reference-key slices with their recording counts](figures/key-mode-slices.png)

*Figure 5. Agreement for each of the 24 reference keys, empty keys shown as
undefined. **Most cells hold only one to a few recordings, and SWD repeats each
work twice**, so a single track moves a cell by tens of points. Read this figure
as a coverage map, not as evidence that Drone is better or worse on any key.*

Reporting by factor and intersection follows
[Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993). No
confidence intervals or significance tests are attached to these
recording-level summaries.

### libkeyfinder correction and cost

The withdrawn wrapper result (35/93 exact, 47.6% weighted) is kept only as a
record of an implementation failure and is excluded from all plots. The corrected
rerun's [execution record](data/corrected-kf/receipt.json) covers all 93 results,
their outputs, and the source and decode identities, and pins the corrected
binary, wrapper source and unchanged libkeyfinder library. Scores are recomputed
from the new predictions rather than remapped from the old ones.

The corrected run took **240.64 seconds wall time and 310.38 CPU seconds**, with
**285,577,216 bytes** peak memory across processes, including fresh decoding and
byte comparison. It is not comparable with the historical detector-only timings
and does not support a speed ranking. This wrapper fault does not affect the
separate 69.35% / 60.60% GiantSteps-604 libkeyfinder baseline, which was produced
through Euphoria's Rust path on a different population.

## Decision

No weights, architecture, production model, dataset roles or release thresholds
change. The comparison shows that a pretrained alternative (madmom) agrees more
often on this panel, chiefly on classical music, and that Drone's weaknesses here
(major keys in Winterreise, trance and dj-tools tags) need a larger independent
panel before any conclusion.

## Limitations

- Two narrow populations: one classical song cycle and 45 development-exposed
  GiantSteps previews. Combined averages are not a population estimate.
- Slices were defined after results were known; many cells hold one to five
  recordings; SWD performances repeat works, so recording-level uncertainty is
  not work-level uncertainty.
- No long DJ tracks, no BPM, no verified full-song durations, no modern genre
  taxonomy.
- Exposure of madmom's and libkeyfinder's development data to these recordings
  has not been assessed.
- MusicFM was not run. The published MusicFM+MLP submission needs a fine-tuned
  checkpoint; a check of the official Hugging Face repository on 13 November 2026
  found no checkpoint files (snapshot last modified 30 October). The
  [API metadata](data/musicfm-availability.json) and
  [check record](data/musicfm-availability-receipt.json) are listed in the
  artifact index. Literature numbers are not measurements on this panel, and the
  repository's own MusicFM teacher is a different model, so it is not
  substituted. A future run must record the checkpoint it actually uses.
- The original run recorded less provenance than the corrected rerun; archive
  checksums establish the identity of the evidence files, not how the first run
  was executed.

## Reproduce

Metadata analysis and figures, without running any detector:

```bash
uv run python -m tools.report_key_references
uv run python -m tools.report_key_references --plot
uv run python -m tools.report_key_references --slices
uv run python -m tools.report_key_references --slices --write --plot
```

The first command verifies the active and historical evidence against
[the artifact index](data/artifact-index.json) and recomputes the stored analysis;
`--plot` regenerates Figure 1. `--slices` rebuilds the metadata fields from source
evidence before joining predictions and checks that the slice outputs reproduce;
with `--write` it rewrites them and their figure inputs. A new detector run can
publish new evidence with `--publish-corrected --seal-index corrected --write
--plot`; for this correction, `--publish-corrected` rebuilt the combined record
from the preserved historical rows and the corrected archive without inference.

For a new detector run, see `tools.benchmark_sota_4way --help` and supply the
SWD and GiantSteps paths, an explicit checkpoint, the corrected CLI paths and a
fresh work directory. Record source, dependency, model and input identities,
valid counts and failures. A missing optional detector is reported as
unavailable, not replaced by another algorithm.
