# MusicFM fine-tuning did not produce a stronger key teacher

Full fine-tuning of the pretrained MusicFM-MSD model for 8,280 updates selected
a checkpoint that scored 83/126 Beatport and 166/250 FMA exact keys, below both
matched K10 controls (best: 89/126 and 176/250). It failed the success threshold
fixed before the run, so there was no confirmation evaluation, distillation or
release change. A follow-up that refreshed the checkpoint's BatchNorm statistics
also failed. This is a negative result for one fixed teacher recipe, not a
rejection of pretrained music representations in general. Study dates: November
12–13, 2026.

## Question

Can full fine-tuning of MusicFM-MSD produce a key-detection teacher worth
pursuing beyond the compact K10 model? The [protocol](PROTOCOL.md) frames this as
a system comparison: architecture, context length, augmentation, pretraining and
optimizer all differ from K10, so it does not isolate the effect of the
representation. Drone's product remains the small CPU model; a
328,957,080-parameter teacher is a research and training candidate, not something
to run on an ordinary PC.

## Method

### Data

The model was fitted on 1,229 Beatport and 4,033 FMA recordings, and selection
used the same 126 Beatport and 250 FMA recordings as the two K10 controls. No
calibration, confirmation, GiantSteps or private DJ audio was evaluated. Each
optimizer update drew 32 microbatches of size one from per-domain streams with
nominal Beatport share 1230/2206 and a fixed interleave, 264,960 fitting draws
in total. Because each microbatch has size one, BatchNorm sees single examples
rather than a true batch of 32.

### Model and training

Inputs were uniform random 12-second crops from anywhere in each recording,
through the original 24 kHz mono frontend and MSD normalization statistics, with
no pitch augmentation. The final Conformer hidden layer was averaged over time
and fed to a linear 24-key head. All active backbone parameters were trained.

Parameters, frontend, gradients and Adam moments were FP32; the training encoder
and head ran under BF16 autocast, evaluation in FP32, with TF32 off and eager
attention. AdamW used learning rates `1e-5` (backbone) and `1e-4` (head), weight
decay `0.01` and cosine decay over 120 events. Training random seeds were set to
13 after model construction (which itself resets to 142) and strict loading of
the pretrained checkpoint. There was no early stopping, resume, seed search, EMA
or recipe change during the fit.

### Evaluation and selection

Every 69 updates (120 events), all 376 selection songs were evaluated by tiling
each whole recording in disjoint 12-second contexts plus its actual tail, and
averaging logits in FP64 weighted by the samples each context covers. This covers
the whole recording but does not attend across it. The selector maximized
equal-domain exact accuracy, then equal-domain symmetric MIREX score, then took
the earliest tie. All 8,280 updates ran even though the winning checkpoint was
early.

### Success threshold

Fixed before the run: the selected teacher had to beat **each** K10 control by at
least five FMA exact cases (at least 181/250, 72.40%) and show no Beatport
decline in exact, symmetric or asymmetric MIREX against either control.

## Results

### Training and selection

![All 120 events: training loss, Beatport/FMA exact and symmetric MIREX](figures/teacher-learning.png)

Each point is one evaluation event; the loss is the mean over the preceding 69
updates. Look at the gap: training cross-entropy falls from 3.077 at the first
event to 0.477 at the last, while selection accuracy peaks early (dashed line,
update 828) and then declines to 61/126 Beatport and 146/250 FMA exact at the
final update. The divergence is an observation, not proof of any particular
cause. [SVG](figures/teacher-learning.svg).

| Selected system | Best update | BP exact | BP symmetric MIREX | BP asymmetric MIREX | FMA exact | FMA symmetric MIREX | FMA asymmetric MIREX |
|---|---:|---:|---:|---:|---:|---:|---:|
| K10, restricted fitting pool | 3,864 | 86/126 (68.25%) | 76.43% | 72.86% | 172/250 (68.80%) | 74.00% | 73.60% |
| K10, expanded fitting pool | 5,796 | 89/126 (70.63%) | 77.14% | 75.16% | 176/250 (70.40%) | 75.84% | 75.04% |
| MusicFM, expanded fitting pool | 828 | 83/126 (65.87%) | 73.25% | 71.27% | 166/250 (66.40%) | 73.36% | 71.36% |

The K10 rows are the matched joint-training controls from the
[K10 v2 pilot](../k10-v2-pilot/REPORT.md), not the released K10 model. The
expanded K10 control also failed its own improvement threshold and was not
promoted. The reproducer checks every integer numerator and denominator against
the recorded metrics.

![Selected MusicFM exact accuracy against both K10 controls and fixed thresholds](figures/teacher-controls.png)

MusicFM's FMA exact count (166) falls well short of the 181/250 line, and its
Beatport bars sit below both controls on all three measures. Both requirements
fail against both controls. [SVG](figures/teacher-controls.svg).

### Retrospective diagnosis by genre, recording duration and BPM

The aggregate result does not say whether MusicFM is weaker on every kind of
music. We joined its saved predictions, and those of the expanded K10 control,
to the Beatport and FMA metadata catalogs. This uses the same 126 Beatport and
250 FMA recordings that selected both checkpoints across 120 events, so it is a
retrospective look at a heavily reused development set. It adds no inference,
training, independent test or new threshold. The
[analysis file](../v2-data-contract/data/development-slices.json.gz) contains the
predictions and the 376 development reference keys only; confirmation references
were not read. Exact misses and abstentions stay in the denominator (all recorded
predictions were valid).

| Development cohort | Both exact | K10 only exact | MusicFM only exact | Both miss | Groups |
|---|---:|---:|---:|---:|---:|
| Beatport, 126 | 70 | 19 | 13 | 24 | 126 single-recording proxies |
| FMA, 250 | 150 | 26 | 16 | 58 | 100 existing components |

These outcomes partition each cohort and account for MusicFM's net losses of six
Beatport and ten FMA exact cases, without hiding where it helps. "Both miss" can
still earn harmonic partial credit. No confidence intervals or significance tests
are reported for this single-seed, reused development comparison.

#### Genre

![Exact success counts by native Beatport workbook genre and FMA top genre](figures/development-native-genres.png)

Each row gives exact successes / recordings for both systems and the number of
groups; asterisks mark fewer than ten recordings and empty categories show NA.
Rows follow the native label order, not the winner. The analysis file also keeps
API genres and FMA's overlapping hierarchy tags.

MusicFM is not uniformly worse. On Beatport workbook genres it gets **Deep House**
11/12 exact versus K10's 9/12 and **Techno** 8/14 versus 6/14, but **Electro House**
7/11 versus 10/11 and **Breaks** 5/10 versus 8/10. These describe these particular
recordings; they are not reliable genre rankings or a basis for routing between
models.

FMA's top-level Hip-Hop field covers only six records, while the
hierarchy-inclusive Hip-Hop tag covers 16, so the choice of field matters:

| FMA hierarchy-inclusive tag | Records / groups | K10 exact | MusicFM exact |
|---|---:|---:|---:|
| Electronic | 93 / 38 | 57/93 | 56/93 |
| Hip-Hop | 16 / 8 | 11/16 | 10/16 |
| House | 9 / 4 | 7/9 | 7/9 |
| Techno | 6 / 3 | 3/6 | 3/6 |
| Afrobeat | 0 / 0 | NA | NA |

These tags overlap and include ancestors, so the rows do not partition FMA. The
legacy Afrobeat tag is not contemporary Afrobeats. The top-level genre is missing
for 143/250 records and 16/250 have no genre tags; those predictions are kept.

#### Duration

![Paired exact outcomes by actual evaluation duration](figures/development-duration-pairs.png)

Bars split each duration bin into both-correct, K10-only, MusicFM-only and
both-miss; counts are printed even for thin segments, and empty bins stay
visible. The Beatport 2–<5 minute bin is almost entirely two-minute previews.

K10 evaluated true-length CQT features derived from 22,050 Hz audio; MusicFM
covered each 24 kHz recording with 12-second contexts plus tails. Both
model-specific durations fall into the same bin for every recording. A 12-second
context does not turn a five-minute recording into a 12-second sample.

In the FMA **five to ten minute** bin (37 recordings, 27 groups), K10 gets
**27/37** exact and MusicFM **29/37**: both correct 26, MusicFM only 3, K10 only 1,
both miss 7. In the larger **two to five minute** bin, K10 gets **126/179** and
MusicFM **117/179**. The two recordings over ten minutes are correct under both
systems, too few for any long-track claim. Genre, duration and group are
confounded: only five FMA selection recordings combine the House tag with five to
ten minute duration, so this says nothing specific about long house or techno.

#### BPM

![Paired exact outcomes by publisher-reported Beatport BPM, including unknown](figures/development-bpm-pairs.png)

The same paired split by publisher BPM band, with unknown BPM as its own bar; the
unknown bar is where MusicFM loses most.

Only 60/126 Beatport development records have a publisher BPM. On those 60, K10
gets **44/60** exact and MusicFM **45/60**; on the 66 without BPM, K10 gets
**45/66** and MusicFM **38/66**. Dropping unknowns would hide the overall decline.
This does not show that missing BPM causes errors or that tempo is the mechanism;
publisher BPM is not a verified tempo reference, and all 250 FMA BPM values are
unknown.

### Checkpoint correctness

During the fit, the selected checkpoint was reloaded strictly and evaluated with
encoder batch 1 and batch 16 across all 376 songs (5,893 contexts). The maximum
pooled-logit difference was **1.2904405593e-6**, within the fixed `1e-5` absolute
tolerance, with **zero changed keys**. The reproducer checks every saved 24-class
vector, their order and context lengths, and that the batch-1 predictions equal
the selection metrics' predictions.

A later independent audit hashed both checkpoint files in place, without loading
them into Torch. Both matched the training record:

| Artifact | Size | Integrity check |
|---|---:|---|
| Selected checkpoint | 1.23 GiB | Matched the training record |
| Final checkpoint | 1.23 GiB | Matched the training record |

The audit establishes which bytes are on disk; it did not reload the model a
second time. Exact sizes and checksums are in the
[checkpoint audit records](data/terminal-index.json).

The training wrapper writes its completion marker before a final output check,
so an exception at that point could leave both a completion marker and a failure
record. The marker alone is therefore insufficient evidence of success. The audit
also confirmed that no failure record exists and that all outputs and resource
limits were as expected; nothing indicates the run took that path. The wrapper is
unchanged here.

### Cost

![Whole-job wall and CPU cost, memory peaks and recorded validation timings](figures/teacher-cost.png)

Left: whole-job wall and CPU time. Middle: peak CUDA and host memory. Right:
evaluation time for each of the 120 batch-16 selection events, which varied
between roughly 70 and 157 s on the shared server, against the separate batch-1
proof (dashed, 208.28 s); the two are different phases, not a paired speedup.
No per-step GPU utilization trace exists. [SVG](figures/teacher-cost.svg).

| Resource | Observed | Fixed limit |
|---|---:|---:|
| Whole-job wall time | 31,934.96 s (**8 h 52 m 15 s**) | 20 h |
| Total CPU time | **33,450.7 s** (9.29 CPU h) | 100,000 s |
| Peak host memory | **3.080 GiB** | 8 GiB |
| Torch peak allocated / reserved CUDA memory | 11.326 / 11.492 GiB | 12 GiB allocator limit |
| Output files | 2,650,589,359 bytes (2.469 GiB) | 8 GiB |

This ran on a shared GPU server; these are neither compact-model CPU inference
costs nor isolated kernel timings. The earlier 10.92-hour projection (below) was
a planning estimate, and the observed 8.87-hour job does not establish a
controlled speedup.

The selected batch-1 proof's per-song times sum to **208.280 s**. The slowest
complete batch-16 event was **157.320 s** (update 5,934). A future diagnostic that
needs a fresh batch-1 baseline plus candidate batch-1 and batch-16 passes can
budget from `2 × 208.280 + 157.320 = 573.880 s` before loading, hashing,
serialization or contingency. This is arithmetic, not a measurement.

### BatchNorm refresh diagnostic: negative

Microbatch-one training means BatchNorm's running statistics may not match
evaluation behavior. As a cheap test of that hypothesis, the
[BatchNorm protocol](BN_PROTOCOL.md) replaces the input statistics of the selected
checkpoint's 18 BatchNorm layers, leaving every learned parameter and unrelated
buffer unchanged. It uses fitting audio only, deterministic crops, FP64
weighted population moments, and replaces layers in order from input to output.
The candidate state was fixed before any validation audio or reference key was
read, and the same success threshold applied: at least 171/250 FMA exact (+5)
with no Beatport decline.

Implementation check: two full traces and 36 stopped-prefix comparisons matched
with **zero maximum absolute difference**, and timing covered 288 stopped
prefixes across 16 fixed clips and all 18 layers. A first attempt under a
1,800-second limit could not fit its required safety margin:

| Budget item | Seconds | Meaning |
|---|---:|---|
| Projected full calibration | 675.66 | 512 × the sum of per-layer maximum observed clip costs |
| Required total margin | 2,638.41 | Doubled calibration and comparison costs, measured overhead, 120 s contingency |
| Time remaining under the limit | 1,753.43 | |
| Qualification run itself | 47.63 | Complete wall time, not calibration time |

That attempt stopped without calibrating or scoring (peak memory 3,550,330,880
bytes, 51.04 CPU seconds). An earlier start had failed on a missing `einops`
module because it used the wrong Python environment; it was fixed and all nine
original module hashes were rechecked. With the time limit raised to 3,600 s, a
next attempt ran out of its 12 GiB CUDA allocator limit (11.54 GiB resident when
a 1.17 GiB allocation was requested). With the allocator limit raised to 16 GiB on
an otherwise idle GPU, the diagnostic completed. None of these limit changes
touched the threshold, checkpoint, panel, estimator or interpretation, and none
of the failed starts is an accuracy result.

The completed run took **707.6 s** wall, **843.28 s** CPU and **6,219,206,656 bytes**
peak host memory. The candidate reloaded strictly and the batch-1 versus batch-16
check had **zero argmax disagreements**.

| Panel | Selected checkpoint | BatchNorm-refreshed | Change |
|---|---:|---:|---:|
| Beatport exact, 126 | 83 (65.87%) | 76 (60.32%) | **−7 (−5.56 pp)** |
| FMA exact, 250 | 166 (66.40%) | 169 (67.60%) | **+3 (+1.20 pp)** |
| Beatport symmetric MIREX | 73.25 | 68.25 | −5.00 |
| FMA symmetric MIREX | 73.36 | 74.40 | +1.04 |

The candidate failed both conditions: it lost seven Beatport exact answers and
gained only three FMA ones. Even meeting the minimum (171/250 FMA with Beatport
at 83/126) would have left it below the expanded K10 control's 176/250 and 89/126.
This closes this specific sequential refresh; it does not rule out a role for
BatchNorm mismatch or predict what a different calibration population, estimator
or checkpoint would do. Sequential population recalculation has precedent in
Appendix A.3 of [Rethinking "Batch" in BatchNorm](https://arxiv.org/pdf/2105.07576),
which supports its plausibility, not an expected gain here.

The executed code included two changes beyond its recorded base revision; the
[source lock](data/bn-diagnostic/v4/source-lock.json) and
[patch](data/bn-diagnostic/v4/source-overrides.patch) reconstruct it. The
[run record](data/bn-diagnostic/v4/run.json) gives resource use, and the
[completion record](data/bn-diagnostic/v4/controller.json) has no final output
size, so output-limit compliance rests on the running check alone.

### Earlier disposable cost study and the rotary-cache bug

Before the real fit, a disposable study measured cost on actual cached fitting
waveforms, 32-microbatch updates and all 376 complete selection songs. Each run
did only three optimizer updates with synthetic targets, and no weights were kept;
it measures cost and numerical correctness, not accuracy. It found a precision
bug that the earlier [16-window resource probe](../musicfm-preflight/REPORT.md)
could not.

The check required that evaluating the encoder at batch 16 reproduce batch 1:
finite 24-way outputs, maximum absolute pooled-logit error ≤ `1e-5`, and zero
changed keys, replaying the separate Beatport and FMA evaluation schedules
including partial batches.

| Measurement | Original implementation | After rotary-cache repair |
| --- | ---: | ---: |
| First 32-microbatch update | 3.46544 s | 3.56225 s |
| Second / third update | 2.49380 / 2.45074 s | 2.54529 / 2.53705 s |
| Input wait before first update | 0.06916 s | 0.06181 s |
| Mean of the two later updates | 2.47227 s | 2.54117 s |
| Batch-1 evaluation, all 376 songs | 247.44495 s | 233.73865 s |
| Batch-1 frontend / batch-16 encoder evaluation | 154.76148 s | 147.80360 s |
| Both evaluation passes, total CPU | 401.82286 s | 381.25761 s |
| Full cache integrity scan | 161.28303 s | 160.16988 s |
| Model setup and load | 7.77894 s | 8.10650 s |
| Total wall / total CPU | 583.60354 / 514.88 s | 562.63631 / 495.34 s |
| Maximum CUDA allocated / reserved | 12,161,753,088 / 12,308,185,088 B | same |
| Maximum sampled host memory | 3,314,368,512 B | 3,315,097,600 B |
| Maximum pooled-logit error | **0.0012251087: failed** | **3.230626e-7: passed** |
| Failing songs / changed keys | 5 / 0 | 0 / 0 |

![Disposable update costs, full-song validation costs and retained numerical failure](cost-results.png)

Left: the three update times in each run. Middle: one complete 376-song
evaluation pass (5,893 windows) at batch 1 and with the batch-16 encoder, in the
repaired run. Right: the batch-1 versus batch-16 error per song, sorted by the
original run's error; the five original failures sit near `1e-3`, far above the
dashed `1e-5` threshold, and after the repair every song is near `1e-7`.
[SVG](cost-results.svg).

The five failures were exactly the first five Beatport songs (`1059355`,
`1077935`, `11077`, `1147737`, `1274431`). Three have no tail and two have
11.97 s and 11.98 s tails, while all short tails passed, so short tails are not
the cause. The worst value was class 6 on `beatport:1059355` (batch-1 logit
0.24281444 versus batch-16 0.24403955). The other 371 songs had errors around
`1e-7`. Zero changed keys did not relax the numeric threshold.

The cause is in the
[archived Transformers source](../musicfm-preflight/evidence/transformers-conformer.py.txt.gz):
the rotary-embedding cache is reused whenever the sequence length matches (lines
143–144) but is built in the hidden-state dtype (lines 154–155), with no dtype or
device check. The last BF16 training crop has 300 encoder positions, and the first
five batch-1 evaluation songs also produce only 300-position contexts; the sixth
song's 11.2 s tail changes the length and forces an FP32 rebuild. So batch-1
evaluation could reuse a BF16 cache at the start of its nominal FP32 pass. This
fits the sharp five-song boundary, though the run did not log rotary tensor
dtypes. A batch-1-only run started in the meantime was stopped before completing
any update (55.45517 s wall) once this was found; it is not a result.

The repair is confined to the teacher wrapper: it verifies the rotary cache
contract, clears the cached length and tensors together on train/eval switches
and on device or dtype changes, and checks for stale dtype or device when
precision changes within a mode. CPU regression tests reproduce the bug and cover
both directions of the BF16/FP32 transition. Only `experiments/musicfm_teacher.py`
changed; initialization, inputs and settings were identical, and the repaired run
passed.

From these timings, the repaired steady update time × 8,280 updates, plus 120
batch-16 evaluation events, setup and a final selected-checkpoint check, gave a
planning estimate of **10.92 h**. Only two steady updates informed it, so it was a
budget input, not a prediction.

## Decision

The selected MusicFM teacher failed the fixed threshold against both K10
controls, and the BatchNorm refresh failed the same threshold. The saved decision
records `continuation_earned: false`, `confirmation_scored: false` and
`release_or_distillation_authorized: false`. There is no seed retry,
reinterpretation of the threshold or release claim, and no distillation from this
candidate. No further GPU time is planned for this teacher; the implementation
and measured costs are kept. Priority moves to cross-corpus exposure checks and
broader evaluation of deployable candidates. A future representation candidate
needs the same complete-domain controls and a decision rule fixed in advance, and
more labelled data should target full-length DJ recordings, contemporary genres
and reliable metadata rather than more short previews.

## Limitations

- One seed and one recipe, selected repeatedly on the same 376-recording
  development panel. There is no seed uncertainty and no independent estimate of
  full-length DJ performance.
- The study does not identify why selection accuracy fell while training loss
  kept dropping. Microbatch-one BatchNorm was a plausible suspect; the tested
  refresh did not rescue the checkpoint.
- The development panel has little long-track support: Beatport inputs are about
  two-minute previews, and only five FMA selection recordings are five to ten
  minute House. The [K10 v2 duration breakdown](../k10-v2-pilot/REPORT.md#accuracy-by-track-duration)
  describes this gap. Full 12-second contexts tile the song but give no long-range
  attention.
- Genre, duration and BPM breakdowns are retrospective, small and confounded.
- Label ambiguity, dataset provenance and overlap between MusicFM pretraining
  data and the evaluation recordings remain open.
- All costs come from a shared GPU server and do not describe CPU deployment.

## Reproduce

From the repository root, without running inference, reading audio or exporting
labels:

```bash
python -m tools.musicfm_teacher_result_report
python -m tools.musicfm_teacher_result_report --plot
python -m tools.musicfm_teacher_cost_report
python -m tools.musicfm_teacher_cost_report --plot
python -m tools.check_musicfm_bn_v4
uv run python -m tools.analyze_development_slices --check
uv run --extra plots python -m tools.plot_development_slices
python -m pytest -q tests/test_musicfm_teacher_result_report.py tests/test_musicfm_teacher_cost_report.py
```

`--plot` regenerates the figures (needs the `plots` extra), and
`plot_development_slices --write` rewrites the slice figures.
`check_musicfm_bn_v4 --reconstruct-source` additionally rebuilds the executed
BatchNorm code in a temporary directory (requires the historical Git revision).

The [completed-fit evidence index](data/terminal-index.json) and
[cost-study evidence index](data/cost-index.json) bind compressed archives of the
original JSON records (metadata, predictions and logit vectors; no audio,
checkpoints or reference-label exports). The
[BatchNorm metadata bundle](data/bn-diagnostic/terminal-metadata.tar.gz) holds the
diagnostic source, configuration and execution records. The
[result verifier](../../../tools/musicfm_teacher_result_report.py) checks
selection, control membership, the threshold, numerical vectors, the checkpoint
audit and resource limits, with tamper tests for each. Hash consistency shows the
archives agree with the retained records; it is not an independent attestation
of the original run.
