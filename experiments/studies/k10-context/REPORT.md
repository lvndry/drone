# K10 training crop length: 65 versus 320 frames

Training K10 on 65-frame crops instead of 320-frame crops did not meet the pre-set
threshold. On FMA the shorter crop gained one correct track (+0.40 points), well
short of the five required, though Beatport exact and both Beatport MIREX scores
improved slightly. The fresh 320-frame control also scored below the earlier GPU
control. No further seeds, release change or calibration followed. The study ran
in November 2026 on CPU.

## Question

Is a shorter training crop a useful lever for this compact key classifier when the
number of updates is held fixed? The study compares 65 and 320 CQT frames (about
12.07 and 59.44 seconds). Changing crop length also changes how much audio each
update sees, where crops can start, pooling, batch-normalization statistics and
gradient noise, so this is not a pure test of temporal context. It is also not a
teacher-architecture comparison or a CPU deployment option, and it does not
reproduce the MusicFM waveform and mel frontend.

## Method

The [protocol](PROTOCOL.md) was fixed before the run. Both arms are fresh seed-13
KeyCNN models with the same 109,128-parameter architecture and the same initial
weights. Both use 1,229 Beatport and 4,033 FMA fitting recordings, Beatport
sampling weight 1230/2206, the same per-recording seeds and pitch-shift
augmentation, batch size 32 and a uniform rule for legal crop starts. Because legal
starts depend on crop length, the two arms do not see identical audio windows.

Training: FP32 AdamW on CPU (learning rate 1e-3, weight decay 1e-4), the same cosine
schedule, 8,280 updates with 120 validation evaluations, and no early stopping.
Each arm used two CPU threads. The 320-frame arm ran first.

Every 69 updates the model was evaluated on the same 376 full-length validation
recordings (372,532 frames): 126 Beatport and 250 FMA. The checkpoint with the
highest mean exact accuracy over the two domains was selected, with mean symmetric
MIREX as tie-break and the earliest event winning remaining ties. Both MIREX
conventions are reported. These validation sets have been used repeatedly during
development. FMA is split by audited artist and album groups; Beatport has
duplicate-ID and exact-audio checks but incomplete artist information, so it is
not artist-disjoint. Beatport inputs are about two-minute previews.

Pre-set success threshold: the 65-frame arm had to gain at least 5 FMA exact cases
over the fresh 320-frame control, with no decline in Beatport exact or either
Beatport MIREX score. Separately, the fresh 320-frame control had to match or beat
the earlier GPU control from the [v2 pilot](../k10-v2-pilot/REPORT.md) on exact and
both MIREX scores in both domains.

## Results

| Selected arm | Update | BP exact | BP symmetric / asymmetric MIREX | FMA exact | FMA symmetric / asymmetric MIREX |
|---|---:|---:|---:|---:|---:|
| 320 frames | 2,898 | 88/126 (69.84%) | 76.11% / 74.52% | 174/250 (69.60%) | 75.20% / 74.60% |
| 65 frames | 7,176 | 89/126 (70.63%) | 77.38% / 75.79% | 175/250 (70.00%) | 75.32% / 74.52% |

![Full-song inner exact curves and selected events](figures/selection-curves.png)

*Figure 1. Exact accuracy on the full-length validation recordings over training
for both arms, with the selected updates marked. The two curves overlap for most
of training.*

The FMA gain was +1 case against the required +5, so the threshold failed. The
control comparison also failed: the 320-frame CPU control lost 1 Beatport and 2 FMA
exact cases against the earlier GPU control, with symmetric / asymmetric weighted
score gaps of -13 / -8 on Beatport and -16 / -11 on FMA (each unit is 0.1 track of
credit). That comparison also differs in hardware (CPU versus CUDA) and thread
count, so it does not test exact reproduction of the earlier run. Neither threshold
was relaxed.

Between the selected checkpoints, 6 Beatport tracks became correct and 5 became
wrong (83 correct in both, 32 wrong in both). On FMA, 9 became correct and 8 became
wrong (166 correct in both, 67 wrong in both).

![Selected paired metric differences and descriptive intervals](figures/paired-differences.png)

*Figure 2. 65-frame minus 320-frame differences with 95% bootstrap intervals. Every
interval includes zero.*

| 65 minus 320 | Exact change [95% interval], points | Symmetric MIREX change [interval] | Asymmetric MIREX change [interval] |
|---|---:|---:|---:|
| BP | +0.79 [-3.97, +6.35] | +1.27 [-3.33, +5.95] | +1.27 [-3.33, +5.95] |
| FMA | +0.40 [-2.79, +3.67] | +0.12 [-2.17, +2.51] | -0.08 [-2.59, +2.50] |

Intervals use 10,000 paired bootstrap replicates (seed 20261112) over 126 Beatport
recordings and 100 FMA metadata clusters. They describe one seed on reused
validation data, not training-seed variation or independent test evidence.

### Accuracy by track duration

| Domain / duration | Tracks / groups | 320 correct, 65 correct |
|---|---:|---:|
| BP under 2 minutes | 5 / 5 | 4, 4 |
| BP 2 to under 5 minutes | 121 / 121 | 84, 85 |
| BP 5 minutes or more | 0 / 0 | n/a |
| FMA under 2 minutes | 32 / 18 | 21, 21 |
| FMA 2 to under 5 minutes | 179 / 80 | 124, 123 |
| FMA 5 to 10 minutes | 37 / 27 | 27, 29 |
| FMA over 10 minutes | 2 / 2 | 2, 2 |

Durations come from the actual sample counts (600 seconds counts as 5 to 10
minutes). Metadata groups can span bins, so group counts should not be summed.
The FMA 5-to-10-minute bin is not a house-music panel: it contains only five House
examples in three groups, none from 8 to 10 minutes and none tagged deep or melodic
house. This study therefore cannot measure accuracy on long house tracks, the
main intended use.

### Cost and audio exposure

![Completed arm cost and actual sampled audio exposure](figures/cost-and-exposure.png)

*Figure 3. Training cost per arm and the number of non-padding frames each arm saw.
The 65-frame arm is cheaper but sees about 4.9 times less audio.*

The pair took 13,902.322 seconds (3.862 hours) in total, using 26,496.92 CPU-seconds
and a peak of 3,014,115,328 bytes (2.807 GiB) of memory. The 320-frame arm took
11,407.757 seconds and the 65-frame arm 2,485.871 seconds, each including the
reload check of its selected checkpoint. These are training-run timings on a shared
machine, not product performance estimates.

Both arms sampled all 5,262 fitting recordings across 264,960 draws with identical
per-recording counts. Non-padding frames seen were 84,532,873 for 320 frames and
17,222,313 for 65 frames, a 4.908 times difference. Lower training cost therefore
comes with less audio seen; this does not isolate the effect of context.

An earlier CPU feasibility check ([resource protocol](RESOURCE_PROTOCOL.md)) ran three
batch-32 updates per crop size with random targets, then one pass over all 376
validation recordings with no predictions kept. It took 34.826 seconds, 56.44
CPU-seconds and 2,464,256,000 bytes of peak memory. Update times were
1.2105 / 1.1403 / 1.0679 seconds at 320 frames and 0.1397 / 0.1410 / 0.1296 seconds
at 65 frames; the full validation pass took 12.196 and 10.394 seconds.

Each selected checkpoint was reloaded into a fresh KeyCNN and re-run on all 376
validation inputs, giving identical scores and predictions. Both arms had the same
initial weights, recording and seed order, per-recording coverage and data sources;
only the crop windows differ, by design.

## Decision

Do not continue with 65-frame training: the FMA gain of one case is far below the
five-case threshold. No winning arm, additional seed or longer schedule was chosen
from these results, and the release model is unchanged.

## Limitations

- One seed on validation sets already used during development. The intervals all
  include zero, so the data do not show that the two crop lengths are equivalent
  either.
- Crop length changes audio exposure, crop positions and normalization statistics
  together; the effect of context alone is not isolated.
- The fresh CPU control did not match the earlier GPU control.
- Beatport validation is not artist-disjoint, and FMA metadata may not reveal every
  near-duplicate.
- There is no long house-music panel.

## Reproduce

Checksums and sizes for all evidence files are in the [archive index](data/archive-index.json).
No weights, CQT arrays or audio are committed. The executed training source is kept
as [a compressed copy](data/executed-context-source.py.gz). Verify metrics, selection,
thresholds, intervals and duration counts and redraw the figures without training:

```sh
python -m tools.check_k10_context_report
python -m tools.plot_research
python -m pytest -q tests/test_k10_context_report.py
```
