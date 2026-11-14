# drone-v2 release evaluation

This study picks the drone-v2 checkpoint from the shift12 recipe of the
[frontier study](../frontier-distillation/REPORT.md) and measures it through the
shipped Python package, from original audio files to keys. Study date: 14
November 2026.

## Choosing the checkpoint

The recipe trained three seeds. The choice used development data only: FMAKv2
`valid` and `holdout` and the development half of GiantSteps+ (794 tracks). The
GiantSteps+ test half was not used ([data/seed-selection.txt](data/seed-selection.txt)).

| Seed | Development exact, one pass | Nine-shift averaging vs one pass (95% interval) |
| --- | ---: | ---: |
| 13 | 75.19 | +0.76 (-0.50 to +2.02) |
| 15 | 75.31 | +0.25 (-0.76 to +1.26) |
| 42 | **77.20** | -0.76 (-1.76 to +0.13) |

Seed 42 is best on development data and on the internal validation score that
selected its epoch (78.65 MIREX). Averaging over nine pitch shifts does not help
it, so drone-v2 runs one forward pass per track, the same cost as drone-v1.

## Accuracy through the package

`scripts/evaluate_release.py` runs `drone.Predictor` on each original file:
ffmpeg decoding to 22,050 Hz mono, the librosa constant-Q transform of the whole
track, one forward pass. `scripts/score_release.py` scores the predictions
([data/release-scores.json](data/release-scores.json)); libkeyfinder predictions
come from the frontier study's corrected wrapper.

| Panel | libkeyfinder | drone-v1 | drone-v2 | drone-v2 minus drone-v1 (95% interval) | drone-v2 minus libkeyfinder |
| --- | ---: | ---: | ---: | ---: | ---: |
| FMAKv2 `valid` (250) | | 62.4 | 72.4 | +10.0 (+4.8 to +14.8) | |
| FMAKv2 `holdout` (250) | | 68.4 | 77.6 | +9.2 (+4.8 to +13.6) | |
| GiantSteps, original labels (604) | 60.4 | 68.5 | 69.9 | +1.3 (-1.2 to +4.0) | +9.4 (+6.1 to +12.8) |
| GiantSteps+ primary key (571) | 69.9 | 77.4 | 80.7 | +3.3 (+0.7 to +6.1) | +10.9 (+7.4 to +14.4) |
| GiantSteps+ single-key (457) | 76.2 | 83.4 | 88.0 | +4.6 (+2.0 to +7.4) | +11.8 (+8.1 to +15.5) |
| GiantSteps+ single-key, test half (226) | 74.8 | 81.0 | 87.2 | +6.2 (+2.2 to +10.6) | +12.4 (+7.1 to +18.1) |

The package reproduces the frontier study's cached-feature scores for seed 42
exactly on every GiantSteps panel. The FMAKv2 panels share their annotation
process with the training labels and contain some recordings DATA.md reserves
for confirmation (see the frontier study), so they measure progress on FMA, not
independent accuracy.

By FMA genre (exact, drone-v1 to drone-v2): electronic 59.3 to 69.2 (91 tracks),
pop 73.2 to 87.5 (56), rock 66.7 to 70.8 (48), folk 69.4 to 77.8 (36), hip-hop
78.6 to 85.7 (14). Smaller genres are in the data file; most have too few
tracks to read.

## Speed

On an Apple M5, one thread, 19 GiantSteps previews (two minutes each), the
median time per track through the Python package is 302 ms: 242 ms to decode
and compute the spectrogram, 57 ms for the network. drone-v1 has the same
architecture and therefore the same cost.
