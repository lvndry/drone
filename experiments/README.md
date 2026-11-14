# Drone research

This is the index of Drone's research studies. Each study keeps its question,
method, results, figures and decision in one `REPORT.md`; the adjacent protocol
records the rules fixed before evaluation, and `data/` keeps the measured
evidence. Dataset coverage (genres, durations, tempo metadata, split roles and
reference provenance) is described in the [corpus inventory](../DATA.md).

Status values: **current** (in use or the latest word on its question),
**superseded** (replaced by a later study or release), **rejected** (the
pre-set success threshold was missed and the idea was not continued), **open**
(no final result yet).

## Release lineage

| Study | Question | Result | Decision | Status |
|---|---|---|---|---|
| [Compact-model research](studies/compact-models/REPORT.md) | Can a corrected small CNN beat libkeyfinder, and which features and pooling matter? | Time-only pooling beat global pooling by about 22 MIREX points. The 47,284-parameter octave model scored 68.99% MIREX on 1,000 FMAK tracks and 70.40% on GiantSteps, versus libkeyfinder's 67.82% and 69.35%; both paired 95% intervals include zero. | Keep libkeyfinder; port K10 instead of extending this design. | Superseded by the K10 release |
| [K10 release (`drone-v1`)](../docs/training.md) | Does our own training of the published K10 architecture beat libkeyfinder on GiantSteps? | 109,128 parameters. Shipped seed 13: 74.8% MIREX / 68.5% exact; three-seed mean 73.58% / 66.8%, versus libkeyfinder 69.35% / 60.60%. GiantSteps was used in development. | Released as `drone-v1`. | Current |

## Accuracy experiments

Every row below starts from the K10 release recipe and changes one thing.

| Study | Question | Result | Decision | Status |
|---|---|---|---|---|
| [K10 multi-domain fitting](studies/k10-multidomain/REPORT.md) | Does adding FMAK fitting data improve accuracy outside EDM? | Three-seed mean gain: FMAK +1.60 MIREX points, Beatport +2.38. The pre-set threshold was +2.00 on FMAK. | Keep release weights. Near threshold (see below). | Rejected |
| [K10 EMA](studies/k10-ema/REPORT.md) | Does training-time weight averaging improve accuracy? | EMA with BatchNorm recalibration lost 1.84 FMAK MIREX points against ordinary weights. | Stop; no confirmation seeds. | Rejected |
| [Structured key targets](studies/k10-structured-targets/REPORT.md) | Does a loss that gives partial credit to harmonic neighbours help? | The fixed seed-13 run lowered both MIREX and exact accuracy. | Stop; no confirmation seeds. | Rejected |
| [K10 capacity and resolution](studies/k10-capacity/REPORT.md) | Does more capacity, or a finer frequency grid at equal size, help? | Width 2 (361,080 parameters) and the matched-size finer grid both missed their thresholds; the finer grid lost 15 FMA exact cases in one fresh GPU pair. | Stop. | Rejected |
| [Pitch-circle sharing](studies/pitch-circle/REPORT.md) | Does sharing convolutions around a pitch-class circle help with fewer parameters (73,506)? | Seed-13 run lost 0.40 FMAK exact points. | Stop; no confirmation seeds. | Rejected |
| [Tonal key ranker](studies/tonal-key-ranker/REPORT.md) | Can fixed harmonic summaries plus a small tree model beat the CNN? | Substantially lower accuracy. | Stop this route. | Rejected |
| [Frozen temporal heads](studies/k10-temporal/REPORT.md) | Do full-track hidden features and their order improve exact accuracy? | The ordered head lost 5.60 FMAK exact points and did not beat a shuffled head. | Stop. | Rejected |
| [Fixed SAM](studies/k10-sam/REPORT.md) | Does one sharpness-aware optimizer recipe help? | Seed-13 run lost weighted and exact accuracy. | Stop; no confirmation seeds. | Rejected |
| [Fixed probability ensemble](studies/k10-ensemble/REPORT.md) | Is an average of joint seeds 13/37/73 a stronger teacher for distillation? | Beatport improved; FMAK regressed against the canonical member. | No distillation. | Rejected |
| [K10 crop context](studies/k10-context/REPORT.md) | Are 65-frame training crops better than 320-frame crops at equal updates? | Short crops gained one FMA exact case; five were required. The fresh 320-frame control also failed its historical-score check. | Stop. | Rejected |
| [K10 v2 matched-data pilot](studies/k10-v2-pilot/REPORT.md) | Does a larger nested FMA fitting pool help at equal updates? | FMA exact 172/250 to 176/250 (+1.60 points) against a +2.00 threshold. Only five long house tracks in the inner panel. | No confirmation. Near threshold (see below). | Rejected |
| [MusicFM key teacher](studies/musicfm-key-pilot/REPORT.md) | Does fine-tuning MusicFM give a stronger teacher than K10? | Selected checkpoint: 83/126 Beatport and 166/250 FMA exact, below both K10 controls. | No distillation or release. | Rejected |
| [MusicFM source and resource preflight](studies/musicfm-preflight/REPORT.md) | Can MusicFM be loaded and trained within available GPU memory? | Strict loading and a bounded 21-update synthetic GPU probe passed. No accuracy result. | Allowed the teacher pilot above. | Complete |
| [MusicFM crop I/O](studies/musicfm-crop-io/REPORT.md) | Can random waveform crops be read without full decoding? | 144/144 crops matched on a small sample; the corpus run stopped at file 2,289 after 2,288 passes. | No complete cache. | Rejected |
| [Myna frozen-embedding pilot](studies/myna-key-pilot/REPORT.md) | Does a frozen pretrained Myna encoder with a trained key head beat matched K10? | Encoder, playback-rate augmentation and complete-track path qualified. No accuracy result yet. | Run the fixed accuracy pilot. | Open |

### Thresholds and statistical power

Most pilots use a single seed and a pre-set threshold of 1 to 2 points on
validation panels of 126 (Beatport) or 250 (FMAK) tracks. One track is 0.79
points on 126 tracks and 0.40 points on 250, so these thresholds correspond to
about one to five tracks. Results close to the threshold are therefore
inconclusive rather than negative. In particular, the two +1.60 results for
more fitting data ([multi-domain](studies/k10-multidomain/REPORT.md) and
[v2 pilot](studies/k10-v2-pilot/REPORT.md)) fell 0.40 points, about one FMAK
track, short of +2.00. The decisions above follow the fixed thresholds; the
evidence does not show that more data fails to help.

## Data and evaluation

| Study | Question | Result | Decision | Status |
|---|---|---|---|---|
| [FMAK split protocol](studies/data-split/PROTOCOL.md) | How should FMAK be split so evaluation is independent of fitting? | Fixed before any FMAK scoring: 1,000 final rows chosen by audio-hash group, 4,489 development rows. Artist/album quarantine was added later, in the compact-model adaptation stage. | Used by the compact-model and early K10 studies. | Superseded by the v2 data contract |
| [v2 data contract and coverage](studies/v2-data-contract/REPORT.md) | Which data, genres, durations and metadata do we actually have, and which recordings were used where? | Collections cover 1,355 Beatport, 5,489 FMA, 604 GiantSteps and 1,000 GTZAN IDs. The GTZAN overlap audit leaves 167 review flags among 769 primary queries, plus one incomplete query in the wider cohort. | Use as the exposure record; no GTZAN recording is accepted for scoring yet. | Current |
| [Beatport / GiantSteps overlap audit](studies/bp-gs-overlap/REPORT.md) | Do Beatport training recordings overlap GiantSteps? | Of 1,486 Beatport IDs: 0 identity matches, 539 unresolved reviews, 947 with no detected overlap. | No overlap-free claim. | Current |
| [Key detector reference comparison](studies/key-sota-4way/REPORT.md) | How does `drone-v1` compare with madmom and libkeyfinder on 48 Winterreise and 45 GiantSteps recordings? | Exact: madmom 72/93, Drone 63/93, corrected libkeyfinder 57/93. Weighted: 82.9%, 75.7%, 73.7%. MusicFM was not run. | Descriptive only; madmom leads on this panel. No release change. | Current |

## CPU and native inference

| Study | Question | Result | Decision | Status |
|---|---|---|---|---|
| [K10 CPU baseline](studies/k10-cpu/REPORT.md) | What costs CPU time and memory? | Frontend median 353 ms versus ONNX 35.6 ms; long-track activations need much memory. | Optimize activation memory. | Current |
| [K10 chunked inference](studies/k10-chunked/REPORT.md) | Can activation memory be bounded without discarding evidence? | All 376 keys unchanged; paired peak RSS 865 to 129 MiB; latency +1.91%. | Proceed to native testing. | Current |
| [Native cached-feature inference](../research/drone-native-cached/REPORT.md) | Does a Rust implementation keep predictions and bound memory? | All 376 keys unchanged; peak RSS 828 to 95 MiB; latency +1.84%. Cached features only. | Measure the frontend separately. | Current |
| [Native CQT frontend](../research/drone-native-cqt/REPORT.md) | Can a native CQT reproduce the Python features on the same PCM? | All 82,562,688 core feature values and 9,024 logits match bit for bit on 376 tracks (Linux). | Qualifies the tested Linux configuration. | Current |
| [Native composed pipeline](../research/drone-native-cqt/composed/REPORT.md) | Does native CQT plus full or chunked model keep predictions, and what does it cost? | Linux: 376/376 tracks with zero key changes; on 20 FMAK tracks, about 27% lower median latency than Python. Mac: all 376 feature cores reproduced after a repair. | Prefer chunked inference for long inputs; production integration still needs work. | Current |

## Study structure

Use `studies/<name>/REPORT.md` for the question, method, results, decision,
limitations and reproduction. Place figures beside that report with a caption
saying what to look at. State metrics, units, sample size, uncertainty, selection
policy and limitations. Keep negative results and missed thresholds.

Write reports for an outside reader using descriptive experiment names. Explain
what changed and why it matters. Keep commit IDs, checksums and process details
in the data files, not in prose or table headings.

Every accuracy report must identify its corpus, version and fixed cohort, show
genre and analyzed-duration coverage, and report available tempo metadata plus
unknowns. Include per-slice recording counts and failures alongside accuracy;
keep original tags when showing derived genre groups. Separate full-song
metadata from the audio actually analyzed, and repeated recordings from distinct
works. State missing slices directly rather than implying all-genre or long-track
coverage from an aggregate score.

Store the frozen protocol as `PROTOCOL.md`, small JSON evidence under `data/`,
and figures under `figures/`. Do not commit audio, caches or training checkpoints.
Historical evidence stays unchanged when files move. Python experiment modules
remain importable here; reusable command-line tools belong in `tools/`.

## Regenerate figures

```bash
uv sync --locked --extra plots
uv run --extra plots python -m tools.plot_research
```

The generator reads checked-in JSON and writes PNG/SVG files into each study.
[Figure inputs](studies/figure-inputs.json) records data, generator and lockfile
hashes. A figure does not change a decision.
