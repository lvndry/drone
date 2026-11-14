# Tonal summaries with gradient-boosted trees: fast but much less accurate

A key classifier built from fixed harmonic summary features and 128 gradient-boosted
trees trained in seconds but was far less accurate than K10. Symmetric MIREX fell
21.11 points on Beatport and 12.20 points on FMAK, and exact accuracy fell 20.63 and
16.40 points. It failed its pre-set threshold and the approach was stopped. The study
ran in November 2026.

## Question

Can fixed harmonic summaries and a small nonlinear decision model beat a learned CNN
when labels are limited? This tests a broad classifier idea without external weights
or a teacher. It differs from the earlier 48-parameter DSP score fusion: the trees
learn directly from audio-derived features, with no neural or native key scores as
input.

## Method

The study uses the same 2,206 fitting and 376 validation recordings and the existing
FP16 CQT caches. Each track yields 24 candidate rows, one per key, of which one is
correct, giving 52,944 fitting rows from 2,206 labelled tracks. Candidate groups never
cross roles, and the existing artist, album and audio-duplicate separation is kept.

Each candidate row has a fixed 24-dimensional descriptor: whole-track chroma relative
to the candidate key, low-register tonic, third and fifth evidence, scale contrasts for
each quarter of the track, the bass tonic at the ending, the candidate mode, a
half-bin tuning proxy and relative-key contrasts. Features never use the true key. Frame
normalization and summaries are deterministic, and tree split borders are fitted on
fitting rows only. The tuning proxy is a feature, not a tuning estimate or correction.

The ranker is CatBoost 1.2.10 with grouped `QuerySoftMax:beta=1` loss, 128 symmetric
trees of depth 4, learning rate 0.05 and the other settings in the
[protocol](https://github.com/lvndry/drone/blob/research/tonal-key-ranker-v1/experiments/studies/tonal-key-ranker/PROTOCOL.md).
There was no teacher, validation-based early stopping, feature search or tree-count
selection. Representation and classifier change together, so this compares two whole
systems rather than isolating trees versus CNNs.

The seed-13 joint K10 control was rechecked against the archived history, predictions
and checkpoint weights. Pre-set success threshold: at least +1 FMAK symmetric MIREX
point, no Beatport MIREX decline and no exact-accuracy decline in either domain. No raw
audio, final-test, calibration or private recordings, and no GPU, were used.

## Results

| Arm | BP symmetric / asymmetric MIREX | BP exact | FMAK symmetric / asymmetric MIREX | FMAK exact |
|---|---:|---:|---:|---:|
| Ordinary K10 (verified control) | 76.11 / 73.33% | 68.25% | 75.48 / 74.48% | 68.40% |
| Tonal ranker | 55.00 / 53.81% | 47.62% | 63.28 / 60.88% | 52.00% |

![Paired K10 and tonal-ranker validation results](figures/accuracy.png)

*Figure 1. The tonal ranker is far below K10 on weighted and exact accuracy in both
domains. The axis starts low enough to show every score.*

Symmetric MIREX changed by -21.11 points on Beatport and -12.20 on FMAK; exact accuracy
by -20.63 and -16.40. Paired error movements under both MIREX conventions are in the
[candidate result](data/candidate-result.json).

Preparing features for all 2,582 cached tracks, including the control checks, took
7.86 seconds with 553.61 MiB peak memory. Training, export, reload and validation
scoring took 5.40 seconds with 641.50 MiB peak memory, using four CPU threads. These
times exclude audio decoding and CQT computation and are not per-track inference
latency; the memory figures include research verification code and are not a minimal
deployment footprint.

## Decision

Stop this approach: no confirmation seeds, feature revisions, tree or learning-rate
sweeps, export or release changes. Fast training does not make up for the accuracy
loss. Broader audited data and learned representations are more promising routes to a
large gain than tuning these features on the same validation data.

## Limitations

- One configuration on validation data used repeatedly during development.
- The result rejects this descriptor with 128 trees. It does not show that all tree
  models are unsuitable, does not identify whether the features or the classifier
  caused the loss, and is not a fair benchmark of tabular classifiers.
- Whole-track summaries lose chord progression and phrasing, and relative major and
  minor keys can share pitch distributions. These are plausible causes, not tested ones.

## Reproduce

The executed source (feature extractor, runner, pinned optional dependencies, model
loader, synthetic compatibility checks and protocol) is preserved at the git tag
`research/tonal-key-ranker-v1`. The [candidate result](data/candidate-result.json)
records checksums for the source, protocol, features, control, model and validation
outputs, the resolved parameters, both MIREX conventions and the threshold outcome.
[Runtime package versions](data/runtime-packages.json) list the software used for the
run. Prepared features, the trained model and validation logits are kept outside the
repository. The [source record](../../REPORT_PROVENANCE.md#tonal-key-ranker) lists the
exact source identities. Regenerate Figure 1 with
`uv run --extra plots python -m tools.plot_research`.
