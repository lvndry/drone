# Drone research

Each study keeps its research question, method, results, figures and decision
together. Start with its report; the adjacent protocol records the rules fixed
before evaluation, and `data/` retains measured evidence and provenance.

| Study | Problem | Result and decision |
|---|---|---|
| [K10 CPU baseline](studies/k10-cpu/REPORT.md) | What actually costs CPU time and memory? | Frontend median 353 ms versus ONNX 35.6 ms; long-track activation memory warrants optimization. |
| [K10 chunked inference](studies/k10-chunked/REPORT.md) | Bound activation memory without discarding evidence | All 376 keys agree; paired RSS 865 → 129 MiB, latency +1.91%. Proceed to native integration testing. |
| [Native cached-feature CPU study](../research/drone-native-cached/REPORT.md) | Validate the Rust implementation and its memory cost | All 376 keys agree; paired RSS 828 → 95 MiB, latency +1.84%. Cached-feature result; the later [composed study](../research/drone-native-cqt/composed/REPORT.md) measures the frontend separately. |
| [K10 context and audio exposure](studies/k10-context/REPORT.md) | Compare 65- versus 320-frame training crops at matched updates | Short crops gain one FMA exact case, below the required five; historical control guard also fails. No continuation or promotion. |
| [K10 v2 matched-data pilot](studies/k10-v2-pilot/REPORT.md) | Test nested expanded fitting data at equal updates | FMA exact +1.60 points misses fixed +2.00 gate; no confirmation. Five long-house inner examples cannot establish primary-use accuracy. |
| [K10 multi-domain fitting](studies/k10-multidomain/REPORT.md) | Improve broader-domain accuracy with more fitting data | FMAK mean +1.60 points misses the fixed +2.00 gate. Keep release weights. |
| [K10 EMA](studies/k10-ema/REPORT.md) | Improve accuracy through training-time averaging | Pilot loses accuracy; stop confirmation. Keep release weights. |
| [Structured key targets](studies/k10-structured-targets/REPORT.md) | Use harmonic error structure in the training loss | Fixed pilot reduces MIREX and exact accuracy; no confirmation. |
| [K10 capacity and resolution](studies/k10-capacity/REPORT.md) | Test useful capacity allocation for a compact model | Width 2 and the matched-size finer grid both miss their accuracy gates; the latter loses 15 FMA exact cases in one fresh GPU pair. |
| [Pitch-circle sharing](studies/pitch-circle/REPORT.md) | Retain octaves while sharing transposition patterns | Seed-13 pilot rejected; FMAK exact −0.40 points, no confirmation. |
| [Tonal key ranker](studies/tonal-key-ranker/REPORT.md) | Test candidate-relative harmonic summaries with a small nonlinear scorer | Fixed descriptor/tree recipe loses substantially; stop this route. |
| [Frozen temporal heads](studies/k10-temporal/REPORT.md) | Test whether full-track hidden features and order improve exact accuracy | Ordered head loses 5.60 FMAK exact points; candidate and mechanism gates fail. |
| [Fixed SAM](studies/k10-sam/REPORT.md) | Improve accuracy through one sharpness-aware update recipe | Canonical pilot loses weighted and exact accuracy; no confirmation or rescue. |
| [Fixed probability ensemble](studies/k10-ensemble/REPORT.md) | Qualify a stronger own-data teacher from joint seeds 13/37/73 | Beatport improves, canonical FMAK regresses; teacher and separate BP-reference gates fail. No distillation. |
| [BP / GiantSteps overlap audit](studies/bp-gs-overlap/REPORT.md) | Reconcile source/model exposure and test cached recording overlap | 0 identity, 539 unresolved review, 947 no-detected; no overlap-free claim. |
| [MusicFM crop I/O qualification](studies/musicfm-crop-io/REPORT.md) | Test bounded random-access waveform crops against full decoding | Initial 144/144 crops pass; corpus attempt stops at file 2,289 after 2,288 passes. No complete cache or training activation. |
| [MusicFM source/resource preflight](studies/musicfm-preflight/REPORT.md) | Establish teacher provenance and a bounded execution path before fine-tuning | Imports/strict loading and a bounded21-update synthetic GPU resource probe passed; no task-accuracy result or qualified teacher. |
| [Compact-model research](studies/compact-models/REPORT.md) | Establish corrected features, pooling and small-model controls | Historical screen, octave/CQT stage and exposed final evaluations; not the current release model. |

The shared [FMAK split protocol](studies/data-split/PROTOCOL.md) records grouping,
quarantine and evaluation roles. The [research guide](../docs/RESEARCH_TRAINING.md)
and [training guide](../docs/training.md) connect the studies to the code.

## Study structure

Use `studies/<name>/REPORT.md` for the problem, method/comparison, measured
results and discussion/decision. Place figures beside that report, with captions
where the results are discussed. State metrics, units, sample size, uncertainty,
selection policy and limitations. Keep negative results and failed gates.

Store the frozen protocol as `PROTOCOL.md`, small JSON evidence under `data/`,
and figures under `figures/`. Do not commit audio, caches or training checkpoints.
Historical evidence stays unchanged when files move. Python experiment modules
remain importable here; reusable command-line machinery belongs in `tools/`.

## Regenerate figures

```bash
uv sync --locked --extra plots
uv run --extra plots python -m tools.plot_research
```

The generator reads checked-in JSON and writes PNG/SVG files into each study.
[Figure inputs](studies/figure-inputs.json) records data, generator and lockfile
hashes. Measurements distinguish cached-feature inference from native application
cost, training-seed variation from uncertainty over unseen recordings, and reused
development data from independent evaluation. A figure does not change a gate.
