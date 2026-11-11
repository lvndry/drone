# Drone research

Each study keeps its research question, method, results, figures and decision
together. Start with its report; the adjacent protocol records the rules fixed
before evaluation, and `data/` retains measured evidence and provenance.

| Study | Problem | Result and decision |
|---|---|---|
| [K10 CPU baseline](studies/k10-cpu/REPORT.md) | What actually costs CPU time and memory? | Frontend median 353 ms versus ONNX 35.6 ms; long-track activation memory warrants optimization. |
| [K10 chunked inference](studies/k10-chunked/REPORT.md) | Bound activation memory without discarding evidence | All 376 keys agree; paired RSS 865 → 129 MiB, latency +1.91%. Proceed to native integration testing. |
| [K10 multi-domain fitting](studies/k10-multidomain/REPORT.md) | Improve broader-domain accuracy with more fitting data | FMAK mean +1.60 points misses the fixed +2.00 gate. Keep release weights. |
| [K10 EMA](studies/k10-ema/REPORT.md) | Improve accuracy through training-time averaging | Pilot loses accuracy; stop confirmation. Keep release weights. |
| [Structured key targets](studies/k10-structured-targets/REPORT.md) | Use harmonic error structure in the training loss | Fixed pilot reduces MIREX and exact accuracy; no confirmation. |
| [K10 capacity](studies/k10-capacity/REPORT.md) | Earn a stronger teacher through one wider model | Width 2 loses accuracy under the same recipe; no teacher or product tier earned. |
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
