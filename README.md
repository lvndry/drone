# drone

Drone is a public, reusable musical-key classifier with a Python API, CLI and
exportable model. It predicts 24 classes: 12 tonics, each in major or minor.
[Euphoria](https://github.com/lvndry/euphoria) is its main consumer; other audio
applications can use Drone independently.

**Model card:** [lvndry/drone on Hugging Face](https://huggingface.co/lvndry/drone).
The bundled artifacts and their provenance are documented in
[weights/README.md](weights/README.md).

## Current state

| Question | Answer |
|---|---|
| Released model | `drone-v1`: our own training of the published K10 architecture (MIREX 2026, MIT-licensed `jhurliman/jams`), a three-block CNN that pools over time only. |
| Size | 109,128 parameters. Runs on CPU. |
| Shipped checkpoint accuracy | Seed 13: **74.8% symmetric MIREX / 68.5% exact** on GiantSteps (604 two-minute EDM previews, 24 major/minor key labels). See [weights/README.md](weights/README.md). |
| Recipe accuracy | The same recipe at seeds 13, 37 and 73 averages **73.58% MIREX / 66.8% exact**. This is the recipe mean, not the score of the shipped checkpoint. |
| Baseline | libkeyfinder 2.2.8 as run inside Euphoria scores 69.35% MIREX / 60.60% exact on the same 604 files. |
| Training | 1,363 Beatport EDM previews (1,237 fitting, 126 validation), 320-frame crops, +/-4 semitone shifts, AdamW, 120 epochs. Details in [docs/training.md](docs/training.md). |
| Research index | [experiments/README.md](experiments/README.md) lists every study with its question, result and decision. |

GiantSteps was used during development, so it is not a held-out test for this
model. On a separate 93-recording reference panel (48 Winterreise performances
and 45 GiantSteps tracks), madmom's KEY_CNN agrees exactly on 72/93 recordings,
Drone on 63/93 and corrected libkeyfinder on 57/93: **madmom leads on that
panel** ([reference comparison](experiments/studies/key-sota-4way/REPORT.md)).

The research goal is a compact CPU model that beats commercial DJ key detection
on full-length club music; 90% exact accuracy is an aspiration, not a measured
result. The release was trained on roughly two-minute EDM previews, and accuracy
on 5 to 10 minute house, deep-house and melodic-house tracks has not been
established (see the [matched-data pilot](experiments/studies/k10-v2-pilot/REPORT.md)).
Softmax outputs are uncalibrated model scores, not validated probabilities of a
correct key. Features and checkpoints from before K10 are incompatible and must
be regenerated.

## Setup

Install [uv](https://docs.astral.sh/uv/) and `ffmpeg` (`brew install ffmpeg` on
macOS, or `apt install ffmpeg` on Ubuntu), then:

```bash
uv sync --locked
uv run pytest
```

`pyproject.toml` defines the package and dependencies; `uv.lock` pins the resolved
environment. `.python-version` selects Python 3.12. PyTorch backend/index choices
for a training host must match its GPU and driver; see the
[official uv/PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
Python inference defaults to CPU. The Python package is named `drone-key`,
imported as `drone`; Euphoria's eventual native integration should consume an
exported model rather than embed Python/PyTorch.

## Data and training

```bash
bash fetch.sh                 # Beatport audio + labels and held-out GiantSteps
uv run train.py prepare       # versioned, content-addressed feature caches
uv run train.py train --device cpu --threads 16
uv run train.py evaluate data/runs/<run>/drone.pt --output data/results.json
```

`fetch.sh` requires `curl`. It verifies Beatport archives against the checksums
published on Zenodo, pins GiantSteps annotations to a repository revision, and
verifies each test preview against its upstream MD5. Downloads are dev-time only;
the roughly 2.1 GB training archive plus test audio are not packaged. Use
`bash fetch.sh --dataset train` or `--dataset test` for one dataset, and `--data`
or `DRONE_DATA` to choose a data directory. See [DATA.md](DATA.md).

The network is a three-block CNN (16/32/64 filters) whose head pools over time
only, leaving a 24x1 readout so absolute frequency position reaches the final
layer. At width 1.0 it has 109,128 parameters; `--width` scales the channels
for smaller or larger profiles, which must each be trained and evaluated
independently. We adopted K10 after our own controlled screen showed that
pooling over time only beat a global-pooling control by about 22 MIREX points;
the reasoning is in the "Why K10" section of [docs/training.md](docs/training.md).

Training follows the K10 recipe: 320-frame crops, uniform +/-4 semitone shifts
applied as exact feature re-slices with a transposed label, AdamW at 1e-3,
cosine decay over 120 epochs, batch 32, plain cross-entropy. A fixed seeded
stratified validation split selects checkpoints; changing the training seed
leaves the split fixed. Label smoothing is available but off in the released
run. The training command does not score the test set; ID and content overlap
checks reject training/test leakage, and evaluation does not need training
features.

Every run writes `run.json` with source hashes, Git revision, dependency-lock
hash, data manifests, split IDs, configuration, per-epoch metrics, timing, and
checkpoint hash. Checkpoints include the model configuration, label schema,
feature spec, and crop length. Evaluation can write per-track predictions and
both symmetric and asymmetric MIREX scores. The default symmetric convention
awards fifths in both directions, matching Euphoria's benchmark; compare like
scoring conventions.

## Optional correction fine-tuning

`prepare-corrections` and `finetune` are an experimental, opt-in workflow for
explicit human key annotations mixed with the base training data. Euphoria's
JSONL export is one compatible adapter, not Drone's core data source; other tools
can produce the same portable format described in [DATA.md](DATA.md#personal-corrections).
Drone does not automatically collect annotations, scan private libraries or upload
user audio. Only explicitly supplied local files participate. No improvement on
real corrections and no GiantSteps nonregression is claimed; synthetic tests
validate the pipeline only. Keep a copy of the starting checkpoint and compare
the written report before using tuned weights. The commands do not replace
bundled release weights.

After preparing the base train/test data above, save the JSONL output of
`euphoria corrections export` and run:

```bash
uv run --extra corrections train.py prepare-corrections \
  --corrections corrections.jsonl --audio /path/to/library
uv run train.py finetune weights/drone-v1.pt --device cpu --threads 4
```

BLAKE3 is needed only for matching and hashing in this optional adapter. Base
inference, ordinary training, export and JSONL schema parsing do not require it;
`uv sync --extra corrections` installs it. Existing helper and CLI names remain
compatible. Matching uses Euphoria's BLAKE3 hash of every file byte, including
tags, so renamed or nested files still match. Tempo-only corrections are counted
and skipped. A deterministic hash rule holds roughly one fifth of matching files
out, and both parts must be nonempty. The split is stable only for identical file
bytes: retagging or re-encoding changes identity and can change the split. It
does not establish artist, recording or near-duplicate independence.

Fine-tuning defaults to 10 epochs, lr 1e-4, batch 32, eight correction examples per
batch mixed with the remaining base data (the final partial batch may have a
higher correction fraction). It keeps the last epoch and never selects on either
held-out score. Before any scoring or gradient update, the starting checkpoint's
known train/validation exposure and the new fitting data must be disjoint from
GiantSteps and the correction held-out set. Known overlap fails with an error;
no sample is reassigned. Missing historical checkpoint records mean exposure is
unknown, not that a set was unseen.

Each fresh `data/runs/finetune-.../` directory contains `drone.pt` and
`finetune.json`: configuration, inherited exposure records, per-epoch loss,
starting/tuned scores and paired improved/worsened/unchanged MIREX credit counts
on both sets. This fixed before/after comparison is deliberate; do not repeatedly
tune settings against the reported evaluation scores. See
[the detailed workflow](docs/training.md#11-optional-annotation-fine-tuning)
and [data limitations](DATA.md#personal-corrections).

## Features

Audio is decoded to mono 22.05 kHz PCM and transformed with a constant-Q
transform (24 bins per octave, 208 bins from four semitones below C1, 4096-sample
hop, log1p amplitude). The 192 core bins cover C1 through the top of the
eighth octave; the 16 outer bins are the +/-4 semitone shift pad. A pitch shift
is therefore an exact re-slice of this grid (no resampling, no interpolation
error), and the same shift transposes the label.

Inference reads the full track at natural length; training crops 320 frames
uniformly from each track. The model input is `(batch, 192, frames)`. Pitch
shifts preserve major/minor mode and therefore do not rebalance mode counts.

## Inference

The `drone-v1` weights ship in `weights/` (and packaged in the wheel):

```bash
uv run python -m drone track.mp3 --threads 1
```

For multiple files, load once and reuse the predictor:

```python
from drone import Predictor

predictor = Predictor()  # resolves drone-v1 by default
print(predictor.predict("first.mp3"))  # tonic, mode, softmax probability
print(predictor.predict("second.mp3"))
```

`drone.predict(path)` is also available for a single file. `DRONE_WEIGHTS`
overrides the default artifact; otherwise drone checks packaged
`drone/weights/drone-v1.pt`, then the checkout's `weights/drone-v1.pt`.
No model is downloaded automatically. A missing or incompatible artifact
produces an actionable error. Softmax probabilities are not calibrated
accuracy estimates. Audio feature caches are keyed by file contents and
feature specification.

## Export and CPU measurement

```bash
uv run --extra export python -m drone.export weights/drone-v1.pt data/drone-v1.onnx
uv run --extra export python -m drone.benchmark weights/drone-v1.pt \
  --onnx data/drone-v1.onnx --audio track.mp3 --threads 1 --output data/cpu.json
```

Export verifies ONNX/PyTorch output parity at several batch sizes and writes a
JSON sidecar containing the feature, model, and label contracts. It exports the
network only: decoding, feature extraction, normalization, and class mapping must
also match in a native host. ONNX is optional and adds no dependency to core Python
inference. The CPU session helper uses sequential execution, one intra-op thread,
and disabled worker spinning by default. Euphoria must bound total concurrency
across analysis workers as well.

For a native host that runs the network itself rather than through an ONNX
runtime, export the tensors as safetensors:

```bash
uv run python -m drone.safetensors weights/drone-v1.pt data/drone-v1.safetensors
```

It stores every learned parameter and BatchNorm running statistic in float32
under its PyTorch name, with the feature spec, label schema and the checkpoint's
SHA-256 in the header metadata. Choose a new output path: the exporter refuses
to overwrite existing files, including the source checkpoint. It needs only
PyTorch, not the export extra.

Benchmarks separate model loading, warm forward latency, cold decoding/features/
forward time, and warm feature-cache time. They report parameter and artifact size,
thread count, and hardware. Parent-process CPU timing excludes ffmpeg; it is not a
whole-system CPU or energy measurement. No CPU latency or accuracy guarantee is
claimed before measuring a trained artifact on representative deployment hardware.

## Development

The [research index](experiments/README.md) lists the questions, methods, results
and decisions behind the models. Each study report includes its figures, evidence
and reproduction instructions.

```bash
uv sync --locked --all-extras
uv run --extra export pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check .
uv build
```

Tests cover all key encodings and relationships, CQT tone placement and octave
specificity, the shift re-slice contract, content-cache invalidation, checkpoint
versions, ONNX parity, and a synthetic prepare/train/evaluate/predict run.
They validate pipeline correctness, not real-world musical-key accuracy.
All Python functions, including research scripts and tests, have parameter and
return annotations. CI enforces their presence with Ruff's `ANN` rules and checks
compatibility with `ty`, which needs `--all-extras` because research modules import
optional packages such as nnAudio.

Code: MIT. Dataset provenance and audio-use notes: [DATA.md](DATA.md).
