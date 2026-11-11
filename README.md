# drone

A small neural classifier for the musical key of an audio track, intended for
[Euphoria](https://github.com/lvndry/euphoria). It predicts 24 classes: 12 tonics,
each in major or minor.

**Status:** the K10 (MIREX 2026) architecture is trained and confirmed. The
`drone-v1` weights (109,128 parameters) score a **73.58% MIREX three-seed mean**
on the held-out GiantSteps 604 (66.8% exact), versus 69.35% for the native
libkeyfinder reference on the same files. Pre-K10 features and unversioned
checkpoints are incompatible and must be regenerated. The full training story
is in [docs/training.md](docs/training.md).

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
independently.

Training follows the K10 recipe: 320-frame crops, uniform +/-4 semitone shifts
applied as exact feature re-slices with a transposed label, AdamW at 1e-3,
cosine decay over 120 epochs, batch 32, plain cross-entropy. A fixed seeded
stratified validation split selects checkpoints; changing the training seed
leaves the split fixed. Label smoothing is available but off in the confirmed
run. The training command does not score the held-out test set; ID/content
overlap checks reject training/test leakage, and evaluation does not need
training features.

Every run writes `run.json` with source hashes, Git revision, dependency-lock
hash, data manifests, split IDs, configuration, per-epoch metrics, timing, and
checkpoint hash. Checkpoints include the model configuration, label schema,
feature spec, and crop length. Evaluation can write per-track predictions and
both symmetric and asymmetric MIREX scores. The default symmetric convention
awards fifths in both directions, matching Euphoria's benchmark; compare like
scoring conventions.

## Features

Audio is decoded to mono 22.05 kHz PCM and transformed with a constant-Q
transform (24 bins per octave, 208 bins from two semitones below C1, 4096-sample
hop, log1p amplitude). The 192 core bins cover C1 through the top of the
eighth octave; the 16 outer bins are the +/-4 semitone shift pad. A pitch shift
is therefore an exact re-slice of this grid — no resampling, no interpolation
error — and the same shift transposes the label.

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

Benchmarks separate model loading, warm forward latency, cold decoding/features/
forward time, and warm feature-cache time. They report parameter and artifact size,
thread count, and hardware. Parent-process CPU timing excludes ffmpeg; it is not a
whole-system CPU or energy measurement. No CPU latency/accuracy guarantee is claimed
before measuring a trained artifact on representative deployment hardware.

## Development

```bash
uv sync --locked --all-extras
uv run --extra export pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Tests cover all key encodings and relationships, CQT tone placement and octave
specificity, the shift re-slice contract, content-cache invalidation, checkpoint
versions, ONNX parity, and a synthetic prepare/train/evaluate/predict run.
They validate pipeline correctness, not real-world musical-key accuracy.

Code: MIT. Dataset provenance and audio-use notes: [DATA.md](DATA.md).
