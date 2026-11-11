# drone

A small neural classifier for the musical key of an audio track, intended for
[Euphoria](https://github.com/lvndry/euphoria). It predicts 24 classes: 12 tonics,
each in major or minor.

**Status:** corrected training/inference pipeline and three experimental model
sizes. No pretrained weights or verified accuracy results ship yet. Earlier
features and unversioned checkpoints are incompatible and must be regenerated.
Larger profiles are capacity experiments, not demonstrated accuracy upgrades.

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
uv run train.py train --model-size small --device cpu --threads 1
uv run train.py evaluate data/runs/<run>/drone.pt --output data/results.json
```

`fetch.sh` requires `curl`. It verifies Beatport archives against the checksums
published on Zenodo, pins GiantSteps annotations to a repository revision, and
verifies each test preview against its upstream MD5. Downloads are dev-time only;
the roughly 2.1 GB training archive plus test audio are not packaged. Use
`bash fetch.sh --dataset train` or `--dataset test` for one dataset, and `--data`
or `DRONE_DATA` to choose a data directory. See [DATA.md](DATA.md).

The initial sizes share one architecture and feature contract:

| Profile | Parameters | Approximate FP32 parameter storage | Intended experiment |
|---|---:|---:|---|
| small | 331,704 | 1.27 MiB | Default CPU candidate |
| medium | 1,318,744 | 5.03 MiB | Capacity/latency comparison |
| large | 5,258,904 | 20.06 MiB | Optional GPU/teacher candidate |

Storage excludes metadata, buffers, and the inference runtime. GPU speed and
accuracy gains have not been established. Select CUDA explicitly for GPU runs:

```bash
uv run train.py train --model-size large --device cuda --batch-size 64
```

Training uses cross-entropy, AdamW, cosine decay, and a fixed seeded stratified
validation split. Changing the training seed leaves the split fixed. Learning
rate, dropout, weight decay, and label smoothing are configurable; their defaults
are a baseline, not an optimized recipe. Validation alone selects checkpoints;
the training command does not score the held-out test set. ID/content overlap
checks reject training/test leakage. Evaluation does not need training features.

Every run writes `run.json` with source hashes, Git revision, dependency-lock hash,
data manifests, split IDs, configuration, per-epoch metrics, timing, and checkpoint
hash. Checkpoints include the model configuration, label schema, feature spec,
and time resolution. Evaluation can write per-track predictions and both symmetric
and asymmetric MIREX scores. The default symmetric convention awards fifths in
both directions, matching Euphoria's benchmark; compare like scoring conventions.

## Features

Audio is decoded to mono 22.05 kHz PCM. A 4096-point Hann STFT with 512-sample hop
is grouped into 33 semitone bands from A1 (55 Hz) through F4 (about 349 Hz).
This is an STFT-derived log-frequency representation, **not a constant-Q transform**.
Energy is log-compressed, time is independently interpolated per band to 300 frames,
and each track is z-scored. The resulting model input is `(batch, 33, 300)`.
Resampling time to 300 frames means the final time step depends on track duration.

Training translations are sampled uniformly from -6 through +5 semitones. The
same shift changes the label and frequency bands; out-of-range bands are discarded
and padded at the feature floor before renormalization. This is an approximation
of pitch shifting, not artifact-free audio transposition. Pitch shifts preserve
major/minor mode and therefore do not rebalance mode counts.

## Inference

Supply a compatible trained checkpoint:

```bash
uv run python -m drone track.mp3 --weights data/runs/<run>/drone.pt --threads 1
```

For multiple files, load once and reuse the predictor:

```python
from drone import Predictor

predictor = Predictor("data/runs/<run>/drone.pt", device="cpu")
print(predictor.predict("first.mp3"))  # tonic, mode, softmax probability
print(predictor.predict("second.mp3"))
```

`drone.predict(path, weights=...)` is also available for a single file.
`DRONE_WEIGHTS` overrides the default artifact location; otherwise drone checks
packaged `drone/weights/drone-v1.pt`, then the checkout's `weights/drone-v1.pt`.
No model is downloaded automatically. A missing or incompatible artifact produces
an actionable error. Softmax probabilities are not calibrated accuracy estimates.
Audio feature caches are keyed by file contents and feature specification.

## Export and CPU measurement

```bash
uv run --extra export python -m drone.export data/runs/<run>/drone.pt data/drone-small.onnx
uv run --extra export python -m drone.benchmark data/runs/<run>/drone.pt \
  --onnx data/drone-small.onnx --audio track.mp3 --threads 1 --output data/cpu.json
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

Tests cover all key encodings and relationships, synthetic-tone frequency mapping,
time interpolation, augmentation, content-cache invalidation, checkpoint versions,
all model sizes, ONNX parity, and a synthetic prepare/train/evaluate/predict run.
They validate pipeline correctness, not real-world musical-key accuracy.

Code: MIT. Dataset provenance and audio-use notes: [DATA.md](DATA.md).
