---
license: mit
library_name: pytorch
pipeline_tag: audio-classification
tags:
  - audio
  - music
  - key-detection
  - music-information-retrieval
  - dj
  - constant-q-transform
  - cnn
  - onnx
  - safetensors
model-index:
  - name: drone-v2
    results:
      - task:
          name: Musical key detection
          type: audio-classification
        dataset:
          name: GiantSteps Key with GiantSteps+ expert labels, single-key tracks (457)
          type: giantsteps-plus
        metrics:
          - name: Exact accuracy
            type: accuracy
            value: 87.96
            verified: false
      - task:
          name: Musical key detection
          type: audio-classification
        dataset:
          name: GiantSteps Key, original labels (604)
          type: giantsteps-key
        metrics:
          - name: Exact accuracy
            type: accuracy
            value: 69.87
            verified: false
          - name: MIREX weighted score
            type: other
            value: 75.45
            verified: false
---

# drone-v2: musical key detection on a CPU

drone-v2 tells you the musical key of a song: one of 24 keys, the 12 tonics
(C, C#, D ... B) each in major or minor. It is a 109,128-parameter
convolutional network that reads a constant-Q spectrogram of the whole track
and runs in about 60 ms on one CPU core for a two-minute track. It was built for DJ software
([Euphoria](https://github.com/lvndry/euphoria)) and works anywhere you can
run PyTorch, ONNX Runtime, or a few hundred lines of native code.

| | |
|---|---|
| Output | One of 24 keys, with a softmax score for each |
| Size | 109,128 parameters; 0.44 MB of float32 weights |
| Speed | About 0.3 s per two-minute track on one core of an Apple M5, of which 57 ms is the network |
| Source | [github.com/lvndry/drone](https://github.com/lvndry/drone) (MIT) |
| Previous version | `drone-v1`, still in this repository |

## How accurate is it?

All detectors below were run on the same 604 GiantSteps previews (two-minute
excerpts of electronic dance music) and scored against the same labels.
**Exact** means the predicted key equals the reference key.

| Detector | Original GiantSteps labels (604) | Expert labels (571) | Expert labels, single-key tracks (457) |
|---|---:|---:|---:|
| libkeyfinder 2.2.8 | 60.4% | 69.9% | 76.2% |
| madmom KEY_CNN | 67.9% | 77.4% | 83.4% |
| drone-v1 | 68.5% | 77.4% | 83.4% |
| **drone-v2** | **69.9%** | **80.7%** | **88.0%** |

Why three columns? The original GiantSteps labels agree with the
[GiantSteps+ expert re-annotation](https://zenodo.org/records/4153506) on only
82% of tracks, so part of every detector's "error" on the original labels is
label error. The single-key column keeps only tracks the experts marked with
one key throughout, the cleanest reference available.

On the expert single-key tracks, drone-v2 is ahead of libkeyfinder by 11.8
points (paired 95% interval 8.1 to 15.5) and of drone-v1 by 4.6 points
(2.0 to 7.4). On the half of those tracks never used for any development
decision, the gaps are 12.4 and 6.2 points.

Outside dance music, on 500 tracks of mixed genres from the Free Music Archive
([FMAKv2](https://zenodo.org/records/12759100)), drone-v2 scores 75.0% exact
against drone-v1's 65.4%: 72.4% and 77.6% on the two 250-track panels, a gain
of 9 to 10 points on each.

### What these numbers do not say

- **Not compared with Mixed In Key.** Published Mixed In Key results (85 to
  89% exact) come from other tracks, labels and scoring, so they are not
  comparable with this table.
- **GiantSteps is not a blind test for this model family.** It was used while
  developing Drone, though never to train it. The 604 previews share no track
  with the training set.
- **On the original GiantSteps labels**, a larger published model (KeyMyna,
  22M parameters) reports 72.0% exact, above drone-v2's 69.9%.
- **Previews, not full mixes.** GiantSteps tracks are two-minute excerpts and
  most training audio is 30-second clips or previews. Full-length tracks work
  (the network averages over time), but accuracy on 6 to 10 minute club tracks
  has not been measured separately.
- **One key per track.** A track that changes key gets the key the network
  hears most.
- **The scores are not probabilities.** The softmax output ranks keys; a score
  of 0.8 does not mean the key is right 80% of the time.

## Use it

### Python

In a project managed by [uv](https://docs.astral.sh/uv/):

```bash
uv add "drone-key @ git+https://github.com/lvndry/drone"
```

With pip:

```bash
pip install "drone-key @ git+https://github.com/lvndry/drone"
```

```python
from drone import Predictor

predictor = Predictor()  # loads drone-v2; reuse it for many files
tonic, mode, score = predictor.predict("track.mp3")
print(tonic, mode)  # e.g. "A minor"
```

`ffmpeg` must be installed: the package decodes audio with it. From a shell:
`python -m drone track.mp3`.

### Any language, with ONNX

`drone-v2.onnx` takes the spectrogram and returns 24 logits. You compute the
spectrogram yourself; this is the exact recipe, checked against the Python
package on 20 of 20 test tracks:

```python
import librosa
import numpy as np
import onnxruntime as ort

TONICS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

samples, _ = librosa.load("track.mp3", sr=22050, mono=True)
lowest = librosa.note_to_hz("C1") * 2 ** (-4 / 12)  # four semitones below C1
transform = librosa.cqt(
    samples, sr=22050, hop_length=4096, fmin=lowest, n_bins=208, bins_per_octave=24
)
spectrogram = np.log1p(np.abs(transform))
features = spectrogram[8:200].astype(np.float32)[None]  # (1, 192, frames)

session = ort.InferenceSession("drone-v2.onnx")
logits = session.run(["logits"], {"features": features})[0][0]
key_class = int(logits.argmax())
print(TONICS[key_class // 2], "minor" if key_class % 2 else "major")
```

### The contract, for a native port

| Step | Specification |
|---|---|
| Audio | Mono (average the channels), 22,050 Hz |
| Spectrogram | Constant-Q transform as in `librosa.cqt` 1.0: 208 bins, 24 per octave, lowest bin four semitones below C1 (25.9565 Hz), hop 4,096 samples, default Hann window and filter scale |
| Values | `log1p` of the magnitude, float32 |
| Network input | Rows 8 to 199 of the spectrogram, shape `(batch, 192, frames)`, at least 8 frames (about 1.5 s) |
| Output | 24 logits. Class `k` is tonic `TONICS[k // 2]`, minor when `k` is odd: 0 is C major, 1 C minor, 2 C# major ... 23 B minor |

`drone-v2.safetensors` holds every weight and BatchNorm statistic in float32
under its PyTorch name, for hosts without PyTorch or ONNX Runtime. The
architecture is three blocks of two 3x3 convolutions (16, 32, 64 channels), each
convolution followed by BatchNorm and ELU, each block ending in 2x2 max pooling;
then an average over time only, which keeps the 24 frequency rows, and one
linear layer from 64 x 24 features to 24 keys. See
[`drone/model.py`](https://github.com/lvndry/drone/blob/main/drone/model.py).
Euphoria's [Rust implementation](https://github.com/lvndry/euphoria) is a
complete example.

### Files

| File | For |
|---|---|
| `drone-v2.pt` | The Python package and PyTorch. Includes training provenance in its metadata |
| `drone-v2.onnx` | ONNX Runtime in any language. Opset 18, dynamic batch and length; matches PyTorch within 2e-5 |
| `drone-v2.json` | The ONNX file's contracts (features, labels, input layout) |
| `drone-v2.safetensors` | Native implementations |
| `drone-v1.*` | The previous release, unchanged |

## How it was trained

- **Data:** 6,263 labelled tracks: 4,988 from FMAKv2 (Free Music Archive tracks
  of many genres with key labels) and 1,275 Beatport dance-music previews from
  GiantSteps-MTG-Key. The training set shares no track ID with the 604
  GiantSteps Key previews.
- **Distillation:** three networks first trained on the labels alone were
  averaged into a "teacher". drone-v2 learned half from the human labels and
  half from the teacher's softer predictions, and also from the teacher's
  predictions on 24,827 unlabelled 30-second clips (FMA medium).
- **Pitch shifting:** every training excerpt was shifted by up to 12
  semitones either way, with its key label moved to match, so the network
  learns every key from every song.
- **Recipe:** AdamW (learning rate 1e-3, weight decay 1e-4), cosine schedule,
  120 epochs, batch 32, 60-second crops. Seed 42; the checkpoint from epoch 88
  was chosen on a held-back 10% of the training data.

The full study, including 13 experiments and everything that did not work
(larger networks, more unlabelled audio, other spectrogram views, label
cleaning), is in
[`experiments/studies/frontier-distillation`](https://github.com/lvndry/drone/blob/main/experiments/studies/frontier-distillation/REPORT.md).
The release evaluation is in
[`experiments/studies/drone-v2-release`](https://github.com/lvndry/drone/blob/main/experiments/studies/drone-v2-release/REPORT.md).

## License

The weights and code are MIT-licensed. The training audio and labels keep their
own terms: Free Music Archive audio under each track's Creative Commons
licence, FMAKv2 labels under CC BY 4.0, GiantSteps+ annotations under
CC BY-SA 4.0, and GiantSteps-MTG-Key, whose publisher states no licence. The
K10 architecture follows the MIT-licensed MIREX 2026 submission.
