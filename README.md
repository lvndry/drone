# drone

A 0.33M-parameter classifier that answers one question: **what key is this track in?**

Given audio, it outputs a probability over the 24 keys (12 tonics × major/minor). It was built to power key detection in [euphoria](https://github.com/lvndry/euphoria) (whose [benchmark](https://github.com/lvndry/euphoria/blob/main/docs/concepts/accuracy.md) uses it against libkeyfinder, Mixed In Key and others), but it is a standalone model — anyone can use the weights.

```
audio ──► CQT (33 semitones × 300 frames) ──► CNN ──► 24-way softmax ──► "C minor"
```

## Why it exists

Template matching (the approach in libkeyfinder, keyfinder, Essentia) compares the spectral content against hand-built key profiles. It is transparent and fast, but it caps out: on the 604-track GiantSteps benchmark, libkeyfinder reaches 69.4% MIREX with tuning correction. A small neural classifier trained on labelled music learns the actual statistical signature of keys and does materially better. drone is that classifier, kept as small as possible so the weights ship in any app.

## Results

Held-out [GiantSteps Key Dataset](https://zenodo.org/records/1101082) (604 tracks, never seen in training):

| System | Exact | MIREX |
|---|---:|---:|
| libkeyfinder + tuning correction | 60.6% | 69.4% |
| Mixed In Key 8.3 (commercial) | 69.4% | 75.7% |
| Korzeniowski & Widmer 2017 (published) | ~67% | 74.6% |
| **drone** | **see below** | **see below** |

MIREX scoring: exact match = 1.0, relative major/minor = 0.3, parallel = 0.2, relative fifth = 0.5, other = 0 — the same weights euphoria's benchmark uses.

> Full numbers and the complete run record (config, seed, timings) land in `TRAINING.md` when the reference run finishes.

## The model

| | |
|---|---|
| Architecture | 3 Conv2d (32/64/128 filters) → global avg pool → dropout 0.3 → linear 24 |
| Parameters | 0.33M (≈1.3 MB in float32, ≈0.3 MB quantized) |
| Input | 33 × 300: a log-frequency spectrogram, one semitone per row (A1 = 55 Hz up to A7), one frame every 0.023 s, per-track z-scored |
| Output | 24 softmax classes: `tonic_index * 2 + mode` (0 = C major … 23 = B minor) |
| Inference | < 1 s per track on a laptop CPU; no GPU required |

The design follows [Korzeniowski & Widmer, "Estimating the Musical Key with Convolutional Neural Networks" (2017)](https://arxiv.org/abs/1706.02921), whose central finding — that a single small model trained on a few thousand tracks across genres matches genre-specialised ones — is what makes a 0.3M-parameter key classifier viable at all.

## How it was trained

### Data

- **Training:** 1,363 of the 1,486 tracks from the [Beatport EDM Key Dataset](https://zenodo.org/records/1101082) (CC BY-SA 4.0 annotations; 2-minute previews, 933 minor / 430 major labels, 25 of which are ambiguous "X | Y" labels resolved to the first option). The 129 tracks without a clean label are skipped.
- **Test (held out, never used for training or model selection):** all 604 tracks of the [GiantSteps Key Dataset](https://github.com/GiantSteps/giantsteps-key-dataset), whose labels are Beatport user corrections. Intersection with the training set: zero (verified by track ID).
- **Validation:** every 10th training track (~136), used only to pick the best epoch.

See [DATA.md](DATA.md) for the provenance and licensing position on the preview audio.

### Features

One STFT (4096-bin, 512-hop, Hann window) per track at 22.05 kHz. Every FFT bin is assigned to its nearest semitone and summed, giving the 33-semitone CQT; amplitudes are log-compressed (`log1p`) and the spectrogram is time-resampled to 300 frames. The semitone-rows representation is chosen so that **pitch shifting is a circular shift of rows** — see augmentation.

### Training loop

- PyTorch, cross-entropy, AdamW (lr 1e-3, weight decay 1e-4), cosine LR decay, 120 epochs, batch 64, seed 13.
- **Augmentation:** each training sample is pitch-shifted by a random amount of −6…+6 semitones by rolling the CQT rows, with the label rotated the same way. This is exact (no resampling artifacts) and it re-balances the class distribution: without it, minor keys would be trained ~2× as often as major.
- Checkpoints are saved when validation MIREX improves; the best is the shipped one.

### Compute

Trained on the CPU of a workstation (AMD Ryzen Threadripper PRO 9955WX, 16 cores / 32 threads, 188 GB RAM). The two GPUs on the box (RTX PRO 6000, RTX 5090) were occupied by unrelated vLLM serving and were deliberately **not** used — the model is small enough that a GPU buys almost nothing here. Total wall time: feature extraction ≈ 10 min, training ≈ a few hours on CPU. No GPU required to reproduce.

### Optimizations

- **CQT caching:** each track is decoded (ffmpeg) and transformed exactly once, then cached as a `.npy`; pitch-shift augmentation operates on the cached spectrogram, so retraining never touches the audio again.
- **Row-rolled augmentation:** pitch shift as a `np.roll` on 33 rows instead of resampling 1.7M audio samples per track.
- **Strided STFT via fancy indexing:** one gather instead of a sliding-window view, so the windowed frames are a writable array (and faster).
- **Small model by construction:** 330k parameters keep the weights under 1.5 MB, which is the whole point of shipping the model in a CLI, a desktop app, and (eventually) an HF hub page.

## Evaluation against the state of the art

The comparison set mirrors euphoria's accuracy page, so the numbers are directly comparable to the product's published benchmark:

| System | Class | Notes |
|---|---|---|
| libkeyfinder | open-source template matching | what euphoria ships today |
| MadKey / Essentia | open-source | template matching |
| Korzeniowski & Widmer 2017 | paper | CNN on 12-bin log-freq spectrogram, GiantSteps+Billboard training |
| MusicalKeyCNN (a1ex90) | open-source reimplementation | PyTorch, same datasets, ~73.5% MIREX published |
| Mixed In Key 8.3 | commercial | the benchmark's strongest baseline |
| MIREX 2026 submission | challenge SOTA | MusicFM encoder, 1.3 GB — not a shipping candidate |

drone's position: it targets the **Korzeniowski/Widmer / MusicalKeyCNN band** (≈73–75% MIREX on this benchmark) with a model small enough to embed, i.e. between template matching and the large-encoder frontier. The FMAK generalization check (5,489 songs, 17 genres) is the open question — tracked in euphoria issue #21.

## Reproducing

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # torch, numpy
bash fetch.sh                          # downloads the 2.1 GB training set + labels to data/
python train.py prepare                # decode + CQT features (cached)
python train.py train                  # 120 epochs, keeps best validation checkpoint
python train.py evaluate runs/<ckpt>   # reports held-out GiantSteps 604 exact/MIREX
```

The trained weights are committed to [`weights/`](weights/) and published on [Hugging Face](https://huggingface.co/lvndry/drone).

## Using it

```python
import drone

print(drone.predict("track.mp3"))  # -> ("C", "minor", confidence)
```

Inference is a single forward pass over one 33×300 tensor; no audio dependencies beyond decoding to mono PCM.

## License

Code: MIT. Weights: trained on the CC BY-SA 4.0 Beatport EDM Key Dataset annotations and the preview audio described in [DATA.md](DATA.md).
