# Myna representation: qualification before the accuracy pilot

The pretrained Myna encoder, a playback-rate augmentation and the complete-track
feature extraction path all passed their numerical checks, so one controlled
accuracy pilot can proceed. Three length-preserving pitch-shift methods failed
their synthetic checks and are not used. No key-detection accuracy has been
measured yet. Study date: November 13, 2026.

## Question

Can a pretrained music representation improve key classification beyond Drone's
compact model? This study prepares that test with the released 21.4-million-
parameter [Myna encoder](https://huggingface.co/oriyonay/myna-vertical), kept
frozen while a separate key head is trained. Before spending compute on embedding
extraction or training, it checks four things:

1. the pinned encoder loads strictly and batching does not change its output;
2. a bounded-memory frontend reproduces the full-track mel frames;
3. a pitch or rate augmentation changes pitch correctly without distorting the
   signal;
4. the cost of extracting features for the full corpus and training the head.

This is a research candidate. It does not replace Drone's released model, and
none of the GPU measurements here say anything about inference on an ordinary PC.

## Method

### Model and environment

The key head follows the published [KeyMyna design](https://arxiv.org/html/2604.10021v1):
384 input features, two 4,096-unit hidden layers with ReLU, dropout 0.99 after
the first hidden layer, and 24 output logits. Its 18.5 million parameters make
the complete research system much larger than the compact product model. We use
our own fitting/selection protocol and head implementation. The Myna pretraining
corpus has not been fingerprint-audited against our evaluation recordings, so
possible overlap is unknown.

The environment uses Python 3.12 with an exact uv lock. Torch and Torchaudio are
both 2.11.0 CUDA 13.0 builds, a pair documented in the
[official installation matrix](https://pytorch.org/get-started/previous-versions/).
Every wheel was size- and SHA-256-checked and installed offline, and the
numerical checks re-verify installed files before importing the ML runtime.
Version compatibility alone does not prove the kernels work; the numerical
checks below are what establish that.

### Input and numerical contract

- Input is finite mono float32 PCM at 24 kHz, decoded by the existing FFmpeg
  preparation pipeline. Stereo is downmixed by the arithmetic channel mean.
- Torchaudio resamples to 16 kHz with `sinc_interp_hann`, filter width 6 and
  rolloff 0.99. This is not claimed to match the upstream file loader.
- The frontend is the pinned upstream nnAudio mel transform: 128 bands,
  2048-point FFT, 512-sample hop, centered reflection padding, power 2, with no
  added logarithm or normalization.
- All complete 196-frame windows are kept; a partial final window is dropped,
  as in the upstream code. There is no cap on window count.
- The frontend runs on CPU; the encoder runs in FP32 on GPU with TF32 disabled.
  Batch sizes 1, 8 and 16 receive identical windows.
- Tolerances are 1e-5 absolute and 1e-4 relative, fixed in advance. A failure
  is kept as a failure; tolerances are not relaxed afterwards.

The blocked frontend gives each frame its complete 1024-sample context and keeps
reflection padding at the true recording boundaries. Unit tests check frame
centers, short-input rejection and final-block handling, and the qualification
additionally compares full and blocked tensors on real inputs.

No reference keys, development rankings, fresh evaluation tracks or private DJ
audio were used in any of these checks. Resource use was capped (600 s wall,
1,200 CPU seconds, 4 GiB host memory, 2 GiB Torch GPU memory, 64 MiB output) on
a shared GPU server. These are sampled limits, not hard isolation.

## Results

### Encoder and frontend

The first numerical check passed in 14.4 seconds. The encoder loaded strictly.
On a ten-minute synthetic waveform, full and blocked mel tensors were exactly
equal. On a real fitting-set recording they differed by at most 0.000916
absolute, within the combined absolute-plus-relative tolerance. All batch-size
comparisons passed; the largest error was 0.069 of its allowed tolerance.

| Resource (first check) | Measurement |
|---|---:|
| Sampled host memory (monitor plus worker) | 1,594 MiB |
| Torch GPU allocated peak | 540 MiB |
| Torch GPU reserved peak | 568 MiB |
| Sampled whole-process GPU memory | 1,216 MiB |
| Minimum sampled free GPU memory on the device | 8,885 MiB |

The 120-second fitting recording produced 19 complete windows. Forward calls
took 68.3 ms at batch 1, 11.7 ms at batch 8 and 7.87 ms at batch 16. These are
not clean batching speedups: batch 1 ran first and absorbed cold-start effects.
Three synthetic head updates took 155 ms. Environment setup took 41.3 seconds
and downloaded 2.77 GB of wheels.

### Pitch augmentation: three length-preserving methods failed

The KeyMyna paper trains with pitch-shift augmentation. We tested three
length-preserving pitch shifters on synthetic tones and chords over 13 shifts
(−6 to +6 semitones), with a pre-set threshold of five cents pitch error per
component and preserved amplitude.

| Method | Outcome |
|---|---|
| Phase vocoder plus polyphase resampling | Failed amplitude: at +3 semitones a 440 Hz tone kept only 32.4% of its input RMS. Negative shifts kept amplitude, so the distortion depends on direction. |
| FFmpeg RubberBand R2 engine, quality setting | Preserved amplitude and sample counts but failed the five-cent check on every nonzero shift of the bass fixtures. |
| RubberBand offline R3 engine (C API, study pass, single thread) | Every individual tone passed (worst error 0.089 cents), amplitudes and sample counts passed, but local peaks in the bass chord differed by up to 16.1 cents, so the five-cent chord threshold failed. |

![Synthetic pitch augmentation diagnostics](figures/pitch_diagnostics.png)

Each panel plots one check against the requested shift; dashed lines are the
pre-set bounds. Look at where each method's line crosses its bound. The chord
panel measures local FFT peaks, not key accuracy. Data:
[pitch_diagnostics.csv](data/pitch_diagnostics.csv).

A reviewer questioned whether local peak interpolation fairly tests whether a
chord's notes survive. We therefore fixed a second, exploratory energy test
before running it (motivated by the R3 failure, so it is not an independent
confirmation). In the central ten seconds of each twelve-second A-major and
A-minor chord at all 13 shifts, at least 90% of spectral power had to fall inside
±50-cent bands around the three expected notes, and at least 20% inside each
band. Analytic correct chords had to pass; a chord shifted one extra semitone and
a chord with its middle note removed had to fail.

All 26 control sets behaved as intended. All 24 nonzero-shift R3 cases failed:
only 14.5–47.3% of their power lay in the expected bands. Both unshifted cases
passed. The R3 branch was stopped with no further threshold or setting search.

![Prospective pitch-energy diagnostic](figures/pitch_energy_v2.png)

The left panel shows the fraction of power inside all expected bands (threshold
0.90), the right panel the weakest single band (threshold 0.20); R3 falls below
both for every nonzero shift. Data: [pitch_energy_v2.csv](data/pitch_energy_v2.csv).

### Playback-rate augmentation passed

The adopted augmentation changes playback rate with
[polyphase resampling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html).
Like changing a turntable's speed, it shifts pitch and tempo together, so it is a
different intervention from the paper's length-preserving shift. There is no gain
normalization or cropping; every complete window of the resulting recording is
kept. Evaluation always uses the original audio.

On a 24-second synthetic set, all 78 fixture/shift cases passed: worst tone error
0.153 cents, RMS ratios 0.9987–1.0013, and every expected chord note kept at least
33.3319% of total spectral power. All 26 control sets behaved as intended.

### Complete recordings

Two fitting recordings, chosen before any model ran and without reference keys:

| Corpus and record | Original duration | Views tested |
|---|---:|---|
| Beatport `100066` | 120 seconds | All 13 shifts |
| FMAK `082660` | 35.9 minutes | Original and ±6 semitones |

At the slowest rate the long recording became 50.8 minutes with 485 complete
windows (343 at the original rate). Full and blocked frontends passed the
numerical bound in every view, and the encoder embeddings were bitwise equal in
all 16 real-recording views. This shows the code computes what it should; it says
nothing about whether predicted keys are correct.

| Workload | Output duration | Windows | Rate change | Blocked frontend | GPU encoder |
|---|---:|---:|---:|---:|---:|
| Two-minute record, all 13 views combined | 26.6 minutes | 251 | 0.430 s | 3.292 s | 0.090 s |
| Long record, original | 35.9 minutes | 343 | 0.125 s (identity copy and check) | 4.393 s | 0.095 s |
| Long record, −6 semitones | 50.8 minutes | 485 | 1.244 s | 6.206 s | 0.134 s |
| Long record, +6 semitones | 25.4 minutes | 242 | 1.023 s | 3.156 s | 0.067 s |

Measured on a shared server with one numerical CPU thread and an RTX 5090
encoder, after warming the actual batch shapes. Loading, hashing, base
resampling and cache writing are excluded. The CPU frontend dominates; a faster
encoder alone would recover little.

The head training loop (weighted cross-entropy, finite checks, non-fused Adam,
the same functions the trainer uses) ran 100 updates of batch 64 in 0.224 s,
447 updates per second, after ten warm-up updates, on synthetic features.
Evaluation grouped synthetic features into the actual 376 per-song window
counts: batch 64 took 0.183 s, batch 1 took 3.135 s, and the pooled logits agreed
within tolerance.

The whole check took 58.7 seconds. Worker peak host memory was 3.60 GiB; the
monitor's sampled peak was 2.73 GiB because sampling missed a short allocation.
Torch allocated/reserved peaks were 620/680 MiB, against 1,328 MiB sampled
whole-process GPU memory (the difference is CUDA context and libraries). The
run included full-matrix reference computations, so it overstates the memory a
streaming-only extractor would need.

![Measured extraction stages and memory counters](figures/runtime_qualification.png)

Left: per-workload time split into playback-rate change, CPU mel frontend and
GPU encoder; the frontend is most of every bar. Right: the host and GPU memory
counters disagree, which is why each is reported separately. Data:
[runtime-qualification.json](data/runtime-qualification.json) and
[runtime-resources.json](data/runtime-resources.json).

### A resampler length edge case

The pinned Torchaudio resampler converts its output length to float32 before
taking a ceiling, which can round down on long inputs. A metadata-only check
found 20 affected recordings: nineteen lose one sample and the longest loses two.
Across all augmented views 216 sample counts change, but no mel-frame or
complete-window count changes. The cache metadata records the library's actual
behavior; we did not patch the library or pad audio.

## Decision

Proceed to one accuracy pilot with the frozen Myna encoder, the KeyMyna-style
head and playback-rate augmentation. The length-preserving pitch shifters are
not used. The [pilot protocol](PROTOCOL.md) fixes the objective, controls and
decision rule.

The planned cache holds 2,152,026 fitting vectors (all 13 views of every fitting
track, every complete window) plus 10,892 original-only selection vectors:
3.32 GB of float32 before index overhead. Per-view window counts enter the loss
weights so long recordings do not dominate. Measured per-duration costs project
about 7.9 CPU-hours for the frontend and 2.0 CPU-hours for rate changes, an ideal
2.5 hours on four workers plus about twelve minutes of encoder time. These are
forecasts; corpus-scale parallel scaling, hashing and cache writing are not
measured.

Before any accuracy was seen, the pilot fixed 100 complete passes, about 3.36
million optimizer updates, predicted at 2.1 hours of head computation plus
evaluation and checkpointing.

The first corpus extraction stopped safely after 275 recordings when another job
on the shared GPU reduced free memory below the protected reserve. That partial
cache cannot be used for training. A single fresh extraction will run when
enough GPU memory is free, and head training may start only from a completely
validated cache. This interruption says nothing about model quality.

## Limitations

- No accuracy has been measured. Nothing here supports a release, replacement or
  distillation decision.
- All augmentation checks use synthetic tones and chords with engineering
  thresholds, not perceptual standards or measured key errors on songs. The R2
  and R3 failures apply to the tested configurations only.
- Playback-rate augmentation changes tempo as well as pitch and is not the
  augmentation used in the KeyMyna paper.
- The two real recordings test numerical execution only. Neither supports a
  genre- or length-specific conclusion; no verified genre or BPM was used.
- Overlap between Myna's pretraining data and our evaluation recordings is
  unaudited.
- Timings come from a shared server GPU and do not qualify ordinary-PC inference.

## Reproduce

The figures are rendered from the recorded measurements, without running audio
models:

```sh
uv run --extra plots python -m tools.plot_myna_qualification
uv run pytest -q tests/test_myna_qualification.py tests/test_myna_pilot.py tests/test_myna_extract.py
```

The [rendering script](../../../tools/plot_myna_qualification.py) reads the raw
diagnostics in [data/](data/).
