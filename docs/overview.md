# Drone: an overview

This guide explains what Drone is, how it works, what was learned building it,
and how to find your way around the repository. It is written for someone new to
the project; the detailed evidence lives in the reports it links to. Last
updated on 14 November 2026, when `drone-v2` shipped.

## What Drone is

Drone tells you the musical key of a song: one of 24 keys, the 12 tonics (C, C#,
D ... B), each in major or minor. It is a small neural network, 109,128
parameters, that runs on an ordinary CPU in a fraction of a second per track.

It exists for [Euphoria](https://github.com/lvndry/euphoria), a DJ library tool,
which runs it in Rust to decide the key of every track. DJs use keys for
harmonic mixing: two tracks in compatible keys blend without clashing. Drone is
also a standalone Python package, a command-line tool and a published model
([huggingface.co/lvndry/drone](https://huggingface.co/lvndry/drone)) that any
project can use.

The current release is `drone-v2`. The previous one, `drone-v1`, stays in
`weights/`.

## How it works, in four steps

1. **Decode.** The audio is converted to mono at 22,050 samples per second.
2. **Spectrogram.** A constant-Q transform (CQT) turns the sound into a picture
   of which pitches are present over time: 24 rows per octave, eight octaves from
   the note C1, one column every 0.19 seconds. Unlike an ordinary spectrogram, its
   rows are spaced like piano keys, so moving a song up a semitone moves the
   picture up exactly two rows. Values are compressed with `log1p`.
3. **Network.** Three blocks of small 3x3 convolutions (16, 32, 64 channels)
   read the picture. At the end the network averages over *time only*, keeping
   the frequency axis, because a key is defined by which absolute pitches are
   used. (Averaging over frequency too made the network blind to transposition
   and stuck at chance: an early lesson.)
4. **Answer.** A final layer scores all 24 keys; the highest wins. Class `k` is
   tonic `k // 2`, minor when `k` is odd. The scores are not calibrated
   probabilities, but the gap between the best two is a useful confidence signal.

The architecture is not ours: it is K10, published for the MIREX 2026 evaluation
under the MIT licence. What Drone added is the training recipe, the data
discipline and a CPU implementation.

## Where it stands

All detectors run on the same 604 GiantSteps previews (two-minute excerpts of
electronic dance music), exact key:

| Detector | Original labels (604) | Expert labels, single-key tracks (457) |
|---|---:|---:|
| libkeyfinder 2.2.8 (what Euphoria used before) | 60.4% | 76.2% |
| madmom KEY_CNN (strongest published compact model) | 67.9% | 83.4% |
| drone-v1 | 68.5% | 83.4% |
| **drone-v2** | **69.9%** | **88.0%** |

For a DJ, a key a fifth away still mixes. Counted that way, drone-v2 gives a
usable key on 90.6% of the cleanly labelled tracks, against 85.6% for
libkeyfinder. On 309 afro house tracks from a real DJ library it agreed with
rekordbox within a fifth on 95.5% of tracks, and on 99% of the tracks it rated
high confidence.

Outside dance music, on 500 mixed-genre Free Music Archive tracks, drone-v2 is
exact on 75.0% against drone-v1's 65.4%.

What is not measured: full-length club tracks separately from previews, most DJ
genres beyond electronic music (hip-hop, R&B, reggae, bouyon), and any direct
comparison with Mixed In Key, whose published figures use other tracks and
labels.

## The story so far

All of this happened between 11 and 14 November 2026.

1. **Compact models.** A first 47,284-parameter CNN learned that pooling over
   time only is essential (about 22 points of MIREX score better than pooling
   everything), but it did not clearly beat libkeyfinder.
2. **drone-v1.** Porting the published K10 architecture and training it on 1,363
   Beatport dance-music previews gave the first release: 68.5% exact on
   GiantSteps against libkeyfinder's 60.6%.
3. **A dozen tweaks, mostly rejected.** Weight averaging, sharpness-aware
   training, soft targets that reward near misses, more capacity, a
   pitch-circle architecture, a tree model on tonal features, temporal heads,
   ensembles as teachers, shorter crops and more fitting data each missed a
   threshold fixed in advance. (The two "more fitting data" pilots missed it by
   less than one track, so they do not show that more data fails to help.) Two large pretrained music models, MusicFM and
   Myna, did not make better teachers either.
4. **Data discipline.** The corpus inventory ([DATA.md](../DATA.md)) and audits
   of overlap between datasets, so test tracks are never in training, and every
   track's role (fitting, selection, confirmation) is recorded.
5. **CPU and native code.** The network and the CQT were reimplemented in Rust
   and matched the Python reference bit for bit on 376 tracks; computing long
   tracks in chunks cut peak memory from 865 to 129 MiB without changing a
   single key.
6. **The frontier study.** Fourteen experiments, judged on expert labels with
   paired statistics, found the recipe behind drone-v2 and the point where
   further work stops paying.
7. **drone-v2.** Released in this repository, on Hugging Face, and inside
   Euphoria, where it replaced libkeyfinder as the key detector.

## What we learned

**The labels matter as much as the model.** The original GiantSteps labels
agree with a later expert re-annotation on only 82% of tracks. Scored against
expert labels, every detector gains 15 to 18 points, and the gaps between them
widen. Benchmark keys are listeners' judgements, not producers' statements.

**Training recipe beat architecture.** drone-v2 has exactly drone-v1's network.
The gain came from two training changes: distillation, where the network learns
half from the human label and half from the averaged predictions of three
earlier networks (including on unlabelled audio), and pitch-shift augmentation of
up to 12 semitones, so every song is heard in every key. Wider networks, longer
crops and more unlabelled audio did not help.

**Most ideas do not survive a fair test.** Every comparison here uses three
training seeds, compares the same tracks before and after, puts a 95% interval on
the difference, chooses on development data and keeps a test half that decides
nothing. Under those rules, most plausible improvements were noise.

**There is a ceiling, and part of it is the music.** From about 88% exact on
expert-labelled tracks, nothing moved the number: not more data, larger models,
other feature views, cleaner training labels, or a different model family. Two
very different models make largely the same mistakes, and about half of those
tracks show independent signs that their key is debatable (modal music,
microtonal material, two human sources disagreeing).

**Confidence is worth as much as accuracy.** For a DJ, knowing which keys to
double-check matters. drone-v2's margin between its best two keys sorts tracks
well: on held-out previews a key rated high is exact 84% of the time and one
rated low 26%.

**Small and native is achievable.** The whole model is 441 KB of weights. In
Euphoria it runs in plain Rust, with no inference library or download, at about
0.3 CPU seconds per two-minute track, and gives the same keys as the Python
package on all 604 benchmark tracks.

## How to explore the repository

### Read in this order

1. [README.md](../README.md): installing, predicting, training the drone-v1
   recipe, exporting.
2. [weights/MODEL_CARD.md](../weights/MODEL_CARD.md): what drone-v2 is, how
   accurate, and how to use it from any language.
3. [experiments/README.md](../experiments/README.md): the research index. Every
   study in one table with its question, result and decision.
4. [The frontier study](../experiments/studies/frontier-distillation/REPORT.md):
   how drone-v2 was found, including everything that failed.
5. [The drone-v2 release evaluation](../experiments/studies/drone-v2-release/REPORT.md):
   how the checkpoint was chosen and measured end to end.
6. [DATA.md](../DATA.md): which datasets exist, what each covers, and which
   recordings were used for what.

### The folders

| Folder | What it holds |
|---|---|
| `drone/` | The package: features (`features.py`), network (`model.py`), key labels and scoring (`keys.py`), checkpoints, ONNX and safetensors export, CPU benchmark |
| `weights/` | Shipped models: `drone-v2` (default) and `drone-v1`, as PyTorch, ONNX and safetensors, plus the model card |
| `train.py` | Training and evaluation for the drone-v1 recipe on Beatport data, and fine-tuning on Euphoria's key corrections |
| `experiments/studies/` | One folder per study: `REPORT.md`, often a `PROTOCOL.md` fixed before results, `data/` with the measured evidence, `figures/`, and sometimes `scripts/` |
| `experiments/*.py` | The Python code behind the earlier studies, kept importable |
| `tools/` | Command-line research tools: dataset audits, benchmarks, exports, figure generation |
| `research/` | The native (Rust) CPU studies: cached-feature inference, the native CQT, and the composed pipeline |
| `tests/` | Unit tests, and checks that pin reports to their evidence files |
| `docs/` | The drone-v1 training write-up, a literature review of methods, the historical compact-model pipeline, and this guide |

### Where things are not

- **Audio and caches** are never committed; `fetch.sh` downloads the public
  datasets. The large research runs used a GPU server, and their scripts read
  data locations from environment variables.
- **drone-v2's training recipe** is not in `train.py`. It lives in the frontier
  study's [scripts](../experiments/studies/frontier-distillation/scripts/), run in
  the order of its `round*.sh` files.
- **Euphoria's Rust implementation** of drone-v2 lives in the Euphoria repository
  (`crates/euphoria-core/src/detect/drone/`), not here. The Rust code in
  `research/` is the earlier research version.

### Try it

```bash
uv sync --locked
uv run python -m drone track.mp3          # key of one file
uv run --all-extras pytest                # the test suite
```

## Open directions

The model work has reached diminishing returns on public benchmarks. The ideas
most likely to matter next are about data, not architecture:

- **Keys stated by producers** (sample packs, stem releases, label metadata) as a
  clean test set for DJ genres.
- **Euphoria's key corrections** as training labels from real libraries; the
  fine-tuning path already exists in `train.py`.
- **Real-library checks** against other detectors, genre by genre, like the
  afro house comparison above.
- A much larger teacher (SheetSage2) has the best published benchmark score, but
  its non-commercial licence may rule it out for a shipped model.
