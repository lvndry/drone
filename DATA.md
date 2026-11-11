# Training data provenance

The weights that ship with euphoria were trained on:

- **Beatport EDM Key Dataset** (1,486 two-minute previews), Zenodo DOI
  [10.5281/zenodo.1101082](https://doi.org/10.5281/zenodo.1101082).
  Annotations by Eduard Mas Marín, revised and expanded by Ángel Faraldo,
  CC BY-SA 4.0 as published. The audio files are preview clips served by
  Beatport for listening; they are a dev-time download only and are never
  shipped or redistributed with the weights.
- Pitch-shift augmentation (±semitones) of the same tracks at training time.

The test set, the **GiantSteps Key Dataset** (604 previews), is never used
for training or model selection. Its intersection with the training set is
empty.

Residual risk: the preview audio is not explicitly licensed for
redistribution, and the shipped artifact is a model trained on it, not the
audio itself. Should Beatport object, the fallback is to keep libkeyfinder
as the shipped detector and drop the trained weights.
