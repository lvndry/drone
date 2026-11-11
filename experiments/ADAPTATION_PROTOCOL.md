# One small-model data-adaptation pilot

Declared after the frozen Beatport-trained candidate scored 52.68% symmetric MIREX
on FMAKv2 development (control 33.27%), before any adaptation fitting.

Hypothesis: limited training-domain coverage contributes to the generalization
gap. Test additional labels before increasing model capacity or changing features.

- Preserve the original 1,000-row final holdout. No calibration/final scoring.
- Official FMA metadata from `mdeff/fma`'s linked `fma_metadata.zip`, upstream SHA-1
  `f0df49ffe5f2a6008d7dc83c6915b31835dfe733`, supplies primary artist and album IDs.
- Form connected components across exact audio hashes, primary artist IDs and
  positive album IDs. Quarantine development components shared with the final
  holdout and rows without a valid primary artist ID. The audit leaves 1,385
  eligible development rows and quarantines 3,104. All 5,489 have artist metadata.
- Assign eligible components by SHA-256 of
  `fmak-artist-album-audio-purged-70-20-10-hash-v1:<component>`: first 70% of the
  hash range fits, next 20% validates, remaining 10% reserves calibration.
  Component hashing is label-independent; actual row counts need not be 70/20/10.
- Deduplicate fitting audio; reject conflicting labels on identical eligible audio.
  Join only FMAK fitting rows with the clean 1,230 Beatport fitting recordings.
  Combine FMAK validation with the existing 126 Beatport validation recordings.
  Dataset-prefix identities. Assert exact audio and group separation.
- Fixed model: 347,064 parameters, frequency head, LR 0.0003, dropout 0.3,
  AdamW weight decay 0.0001, smoothing 0.1, bounded semitone augmentation,
  120 epochs, batch 64, seed 13, CPU 16 threads. Start from scratch; no fine-tuning.
- Select checkpoint by combined validation per-track symmetric MIREX; ties keep
  the earlier epoch. Report each domain separately against the frozen candidate.
- One recipe, one initial seed, ten-minute run cap. Qualify for later confirmation
  only with FMAK validation gain >=2 points and Beatport validation loss <=2 points.
  No GiantSteps rerun for the adapted model. Native Euphoria reference evaluation
  is independent of this neural recipe and remains in flight.

This is development research. Known primary artist/album groups do not establish
remix/cover independence, and Beatport lacks matched artist metadata. Report these
limitations; do not claim a new cross-artist or SOTA benchmark from this pilot.
