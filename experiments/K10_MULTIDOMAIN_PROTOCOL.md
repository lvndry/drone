# One controlled K10 multi-domain comparison

Declared before extraction or fitting. Production K10 is the release candidate;
the earlier DSP residual failed its development gate. Test additional domain
labels without changing the backbone, optimizer recipe or model size.

## Audit and fixed identities

The shipped seed-13 run used 1,363 Beatport rows, 1,356 unique audio hashes,
1,237 fitting rows and 126 validation rows. Seven duplicate groups exist and two
audio hashes cross fitting/validation. This affects development checkpoint
selection; it does not demonstrate contamination of GiantSteps. Historical scores
remain historical observations, not results from the corrected protocol.

`train.evaluate_tracks` also padded mixed-length tracks, despite describing
unpadded inference. This can change predictions with batch composition. A failing
regression test demonstrates it. Group equal-length arrays within each bounded
batch, run at actual length, and restore original order. Train/test augmentation
and the feature spec stay unchanged. Compare the shipped checkpoint's old and
corrected development predictions; do not tune against GiantSteps.

Use the already fixed 2,582-row octave development manifest at
`~/drone-experiments/octave-pilot-20261111-v1/data/manifest.json`, SHA-256
`5f995f516430332a37f591ec299a36f9a46c16b2cd01cd887700837fd600bac9`.
It contains Beatport 1,230 fitting / 126 validation and artist/album/audio-purged
FMAK 976 fitting / 250 validation rows. Preserve all IDs, labels and roles; verify
content hashes and no audio overlap. Never extract calibration, final, SWD or
private-library audio in this experiment. FMAK roles retain the previous group
audit; this experiment does not create a new pristine test set.

## Two arms, one difference

- Control: fit the 1,230 Beatport rows only.
- Candidate: fit those same rows plus the fixed 976 FMAK fitting rows.
- Both use the same 376 validation rows and select the earliest checkpoint with
  best combined per-track symmetric-fifth MIREX. Also report each domain's exact,
  symmetric/asymmetric MIREX and mode accuracy. This comparison changes training
  data only; the control also uses FMAK validation for checkpoint selection.
- K10 width 1.0, 109,128 parameters, original librosa 24-bin/octave CQT (208
  padded / 192 core bins), hop 4096, log1p magnitude and float16 cache. Reuse only
  matching-profile valid cached arrays. Record dependency/source/cache identities.
- Random 320-frame crop, uniform shift -4..4 semitones, label minus shift.
  Seed 13, fresh initialization per arm, FP32 AdamW LR 0.001, decay 0.0001,
  plain cross-entropy, batch 32, cosine horizon 120, patience 25.
- Epoch-boundary records/checkpoints; incomplete runs never pass a gate.
  Feature preparation cap 20 minutes with eight CPU workers; each fit cap
  20 minutes. Maximum pilot stage 65 minutes including diagnostic scoring.
- GPU 0 only, >=8 GiB free at start, allocator fraction 0.04, four CPU threads
  in training; preserve both vLLM instances. No larger-model sweep.

## Decision

The joint seed-13 candidate qualifies only if FMAK validation MIREX improves by
at least 2 points versus the corrected fresh control and Beatport loses no more
than 2 points. Both arms must complete. A qualifying result earns a paired
confirmation of both arms with seeds 37 and 73, fixed recipe and validation IDs.
Require mean FMAK gain >=2 points and mean Beatport loss <=2 points. Repeating
seed 13 is a reproducibility check, not an independent seed.

Confirmation is bounded to 85 minutes for four fits (two remaining seeds times
two arms), retaining the 20-minute cap per fit. If the mean gate passes, retain
the canonical joint seed-13 checkpoint for later freezing; do not pick the best
confirmation seed or introduce an ensemble after seeing their results.

No default weight replacement follows development results. Freeze any qualifying
candidate before an independently audited external protocol. Prior GiantSteps
and FMAK final scores are exposed; additional scores there are diagnostics, not
independent final proof. SWD's 48 canonical recordings / 24 compositions provide
a classical stress test only; alternate WAVs must not be counted as new tracks.
