# FMAKv2 protocol, declared before model scoring

The small-model recipe was selected using the fixed Beatport development split.
FMAKv2 audio preparation and its 100-file runtime pilot do not fit or evaluate a
model. No FMAK model scores have been observed when declaring this protocol.

## Partition

- Use the official 5,489-row FMAKv2 annotation file (Zenodo 12759100, MD5
  `3b2d16784ffbda850c8ddf0519478bfd`) and the original FMAK audio archives from
  Zenodo 10719860. Pin each archive's upstream MD5 and each extracted audio SHA-256.
- Exclude exact audio overlap with Beatport training/validation, if any, and report
  excluded IDs. IDs are dataset-scoped; matching numeric FMA/Beatport IDs alone is
  not evidence of matching audio.
- Group by exact audio SHA-256. Sort groups by the SHA-256 of
  `drone-fmak-holdout-v1:<audio-sha256>`, independent of labels and model outputs.
- Assign whole groups to the final holdout until exactly 1,000 rows are selected,
  skipping a group if it would exceed the remaining quota. Other rows form the
  development corpus (4,489 if no exclusions). Save all identities and fingerprints
  before running a model.
- Do not use the final 1,000 rows for model selection, calibration, pseudo-labels,
  training, or representation/optimizer decisions. Initially, evaluate the frozen
  Beatport-trained candidate/control only on FMAK development for diagnosis.
- Artist/remix grouping is not established by exact hashes. Record that limitation
  until trustworthy artist/album metadata and near-duplicate checks are available.
  Keep duplicate-audio groups together even if their supplied labels disagree;
  report such disagreement rather than silently relabeling the corpus.

Future training on FMAK is limited to the development corpus and requires a
separate saved train/validation/calibration split. Preserve the final 1,000 rows.
A model evaluated under this partition is not directly comparable to papers
reporting results over all 5,489 tracks or a different 1,000-track partition.

## Reporting and promotion

Report exact accuracy, symmetric/asymmetric MIREX, mode accuracy, confusion and
per-key support, dataset coverage and provenance. Model/feature revisions must be
frozen before a final-holdout evaluation. Use a larger model only after a measured
small-model generalization failure justifies a specific, budgeted capacity test.

GiantSteps has already been used for one external comparison of the selected
small recipe and control. Report that exposure; future recipe selection uses
Beatport/FMAK development, not repeated GiantSteps test scores. Keep Euphoria's
existing detector until a candidate clears the declared quality and CPU gates.
