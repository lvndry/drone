# Data provenance

No trained weights ship in this checkout. The baseline training pipeline uses:

- [Beatport EDM Key Dataset](https://doi.org/10.5281/zenodo.1101082): 1,486
  two-minute previews with annotations by Eduard Mas Marín, revised and expanded
  by Ángel Faraldo. The published annotations are CC BY-SA 4.0. Unparseable labels
  are skipped; ambiguous `X | Y` labels currently use the first option. Counts and
  exact accepted labels are recorded in the prepared manifest, not assumed.
- [GiantSteps Key Dataset](https://github.com/GiantSteps/giantsteps-key-dataset):
  held-out evaluation data. The downloader pins annotations to revision
  `6bcd492c825ac9b8597bc650a5f6fd18b6c43d2b` and checks preview MD5 values.

`bash fetch.sh` fetches both datasets, including training labels, and writes
`data/sources.json`. Zenodo archive checksums and the GiantSteps archive SHA-256
are pinned in `drone/download.py`. Partial or mismatching downloads are never
installed as verified inputs. The loader fails for missing labelled audio.

Preparation and training check track-ID separation; training/evaluation also check
exact audio hashes across training and test bundles. These checks do not establish
artist-, album-, remix-, or near-duplicate independence. Broader corpus evaluation
must audit those relationships and any pretrained teacher's data exposure.

Training selects checkpoints using only a fixed training/validation split.
GiantSteps metrics are computed only by the explicit `evaluate` command. Do not
use repeated test evaluations to select architectures or hyperparameters.

Preview audio is dev-time input and is never included in package builds. The
annotation license does not itself establish redistribution rights to the audio;
retain provenance and review the distribution terms for any future model release.
