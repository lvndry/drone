# Data provenance

Bundled weight provenance is documented in [weights/README.md](weights/README.md).
The baseline training pipeline uses:

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
Ordinary training does not score GiantSteps; its metrics come from the explicit
`evaluate` command. Optional correction fine-tuning deliberately reports one
starting/tuned comparison on GiantSteps and correction-held-out, keeps the last
epoch, and never selects checkpoints on those scores. Do not use repeated test
evaluations to select architectures or hyperparameters.

Preview audio is dev-time input and is never included in package builds. The
annotation license does not itself establish redistribution rights to the audio;
retain provenance and review the distribution terms for any future model release.


## Research reference versions

Research references use [FMAKv2](https://zenodo.org/records/12759100), derived from
[expert-annotated FMAK](https://zenodo.org/records/10719860). Its release reports
about 200 revisions not verified by the original annotators; this does not prove
errors. [Spotify URI matches](https://github.com/stellaywong/fma_keys) support analyzer
comparison, not Spotify-generated reference keys. Targets remain pinned. See the
[coverage/provenance audit](experiments/studies/v2-data-contract/REPORT.md#current-genre-duration-and-reference-provenance).

Beatport's human reference annotations are distinct from store metadata keys;
genre or duration metadata must not replace either references or actual preview
support. The coverage audit keeps native taxonomies and version identities separate.

## Personal corrections

Human annotations are optional explicit inputs, separate from Drone's public base
model data. `train.py prepare-corrections` accepts portable schema 1 JSONL produced
by `euphoria corrections export` or another tool implementing that adapter. Drone
does not collect annotations automatically, read private libraries unless their
folder is explicitly supplied, or upload audio/labels. Only corrected keys are
fitting labels; tempo-only records are counted and skipped.

Each line is a JSON object with `schema_version: 1`, a lowercase 64-hex-character
`content_hash` (full-file BLAKE3), `corrected_key` (`tonic` integer 0–11 and `mode`
`major`/`minor`, or null for a tempo-only record), and integer Unix-seconds
`corrected_at`. Optional `detected_key` uses the same key object or null. Additional
Euphoria display/tempo fields are accepted without becoming key-training targets.
The format and existing helper/CLI names are unchanged. File-byte matching needs
`uv sync --extra corrections` / `drone-key[corrections]`; schema parsing and core
model use remain independent of BLAKE3. Files are matched recursively by
Euphoria's full-file BLAKE3 identity,
including tags. The export has no audio paths; preparation manifests contain local
paths, SHA-256 identities and corrected labels. These personal manifests/audio are
local inputs, not bundled package data. Keep only audio and labels you are
entitled to use; this workflow confers no new redistribution rights.

Matching GiantSteps file bytes is rejected before correction preparation. The
hash split assigns roughly one fifth of matching files to corrections-held-out;
both sides must be nonempty. The split is stable for identical file bytes only.
Retagging/re-encoding may move unchanged musical audio between sides; artist,
album, recording, remix and decoded-audio independence are not established.

Fine-tuning checks known starting-checkpoint train/validation IDs and training
SHA-256 witnesses and actual new fitting data against both evaluation sets before
scoring or gradients. Known overlap fails without silently changing membership.
Missing legacy checkpoint witnesses remain unknown and are listed in
`finetune.json`; passing available checks does not prove unknown historical
exposure absent. Genuine inherited provenance is retained in tuned checkpoints,
while new hashes describe only the filtered base and correction-fit audio actually
sampled. Pre/post scores are descriptive; no measured real-world or public-model
improvement is claimed. Public-model training would require a separate, explicit
annotation provenance/rights and evaluation decision; local inputs are not
silently folded into a published model.
See [the workflow](docs/training.md#11-optional-annotation-fine-tuning).
