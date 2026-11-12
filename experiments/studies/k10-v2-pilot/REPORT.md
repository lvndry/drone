# Paired K10 v2 pilot: concrete review point, no accuracy outcome

**Ready for source/protocol review and bounded first16 preparation. Nothing was
trained, decoded or newly preprocessed.** Frozen assignmentca763c83 is untouched.
The paired driver compares restricted976 versus expanded4035eligible container-unique
FMA records with identical BP1230 exposure and NEW250FMA/126BP selection IDs.
The [protocol](PROTOCOL.md) fixes the new matched8280-update recipe, exact-first
equal-domain selector and +2FMA exact/BP-nondecline gate. It does not reproduce the
old sampler/early-stop trajectory or claim an independent final test.

## Implemented and checked

- Metadata-only plan with exact container, role, numeric-target and source bindings;
  new FMA250, not historicalv1FMA250. Exported selected FMA labels directly match
  pinned officialCSV through `encode_key`; BP source39fe1842 and preparedFMA member
  94f65bb5 are pinned. The publisher's unverified~200v2 modifications are disclosed.
- Separate domain order/per-draw augmentation streams; full8280-step metadata test
  proves147733identicalBP draws/augmentation seeds in both arms and117227FMA draws.
- Fresh109128parameter K10, unchanged CQT/crop/hard-target optimizer, equal updates,
  cadence and scheduler; no independent early stopping. Checkpoint exact-first ties
  use rational equal-domain scores, then symmetric MIREX, then earliest event.
- Strict feature/label/canonical-identity boundaries. Training cannot load calibration
  or confirmation. Gate checks source/config/runtime/device/ordered IDs and both
  arms' completion, and never scores confirmation. FMAweighted changes report-only.
- First16 normal-preprocessing helper plus Linux outer process-group resource guard;
  no permit supplied and no job launched. First16 output is explicitly non-trainable.

## Actual cache availability and bounded next action

Stat-only inventory of the5641pilot IDs found **2475existing files**, all in the
frozen v1 CQT root:1356BP plus1119FMA. Missing: **2965expanded-fit FMA +201inner FMA
=3166**. Old976FMA and all BPfit/selection caches exist. File existence/size is not
content or decode verification; normal preprocessing checks those next.

Existing logical cache bytes: **947443968**. Metadata-duration projection for
missing208×float16 full-time arrays: **1513196736bytes (~1.51GB)**. Total logical
cache footprint is therefore roughly2.46GB; hardlinks avoid recopying existing data.
Reserve3GiB additional workspace for bulk to cover copy fallback/frame variance and
scratch, subject to actual first16 measurements. Legacy5489feature arrays are not
assumed physically compatible; only the matching208-bin cache is reused.

The fixed first16 has12missing fitting tracks across duration quartiles including
2157s longest, plus4verified-cache fitting examples. Caps:2workers onCPU10–11,
900wall/1800CPU seconds,4GiBRSS,128MiB CQT,256MiBWAVscratch. No current calibration or
confirmation audio. Root can execute the reviewed guard with a hash-bound
preprocessing permit; its controller emits actual PID/log/exit/resource evidence.

A provisional old-measurement proxy (20public-track cold frontend median0.353s,
p95~1.0s) gives about19–53CPU-min for3166cache misses, ideally9–27wall-min at2workers.
This is **not a measured total-preparation forecast**: duration distribution differs,
longest pilot track exceeds that benchmark, and2475cache hits still need canonical
waveform decode/hash. First16 must estimate cold/reuse seconds per audio-second by
stratum; sum those over the full inventory before a bulk launch. A prospective90min
wall/3CPU-hour bulk envelope is a cap proposal, not performance evidence. The model
pair has its own fixed20min/arm cap; expanded undertraining at equal updates remains
an explicit interpretation limit.

The genuine remaining feature prerequisite is the complete5641-row physical manifest
with canonical22050identity/conflict checks. BP612unknown artist IDs are disclosed;
they do not indefinitely block this shared-BP INTERNAL data comparison. BP→GS audit
results affect external interpretation, not this pilot, which never consults GS.

## Cached PCM addendum: identity scope only

Classifier's metadata-only join covered all5489FMA with5413cached PCM hashes:
zero families crossed proposed FMA roles; zero current1230 or release1237BP fitting
IDs matched proposed FMA evaluation-role PCM. No PCM family bridged saved components.
On PRE-overlay assignments, expanded4082rows had70duplicate families/70excess
(4012PCM hashes); confirmation998had5families/6excess(992PCM hashes). These counts
DO NOT replace current4035container-unique fitting/997confirmation units or rewrite
roles/eligibility. Group/family witnesses remain with the source audit for uncertainty.

The exact contract is ffmpeg mono11025Hz little-endianfloat32 stdout SHA, NOT
original stereo/container identity or canonical22050model input. It supplies no
near-duplicate/work clearance. Full source/report hashes and contract are retained
in `data/cached-pcm-addendum.json`; canonical verification belongs to normal pilot
preparation and later separately authorized confirmation preparation.

## Reproduction and next review decision

The compressed plan, numeric export, stat inventory and first16 request are SHA
indexed. Planning/preparation/test code contains no automatic model launch or role
activation. `python -m pytest tests/test_k10_v2_pilot.py -q` exercises pairing,
selection, stale-label/provenance rejection, reserved-audio exclusion, permit binding,
source/config/calibration/order gates and outer process accounting/caps without
models/audio. Ruff and targeted type checks also pass.

The next requested decision is the concrete first16 preprocessing run, followed by
measured bulk preparation and then the fixed K10 pair after a complete manifest.
A successful pilot earns root review for the997-unit confirmation stage, not automatic
release promotion or a model/optimizer search. Existing weights and v1 evidence remain
unchanged; no push/PR or server audio job was performed by this work.
