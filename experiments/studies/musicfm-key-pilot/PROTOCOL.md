# Prospective MusicFM key teacher

Status: implementation in progress; no real teacher fit or task scores. The
completed synthetic resource probe is not an accuracy result or a whole-fit cost.
This file supersedes the obsolete `/tmp/musicfm-v2-real-pilot-proposal.md` draft.

## Question and fixed comparison

Can full fine-tuning of the pinned MusicFM-MSD backbone with a time-mean/linear
24-class head earn a stronger internal teacher than the completed matched K10
controls? Architecture, context, augmentation, pretraining and optimizer differ;
this is a system comparison, not an isolated representation effect.

Use only the approved amended v2 pools: 1,229 Beatport + 4,033 FMA fitting IDs,
126 Beatport + 250 FMA selection IDs. No calibration, confirmation or private DJ
audio. The original role/exposure ledger and the canonical-family amendment
remain binding; no replacements, resplit, chosen duplicate or relabeling.

- Plan gzip SHA256: `c3e59ffb764ca5f51444447b0a18a69543228e60a9e3723d925ddd82b0fc8373`.
- Numeric-label gzip SHA256: `b8942b5913b8ebacdf2fc9e0ebca72a934aaa7c56d4e1a97d0f60fac84354323`.
- Current K10 feature manifest gzip SHA256: `fcbc61081166f1845efe8af1ec2d7f807bb2d0a906dd93728b735e287c855291`.
- Public MSD checkpoint SHA256: `218b483a0256ddef736267425fabb166fd97008983696bb9270def464b47bded`.
- Upstream MusicFM source: `b83ebedb401bcef639b26b05c0c8bee1dc2dfe71`.

The K10 expanded-data pair gained 1.6 FMA exact percentage points and failed its
fixed 2-point gate. Neither seed rescue nor confirmation follows that failure.
It remains a completed comparator for this independently motivated teacher.

## Training recipe to implement before outcomes

Fresh seed-13 head, all active backbone parameters trainable. Original mono 24 kHz
frontend, MSD global mean/std, no waveform normalization or reslice pitch shift.
Uniform whole-track 12-second crops; shorter tracks use their actual waveform.
Exactly 8,280 optimizer updates, 32 accumulated microbatches of size one per
update. Preserve the original paired sampler's domain identity subsequences,
nominal Beatport mass 1230/2206 and total 264,960 draws; freeze a deterministic
within-update interleave and independent crop RNG before execution. Microbatch
one is not batch 32 for BatchNorm.

FP32 parameters, frontend, gradients and Adam moments; BF16 encoder/head autocast,
TF32 off, eager attention. AdamW backbone/head LR 1e-5/1e-4, weight decay 0.01,
betas (0.9, 0.999), epsilon 1e-8, no fused/foreach. Cosine decay over 120 virtual
events of 69 updates. No EMA, early stopping, checkpointing, compile or sweep.
Check finite loss every microbatch and finite accumulated gradients before each
optimizer commit. Abort on failure; never silently skip an update.

## Complete-song selection

Validate all 376 complete songs at all 120 events, FP32 eval mode. The local
window contract is implemented in `experiments/musicfm_key.py`: all disjoint
12-second windows plus the real tail. Tails longer than 1,024 samples use their
own input; smaller tails borrow the final full window as context and receive
only their newly covered samples as aggregation weight. There is no truncation
at two minutes, peer padding, selected-window weighting or ignored outro.

Aggregate logits in FP64 with exact sample-count weights. This is local-context
inference across a complete song, not whole-song attention. Use the shared Drone
class order. Selection is equal-domain exact first, then equal-domain symmetric
MIREX, then earliest event, using exact integer/rational comparisons.

Evaluation batching remains conditional on the separate fixed numerical/resource
probe. The actual trained teacher must subsequently pass a label-blind batch-one
versus chosen-batch logit/key comparison on all 376 songs before any claim that
batching preserves its decisions.

## Input readiness and finite cost

The seek/full-decode corpus qualification stopped at its first failed crop;
there is no usable completed 24 kHz manifest. The proposed replacement decodes
each complete fitting waveform once into an explicitly owned temporary RAM
cache, then slices readonly mmaps. Validation waveforms remain on ordinary disk.
No seek correction, dropped file or lossy waveform cache is allowed.

The RAM-cache implementation must account for its roughly 99 GB fitting cache
separately from private RSS, preserve at least 40 GiB actual MemAvailable and
1 GiB tmpfs headroom, retain 32 GiB ordinary-disk headroom and verify both vLLM
parents/engines throughout. No swap reliance. Pressure aborts only our task and
cleans only its explicitly owned temporary cache while retaining disk receipts.

The earlier eight-hour estimate is withdrawn. Freeze a finite whole-fit budget
only after measuring the real accumulation loop, input pipeline and complete
validation cost. Preserve data, updates, context and validation opportunities;
do not shrink them to meet an unsupported estimate. Exact implementation,
environment, inputs, resource caps and a fresh UTC window precede execution.

## Interpretation and continuation

An internal continuation requires FMA exact at least 5/250 above **each** K10
control and no Beatport exact/symmetric/asymmetric MIREX decline against either.
This is a research continuation bar, not a significance or release claim. Report
paired per-ID errors and grouped descriptive uncertainty with both controls.

Full 5–10 minute house, deep house and melodic house are the primary release
domain. Report fixed duration strata (<120, 120–<300, 300–600, >600 seconds),
genre/missingness and support groups. Only five generic House selection songs
across three groups currently cover that duration range; no deep/melodic-house
panel is verified. That cannot establish primary-domain product acceptance.
A generic teacher win earns separately planned confirmation and distillation,
not release promotion. An incomplete fit is inconclusive; a completed negative
rejects this recipe, not all pretrained representations.
