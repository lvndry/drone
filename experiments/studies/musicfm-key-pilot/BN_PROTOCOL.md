# Prospective MusicFM BN-buffer diagnostic, version 1

Status: prospective protocol; pure schedule and FP64 numerical primitives have been implemented and tested on synthetic inputs only. No checkpoint loading, audio processing, model execution or score inspection performed. This is one inference-statistics intervention, not a new fit, a release candidate, or an amendment to the live teacher's gate.

## Implemented scope and remaining launch blockers

`experiments/musicfm_bn.py` implements the fixed 512-recording schedule from caller-supplied fitting metadata, exact rational recording weights, validated per-recording population moments and streaming weighted-Welford aggregation. `tests/test_musicfm_bn.py` covers deterministic golden membership/crop mappings, rejection and tie handling, actual-length support, exact domain mass, unequal-support/between-mean fixtures, merge-order stability, input validation and atomic failure. Tests use synthetic metadata/NumPy arrays only; the module imports no model or waveform reader and exposes no execution entry point.

This does not qualify the full diagnostic. Source roles, real fitting/cache/crop hashes, final selected checkpoint and terminal gate, runtime implementation and hook correctness, installed-environment identity, protected services, hardware/resource receipts, measured timing admission and evaluator/audit identities remain unfilled or unverified launch blockers. No hooks, model/checkpoint loading, score logic, guardian or server code are included. No diagnostic is launched by these primitives or their tests.

## Start condition and immutable inputs

Start only after the original 8,280-update/120-event run is terminal, its selected checkpoint has passed its required audit, and its original outcome/gate has been written and hashed. Select the checkpoint by that original rule, not by this diagnostic. Work on a separate copy. Preserve the terminal original and its report.

Source is Drone commit `051e4adaf5207abd5f83ad822dbf60f7c7abf412`; MusicFM source is `b83ebedb401bcef639b26b05c0c8bee1dc2dfe71`; cached Transformers source is 4.57.6. Verified source SHA256 anchors:

| File at Drone commit | SHA256 |
|---|---|
| `experiments/musicfm_pilot.py` | `99b62804db4b6ccae95546f67104d0bed560fbac47606177a764cffac8a46104` |
| `experiments/musicfm_inputs.py` | `9f215476a1b97795c4b0577e6ca92888115538122dba9bc116e05d7710e197c0` |
| evidence `source/modules/conv.py.txt` | `585705a6450db374e8c411034e846491f7f48d4ed0516fbf8d3b84855ef0d7d1` |
| evidence `source/model/musicfm_25hz.py.txt` | `9645e51938e7a73689a0792ecb15ee957e0723582c8f040949eadd25036ec804` |
| evidence `transformers-conformer.py.txt.gz` | `78ebb20a47ae2773644fe8f387bbdc65baf8e51a66daabec4627479fcb24f5d0` |

Before execution, seal a single manifest containing actual SHA256 values for: this protocol; diagnostic implementation and qualification tests; all imported numerical/evaluator/guardian sources; terminal outcome; selected checkpoint; original plan and current fitting-ID lists; prepared-cache manifest and selected cache files; MSD statistics/configuration; original environment/provision receipts and installed dependency files; protected service identities; evaluator window schedule and original final logits receipt. Record hardware UUID, driver, Python/NumPy/PyTorch/torchaudio/Transformers versions, backend flags, CPU affinity and thread settings. The installed torch version must match the original `2.15.0+cu132` receipt. Missing pins or any unexpected discrepancy block activation; no invented hashes or substituting current main. Later checkpoint and implementation hashes cannot truthfully be filled in this draft.

## Deterministic fitting-only schedule

Read only canonical IDs, roles, cache hashes and sample counts while constructing the panel. The current fitting pools must contain exactly 1,229 BP and 4,033 FMA recordings; no reserved/selection label access is needed. No genre, outcome, key or duration targeting. A source-ineligible/missing recording aborts; do not replace it with the next-ranked ID.

Define `H(parts)` as SHA256 of ASCII bytes of `json.dumps(parts, ensure_ascii=True, separators=(",", ":"))`, with each ID preserved exactly as its source string. Within each domain (`"beatport"`, `"fmak"`), sort IDs by raw digest `H(["musicfm-bn-posthoc-v1", "membership", domain, id])`, breaking a digest tie by the ASCII JSON-encoded ID bytes; select the first 256.

For each selected recording with N cached 24-kHz FP32 samples, require N>=1,200 and set L=min(N,288000), M=N−L+1. Starting at integer counter k=0, set x to the big-endian integer from `H(["musicfm-bn-posthoc-v1", "crop", domain, id, k])`. Accept the first x below `floor(2^256/M)*M`, and use start=x mod M; otherwise increment k, aborting if k reaches 1024. This is an exact, deterministic rejection-sampling mapping with no Python-RNG dependency. Use the existing immutable cache slice `[start,start+L)`; no decoder, shift, stretch, normalization change or added padding.

Seal the resulting 512-row manifest, including IDs, N/start/L, cache and crop-byte SHA256 values. All passes visit BP's ranked 256 then FMA's ranked 256. This order cannot create EMA suffix bias: no EMA updates occur. It fixes numerical aggregation order. Each BP recording has weight `(1230/2206)/256`; each FMA recording has weight `(976/2206)/256`. Preserve this nominal sampler mixture despite the amended BP count. A longer crop has more positions for estimating its own moments, but no greater total recording weight. The panel is a bounded diagnostic sample, not a powered or comprehensive domain sample.

## Actual graph and observed supports

Use the original `KeyTeacher.forward` and its rotary-cache protections, all modules in eval mode, inference mode, FP32 parameters/frontend/encoder and no autocast. Disable all Dropout through eval; no gradients, optimizer or labels. Preserve original backend settings, including TF32 off and benchmark off; do not silently add a stronger global deterministic policy. Schedule and estimator are deterministic; cross-device bitwise identity is not asserted.

The required observed forward order is:

1. `backbone.conv.conv.0.bn1`, `.bn2`, `.bn3`;
2. `backbone.conv.conv.1.bn1`, `.bn2`, `.bn3`;
3. `backbone.conformer.layers.0.conv_module.batch_norm` through layer 11 in increasing order.

The first six are BatchNorm2d(512), the remaining twelve BatchNorm1d(1024), all eps=1e−5. This is execution order, not assumed module-registration order. Each residual block executes the main branch's bn1/bn2 before the projection bn3; bn3 depends on the block input, not its sibling BNs, so this order is a valid dependency ordering. The next block depends on both branches. Each Conformer target observes the depthwise-convolution output, after input LayerNorm, pointwise convolution and GLU; never collect the layer's external input instead.

At L=288000, target inputs are respectively `(1,512,64,600)`, `(1,512,32,300)`, and `(1,1024,300)`. For shorter inputs, assert time lengths derived from floor(L/240), then two ceiling halvings. BN2d reduces N/F/T; BN1d reduces N/T. At N=1, their supports are 38,400, 9,600 and 300 positions per channel for full crops. Do not equate these supports or assume independent positions.

The wrapper supplies no attention mask; B1 actual-length input has no batch padding. Retain native STFT centering/reflection and convolution padding, including boundary-derived activations. These are part of the original computation, not positions to remove with an invented mask. No manual tail mask or exclusion is introduced. LayerNorm and fixed MSD normalization are not calibrated. Unexpected BN inventory, masking, extra call, skipped call or changed shape aborts.

## FP64 estimator and sequential replacement

For target j, run every scheduled crop through the current copy's already-calibrated prefix. A pre-forward hook receives the target's actual input; copy only that input to CPU, then raise a dedicated, narrowly caught stop exception before target normalization. Other exceptions propagate as failures. Maintain lightweight call-order traces without retaining preceding activations. After all 512 inputs are accumulated, replace only target j's running mean and variance, then proceed to j+1. Thus downstream moments are measured under the new upstream inference statistics. Simultaneous one-pass calibration under old upstream statistics is not equivalent.

For each crop/channel, flatten non-channel axes in fixed N/F/T or N/T order. Compute CPU FP64 two-pass population mean `m_i` and variance `v_i = mean((x−m_i)^2)` with fixed NumPy/thread versions and channel chunks of 64. Do not materialize all crops or all sites. Merge crops in manifest order using FP64 weighted parallel-Welford formulas: for existing mass W/mean m/central moment S and crop mass w, let delta=m_i−m, then `m += delta*w/(W+w)` and `S += w*v_i + delta^2*W*w/(W+w)`; finally `v=S/W`. Initialize directly from the first crop. This equals `sum w_i*(v_i+(m_i−m)^2)` after normalization and includes between-recording mean variation. Never average within-clip variances alone, concatenate clips with unequal recording mass, or apply a Bessel correction. Preserve FP64 moments in the receipt; cast each final vector once to FP32. Reject nonfinite means/variances or negative variance; add no hidden variance floor or extra epsilon.

Only the 36 named `running_mean`/`running_var` tensors may differ. Hash all learned parameters, BN affine parameters, counters (`num_batches_tracked`), other buffers and MSD state before/after; all must be byte-identical. No `train()`, `reset_running_stats()` or generic `update_bn`. PyTorch's native training BN uses biased normalization variance and unbiased running-variance updates; this deliberately different population estimator is not presented as reproduction of native training statistics.

## Qualification and hard bounds, before scores

This protocol is not executable until an isolated implementation passes review and bounded qualification. Qualification may use only the sealed fitting panel and synthetic numerical fixtures; no selection labels/logits. A required unequal-support fixture has channel-major clip A=`[[0,2],[1,1]]`, clip B=`[[6],[5]]`, masses 1/4 and 3/4: expected means `[4.75,4]`, population variances `[4.9375,3]`. Also test the actual rational domain masses p/q: expected means `[p+6q,p+5q]`, variances `[p+25pq,16pq]`. Require error <=1e−12 + 1e−12*abs(expected) on these small FP64 fixtures, plus constant and identical-clip cases. This catches missing between-clip variance and position-weighted pooling.

Verify stopping hooks against an ordinary eval forward on disposable copies, comparing captured target inputs under identical state: require finite, identical shapes and max absolute difference <=1e−5. Use the first ranked crop from each domain and the shortest selected crop (deduplicate); at most three full traces plus 54 stopped-prefix checks. Trace all 18 sites; confirm counters never change and all modules remain eval. Any mismatch fails qualification without changing tolerance.

For timing, use the first eight ranked clips per domain, plus the longest selected clip if absent, across all 18 stopped prefixes; discard these provisional statistics. Reset from the untouched selected checkpoint before the complete pass. Include FP64 CPU reductions, transfers, input verification and storage. A 300-second qualification sublimit sits inside a 1,800-second total wall limit. Require `2 * 512 * sum(max observed per-clip seconds for each site)` plus twice the pinned historical wall cost of the complete required evaluation/audit schedule, plus measured loading/hash/serialization cost and 120 seconds contingency, to fit the remaining wall budget. This is a conservative admission estimate, not a runtime guarantee; the hard watchdog remains authoritative. Missing comparable evaluation timing or an insufficient margin means budget-inconclusive, not a smaller panel or fewer sites.

Limits: at most 9,216 calibration prefix forwards, plus qualification's 306 timing prefixes, 54 correctness prefixes and three full traces; total 10,800 CPU-seconds, 8 GiB owned-host memory, 8 GiB outputs, 12 GiB Torch allocator. Use the original teacher's GPU UUID and unchanged resource guardian contract after its completion: at least 8 GiB global GPU free, 40 GiB global host free and all protected services/start identities preserved. No overlap with the live teacher, device substitution or vLLM restart. Existing heartbeat/resource checks continue during copying and FP64 work. Abort on any cap, missing heartbeat, correctness failure or resource conflict; terminate only owned diagnostic processes and retain the failure receipt. No automatic retry, OOM rescue or backend change.

Seal the complete candidate checkpoint and moment/identity receipt before opening any score. Recheck cost margin before scoring. If correctness or duration cannot be bounded under these constraints, do not run this diagnostic.

## Exactly one paired comparison and its interpretation

Compare the untouched selected checkpoint and this one candidate on the existing 126-BP/250-FMA inner panel, using the frozen whole-song FP32 evaluator, actual tails, support-weighted FP64 logits and its required B1/B16 audit/tolerance. Reuse original terminal baseline logits only if their exact source/checkpoint/schedule/environment identity is verified; otherwise one fresh baseline pass is part of this same comparison. Audit duplicate passes are numerical checks, not alternative candidates. Open no calibration159, confirmation997, GiantSteps, private tracks or new reference panel. No label cleanup, exclusions, momentum sweep or second candidate after results.

Prospective useful-signal gate: at least five additional FMA exact predictions versus the untouched selected checkpoint, with no BP exact, symmetric-MIREX or asymmetric-MIREX decline. Report all paired movements and both domains' weighted-score changes, including failures. This does not alter the original teacher's control-based continuation gate or authorize distillation/release.

If state checks pass, prediction changes can be attributed to this joint buffer intervention in this fixed network. They cannot establish that N1 BN caused the training plateau, isolate a particular BN path, distinguish recent-window instability from precision/dropout/domain differences, or show that frozen-pretrained BN training would help. The estimator, finite fitting panel and all 18 sites change together. The old 376-song panel has already driven selection; a gain is an exploratory signal on an adaptively reused panel, not independent generalization evidence. A null result also cannot falsify a fresh frozen-BN training hypothesis. The result may inform a separately budgeted proposal; it triggers no new fit automatically.

Primary sources: [MusicFM residual/subsampling implementation](https://github.com/minzwon/musicfm/blob/b83ebedb401bcef639b26b05c0c8bee1dc2dfe71/modules/conv.py), [MusicFM preprocessing and encoder](https://github.com/minzwon/musicfm/blob/b83ebedb401bcef639b26b05c0c8bee1dc2dfe71/model/musicfm_25hz.py), [pinned Transformers Conformer](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py), [PyTorch BN estimator semantics](https://docs.pytorch.org/docs/2.15/generated/torch.nn.BatchNorm1d.html). Local antecedent: `/tmp/astra-musicfm-bn-audit.md`; the previously proposed BP→FMA suffix bias remains falsified by the actual interleaver.
