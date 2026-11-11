# Fixed K10 EMA probe v1 — pinned before fitting

Goal: improve accuracy and CPU performance, never trading accuracy for speed.
No automatic weight promotion, architecture changes, teacher, decay sweep or EMA epoch search.

Compare one joint training trajectory per seed with a parameter-only EMA shadow.
Decay is exactly 0.99 after every optimizer step, initialized explicitly to theta_0.
The first update includes theta_0, unlike first-copy AveragedModel behavior.
At the ordinary RAW earliest-best combined-MIREX epoch save EMA parameters together
with contemporaneous ordinary RAW buffers (BN statistics are never averaged).
Only ordinary RAW validation controls checkpoint choice and patience25 stopping.
The baseline optimizer/crops/augmentation/config, 120-epoch horizon and RNG stay unchanged.

Canonical pilot is seed13. Only a completed, exactly reproduced pilot with >=0.01
FMAK MIREX gain over RAW+BN, zero Beatport MIREX decline and zero exact-accuracy
decline in either domain permits seeds37/73 with identical protocol/source/config.
Three-seed means must pass that same method-specific gate AND the distinct original
hurdle: >=0.02 FMAK MIREX above archived BP-only controls and no Beatport decline.
Never substitute the one-point increment for the two-point original hurdle.
Separate accuracy safeguard: EMA+BN must also not lower ordinary RAW joint MIREX
or exact accuracy in either domain on three-seed means. Report this independently
of the method-specific BN-controlled comparison; never substitute it for that gate.
All three conditions must pass before a later external freeze. They do not promote weights.

Before fitting write a BN schedule manifest with id/audio hash/index/start/length/
padding/batch/position. Use exactly the joint2206 fitting rows only, shift0,
center320-frame crops starting max(0,(T-320)//2), right-pad short arrays with zeros
exactly as training does. Order: np.random.default_rng(20261111).permutation(fitting).
Batch32, including final partial30. Use the identical schedule for both copies.
No validation/calibration/final/SWD/private audio enters BN recalibration.

At training end require exact RAW history (epoch/loss/MIREX), selected epoch,
stop epoch, early-stop status, fit/validation IDs, feature SHA, config and per-track
predictions/metrics against source673e280ca244d4187d8c20f9fdaecaa1c75281fd archive.
Require tensor-by-tensor exact checkpoint equality when reference best.pt is present
(or explicitly supplied). No numerical tolerance tuning. Failure ends qualification.
Archive/current source maps are recorded in full, with all hash differences. Reviewed
base c52255761412ef48f2f9af814df67c3d3214ef4f contains annotations plus inference/export
changes outside cached-feature fitting; intentional source allowances are pinned in
ema-reviewed-sources.json. Changed shared-loop integration is minimal optional hooks.

After reproduction, perform exactly one fitting-only BN pass on each selected model
copy: ordinary RAW and EMA. Reset all six BatchNorm2d statistics, momentum=None;
only BN layers train, all dropout stays disabled. Restore original momentum and eval
mode. Preserve Torch CPU/CUDA RNG and use a separate NumPy generator. Source models
are never mutated. Record RAW, RAW+BN and EMA+BN logits/IDs/refs/metrics for all376
validation identities and checkpoint/artifact/source/protocol/schedule hashes.
RAW+BN controls BN effects; EMA+BN minus RAW+BN attributes averaging under this fixed
selection policy. EMA may peak earlier; knowingly forgo independent epoch selection.

Existing device_setup remains: GPU0, >=8GiB free, allocator fraction0.04, TF32 off,
vLLM untouched. Hard20min/seed includes training, BN and final scoring, without
shortening the baseline horizon. External wrapper timeout1250s. At most three fits.
Record total epochs, updates, total and BN timings, status and artifact version.

Evidence: EMA TMLR2024 https://arxiv.org/html/2411.18704v1 (SGD evidence; transfer to
AdamW is a hypothesis); PyTorch averaged-model BN handling:
https://docs.pytorch.org/docs/2.15/optim.html#stochastic-weight-averaging .
