# MusicFM-MSD source and resource preflight

## Problem

A330M pretrained encoder may make a useful key teacher, but the proposed GPU-hour
budget and CPU-student target lack a measured teacher execution path. Establish
artifact provenance, dependency compatibility and a bounded resource probe before
spending on fine-tuning or distillation.

## Method and audit results

This change audits official source/metadata and supplies an offline-by-default
preflight tool. It does not download checkpoint weights or execute MusicFM/GPU
work. See the [fixed prospective protocol](PROTOCOL.md) for source hashes,
frontend/math, resource policies, unresolved compatibility and the exact CLI.

| Finding | Status |
| --- | --- |
| Official source |Pinned`b83ebedb401bcef639b26b05c0c8bee1dc2dfe71` |
| MSD weights |Only135-byte LFS pointer fetched;1,316,802,088byte artifact/SHA pinned |
| Model-card/code licenses |MIT model card; MIT code except Apache2 flash module; notices retained |
| MSD normalization |Pinned mean6.768444971712967/std18.417922652295623 for mel2048 |
|12s synthetic shape |Analytical`[1,288000]→[1,128,1200]→[1,300,1024]`; local boundary tests |
| Current server |Measured metadata: Python3.12.3, Torch2.15.0+cu132; CUDA not initialized |
| Missing dependencies |TorchAudio, Transformers, einops, HFhub, safetensors, tokenizers |
| Candidate stack |TorchAudio2.11.0/Transformers4.57.6/einops0.8.1; static APIs checked, imports unproved |
|330M all-active state |Estimated5.28GB/4.92GiB FP32 weights+gradients+Adam moments, before activations |
| MusicFM parameters/VRAM/step time/accuracy |**Unmeasured** |

The current~4GB pilot cap is insufficient for the nominal conventional full-model
state scenario. A prospective cap policy requests at most12GiB while retaining an
8GiB global-free reserve, reducing the applied cap if memory shrinks before CUDA
allocation. This is a proposal, not authorization or a guarantee of fitting.
Unused pretraining parameters may not receive optimizer states; the tool will
count actual active parameters and first-step state bytes instead of presenting
16×nominal-parameters as an exact loaded-model measurement.

## Executable result

`python -m tools.preflight_musicfm inspect` checks source/stat/config/pointer
integrity and reports estimates plus dependency metadata **without importing
Torch**. A future explicitly reviewed `run-resource --config ...` requires exact
source/protocol/environment bindings, a fresh UTC window, GPU UUID, both protected
vLLM process identities and a verified existing checkpoint. It performs one
synthetic first AdamW update and20 steady updates, saving resource reports only.
The supervisor stops only its own worker on reserve/PID/resource/window failure.

Local tests use CPU tensors for pooling-axis and lazy optimizer-state checks,
plus source/shape/window/reserve contracts and mocked NVML parsing. They do not
prove the330M model loads or fits. Full repository Ruff/type/format and focused
tests are part of verification; no new production dependencies were added.

## Decision and limits

Proceed only to root review of the source/config and a task-local dependency
provisioning plan, followed by a fresh resource-execution window. No training,
weight download, shared-environment change or GPU execution is authorized here.
Static package compatibility is not binary/checkpoint compatibility. Stop on
strict loading or API failure; do not substitute a model or loosen state loading.
A later teacher experiment still needs its own accuracy, split, optimizer and
budget protocol. No native measured study or production/default code changed.
