# MusicFM-MSD source and synthetic resource preflight

Status: **source/metadata audit and local CPU contract tests only**. No checkpoint
weights were downloaded, no MusicFM model constructed, no GPU kernels executed,
no optimizer step on MusicFM, no real audio and no model score. No execution
window or increased GPU cap has been authorized. The tool's default `inspect`
command does not import Torch. `run-resource` requires a prospectively reviewed
source/config/environment/window and an already present hash-verified checkpoint;
it never downloads or installs anything.

## Pinned sources and what is known

| Artifact | Exact identity |
| --- | --- |
| Official MusicFM source |`b83ebedb401bcef639b26b05c0c8bee1dc2dfe71`,2024-02-14 |
| Official model repository |`minzwon/MusicFM`, revision`4513b38bc25ad1d227b1980819b9691ba97f4d87` |
| MSD checkpoint pointer |1,316,802,088bytes; SHA256`218b483a0256ddef736267425fabb166fd97008983696bb9270def464b47bded` |
| Config repository |`facebook/wav2vec2-conformer-rope-large-960h-ft`, revision`6b36ef01c6443c67ae7ed0822876d091ab50e4aa` |
| Candidate Transformers |4.57.6, source`753d61104116eefc8ffc977327b441ee0c8d599f` |
| Candidate TorchAudio |2.11.0, source`34c52a67e8941bbd8e6adaca0eb0b9eabec11d78` |

[Source index](evidence/source-index.json), [artifact/response index](evidence/artifact-index.json)
and [dependency API index](evidence/dependency-source-index.json) preserve raw
source/metadata byte SHA256 values and Git blob/commit identities. Upstream Python
is stored as `.py.txt` evidence, with copyright/license texts, not as a silent
replacement production package. Large source snapshots and verbatim texts with upstream trailing
whitespace use deterministic gzip; compressed and original decoded hashes are both
retained, without rewriting their bytes. The future worker reconstructs exactly those
verified source bytes under its new task-owned output root.

Only small Git metadata/blobs were fetched from the official repositories using
no-checkout, LFS-smudge-disabled clones. The `.pt` file in evidence is a135-byte
LFS pointer, not checkpoint data. No model zoo expansion occurred.

The model card declares MIT. Upstream MusicFM code is MIT except the Apache2
flash module; relevant notices are preserved. The config model card declares
Apache2. Dependency source snapshots carry their original notices and license
files. These declarations do not establish downstream dataset provenance or
reproduce the unpublished key teacher. The author states MSD is outside the
original paper and downstream fine-tuned models/evaluation pipeline are not
provided. The old pre-2024-02-13 FMA checkpoint warning is retained; this proposal
pins the separate MSD artifact instead of choosing a moving download URL.

Primary sources:
[MusicFM](https://github.com/minzwon/musicfm/tree/b83ebedb401bcef639b26b05c0c8bee1dc2dfe71),
[MSD artifact](https://huggingface.co/minzwon/MusicFM/blob/4513b38bc25ad1d227b1980819b9691ba97f4d87/pretrained_msd.pt),
[TorchAudio compatibility](https://docs.pytorch.org/audio/stable/installation.html),
[Transformers4.57.6](https://pypi.org/project/transformers/4.57.6/).

## Architecture, frontend and resource-only adapter

The source uses two residual2D convolution stages with512 channels, each halving
frequency and time; projection is512×32→1024. The encoder is12 Conformer layers,
1024 hidden channels,4096 feed-forward width,16 attention heads and rotary position
embeddings from the pinned config. Source overrides the config's24-layer depth
to12. It also constructs an unused-for-key4096-way pretraining projection, class
token and random quantizer buffers. Do not replace it with a vaguely similar
HuggingFace audio model.

Input is mono24kHz,128-band HTK mel power spectrogram: FFT/window2048,
hop240, periodic Hann, center/reflect padding, no mel normalization, power2,
AmplitudeToDB(power,top_db=None), followed by upstream removal of the final STFT
frame. MSD `melspec_2048_mean=6.768444971712967`,
`std=18.417922652295623` are dataset-level statistics. They are not the FMA stats,
per-song normalization, librosa CQT or K10's FP16 cache contract.

The author's key input window is12s. Fixed synthetic input is therefore
`[1,288000]` float32, producing`[1,128,1200]` mel frames and`[1,300,1024]` final
hidden states. The resource-only head averages TIME (`dim=1`) then applies a
new1024→24 FP32 linear head. Its synthetic target0 has no musical meaning. This
is an explicitly defined cost probe, not the unpublished downstream classifier
or a claim to reproduce published key accuracy. No weights are saved afterward.

Use `is_flash=False` and the stock pinned Conformer path with eager attention.
No flash-attn installation, SDPA substitution, torch.compile, gradient
checkpointing, layer freezing, quantization, optimizer-state compression or
window/batch search in this fixed probe. Such changes need separate identities.
Original `.get_latent(layer_ix=12)` traverses the pretraining projection and emits
hidden states; the worker preserves that encoder path while splitting FP32
preprocessing from BF16 encoder/head autocast. Parameters, gradients and AdamW
moments remain FP32. TF32 is off. BF16 resource success does not prove task
accuracy equivalence to FP32. Upstream's half-input path behaves differently;
this probe does not silently select it.

## Actual server compatibility and unresolved parts

The [read-only server snapshot](evidence/server-metadata.json) records Python3.12.3,
Torch distribution2.15.0/runtime2.15.0+cu132, CUDA build13.2 and
`cuda_initialized:false`. TorchAudio, Transformers, einops, HFhub, safetensors
and tokenizers are absent. The shared environment was not modified.

TorchAudio2.11 explicitly documents stable ABI compatibility with PyTorch2.11
and later, including2.15; an exact TorchAudio2.15 version is not required by that
published policy. Candidate Transformers4.57.6 advertises Python3.9+/Torch2.1+;
the pinned source still contains the required Conformer constructor/forward API.
Candidate einops0.8.1 supplies rearrange/einsum. These are static compatibility
checks, **not successful binary imports or checkpoint loading**.

A future reviewed provisioning step must create a task-local overlay preserving
the exact existing Torch/CUDA build, resolve/pin transitive packages and wheel
hashes, and freeze the complete environment fingerprint. The proposed direct
requirements are not a tested/resolved lock. Do not run pip in the shared venv,
auto-upgrade Torch or install flash-attn. No provisioning is performed by the tool.

Before any CUDA initialization, the worker imports the actual dependencies,
constructs the original model on CPU and strictly loads the approved checkpoint.
The upstream config fetch is intercepted solely to return the exact local pinned
config; HF offline variables prevent implicit downloads. Original model loading
is bypassed (`model_path=''`) and replaced by `torch.load(weights_only=True,
mmap=True)` with the official `model.` prefix mapping and strict state matching.
Stop on unsupported safe loading/mmap, import/API/state mismatch. Never fall back
to unsafe loading, non-strict loading, guessed key remapping or an unpinned config.
This checkpoint's real tensor layout remains unverified until download is approved.

## Analytical bytes versus measured results

The exact storage identity is`4P +12A` bytes for FP32 stored parametersP, active
gradient/Adam-state parametersA:4P weights,4A gradients,8A first/second moments.
This excludes buffers, activation tensors, workspaces, context and allocator
fragmentation. When all330M nominal parameters are active, this is
**5,280,000,000bytes (4.92GiB)** before activations, already above the current~4GB
pilot cap.330M is a nominal scenario, not a counted checkpoint parameter total.
Unused pretraining-head/class-token parameters may receive no gradient or Adam
state; do not label16×all-parameters an exact mandatory total for that path.

The future report counts actual loaded parameters, active optimizer parameters,
parameter/gradient/state tensor bytes after the FIRST optimizer step, CUDA
allocated/reserved/peak values, NVML worker memory, host tree RSS and timings.
Analytical formulas remain under `estimated`; `measured_model_resources` is null
in the current inspection. Nothing here estimates teacher accuracy or asserts
10–20GPUh without measured step/validation/checkpoint/teacher-cache costs.

## Prospective execution envelope — not permission to run

Root must review final source/config/environment and supply a NEW explicit UTC
window of≤900s. Tool hard stops are600 child CPU seconds,600 worker wall seconds,
8GiB sampled worker process-tree RSS and4GiB newly owned output storage (runtime caches redirected into that root); the earlier
UTC deadline wins. Run parent and child on Linux CPUs8–9, all four numeric thread
variables1, Torch intra/inter1. No CPU0/2, services changes or other GPU work.

The root-supplied last-known GPU0 free memory is22971MiB; it is not newly measured
by this source audit. Proposed maximum allocator cap is12288MiB, global free-memory
reserve8192MiB. At launch compute
`min(12288,floor(0.9*(fresh_free_mib-8192)))`; stop if below8192MiB. At22971MiB
this gives12288MiB, leaving10683MiB nominal free. Recheck before CUDA allocation
and only reduce the applied cap if free memory shrank. Do not expand on OOM or
retry with another batch/precision/window. Use actual NVML total memory to set
Torch's per-process fraction, not a guessed physical capacity.

The supervisor requires NVML index0 and binds physical GPU UUID and both root-provided vLLM PIDs plus
`/proc` start times. Every~0.25s it verifies identities, global reserve, worker
NVML usage≤applied-initial-cap+1024MiB context allowance, host RSS/CPU/disk/window.
`nvidia-smi` calls have5s timeouts; sampling gaps are not a hard instantaneous
memory guarantee. Torch's allocator cap does not cover all CUDA context/library
allocations, hence the independent NVML reserve/usage checks. On failure terminate
ONLY this worker process group, preserve reports/log and exit; never touch vLLM,
restart services, kill other tasks or auto-increase the cap.

One first optimizer step followed by20 fixed steady steps uses a single seeded
uniform-noise microbatch and synthetic target. AdamW backbone/head LR1e-5/1e-4,
weight_decay0.01,foreach=False,fused=False; first-step allocation is mandatory.
No real audio, training dataset, key score, accuracy validation or checkpoint save.
Reported step durations include resource/finite-gradient instrumentation and are
not a clean training-throughput benchmark. Any later GPU-hour budget must name
this overhead and separately include epochs/steps/seeds, validation and caching.

## Executable interface and approval bindings

Safe inspection, local or on the server after source deployment:

```sh
python -m tools.preflight_musicfm inspect
```

It verifies all pinned snapshots/pointers and prints analytical state/shape,
proposed and installed package versions, source/guard/index SHAs and environment
fingerprint, with no Torch import or CUDA query. Future reviewed execution:

```sh
taskset -c 8,9 /TASK_LOCAL_OVERLAY/bin/python -m tools.preflight_musicfm \
  run-resource --config /NEW_TASK_ROOT_REVIEW_CONFIG.json
```

The not-yet-issued review JSON must provide tool/source/artifact/dependency-index
guard and protocol SHAs, exact environment fingerprint, explicit start/deadline UTC,
new absolute output root, verified local weights path, physical GPU UUID,
two distinct live protected vLLM PIDs, requested cap≤12288MiB/reserve≥8192MiB,
precision`bf16-autocast-fp32-parameters`,attention`eager`,steady_steps20. It is
validated before GPU access. The private worker additionally requires its live
supervisor process identity. There is deliberately no runnable authorization JSON
or execution window in this change.

Local tests exercise source/pointer integrity, no-Torch default inspection,
reflect/STFT/stride boundary shapes, reserve-policy refusal, time/source/service
validation, actual CPU pooling-axis behavior and lazy Adam moment allocation on
a small synthetic head, plus mocked NVML device/PID isolation. They do not
construct330M parameters or claim real MusicFM/binary/GPU compatibility.
