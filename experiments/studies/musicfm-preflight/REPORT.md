# MusicFM-MSD fits on the shared GPU for one 12-second training microbatch

The pretrained MusicFM-MSD model, with a new 24-class head and AdamW, loaded
strictly and completed 21 synthetic training updates on a shared GPU, peaking at
5.116 GiB of allocated CUDA memory. This shows resource feasibility for one
fixed 12-second microbatch only. It says nothing about key accuracy or the cost
of a full training run. Study date: November 12, 2026.

## Question

Before committing to MusicFM fine-tuning or distillation, does the actual
source, checkpoint and optimizer fit on a GPU that is also serving two other
models? Version declarations alone were not enough to answer this: the first
TorchAudio build failed to import (see Method).

## Method

### Model and probe

The original MusicFM source, statistics, configuration and weights are fixed in
the [initial protocol](PROTOCOL.md); the [CPU-build amendment](CPU_CANDIDATE_AMENDMENT.md)
records the later TorchAudio build choice. Configuration, source and environment
were fixed before the GPU run, and no setting was changed after seeing an outcome.

The checkpoint was loaded on CPU (`weights_only=True`, `mmap=True`) with strict
state matching against the exact official model before CUDA initialization.

The input was one uniform-noise `[1, 288000]` FP32 waveform at 24 kHz (12
seconds), reused for all 21 updates at microbatch size 1. The original frontend
(2048-point mel, hop 240, 128 bands, MSD normalization) produced 1,200 frames,
then 300×1024 hidden states. Preprocessing stayed FP32. The stock eager Conformer
and head ran under BF16 autocast, with FP32 parameters, gradients and Adam moments
and TF32 off. Hidden states were averaged over time and passed to a new 1024→24
linear head with a fixed synthetic target (class 0). The full official encoder
path ran, including computations the head does not use. No real audio, musical
label or task score was involved, and no new weights were saved.

The probe used one GPU, two CPU cores and one numerical thread, with fixed caps:
a 12,288 MiB CUDA allocator limit, at least 8,192 MiB of GPU memory left free
for the other services, 600 CPU seconds, 600 s wall, 8 GiB sampled host memory
and 4 GiB of new output. The two serving processes stayed up throughout and the
GPU was actively serving before launch.

### Dependency environment

| Build | Outcome | Child CPU / wall | Sampled peak RSS | Bytes on disk afterwards |
| --- | --- | ---: | ---: | ---: |
| PyPI TorchAudio 2.11.0 | Import failed: built for CUDA 13.0, while the shared Torch uses CUDA 13.2; the checkpoint was never downloaded | 4.932 s / 6.962 s | 601,124,864 B | 233,937,500 B |
| Official TorchAudio 2.11.0+cpu | Imports, dependency constraints and safe CPU checkpoint layout passed | 16.982 s / 30.669 s | 851,750,912 B | 1,523,119,630 B |

A stable Torch ABI did not imply CUDA-build compatibility. Rather than patching
the version check, we switched to the official CPU-only TorchAudio wheel (341,338
bytes). All other resolved wheels stayed identical; 19 direct and transitive
wheel identities are recorded. The environment reuses the shared Torch
2.15.0+cu132 and NumPy 2.5.4 unchanged, adding TorchAudio 2.11.0+cpu,
Transformers 4.57.6 and einops 0.8.1, installed offline with hash checks. Only
after imports passed was the 1,316,802,088-byte MSD checkpoint fetched and
checksum-verified.

Because TorchAudio is CPU-only here, the probe also tests that its mel frontend
operators run on CUDA tensors through shared Torch at this shape. It is not a
general claim about every TorchAudio CUDA operator.

## Results

The model plus new head has **328,957,080 parameters**. The checkpoint holds
**329,170,226 state elements in 465 tensors**; state includes buffers, so it is
not a parameter count. **316,367,896 elements** received gradients and Adam
state. First AdamW state allocation and 20 steady updates completed.

![Measured update timing and GPU allocation](resource-probe.png)

Left: update 0 (about 743 ms) includes first optimizer-state allocation and
first-use CUDA work; updates 1–20 sit near the 51 ms steady median. Right: Torch
allocated and reserved peaks level off near 5.1 and 5.7 GiB after the first
update, below the sampled NVML maximum. Model loading is outside the timer.
NVML values are sampled MiB; CUDA fields are bytes, shown in GiB. GPU utilization
from the other services was not sampled, so this is not an isolated benchmark.

| Measurement | Observed result |
| --- | ---: |
| First instrumented update | 0.743002 s |
| Median of 20 steady updates | 0.050997 s |
| Maximum CUDA allocated, all 21 updates | 5,493,349,376 B / 5.116 GiB |
| Maximum CUDA reserved, all 21 updates | 6,142,558,208 B / 5.721 GiB |
| Maximum sampled process GPU usage (NVML) | 6,654 MiB / 6.498 GiB |
| Minimum sampled free GPU memory on the device | 16,312 MiB / 15.930 GiB |
| Maximum sampled host RSS | 2,238,050,304 B / 2.084 GiB |
| Total wall / child CPU | 10.568 s / 11.065 s |
| Registered buffer tensor bytes | 951,056 B |
| Analytical FP32 weights + active gradients + two Adam moments | 5,112,243,072 B / 4.761 GiB |

The analytical row is `4P + 12A`, with P stored parameters and A active optimizer
elements. It excludes buffers, activations, workspaces, CUDA context and
allocator effects, so it is not a measured peak. The nominal 330M all-active
estimate was 5.28 GB; actual graph activity reduces it, but it still exceeds the
earlier ~4 GB planning cap before workspace and context are counted.

The setup log also records a warning that the `cache/torch/kernels` directory
could not be created, so kernel caching was disabled during the probe.

## Decision

MusicFM-MSD fine-tuning is feasible on the shared GPU at this microbatch size.
The next step is one finite MusicFM pilot on the v2 data, with bounded 24 kHz
waveform and crop I/O, fixed song-level validation aggregation and measured
validation and checkpoint costs. It uses the frozen v2 inner selection IDs (not
the older v1 FMAK 250) and excludes confirmation and calibration recordings. A
stronger teacher must show better accuracy before any student budget is spent.

The later [complete-song cost study](../musicfm-key-pilot/REPORT.md) found a
rotary-embedding precision bug between training and evaluation that this
fixed-window probe could not reveal, and qualified a repair. This probe therefore
does not establish full-song batching equivalence.

## Limitations

- One repeated synthetic input, one microbatch shape and 21 updates. No key
  accuracy, trained teacher, whole-fit GPU hours, standalone throughput, CPU
  performance or production readiness follows.
- The GPU was shared with active serving, and memory is sampled; brief peaks
  between samples can be missed.
- The input stayed resident on GPU, with no audio decoding or augmentation, so
  real training will cost more per update.
- The primary acceptance domain is 5–10 minute house, deep-house and
  melodic-house tracks. This probe has no labelled evidence for it. A real pilot
  must evaluate whole songs and report duration, genre and group coverage;
  generic FMA Electronic averages and 120-second Beatport previews cannot
  establish it. Long percussive intros and outros may dilute key evidence across
  a whole song, which a pilot should record with per-window diagnostics.
- Private DJ audio stays excluded from training and selection.

## Reproduce

```sh
uv run python -m tools.musicfm_resource_report --plot
uv run pytest -q tests/test_musicfm_preflight.py tests/test_musicfm_provision.py
```

The report tool verifies the recorded configuration, environment, shapes, steps,
precision and resource limits before regenerating the PNG and SVG figure. The
[evidence index](data/runtime-index.json) covers 90 original files (failed
attempts, the import failure, wheel and lock records, all 21 update records and
resource samples) with checksums; it contains no checkpoint, wheel or audio
bytes. The executed provisioning and probe source revisions are recorded there
as well.
