# Prospective GPU1 resource qualification — no task fit

Preparation only. This protocol defines one synthetic feasibility probe, not a
launch permit, an accuracy comparison or proof of unaffected serving latency.
No existing experiment, GPU0 reserve, vLLM setting or shared environment changes.

## Question

Can ordinary K10 and the fixed 108,936-parameter finer-grid candidate execute the
intended training/evaluation recipe in GPU1's currently spare memory? A successful
probe would justify preparing a fresh GPU1 baseline/candidate accuracy pair. The
running CPU320 model would then be contextual evidence, not the matched GPU control.
If either model fails, preserve an inconclusive resource result and use the CPU
option or wait for GPU0; no smaller-batch or larger-budget rescue follows.

GPU1 is the RTX5090, UUID `GPU-9c2004f9-0d41-d7c7-7075-863335543ee8`.
The observed free memory is 9,508 MiB. vLLM uses a .67 utilization target,
FP8 KV cache, prefix caching, max 8 sequences and max model length 32,768.
Those settings do not establish a hard runtime cap or static allocation.
Installed vLLM version/effective defaults were not verified; no such assumption
is needed for a guarded empirical probe. Do not access credentials or restart it.

## Frozen resource policy

- One five-minute wall window, including a 30-second passive baseline. Aggregate
  owned CPU cap 1,200 seconds, whole-tree host RSS 4 GiB, output 64 MiB.
- Admission requires at least 9 GiB free on the exact GPU1 UUID, stable protected
  process starts and global host/disk reserves. CUDA visibility is GPU1 only.
- Each model gets a separate fresh worker. Torch's allocator cap is exactly 2 GiB
  divided by its measured total device bytes. This does not cap all CUDA memory.
  Owned NVML process-memory ceiling is 2.75 GiB.
- GPU1 hard free-memory floor is 6 GiB, with early abort below 6.5 GiB. The 512 MiB
  difference is a reaction margin, not an allocation interlock. Retain all samples
  and any observed overshoot. A sensor failure or stale sample also stops the probe.
- A separate guardian samples at most 0.2 seconds apart using bounded queries;
  workers check before expensive operations. Timeouts, nonfinite values, OOM,
  threshold violations or protected-process changes fail closed. Stop only the
  owned, start-bound process group with short TERM/KILL cleanup; confirm memory
  recovery before the second worker. No restart, retry or cap/precision changes.
- Proposed CPU affinity 10–11, two Torch intra-op/one inter-op thread, numeric-library
  environment threads one and nice19. Validate distinct physical cores and overlap
  against current workers before sealing. Both vLLM processes remain unrestricted.
- Keep GPU0's existing 8 GiB guards and all live source/configuration/deadlines
  unchanged. Preserve both vLLM start identities and their engine children; monitor
  current teacher/context starts without signaling or restarting them.

## Numerical workload and retained evidence

Baseline first, then finer-grid candidate, with independent fresh initialization
at seed13. Use true batch32×192×320, FP32 weights/gradients/evaluation, TF32 off,
cuDNN benchmark off/deterministic on and fixed AdamW defaults. Complete three
synthetic-target optimizer updates, including train/eval transitions, so optimizer
state and backward allocations are observed. No real reference key enters the loss.

Build synthetic selection arrays from only the admitted 376 ordered ID/frame-count
metadata. Exercise the same true-length batch4 grouping, including odd, short and
maximum lengths; do not replace full-song evaluation with training crops. Check
finite logits, gradients, state and complete input/frame coverage. Discard transient
argmax values; compute/retain no task scores, predictions or checkpoints.

Synchronize CUDA before times and peak readings. Retain source/runtime/input-shape
hashes, initial tensor digests, update counts, phase timings, Torch allocated and
reserved peaks, host/aggregate CPU peaks, owned NVML memory and global minimum free
memory. Private final approval must bind exact source/configuration and a finite
UTC window before execution. No audio, reserved/calibration/private inputs or new
weights are acquired. No new dependencies are installed in shared environments.

Polling cannot prevent every allocation burst or guarantee vLLM tail latency;
nice19 governs CPU scheduling, not GPU priority. Existing passive health metrics
may be observed if available, without issuing inference/load-test requests. Success
means feasibility under observed load only. A later task pair needs its own source
review, complete-cost projection, fixed accuracy gates and sealed authorization.
