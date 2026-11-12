# MusicFM whole-song cost qualification exposed and repaired a precision cache bug

## Problem and decision

A short-window resource probe cannot establish the cost or correctness of a real
MusicFM key-training pipeline. The next step exercised actual cached fitting
waveforms, 32-microbatch optimizer updates and all 376 complete selection songs.
It found a precision-dependent state bug that the earlier 16-window probe missed.

The original B1/B16 composed check **failed**: maximum absolute pooled-logit error
**0.0012251087**, with five failing songs and zero changed keys. A proposed B1-only
run was stopped before model setup completed when independent source review found
that B1 itself could reuse BF16 rotary embeddings during FP32 evaluation. A new,
wrapper revision then passed the same fixed complete-song
gate: maximum error **3.230626e-7**, zero key changes, unchanged `1e-5` absolute
limit. The original failure and controlled stop remain preserved.

**This is a disposable cost/numerical study, not trained-key accuracy.** Each
complete run performed only three optimizer updates with synthetic targets. No
teacher weights were saved or reused. A real 8,280-update fit is distinct: it starts afresh from the original checkpoint,
binds its initial state to the probe, and includes complete-song evaluation and
the fixed continuation comparison. No real-fit accuracy result, stronger teacher,
distillation benefit or CPU deployment footprint is established here.

![Disposable update costs, full-song validation costs and retained numerical failure](cost-results.png)

The same 376 songs appear in both numerical series, ordered by the original run's
error. The dashed line is the unchanged absolute gate. No logarithmic display floor
affects gate calculation. The middle panel covers all 5,893 actual song windows;
it is not a single-window forward benchmark. [SVG](cost-results.svg) and the
[verified plotting source](../../../tools/musicfm_teacher_cost_report.py) are
available; reproduction commands appear below.

## Method

Both complete runs use the original pinned MusicFM-MSD checkpoint, original
24-kHz mono mel frontend and MSD global statistics, original Conformer encoder
and pretraining projection, final hidden layer's time mean and a seed-13 linear
24-key head. Class IDs use the shared `2*tonic_index + minor` mapping.
The measured environment retains Torch2.15.0+cu132 and NumPy2.5.4; exact module
paths/hashes and the complete source closure are in the raw configs.
The original checkpoint SHA is
`218b483a0256ddef736267425fabb166fd97008983696bb9270def464b47bded`.
There are 328,957,080 registered model/head parameters; checkpoint state elements
are a different quantity, documented in the [source/resource preflight](../musicfm-preflight/REPORT.md).

Parameters, frontend, gradients and Adam moments remain FP32. Training uses BF16
encoder/head autocast, TF32 off, eager attention, hard cross-entropy scaled by32,
finite loss per microbatch and finite accumulated gradients before optimizer
commit. AdamW uses backbone/head learning rates `1e-5`/`1e-4`, weight decay0.01,
betas0.9/0.999, epsilon1e-8, foreach/fused off. Python/NumPy/Torch training RNGs are
reset to13 after upstream construction's142 reset, strict checkpoint load and
head/device setup. Both runs' full initial state digest is
`d278e0e197d21d98f8a570be7fbb6195e79247c5dbfada2686c34c6e502a507d`.

Each cost run consumes exactly three groups of32 synthetic-target examples from
the predeclared original16 fitting IDs, using deterministic whole-track crop
sampling. It uses actual Adam allocation and updates, not shaped empty moments.
The normal provider returns exact FP32 crops from the hash-verified ordinary-disk
cache. It does not seek-decode each draw, use a full-song fitting LRU, silently
change waveform precision or activate the removed RAM cache.

The 376 evaluation songs contain126 Beatport and250 FMA IDs, with 5,893 windows.
No task scores are computed. Complete songs are tiled in actual12s contexts;
tails longer than1,024 samples use their own context, while smaller tails borrow
the final full context and contribute only their newly covered duration. The
FP64 mean weights each window by its exact support sample count. This covers the
song without providing whole-song attention or selecting harmonic windows.

The B1 reference and candidate both run the original frontend/normalization
individually at B1 in FP32. Candidate encoder batches combine only identical mel
shapes, with no padding or context change. Outputs are reordered before identical
FP64 aggregation. Both replay the actual separate Beatport/FMA evaluation calls,
including partial batches; this avoids qualifying a combined-panel schedule that
checkpoint selection would not use. The fixed gate is finite24-way outputs,
maximum absolute aggregated-logit error ≤`1e-5`, and zero changed argmax IDs.

The host is the shared RTX PRO6000 Blackwell server. Main worker affinity is
CPUs8–9. The executed CachedProvider uses one prefetch thread, inheriting that
affinity; numeric threads1. Four decoder threads and FFmpeg affinity10–13 belong
only to the explicit streaming fallback, not these disk-cache cost runs. ORT is
not used by this teacher probe. The per-process CUDA allocator cap is12GiB, with a global
8GiB free-memory reserve and all four protected vLLM parent/engine start identities
checked by the guardian. This is instrumented work on a shared serving host;
throughout-run foreign GPU utilization was not measured, so these are not isolated
throughput numbers.

## Results and timing boundaries

| Measurement | Original `ef3a735` | Corrected `051e4ada` |
| --- | ---: | ---: |
| First actual32-microbatch update |3.46544s |3.56225s |
| Second / third update |2.49380 /2.45074s |2.54529 /2.53705s |
| Input wait before first update |0.06916s |0.06181s |
| Mean of the two later update times |2.47227s |2.54117s |
| B1 complete376-song evaluation |247.44495s |233.73865s |
| B1-frontend/B16-encoder complete evaluation |154.76148s |147.80360s |
| Both validation passes, aggregate worker/child CPU |401.82286s |381.25761s |
| Full cache integrity scan |161.28303s |160.16988s |
| Model setup / load |7.77894s |8.10650s |
| Complete controller wall / aggregate owned CPU |583.60354 /514.88s |562.63631 /495.34s |
| Maximum CUDA allocated / reserved |12,161,753,088 /12,308,185,088B |same |
| Maximum sampled controller-owned process-tree RSS |3,314,368,512B |3,315,097,600B |
| Absolute composed-logit maximum |**0.0012251087: failed** |**3.230626e-7: passed** |
| Failing tracks / changed keys |5 /0 |0 /0 |

Update timing starts only after all32 inputs resolve and CUDA synchronizes, then
includes H2D, actual forward/backward/Adam, finite-state instrumentation and final
synchronization. Separate training-pipeline time includes initial provider wait
and overlap:8.48340s original,8.71063s corrected. Subsequent resolve waits were
approximately30–38µs in these tiny runs; this is not a corpus-wide I/O forecast.

Evaluation timers include cache slicing, H2D, original frontend, encoder/head,
aggregation and worker watch calls. They exclude the earlier full cache scan and
model setup. B1 outputs reach CPU per window; candidate outputs reach CPU per
flushed group. Controller wall additionally includes startup, guards, GPU queries
and other setup. CUDA byte high-water marks differ from sampled whole-process
NVML allocation; no sampled NVML peak is invented from the CUDA fields. Host RSS
is the research process tree, not a prediction of final CPU classifier memory.

The corrected steady mean ×8,280 updates plus120 complete B16 validation events,
one setup and a final B1/B16 selected-checkpoint audit gives an arithmetic planning
estimate of **10.92h**. It is **not measured fit duration or an11h promise**: only
two later updates informed steady timing, cache/locality and serving contention
may vary, and progress/scoring/checkpoint writes and other long-run costs are not
fully represented. No training horizon, full-song coverage or validation cadence
may be shortened to fit that estimate. Actual full-fit resource use and duration remain separately measured outcomes. The expensive per-draw full-decode projection
belongs to the explicit streaming fallback, not this ordinary-disk cache path.

## The failed gate and precision-cache repair

The original five failures were exactly the first five Beatport songs:
`1059355`, `1077935`, `11077`, `1147737`, `1274431`. Three have no tail; the other
two have nearly complete11.97/11.98s tails. All short tails ≤one second passed.
This contradicts attributing the failure to very short tails. The worst component
was class6 on `beatport:1059355`, B1logit0.24281444 versus B16logit0.24403955.
The remaining371 songs had errors on the ordinary~`1e-7` scale; FMA maximum was
3.230626e-7. Zero changed keys did not satisfy or relax the frozen numeric gate.

The [pinned Transformers source](../musicfm-preflight/evidence/transformers-conformer.py.txt.gz)
(decoded SHA`791d19879014102501be3760f0a3d7ecf9d7335333f662af8e69da27f0125a1a`)
returns the rotary cache whenever
sequence length matches (snapshot lines143–144), but constructs that tensor in
the hidden-state dtype (lines154–155). The lookup does not check dtype/device.
The final BF16 training crop has300 encoder positions. The first five B1
validation songs also only produce300-position contexts; the sixth song's11.2s
tail changes length and forces FP32 cache reconstruction. Candidate B16 is run
after that reference pass. Thus B1 could reuse a BF16 cache at the beginning of
its nominal FP32 evaluation. This source-level state defect fits the sharp first-
five failure boundary; the old run did not retain rotary-tensor dtype traces.

A B1-only cost attempt was therefore stopped through its own verified guardian:
SIGTERM, worker exit−15,55.45517s controller wall. Its terminal receipt records
no completed update/validation and no upstream setup directory. It is an incomplete
controlled stop, not an additional failed accuracy or completed timing experiment.
Its raw receipts are retained with the two complete runs.

The prospective repair is confined to the teacher wrapper. It strictly verifies
the pinned rotary cache contract, clears its length/tensor attributes together
on train/eval transitions and module device/dtype application, and checks stale
dtype/device on same-mode forward precision changes. Repeated same-mode evaluation
keeps a valid cache. Shared dependencies, original frontend, model parameters,
head, windows and schedule are unchanged. This also protects the reverse
FP32-evaluation→BF16-training transition; choosing B1 alone was insufficient.

CPU synthetic regressions reproduce a length-only cache, cover BF16→FP32same-
length→BF16, exact fresh-reference FP32 results, unchanged state/parameters,
same-mode precision changes, module moves, repeated evaluation and rejection of
malformed cache contracts. The corrected complete run had the same initialization,
inputs, module receipts and settings; only `experiments/musicfm_teacher.py` changed
within the bound numerical source closure. Its new passing result supports this
repair without rewriting the old failure. No disposable weights carry into a fit.

## Evidence and reproduction

[Evidence index](data/cost-index.json) binds deterministic gzip containing13
original small UTF-8 JSON files, each preserved byte-for-byte with its own SHA256.
Both complete results include all376 ordered pairs of24-way logits, context
sample/window counts, separate-domain schedule, measured source closures, module
receipts and initialization digests. No waveform, feature array, model weights or
new task labels are bundled. Controlled-stop receipts remain distinct.

Executed numerical commits are `ef3a735ebf79c22c7b87566b6433a31966d3dc7a` and
`051e4adaf5207abd5f83ad822dbf60f7c7abf412`; publication cherry-picks are not relabelled
as measured commits. Original result SHA is
`4037db0d9ef947485f17123ff577d1a33dc537091555653eb6e02711d14b1ef9`;
corrected result SHA is
`adac185c04aa3e34f8b7aad7e84b4614e65f87b32cc1ff194024782afa470733`.
The index and retained config/source hashes are authoritative if a file path moved.

The verifier needs only the Python standard library. Plotting uses the repository's
existing Matplotlib plotting environment; it adds no inference/training dependency.

```sh
python -m tools.musicfm_teacher_cost_report
python -m pytest -q tests/test_musicfm_teacher_cost_report.py
python -m tools.musicfm_teacher_cost_report --plot
```

The verifier checks compressed/decoded/raw hashes, complete vector shapes and
finiteness, every paired gate, whole-song counts, schedule identity, scope and
initialization bindings, source/data invariants, and controlled-stop status.
Tests damage vectors, contexts, schedules and evidence bytes to exercise rejection.
The inline PNG/SVG derive from verified evidence, not transcribed figure labels.

## Limits and next decision

There are no real task-training results here. Full12s contexts tile the entire song
but do not provide long-range attention. Beatport120s previews and broad FMA
Electronic cannot establish the primary5–10minute house/deep-house/melodic-house
acceptance target; only five generic House selection songs across three groups
currently cover that range. No calibration, confirmation or private DJ selection
was used. Internal teacher continuation still requires the separately frozen gain
against both K10 controls and Beatport nondecline; resource feasibility earns no
accuracy exception. The K10 expanded-data pair's failed2-point gate stays failed.

The research teacher is a GPU system. These timings do not qualify its model or
frontend for ordinary PCs, prove a compact student's eventual accuracy, or promise
CPU memory/latency. The next accuracy evidence must come from a separately
completed, source-bound fresh fit and its complete-song evaluations; no such result
is included in this publication.
