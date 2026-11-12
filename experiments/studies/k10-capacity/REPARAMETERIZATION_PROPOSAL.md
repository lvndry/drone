# Training branches with an unchanged deployed K10 graph

Status: local prototype only; no task fit, selection scoring, resource permit or
release integration. The completed finer-grid experiment remains a failed route.
This proposal must not become an automatic architecture sweep or a rescue of that
candidate. The teacher and context experiments retain priority and unchanged gates.

## Question and prior evidence

Can a redundant training parameterization improve compact K10 without increasing
its inference graph? [RepVGG](https://github.com/DingXiaoH/RepVGG) demonstrates the
general mechanism in image classification: parallel linear branches and frozen
batch normalization can be combined into one convolution. Its image accuracy and
timings do not establish musical-key gains. Unlike the finer-grid candidate, this
prototype preserves all original channel counts, pooling, frequency resolution,
receptive fields, time aggregation and readout capacity.

The fixed prototype adds exactly one Conv1x1/BN branch in parallel with each of
K10's six Conv3x3/BN pairs, summed BEFORE the original ELU. There are no additional
activations, identity branches, attention, distillation, loss changes, width
changes or pitch-folding operations. It is inspired by RepVGG, not its full model
or training recipe. Each side BN scale starts at zero. The main branch is an
independent copy of the control; construction preserves the CPU RNG state.
Construct from a CPU source, using one advancing forked CPU RNG stream for all
side kernels, then explicitly move the completed candidate to the training device.
At initialization, finite input logits and main gradients match the control,
including training dropout with a shared RNG state. Side BN scales can learn on
the first step; side convolution gradients can start after those scales change.
Subsequent trained states and stochastic gradients are not claimed identical.

This changes optimization geometry and regularization as well as BN statistics.
It does not add deployed expressivity. The zero-scale start and ordinary AdamW
are a hypothesis, not the recipe validated in the original RepVGG paper. Lack of
gain would reject this recipe, not structural reparameterization in general.

## Conversion and local evidence

For frozen BN, fold each branch using its own running mean/variance, epsilon,
scale and offset. Center-pad the 1x1 kernel to 3x3, then sum kernels and biases.
The exported copy contains six Conv3x3 operations and no branch or BN modules;
it retains the original ELU/pooling/dropout-in-eval/time-mean/linear structure.
Conversion refuses training or mixed modes and never mutates the source model.
This is algebraic equivalence subject to floating-point rounding, not bitwise
equivalence. Padding boundaries are included in the tests.

| Model | Parameters |
|---|---:|
| Ordinary unfused K10 | 109,128 |
| Candidate during training | 117,528 |
| Candidate after BN/branch folding | 108,680 |

An ordinary K10 with its own BN folded also has 108,680 parameters and the same
convolution shapes. Any later speed comparison must fold BOTH models; comparing
a fused candidate with an unfused baseline would confound the mechanism. Equal
graph shapes do not prove equal measured runtime or complete-pipeline memory.

The twelve local tests cover preserved initial function/CPU RNG/source tensors,
initial training logits/main gradients/running statistics, nonzero side gradients,
nontrivial fusion with randomized BN moments/affines in FP64, FP32 fusion after
three actual optimizer steps, serialization, shortest/odd/long lengths (8,65,320,
1615 frames), and rejection of training/mixed modes and noncanonical width.
These use synthetic inputs and have no task accuracy evidence. Run:

```sh
uv run python -m pytest -q tests/test_k10_reparameterization.py
```

## What must happen before a fit

First finish and interpret the active context/teacher experiments. Have an
independent review challenge this candidate and the conversion contract. Then,
only if still justified, freeze a separate source/config-bound proposal using
the same expanded fitting data and old development panel, fresh matched GPU
control, common seed/initial main tensors/draws/crops/optimizer/selector/horizon,
and no reserved/private data. Specify the accuracy/continuation gates BEFORE any
candidate task output. The failed historical control guard from the finer-grid
pair is preserved and cannot be reinterpreted as a pass.

The new branches create extra training activations and convolutions; the previous
resource qualification is insufficient. Qualify synthetic training, AdamW state,
full true-length evaluation AND unfused/fused replay under new reviewed caps and
protected identities before any task fit. Preserve both vLLM instances and live
jobs. There is no permission implied here to weaken reserves, change precision,
alter batches, retry or run more seeds.

Any future task result must retain both unfused/fused selected states, complete
paired development predictions, strict checkpoint replay, actual numeric conversion
errors and changed argmaxes. Accept a deployed candidate only if its frozen full
selection replay has zero changed argmaxes and bounded finite logits; a local
synthetic tolerance alone cannot authorize deployment. Accuracy confirmation and
ordinary-PC CPU qualification remain separate later decisions. No new fit follows
from this document.

### Independent review constraints

The fused state has no BN buffers and is intentionally incompatible with the
current production checkpoint loader. The prototype includes an independent
`FusedKeyCNN` factory and a strict synthetic state reload without access to the
training model. A separately versioned experimental artifact/metadata contract
and fresh-process replay would still be required before any actual fit/export;
do not silently load this state as a production `KeyCNN` checkpoint.

At batch32/T320, six extra branch feature maps contain approximately420MiB of
FP32 values in total (120+120+60+60+30+30MiB). Their lifetimes and actual allocator
peak are not yet measured; this is NOT a prediction of additive peak memory or
an observed OOM. The completed pair's owned-NVML peak was2,648,702,976B, leaving
only290MiB to the unchanged2.75GiB ceiling. A corrected resource qualification
may therefore reject this route on the present shared GPU1. Defer it if so;
neither the prototype nor enthusiasm for it authorizes a precision/batch/reserve
change. The zero-scale side branches still allocate tensors during training.
