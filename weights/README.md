# Weights

`drone-v2` is the default release. `drone-v1` stays available for comparison
and for hosts pinned to it. Both use the same architecture (109,128
parameters), feature contract `cqt24-8oct-c1-pad4-hop4096-v2` and label schema
`tonic-times-two-plus-minor-v1`, so they are interchangeable files.

| File | Contents |
| --- | --- |
| `drone-v2.pt` | PyTorch checkpoint, `format_version: 2`. Seed 42, epoch 88 of the shift12 recipe, with its configuration and 6,263 training IDs in `metadata` |
| `drone-v2.onnx` | ONNX export, opset 18, dynamic batch and time; matches the checkpoint within 1.7e-5 |
| `drone-v2.json` | The ONNX export's contracts and parity error |
| `drone-v2.safetensors` | Every float32 tensor under its PyTorch name, for native hosts |
| `drone-v1.pt`, `drone-v1.onnx` | The previous release, unchanged |
| `MODEL_CARD.md` | The Hugging Face model card |

Exact accuracy, from original audio through the packaged `Predictor`
([evaluation](../experiments/studies/drone-v2-release/REPORT.md)):

| Panel | libkeyfinder 2.2.8 | drone-v1 | drone-v2 |
| --- | ---: | ---: | ---: |
| GiantSteps, original labels (604) | 60.4% | 68.5% | 69.9% |
| GiantSteps+ expert labels (571) | 69.9% | 77.4% | 80.7% |
| GiantSteps+ expert labels, single-key tracks (457) | 76.2% | 83.4% | 88.0% |
| FMAKv2 `valid` (250, mixed genres) | not run | 62.4% | 72.4% |
| FMAKv2 `holdout` (250, mixed genres) | not run | 68.4% | 77.6% |

GiantSteps was used during development of both releases (never for training).
drone-v1 was trained on 1,363 Beatport previews; drone-v2 on 4,988 FMAKv2 and
1,275 GiantSteps-MTG-Key tracks with ensemble distillation and pitch shifts of
up to 12 semitones ([study](../experiments/studies/frontier-distillation/REPORT.md)).

The package loads `drone/weights/drone-v2.pt` when installed, then the
checkout's `weights/drone-v2.pt`. Set `DRONE_WEIGHTS` to use another checkpoint,
for example `weights/drone-v1.pt`. Do not distribute old unversioned
checkpoints: they do not describe the current feature or label contracts.
