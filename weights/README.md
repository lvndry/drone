# Weights

`drone-v1` — the K10 (MIREX 2026) port, confirmed by the 3-seed gate.

| file | contents |
| --- | --- |
| `drone-v1.pt` | PyTorch checkpoint, `format_version: 2`, feature contract `cqt24-8oct-c1-pad4-hop4096-v2`, 109,128 parameters, run `w1.0-seed13-20261111-185528-lr7fvv4k` (seed 13) |
| `drone-v1.onnx` | ONNX export of the same weights, opset 18, dynamic batch; CPU parity verified at export time |

Held-out GiantSteps (604 tracks, single evaluation per seed):

| seed | exact | MIREX |
| --- | --- | --- |
| 13 | 68.5% | 74.8% |
| 37 | 66.1% | 73.0% |
| 73 | 65.7% | 72.9% |
| **mean** | **66.8%** | **73.58%** |

Reference: native libkeyfinder scores 69.35% MIREX on the same files.

Use the default resolution (packaged `drone/weights/drone-v1.pt`, then the
checkout's `weights/drone-v1.pt`) or point `DRONE_WEIGHTS` at your own
checkpoint. Do not distribute old unversioned checkpoints: they do not describe
the current feature or label contracts.
