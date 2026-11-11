# Weights

No pretrained weights are included yet. Train a corrected model with:

```bash
uv run train.py train --model-size small
```

The selected checkpoint is written to `data/runs/<run>/drone.pt`; its run record
is alongside it in `run.json`. Pass that path to `drone.Predictor` or set
`DRONE_WEIGHTS`. For a default artifact, place a verified compatible checkpoint at
`weights/drone-v1.pt`. Wheel builds include this directory as `drone/weights`.

Small, medium, and large artifacts must each be trained and evaluated independently.
Do not distribute old unversioned checkpoints: they do not describe the corrected
feature or label contracts. ONNX exports and their metadata can be generated with
`uv run --extra export python -m drone.export <checkpoint> <output.onnx>`.
