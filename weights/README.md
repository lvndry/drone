# Weights

Trained drone weights land here:

- `drone-v1.pt` — PyTorch state dict (the model behind `drone.predict`)
- `drone-v1.onnx` — same model exported for runtime-agnostic inference (euphoria, mobile, edge)

The run record (data versions, hyperparameters, seed, compute, timings, and the
held-out GiantSteps 604 evaluation against libkeyfinder / Mixed In Key /
Korzeniowski & Widmer) is written to [`../TRAINING.md`](../TRAINING.md) after
each run so every shipped weight traces back to an exact, reproducible run.
