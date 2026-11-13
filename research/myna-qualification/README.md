# Isolated Myna environment

This Linux/Python 3.12 environment pins the runtime used for Myna qualification.
The exact Torch/Torchaudio CUDA pair and all dependencies are recorded in
`pyproject.toml` and `uv.lock`; they do not replace Drone's normal environment.

The consolidated [study report](../../experiments/studies/myna-key-pilot/REPORT.md)
contains the problem, methods, measurements, failed approaches and inline
figures. Execution requires explicit source/input hashes and reviewed resource
bounds through the qualification tools; installing this environment alone does
not authorize training or establish model accuracy.
