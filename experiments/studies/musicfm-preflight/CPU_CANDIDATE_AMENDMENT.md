# Prospective TorchAudio CPU-build provisioning amendment

The first isolated overlay failed its real import gate: official PyPI
TorchAudio2.11.0 was compiled for CUDA13.0, while the unchanged shared
Torch2.15.0 uses CUDA13.2. The version check was not bypassed. All v1 source,
wheel identities, install/error logs and failure report remain preserved; no
checkpoint download followed that failure.

Root authorizes a distinct CPU-build candidate with the same bounded provisioning
policy and fixed CPUs8–9, no GPU visibility and no model construction/inference.
New root: `/home/lysk-admin/drone-experiments/musicfm-preflight-20261112-cpu-v2`.

Pin **torchaudio2.11.0+cpu** explicitly, not a normalized/stripped version:

- Official index:`https://download.pytorch.org/whl/cpu`.
- Official index resolution returned CDN
  `https://download-r2.pytorch.org/whl/cpu/torchaudio-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl`.
- Verified file size341338bytes.
- Verified SHA256`2354248848d06a9ae1e7a12165f800f0dda7df60ecac9fca892322b722b922c0`.

The resolver's official CPU-index metadata omitted size, so a bounded preliminary
wheel download supplied the verified byte count. Preserve that receipt and raw
CPU lock. Compose a candidate lock by replacing ONLY the v1 TorchAudio package
entry with this exact CPU build; preserve all other resolved distribution and
wheel identities. Recheck actual dependency constraints after offline install.
The wheel origin allowance expands only for this exact TorchAudio CPU build and
the two official index/CDN hostnames. No Torch/NumPy selection or installation.

The MusicFM mel/decibel frontend calls core Torch spectrogram/matmul/log/clamp
operators in the pinned TorchAudio source. It is reasonable to expect CUDA tensor
execution through shared Torch even with a CPU-only TorchAudio extension, but
that remains an inference until a separately authorized GPU resource probe.
This stage establishes imports and safe CPU checkpoint layout only. No frontend
formula, official CUDA guard or model source is modified.

The same import-first checkpoint gate,2GiB payload/3GiB storage+sampledRSS/
600 child CPU/1200 wall limits apply. No GPU probe is authorized by this amendment.
