# Authorized isolated dependency and CPU checkpoint stage

Root authorizes task-local dependency provisioning/import verification, then
conditional download of the pinned official MSD checkpoint and safe CPU tensor
layout inspection. No CUDA initialization, model construction, forward/backward,
training, score or source/API workaround is authorized. Active vLLM serving is
preserved; this stage has no GPU visibility.

Use dedicated server root
`/home/lysk-admin/drone-experiments/musicfm-preflight-20261112-v1`.
Read-only1s CPU sample found CPUs8/9 about3%/1% busy; use that fixed pair with
OMP/MKL/OpenBLAS/NumExpr1, leaving4–5 for the independently running cached audit.
Outer supervisor caps600 aggregate child CPU /1200 wall seconds,3GiB owned
storage and sampled RSS. Wheel+checkpoint payload≤2GiB; all caches, temporary
files and installed packages stay under the new root. Before/after disk snapshots
are retained. Root's09:15 metadata showed~65.6GB free disk and GPU0 actively
serving; neither GPU is touched.

Resolve only torchaudio2.11.0/transformers4.57.6/einops0.8.1 and their dependency
closure with uv0.11.1. Explicitly exclude Torch/NumPy from resolution; preserve
shared Torch2.15.0+cu132 and NumPy2.5.4. Capture complete pylock metadata, select
compatible CPython3.12/Linux wheels by packaging tags, download exact URLs/sizes/
SHA256, then install offline with --no-deps/--require-hashes into a new overlay.
Only a task-owned `.pth` references the shared base site-packages read-only.
No install/uninstall operation targets the base. Exact effective dependency
requirements are checked against the overlay+base combination afterward.

Baseline and final base receipts bind Torch/NumPy module paths, source/extension
hashes, versions and full distribution inventory. Import receipts require the
same protected module paths/bytes and no CUDA initialization, direct version
pins, dependency constraints and actual upstream class signatures. CUDA devices
are masked; lazy CUDA initialization raises an error. Import public MusicFM
class definitions and pinned config only, without constructing the model.

Only after imports pass, download revision4513b38's MSD checkpoint, exactly
1,316,802,088bytes/SHA218b483a0256ddef736267425fabb166fd97008983696bb9270def464b47bded.
Use CPU torch.load(weights_only=True,mmap=True), never unsafe fallback. Record
all state tensor names/shapes/dtypes/storage sizes, top-level keys and whether
all keys match the future loader's official model. prefix. State tensor elements
are not automatically parameter counts or active optimizer counts. No model is
constructed and no tensor value quality/accuracy is scored.

Preserve all errors. First resolver attempt rejected redundant --no-build with
--only-binary; its log remains resolve.log. A distinct resolve-v1.log removes only
that CLI conflict and keeps binary-only policy. Any package import, ABI, config,
safe-loader or tensor-layout incompatibility stops the stage for root review;
no version/source/API substitution or automatic workaround.

After success, prepare but DO NOT execute the GPU probe review config with exact
overlay fingerprint/module paths, source/protocol/library hashes, physical GPU0
UUID and both protected vLLM process start identities. The probe still requires
a fresh root-approved UTC window. CPU provisioning results do not establish GPU
fit, throughput, numerical task accuracy or a usable trained teacher.
